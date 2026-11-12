# Qwen3.5-397B-A17B in fp8 on trn1.32xlarge (32 NeuronCores, TP=32): design

Target: `Qwen/Qwen3.5-397B-A17B` (revision `8472618112abcbd45acbcdc58436aff4233c23f7`, 94 safetensors shards, 806.8 GB
bf16, 403.4 B parameters including MTP and the vision tower) on one trn1.32xlarge: 16 Trainium chips = 32 NeuronCores,
16 GiB HBM per core. This is the 35B port (docs/neuron-qwen35-moe.md: same `qwen3_5_moe` architecture, same graphs,
prepared format and runtime backend) generalised from TP=2 to TP=32. Read that document first; only the differences are here.

## 1. Geometry (config.json)

| | 397B-A17B | 35B-A3B (Ornith REAP-50) |
|---|---|---|
| layers | 60 = (3 gdn + 1 gqa) x 15 -> 45 gdn, 15 gqa | 40 (30 / 10) |
| hidden | 4096 | 2048 |
| gdn | 16 key heads x 128, 64 value heads x 128 (4 value heads per key head), conv kernel 4 | 16 / 32 (2 per key head) |
| gqa | 32 q heads x 256 (+ output gate), 2 KV heads x 256, rotary 64 dims, theta 1e7 | 16 / 2 |
| MoE | 512 experts, top-10, intermediate 1024; shared expert 1024 | 128, top-8, 512; 512 |
| vocab | 248320, untied head | same |

Text-model parameters (what is served): experts 60 x 512 x 12.58 M = 386.5 B (97.5 %), gdn blocks 45 x 118.0 M = 5.31 B,
gqa blocks 15 x 104.9 M = 1.57 B, shared experts 0.75 B, routers 0.13 B, lm_head 1.017 B, embeddings 1.017 B (host).
MTP (`mtp.*`) and vision (`model.visual.*`) are ignored, as for the 35B.

## 2. TP = 32, one rank per NeuronCore (decision)

**TP=32 with every expert split along its intermediate axis (ETP = 32, EP = 1)**, replicated KV heads and replicated
linear key heads. Alternatives considered:

* *TP=16 with two cores per chip as one rank*: a rank is one NeuronCore to libnrt (one NEFF instance, one HBM); two cores
  cannot back one rank's tensors. TP=16 on 16 of the 32 cores needs 23.5 GiB of weights per core (table below): does not fit.
  TP=16 x EP=2 (experts 0-255 on even cores, 256-511 on odd) would fit but makes the two cores of a rank pair run different
  data with a data-dependent load (0..10 selected experts each): two NEFF variants or a 10-slot worst case on both, for
  no saving in collectives (all 32 cores still take part in the all-reduce).
* *Expert parallel (EP=32: 16 whole experts per core)*: the same memory, but each core must be compiled for the worst
  case (all 10 selected experts on one core: 10 x 12.6 MB read per layer instead of 10 x 0.39 MB), the load is
  data-dependent, and prefill (`MOE=all`) gains nothing. ETP keeps every rank on the same NEFF with the same static
  work: per layer and rank the gather reads 10 x (64 + 32) rows x 4096 = 3.9 MB.
* **KV heads (2) < TP**: rank r holds q head r (hq = 1, with its output gate) and a **replica of the KV head it attends
  to** (`kv_of(r) = r // 16`): `k_proj` / `v_proj` rows [256 kv, 256 (kv+1)), and its own copy of that head's paged K|V
  cache. No sequence split and no extra collective: the attention is rank-local exactly as at TP=2. Cost: each KV head's
  cache exists 16 times (once per core that uses it), 15.36 KB per token per core - the per-core figure is what matters
  and it is the same as it would be for any layout in which a core attends over one KV head.
* **Linear key heads (16) < TP**: rank r holds value heads 2r, 2r+1 (hl = 2) and a replica of their key head
  (`key_of(r) = r // 2`, hk = 1): q / k rows of `in_proj_qkv` and the matching conv channels are duplicated on the two
  ranks sharing a key head (2 x 128 x 4096 fp8 = 1 MB per layer per rank). The recurrent state is per value head, so
  nothing is duplicated there.
* `out_proj` / `o_proj` / shared-expert down / expert down are row-parallel (column slices) + one all-reduce per
  sub-block, as at TP=2: **2 all-reduces of a [T, 4096] f32 vector per layer, 120 per token**, each over all 32 cores.
* Router (`gate.weight` [512, 4096] f32) replicated on every rank; top-10 on the device, identical on all ranks.
* `lm_head` by vocabulary rows: 7760 rows per rank (248320 / 32, no padding); the host concatenates 32 logit shards.

Code: `Geo.set_parallel` (qwen35_graphs.py) now derives `hk`/`kvl` = max(1, heads / TP) with `key_of` / `kv_of`;
`qwen35_prepare.shard_layer` slices by them. The graph functions are unchanged (they only see per-rank counts), so the
TP=2 35B path is bit-identical to before (tests below).

Per-rank tensors at TP=32 (fp8 codes + f32 row scales):

| tensor | per-rank shape |
|---|---|
| gdn `qkv` | [512, 4096] (128 q + 128 k of key head r//2, 256 v) ; `conv` [512, 4] f32 |
| gdn `z` / `a`, `b` / `o` | [256, 4096] / [2, 4096] / [4096, 256] |
| gdn state | `kstate` f32 [slots, 2, 128, 128], `cstate` f32 [slots, 512, 3] |
| gqa `q` (q + gate) / `k`, `v` / `o` | [512, 4096] / [256, 4096] / [4096, 256] |
| gqa KV | bf16 [pages, 2, 1, 16, 256] = 16 KiB per page per layer |
| moe `egu` / `ed` | [512, 64, 4096] / [512, 32, 4096] fp8, scales [512, 64] / [512, 32] |
| moe shared `sg`, `su` / `sd` ; `rw` | [32, 4096] / [4096, 32] ; [512, 4096] f32 |
| head `w` | [7760, 4096] fp8 |

## 3. Memory per core (exact: `neuron/qwen35_budget.py`, bytes the prepare step writes / the runtime allocates)

| | TP=32 fp8 | TP=32 bf16 | TP=16 fp8 |
|---|---|---|---|
| gdn block (x45) | 4.26 MB | 8.45 MB | 7.42 MB |
| gqa block (x15) | 5.28 MB | 10.50 MB | 8.43 MB |
| moe block (x60), of which experts / router f32 | 210.35 MB (201.52 / 8.39) | 411.86 MB | 412.27 MB |
| **weights, 60 layers** | **12.007 GiB** (12 891 979 680 B) | 23.515 GiB | 23.466 GiB |
| head shard | 0.030 GiB | 0.059 | 0.059 |
| KV, 4096 pages = 64k tokens (15.36 KB / token) | 0.938 GiB | same | same |
| GDN state, 8 slots (6.2 MB per sequence) | 0.052 GiB | same | 0.102 |
| **total** | **13.03 GiB of 16** | 24.6: no | 24.6: no |

**Verdict: the whole 512-expert model fits in fp8 at TP=32 with a 64k-token KV pool, 13.03 GiB of the 16 GiB HBM per
core**, leaving ~3 GiB for the NEFFs' own scratch and I/O tensors. Nothing has to be cut for 64k. What does not fit:

* the config's full 262144-token context: KV 3.75 GiB -> 15.84 GiB, no margin. **Context cap: 64k (4096 pages) as built;
  128k (8192 pages, 1.875 GiB KV, 13.97 GiB total) is the practical maximum** and needs the 8192-page gqa bucket to
  load next to the others (to be confirmed on the device); beyond that the KV must go fp8 (halves it; the 35B doc notes
  the bf16 kernel is the validated one) or the router to bf16 (lossless, the checkpoint is bf16: saves 0.23 GiB).
* bf16 anything for the experts. bf16 attention blocks alone would cost only +0.27 GiB per core (45 x 4.2 + 15 x 5.2 MB)
  and is available as a quality lever if fp8 attention turns out to matter; the build here is all-fp8 as asked.

Host: the prepared set is 32 x 12.04 GiB + 1.9 GiB embeddings = ~387 GiB on the instance store; host RAM (512 GB) holds
the embedding table (mmap) and one layer file per rank thread during the load.

## 4. Decode cost model

Per token and rank the launches read: 45 x (4.26 + 12.7) + 15 x (5.28 + 12.7) MB + 31.8 MB head = **1.06 GB** (12.7 MB
moe = 3.93 gathered experts + 0.39 shared + 8.39 router): about the same bytes per rank as the 35B at TP=2 (1.22 GB), so
the per-launch device times should be near the 35B's (gdn 0.34 ms, gqa 0.38-0.53, moe-gather 0.25 ms) and the token is
again fixed-cost bound: **120 sub-block launches + head**, each with one all-reduce over 32 cores / 16 chips. Estimate
before measuring: 120 x (0.3 ms device + 0.26 ms host launch overhead + C) + ~1 ms head, C = the 32-core all-reduce
latency (section 5): ~70 ms/token at C ~ 0, ~190 ms at C = 1 ms. The measured numbers are in the status log.

## 5. Collectives (measured on the instance, 2026-11-12)

`neuron-ls`: 16 devices x 2 cores x 32 GB, each device linked to 4 neighbours (a 4 x 4 torus over NeuronLink), two NUMA
nodes. `nccom-test -r N allr -d fp32 -n 200 -w 20` (libnrt / collectives 2.33.10.0), average latency of one all-reduce:

| cores | 16 KiB (= one [1, 4096] f32 row) | 64 KiB (4 rows) |
|---|---|---|
| 2 (one chip) | 2.6 us | 4.1 us |
| 8 | 7.8 us | 12.0 us |
| 16 | 8.4 us | 12.2 us |
| **32 (all 16 chips)** | **10.2 us** | **12.7 us** |

(`-r 4` fails inside neuron-bench with this tool version; not pursued.) So the 32-core all-reduce of the residual costs
4x the single-chip one but is still ~10 us: 120 all-reduces per token = **1.2 ms per token of pure collective time**,
negligible next to the launch overhead. The in-graph all-reduce was 44-76 us at TP=2 (docs/neuron-nki-fusion.md) against
2.7 us in nccom-test, so the number to trust is the per-launch time of the compiled sub-blocks (status log). No
all-to-all is needed by this design (ETP + replicated heads: the only collective is the residual all-reduce).

## 6. Prepare at this scale

`qwen35_prepare.pack_layer`: the expert tensors of one layer (512 x [2048, 4096] + 512 x [4096, 1024] bf16 = 12.9 GB)
are read in chunks of 32 experts, quantised **once** (row scales are per gate/up output row and per down input channel,
i.e. independent of the rank slice) and then sliced into the 32 ranks' [512, 64, 4096] / [512, 32, 4096] pieces;
`--jobs` packs several layers in parallel worker processes, a `.done/Lnnn.json` marker per layer makes the run
resumable. `tests/test_qwen35_pack.py`: bit-equal to the per-rank `shard_layer` reference (TP=32 and TP=2; dict and
safetensors paths).

## 7. Status log

**Gate (a), 2026-11-12 07:13 UTC: instance, toolchain, checkpoint**
* `i-03935aa4aec5c547c` (trn1.32xlarge, ap-southeast-2c, capacity block `cr-0bc51abb2b32a0f50`), AMI
  `ami-0231f8b9dcafcd361` (Deep Learning AMI Neuron Ubuntu 22.04 20260115), root gp3 300 GB, 4 instance-store NVMe in
  RAID0 (xfs, 7.0 TB) at `/mnt/scratch`, 128 vCPU, 495 GiB RAM. SSM only.
* The DLAMI ships an OLDER stack than the pin (neuronx-cc 2.22.12471, torch-neuronx 2.8.0.2.11, nki 0.1.0, runtime-lib
  2.29.40.0, collectives 2.29.41.0, tools 2.27.33.0, dkms 2.25.4.0, python 3.10). Installed the pinned set instead
  (neuron/requirements-neuron-build.txt): apt `aws-neuronx-runtime-lib 2.33.10.0-3dcef56f0`, `aws-neuronx-collectives
  2.33.10.0-068180c7a`, `aws-neuronx-tools 2.31.13.0-a9e473f33`, `aws-neuronx-dkms 2.29.0.0` (module reloaded, 16
  devices visible); venv `/opt/nb` (uv, CPython 3.12.16; the pin box has 3.12.3) with neuronx-cc 2.26.6360.0+6f180f47,
  torch-neuronx 2.9.0.2.15.32035+de43f57c, torch-xla 2.9.0, torch 2.9.1, libneuronxla 2.2.17544.0+fb9962bf, nki
  0.5.0+28631259367.ga768afa6, neuronx-distributed 0.19.28492, numpy 2.4.6, protobuf 7.35.1 - exact matches. Kernel is
  6.8.0-1044-aws (pin box: 6.17). Extra, not part of the pin: transformers 5.22.2, tokenizers 0.23.3, safetensors 0.8.0.
* Checkpoint: 107 files, 806 819 566 361 bytes, HF -> `/mnt/scratch/raw` in 12 min (40 parallel files, ~1.1 GB/s
  average, 1.8 GB/s peak), every size equal to the HF manifest; sha256 against the LFS oids: see gate (c).

**Gate (c), 2026-11-12 07:38 UTC: prepared set, graphs, single-launch device validation**
* Raw checkpoint: sha256 of all 94 safetensors shards equal to the HF LFS oids (`verify_raw.sh`: bad=0).
* Prepare (`qwen35_prepare.py /mnt/scratch/raw /mnt/scratch/prepared --tp 32 --weights fp8 --jobs 16 --threads 8`):
  **283 s** for the 60 layers (43-63 s per layer, 16 in parallel), 1962 files, 415 624 443 408 bytes: 32 x (60 layer files of
  205-215 MB + a 31.8 MB head) + 2.03 GB `embed.bin` + tokenizer / config files. Per core 12.01 GiB of weights, as
  budgeted. (First attempt hung: forked pool workers of a process with live OpenMP threads; the pools now use spawn.)
* `qwen35_verify_prepared.py` (2 ranks per layer, all 60 layers, 3192 tensors): 0 failures; dense shards bit-equal to
  the per-rank reference `shard_layer`, expert codes x scales vs the bf16 source rel L2 max **2.65e-2** (E4M3 level),
  heads and embedding bit-equal.
* Archived: `s3://912935854432-neuron-build/qwen397/prepared/Qwen3.5-397B-A17B-tp32-fp8/` - 1963 objects (the set +
  `MANIFEST.sha256` of every file + `VERIFY.log`), 415 624 443 408 bytes, equal to the local tree (128 parallel presigned
  PUTs, ~5 min, ~1.4 GB/s Sydney -> us-east-1).
* Graphs (`qwen35_build_buckets.sh`, TP=32, PAGES=4097, SLOTS=9, fp8, MOE_DEC=gather): gdn / moe / gqa x R {1, 4} and
  Q {64, 256}, gqa P {32, 128, 512, 1024, 2048, 4096}, heads 1/1, 4/4, 64/1, 256/1: every bucket compiled in 9-31 s
  (the unrolled Q=256 gdn recurrence 205 s, the fused layer NEFFs 202 s).
* Device vs CPU mirror, random weights of the real per-rank shapes, one launch on all 32 cores (`test_qwen35_device.py`):
  gdn 2.88e-2, gqa 2.84e-2, moe-gather 3.64e-2, fused layer_gdn_moe 7.09e-2, layer_gqa_moe 4.30e-2; **all 32 ranks'
  outputs bit-identical** (the all-reduce over 16 chips is consistent).
* Real layers (`test_qwen35_real_layer.py`, fused NEFFs, random input + state, f32 oracle `qwen35_moe_ref`):

| layer | device vs mirror | fp8 mirror vs oracle | **device vs oracle** |
|---|---|---|---|
| 0 (gdn) | 3.33e-2 | 6.18e-2 | 5.83e-2 |
| 3 (gqa) | 2.66e-2 | 3.79e-2 | 3.92e-2 |
| 30 (gdn) | 4.16e-2 | 8.10e-2 | 7.35e-2 |
| 58 (gdn) | 4.01e-2 | 7.19e-2 | 6.60e-2 |
| 59 (gqa) | 5.34e-2 | 6.78e-2 | 5.76e-2 |

  The same level as the 35B port (5.5-6.8e-2 per layer): the fp8 format, not the TP=32 sharding.

**Gate (d), 2026-11-12 07:41 UTC: first tokens; reproducibility; oracle**
* Engine (`neuron_engine_cli`, `qwen35_model.cpp` unchanged apart from the two debug switches below, 32 rank threads,
  `NEURON_RT_VISIBLE_CORES=0-31`, `nrt_load_collectives(..., rank, 32)`): 36 NEFFs loaded on every core, **weights
  resident in 62-64 s** (32 threads, files in the page cache), `per core 12.036 GiB weights + 0.989 GiB state` (4096 KV
  pages + 8 GDN slots) - as budgeted, and it loads next to all 36 NEFFs' scratch.
* First generation (greedy, chat template, enable_thinking=false, "Explain in two paragraphs why the sky is blue.",
  200 new tokens): *"The sky appears blue primarily due to a phenomenon known as Rayleigh scattering, which occurs when
  sunlight interacts with the gases and particles in Earth's atmosphere. Sunlight, or white light, is composed of all
  the colors of the visible spectrum, each traveling in waves of different lengths. As this light enters the
  atmosphere, it collides with nitrogen and oxygen molecules; shorter wavelengths, such as blue and violet, are
  scattered much more strongly than longer wavelengths like red and yellow ... Although violet light has an even shorter
  wavelength than blue and is scattered more intensely, the sky does not appear violet to us for two main reasons.
  First, the sun emits less violet light than blue light to begin with. Second, the human eye is significantly more
  sensitive to blue light than to violet ..."* - coherent and correct; decode median 83 ms/token.
* **Oracle** (`test_qwen35_oracle.py`, f32 CPU reference over the 23 prompt + 200 generated tokens, teacher forced,
  540 s): top-1 agreement **194/200**; the 6 misses are near ties (oracle margin 0.007-1.07, the device token is the
  oracle's rank-2 token every time). Same picture as the 35B (107/112).
* **Reproducibility (the section-12 problem of the 35B doc): not reproduced here.** Same prompt, greedy, 64 tokens:
  12 sequential requests in one engine process x 4 processes (plain, `BEHEMOTH_NEURON_QWEN35_BARRIER=1`,
  `BEHEMOTH_NEURON_QWEN35_SYNC=4`, plain again) + 8 further separate plain processes + the first 64 tokens of the
  200-token run = **57 streams from 13 engine processes, all token-identical; no non-finite logits**. The switches make
  no difference to the output (barrier: +5 ms/token). Since the plain path is already deterministic on this
  instance, the non-aliased-state variant was not built. What differs from the trn1.2xlarge where 1 in 6 runs faulted:
  32 ranks instead of 2 (every launch waits on a 16-chip all-reduce, so ranks cannot run ahead of each other by much),
  kernel 6.8 vs 6.17, a fresh driver load; the Neuron packages are the same pinned versions. This does not explain the
  TP=2 fault - it only shows the TP=32 build does not exhibit it in 57 runs. The switches stay in the backend for the
  TP=2 A/B.
