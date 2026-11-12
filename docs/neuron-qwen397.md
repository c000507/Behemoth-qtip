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

## 8. Measurements and speed work (2026-11-12, all 32 NeuronCores, `NEURON_RT_VISIBLE_CORES=0-31`, engine step log)

### 8.1 Baseline: sub-block graphs (2 launches per layer, 121 per token), bucket set `/mnt/scratch/graphs`

| | decode ms/token (median) | tok/s | prefill (Q=256 chunks) |
|---|---|---|---|
| short context (23-token prompt, 200 new) | 82.0 | 12.2 | |
| after an 8k fill (8033 prompt tokens) | 90.6 | 11.0 | 8033 tokens in 18.3 s = 440 tok/s (495 after the first chunk) |
| after a 32k fill (32033) | 147.5 | 6.8 | 469 tok/s |
| after a 64k fill (64033, the 4096-page bucket) | 190.0 | 5.3 | 64033 in 138.7 s = 462 tok/s |

64k of context prefills and decodes correctly through the whole page ladder (coherent continuation of the same answer).
Oracle 194/200 and the 57-stream reproducibility test (section 7) were taken on this configuration.

### 8.2 Where the 82 ms went

`layer_bench` (one NEFF launched 200 times on 32 ranks, drained) vs the engine's per-rank profile (`r0` / `rmax` in the
step log: time inside `nrt_execute` per sub-block kind):

| NEFF | layer_bench ms/launch | in the engine |
|---|---|---|
| gdn 1x1 | 0.313 | 0.65-0.82 |
| moe 1x1 (gather, 10 slots) | 0.361 | 0.60-0.70 |
| gqa 1x1, P = 32 / 512 / 4096 pages | 0.315 / 0.569 / 3.67 | 0.61-0.67 (P=32) |
| head 1 row | 0.359 | 0.5 |
| fused layer_gdn_moe / layer_gqa_moe | 0.580 / 0.499 | |
| prefill Q=256: gdn / moe (all experts) / gqa P=32 / gqa P=4096 | 3.06 / 4.91 / 1.14 / 7.28 | (prefill is device-bound: 450 ms per chunk predicted, 520 measured) |

Sum of the device times for a token: 45 x (0.313 + 0.361) + 15 x (0.315 + 0.361) + 0.36 = **40.8 ms**; the engine spent 80
ms inside the same `nrt_execute` calls. `cpp/bind_bench.cpp` isolates why: launching the same NEFF with a different
tensor set every time costs nothing (0.312 ms vs 0.318), but **alternating between two loaded models on a core costs
+0.2-0.3 ms per launch** (gdn <-> moe alternation 0.50-0.57 ms per launch; even two instances of the same NEFF: 0.51).
The sub-block plan switches model on every launch (gdn, moe, gdn, moe, ..., gqa, moe): ~120 switches = ~35 ms per
token. Collectives are not the cost (nccom 10 us; in-graph they are inside the 0.31-0.36 ms). Host-side work is
negligible (input writes 0.04 ms, tensor-set lookup 2 ms, logits read 0.02 ms, sampling 1-3 ms).

### 8.3 Levers, in the order found

| lever | decode ms/token, short context | tokens vs baseline |
|---|---|---|
| baseline sub-blocks | 82.0 (12.2 tok/s) | reference |
| per-launch blocking read (`..._SYNC=4`) / inter-rank barrier (`..._BARRIER=1`) | 84.6 / 87.0 | identical |
| fused layer NEFFs (61 launches, 30 model switches), as first measured | 97-112, very jittery (45-129 ms steps) | identical |
| + **CPU idle states held off** (`/dev/cpu_dma_latency` = 0 us; now done by the backend itself) | fused **48.6** (20.6 tok/s), stable; sub-blocks unchanged at 80-83 | identical |
| + **segment NEFF: 4 whole layers (gdn, gdn, gdn, gqa + their MoEs) per launch**, 15 launches per token, one model for the whole stack (no model switches), decode routing without `torch.topk` | **28.6** (35.0 tok/s); 23.6 ms of it in the 15 launches (1.5 ms each) + head 0.5 + sampling / engine 3-4 | identical (64 tokens) |

* The fused-layer non-finite fault of the 35B doc (section 10) does **not** occur at TP=32 on this instance: fused
  graphs gave 12/12 identical 64-token streams and the same tokens as the sub-block run over 200 tokens.
* Jitter: every launch is a rendezvous of 32 rank threads in a sub-millisecond all-reduce, so one thread waking from a
  deep C-state stalls all 32. The fused plan (longer launches, fewer of them) was hit hardest. The backend now opens
  `/dev/cpu_dma_latency` and holds 0 us for the life of the process (`BEHEMOTH_NEURON_CPU_DMA_LATENCY=-1` disables; the
  device node must be writable: root, or `chmod 666 /dev/cpu_dma_latency`).
* Segment graphs (`qwen35_graphs.py ... seg_gdn_gdn_gdn_gqa R 1 P`, `segment()`; weights `l<i>.a.*` / `l<i>.m.*`, states
  `l<i>.<state>`; backend `segment_graph()` in the launch plan, `BEHEMOTH_NEURON_QWEN35_NOSEG=1` falls back): the layer
  pattern (3 gdn + 1 gqa) repeats 15 times, so one NEFF serves the whole stack and the core never changes model
  between launches. `tests/test_qwen35_segment.py`: bit-identical to the sub-blocks on the CPU. Compiles in 24 s.

### 8.4 Bug found: `torch.topk` in the router is non-finite on the device for 4 / 8 rows

The first 4-stream run stopped with "Nonfinite model logits" at the first 4-row decode step. Bisection on the device
(`MOE_DEBUG` stage outputs through `test_qwen35_device.py`): the shared expert and the softmax probabilities are fine,
the combine weights are NaN - `torch.topk(probs [4, 512], 10)` (and [8, 512]) returns garbage on trn1 with neuronx-cc
2.26, while [1, 512], [64, 512] and [256, 512] are fine (and [4, 128] k=8 was fine on the 35B). Fix: `ROUTE=rank`
(`route()`): top-k membership by pairwise counting (`beats[e] = #{e' : p[e'] > p[e] or (p[e'] = p[e] and e' < e)} < k`),
no topk op; CPU-identical to the topk form. Device vs mirror with it: moe gather R = 1 / 4 / 8 / 16: 3.64e-2 / 3.30e-2 /
3.20e-2 / 3.20e-2. `ROUTE_DEC=rank` in `qwen35_build_buckets.sh` builds every decode bucket this way (prefill keeps
topk: fine at Q = 64 / 256, and the [T, E, E] comparison would be large there).

### 8.5 HBM headroom limits the bucket set

Loading 54 NEFFs including R = 8 / 16 buckets at 2048 / 4096 pages failed with HBM exhausted: the NEFFs' shared
scratchpad is the maximum over the loaded NEFFs (gqa 16 rows x 4096 pages gathers 16 x 65536 keys x 512 f32 = 2.0 GiB)
and their DMA spill rings add up (1.26 GiB). With 12.04 GiB of weights + 1.04 GiB of state per core there are ~2.9 GiB
for code + scratch + rings. Served ladder: R=1 up to 4096 pages, R=4 up to 2048, R=8 up to 1024, R=16 up to 512 (the
4096-page pool is shared by all streams anyway: 16 streams average 256 pages).

### 8.6 What a launch costs (micro NEFFs, `tests/bench_allreduce_graph.py`, layer_bench, 32 ranks, drained)

| NEFF | ms/launch |
|---|---|
| x + w, no collective | 0.086 |
| 1 all-reduce of [1, 4096] f32 over 32 ranks | 0.285 |
| 4 / 16 all-reduces in one launch | 0.327 / 0.727 (0.52 with an RMSNorm between them) |
| 4 all-reduces of [8, 4096] | 0.430 |
| 1 / 4 all-reduces of [256, 4096] f32 (4 MB, the prefill chunk) | 0.545 / 2.70 |

* A launch that contains a collective pays **~0.2 ms once** (the 32-rank rendezvous / collective setup), each further
  all-reduce of a decode row costs only **15-30 us**. nccom-test's 10 us is the steady-state figure; the per-launch
  0.2 ms is what made 121 single-all-reduce launches cost 80 ms. Segments amortise it: 20 layers per launch = 3 launches.
* Model switching (+0.2-0.3 ms when consecutive launches use different NEFFs, section 8.2) is the second fixed cost;
  one NEFF for the whole repeating pattern removes it.
* With both gone a layer costs ~0.32 ms inside a segment (19.2 ms for 60 layers in 3 launches): 2 all-reduces ~0.05 ms,
  the rest is the ~100 small ops of gdn + moe (norms, 5 projections, conv step, the recurrence, routing, gather, 3 expert
  contractions), each far from the DMA wall: the weights a rank reads per layer (17 MB incl. the 8.4 MB f32 router)
  need ~0.07 ms at 250 GB/s. This is op-count bound, not byte bound.
* Prefill: an all-reduce of a 256-token chunk (4 MB f32) costs ~0.7 ms -> 120 per chunk = **~85 ms of the 520 ms chunk
  (17 %) is the TP=32 collective**; reducing in bf16 would halve it.

### 8.7 Would another sharding be faster? (measured prototypes of the MoE block alone, R = 1)

| MoE layout per core | ms/launch |
|---|---|
| now: 512 experts x 32-wide slices (ETP=32), gather 10 slices (3.9 MB) | 0.333-0.361 |
| stages of it: router + softmax only 0.280; + top-k weights 0.306; + gate/up contraction 0.311; + activation 0.349; + down 0.371; shared expert alone 0.318 (each is a full launch incl. its all-reduce, 0.285) | |
| TP8 x EP4 shape: 128 local experts x 128-wide slices, 10 slots (15.7 MB worst case) | 0.324 |
| EP=32 whole experts: 16 local experts x 1024 wide, 10 static slots (126 MB worst case) | **1.502** |

* The expert contractions are ~0.06-0.09 ms of the 0.36 ms MoE launch; the rest is launch + all-reduce + router. The
  32-wide tiles are **not** an inefficiency worth chasing: 128-wide tiles measure the same within noise (0.324 vs 0.333).
* **EP=32 whole experts: no-go.** A static graph must provision the worst case on every core (all 10 selected experts
  local: 10 x 12.6 MB), 4.2x slower per MoE launch (+1.15 ms x 60 layers = +69 ms/token). Provisioning fewer slots is
  not exact (uniform routing already puts 3+ experts on one core in 8 % of the tokens; real routers are skewed).
* **TP=2 x EP=16: does not fit** (non-expert weights 8.7 B fp8 / 2 = 4.35 GB + 12.08 GB experts = 16.4 GB > 16 GiB before
  KV / scratch). **TP=8 x EP=4: no-go**: fits on paper (12.08 + 1.09 + KV 0.94 = 14.1 GB) but leaves < 1.5 GiB for NEFF
  scratch + rings where ~2.9 GiB is already tight (8.5); the tile-width gain is nil (above); the attention all-reduce
  shrinking from 32 to 8 ranks saves at most the incremental 15-30 us per layer; and the MoE needs worst-case 10 slots
  of 128-wide slices on every core (4x the bytes of today). Data-parallel attention across EP groups would add an
  all-gather of activations per layer for batch >= 4 and give each stream only 1/4 of the cores for its attention: the
  decode is op-count bound, not attention bound, so nothing to win there either.

### 8.8 The fast decode configuration: 20-layer segments (3 launches + head per token), validated

Bucket set `g4` = sub-block graphs (prefill; fallback) + `seg_(gdn_gdn_gdn_gqa)x5` decode segments for R = 1 (P 32..4096),
R = 4 (P..2048), R = 8 (P..1024); `PAGES=4097 SLOTS=17 ROUTE_DEC=rank`; all 32 cores (`NEURON_RT_VISIBLE_CORES=0-31`,
`Qwen35: 60 layers, TP 32, 63 NEFFs, 4096 pages, 16 GDN state slots; per core 12.04 GiB weights + 1.04 GiB state`).
Larger segments: 4 layers per launch 28.6 ms, 8 layers 29.0 (two NEFFs alternate), **20 layers 24.5**, 60 layers in one NEFF
28.7 (compiles in 401 s; measured next to a running compile job, so not better than 20 in any case).

| single stream | baseline (8.1) | **segments** | prefill tok/s |
|---|---|---|---|
| short context | 82.0 ms, 12.2 tok/s | **24.5 ms, 40.9 tok/s** | |
| 8k (8033 prompt tokens) | 90.6 | **26.7 ms, 37.4 tok/s** | 556 (639 after the first chunk) |
| 32k (32033) | 147.5 | **47.9 ms, 20.9 tok/s** | 546 |
| 64k (64033) | 190.0 | **66.0 ms, 15.2 tok/s** | 564 |

(prefill here is still the sub-block Q=256 path; it gained 20 % from the C-state hold alone.)

* **Token identity**: 12 sequential requests in one process + 6 separate processes (64 tokens): all identical to each
  other and to the baseline stream; the 8k / 32k / 64k continuations (32 tokens) are identical to the baseline's; the
  200-token run equals the baseline for 104 tokens and then takes the other branch of a near tie (oracle margin 0.44 at
  that position). **Oracle on the 200-token segment run: 197/200** (baseline 194/200): the two configurations are
  equally valid fp8 realisations; within a configuration the output is reproducible.
* Decode at long context now grows only with the attention itself (15 gqa layers: dense scores over the bucket's keys +
  the f32 K|V gather, 3.7 ms per layer at 4096 pages in layer_bench): 24.5 -> 66 ms at 64k. That part needs the paged
  NKI attention kernel of the Llama path (bf16 keys, no f32 materialisation); not ported here.

### 8.9 Batching (distinct prompts, 100 new tokens each, segment decode)

| streams | bucket | ms/step | aggregate tok/s |
|---|---|---|---|
| 1 | R=1 seg20 | 24.3-24.6 | 40.9 |
| 2 | R=4 seg20 (2 rows used) | 41.9 | 47.5 |
| 4 | R=4 seg20 | 43.0 | **93.0** |
| 8 | R=8 seg20 | 69.0 | **116** (7.3 rows on average as requests finish: 106) |
| 16 | R=16, sub-block graphs (no R=16 segment was built) | 258 | 59 |
| 4 streams after 8k fills each | R=4 seg20, P=512 | 63.4 | 62.6 (prefill 4 x 8k: 623-645 tok/s) |

8 identical prompts gave 63 ms/step (127 tok/s): the gathered-expert MoE is cheaper when the rows share experts.
Rows of a wide bucket do not always reproduce the R=1 stream token for token (8 / 16 identical rows: all rows equal to
each other, but they leave the R=1 stream at token 22, an oracle near tie of margin 0.007): different bucket, different
rounding order. R=16 is not worth serving on this instance: its MoE (160 gather slots) and attention (16 rows of keys)
are disproportionately slow (208-258 ms/step) and its buckets cost the HBM headroom (8.5). **Serve R in {1, 4, 8}.**

### 8.10 Prefill

Per layer at Q=256 (layer_bench): gdn 3.1-3.2 ms, moe (all experts) 4.9-7.5 ms (varies between runs), gqa 1.0-1.1 ms
(P=32) .. 6.8-7.3 (P=4096); an all-reduce of the [256, 4096] f32 chunk 0.6-0.77 ms.

| lever | result |
|---|---|
| baseline sub-blocks, Q=256 | 440-495 tok/s |
| + C-state hold | 546-575 (639 at 8k after the first chunk) |
| + **prefill segments** (`seg_gdn_gdn_gdn_gqa` at Q=256, 15 launches per chunk; 11 min compile each) | **711 tok/s** at 8k (345 ms per chunk), tokens identical to the baseline |
| MoE chunk size: Q = 64 / 256 / 512 / 1024 -> 3.07 / 7.5 / 7.5 / 17.5 ms per launch | Q=512 costs the same as Q=256 for the MoE (weight-stream bound up to there): the Q=512 probe is in the status log |
| gqa Q=1024 vs 4 x Q=256: 3.08 vs 4.2 ms (P=32), 18.2 vs 27 ms (P=4096) | mild gain |
| all-reduce in bf16 (`AR_BF16=1`): [256, 4096] 0.60 -> 0.50 ms (1 per launch), 2.91 -> 1.53 ms (4 per launch) | halves the incremental collective cost; block level: gdn 3.24 -> 2.98, gqa 1.04 -> 0.81; e2e + numerics in the status log |
| **chunked (matmul-form) gated delta rule** (`GDN_CHUNK`, `gdn_chunks()`, exact block inverse; CPU: equal to the token loop to 5e-6 incl. padding / carried state) | **no speed gain on the device**: 3.96 / 3.38 / 3.27 ms at chunk 64 / 32 / 16 vs 3.24 for the unrolled loop - the loop was never the cost (the projections and the conv over 256 tokens are); compiles in 48-85 s instead of 200 s. **And the device result is wrong**: rel L2 1.15-1.2 vs its CPU mirror at chunk 64 / 32 (the loop form: 3.9e-2), not investigated (suspect: the f32 block-inverse matmuls being auto-cast). Not used; left behind `GDN_CHUNK` (default 0). |
| grouped / capacity-padded expert MoE for prefill | **not built.** Estimate: at Q=256 the all-experts form does 51 GFLOP per core and layer; a capacity-32 grouped form 13 GFLOP plus a 512 x 32 x 4096 token gather and a scatter back, and it needs an exact fallback when one expert receives more than its capacity (stateless MoE: the engine can rerun the launch with the all-experts NEFF after reading an overflow flag). Since Q=512 already halves the MoE cost per token for free, the remaining gain is perhaps 1.5-2x on 55 % of the chunk; this is NKI-kernel work (grouped matmul over a sorted token list), not a tracer change. |

What limits prefill now: per 256 tokens ~340 ms = 15 x (3 gdn x ~3 ms + gqa ~1 ms + 4 moe x ~3 ms); MoE-all and the gdn
projections are roughly equal halves; the collective is ~70 ms of it in f32.

### 8.11 Long-context decode: bf16 contraction of the paged keys (`ATT_BF16=1`)

`gqa_block` gathered the request's pages and converted K|V to f32 before the two contractions (134 MB per row at 4096
pages). With `ATT_BF16=1` the gathered bf16 pages are contracted directly (scores and values in bf16, softmax in f32).
layer_bench gqa 1x1: P=512 0.535 -> 0.411 ms, **P=4096 3.58 -> 1.76 ms**; device vs CPU mirror 2.88e-2 (f32 form 2.84e-2);
CPU mirror vs oracle unchanged (tiny models: 8.7e-3 -> 8.2e-3, 3.59e-2 -> 3.61e-2).

| decode, 20-layer segments | f32 keys (8.8) | **`ATT_BF16=1`** |
|---|---|---|
| short | 24.5 ms | 24.5-27 (same graph cost at P=32; measured next to compile jobs) |
| 8k | 26.7 | 28.0 (ditto) |
| 32k | 47.9 | **33.9 ms, 29.5 tok/s** |
| 64k | 66.0 | **40.0 ms, 25.0 tok/s** |

Tokens: the 8k / 32k / 64k continuations are identical to the baseline's; the short prompt takes the other branch of the
0.007-margin near tie at token 22 (reproducibly: 6/6 sequential runs identical). **Oracle on its 200 tokens: 198/200**
(f32-key segments 197/200, sub-block baseline 194/200).

### 8.13 Final configuration (bucket set `g10`, archived as `...-tp32-fp8-fast512`, 79 NEFFs)

Sub-block graphs R {1, 4, 8} / Q {64, 256, 512}; decode segments `seg_(gdn_gdn_gdn_gqa)x5` R {1, 4, 8} with `ATT_BF16=1
ROUTE=rank`; prefill segments `seg_gdn_gdn_gdn_gqa` at Q=256 (`AR_BF16=1`) and **Q=512 (`AR_BF16=1 ATT_BF16=1`)**;
`PAGES=4097 SLOTS=17`; `BEHEMOTH_PREFILL_CHUNK=512`. Loads next to the weights: `TP 32, 79 NEFFs, 4096 pages, 16 GDN
state slots; per core 12.04 GiB weights + 1.04 GiB state` on cores 0-31.

| | decode ms/token (median) | tok/s | prefill tok/s (after the first chunk) |
|---|---|---|---|
| 1 stream, short context | 24.5 quiet (8.8); 27-33 while the second agent's CPU jobs ran (forward 22-24 ms of it) | 31-41 | |
| 1 stream after 8k | **25.7** | 38.8 | **895** (8033 tokens in 10.6 s incl. the first chunk) |
| 1 stream after 32k | **30.1** | 33.2 | **899** (32033 in 37.2 s) |
| 1 stream after 64k | **36.0** | 27.7 | **870** (64033 in 75.5 s) |
| 4 streams (distinct prompts) | 41.4 per step | **96.5** aggregate | |
| 8 streams (distinct prompts) | 56.2 per step | **128** aggregate (7.2 rows on average) | |
| 4 streams after 8k fills each | 46.1 per step | 61 (2.8 rows on average) | 887 |

vs the first end-to-end numbers of this morning (82 / 91 / 148 / 190 ms, prefill 440-480 tok/s): **3.3x / 3.5x / 4.9x /
5.3x faster decode, 1.9x faster prefill**, same prepared weights.

Checks on this set: 8 sequential identical requests identical (and equal to the `ATT_BF16` segment run whose 200 tokens
scored 198/200 against the oracle); the 8k and 32k continuations (32 tokens) identical to the baseline's, the 64k one
identical for 31 of 32 tokens; 4 concurrent 8k-fill streams identical to each other; HTTP server: ready in 148 s, three
200-token chat completions in 5.3-7.2 s. Oracle through the Q=512 bf16-reduce prefill path (600-token prompt + 100
generated): see the status log.

## 9. Serving the fastest validated configuration

* **Instance**: trn1.32xlarge (16 Trainium, 32 NeuronCores, 512 GB HBM), >= 450 GB of local disk for the prepared set
  (instance-store NVMe), Neuron packages as pinned (aws-neuronx-runtime-lib 2.33.10.0, -collectives 2.33.10.0, -dkms
  2.29.0.0). `/dev/cpu_dma_latency` must be writable by the server process (root, or `chmod 666`), else decode is 2-4x
  slower and jittery.
* **Model**: `s3://912935854432-neuron-build/qwen397/prepared/Qwen3.5-397B-A17B-tp32-fp8/` (1963 objects, 415.6 GB,
  `MANIFEST.sha256`) -> a local directory.
* **Graphs**: `s3://912935854432-neuron-build/qwen397/graphs/Qwen3.5-397B-A17B-tp32-fp8-fast512/` (79 buckets, the set of 8.13:
  `<bucket>/graph.neff` + `meta.json`, `MANIFEST.json` with md5s; `graphs_full.tgz` = the same with compiler logs):
  sub-block graphs R {1, 4, 8} / Q {64, 256, 512}, prefill segments `seg_gdn_gdn_gdn_gqa_1_{256,512}_P`, decode
  segments `seg_(gdn_gdn_gdn_gqa)x5_{1,4,8}_1_P` built with `ATT_BF16=1 ROUTE=rank`; `PAGES=4097` (64k-token KV pool shared by all requests), `SLOTS=17` (16 concurrent sequences on the
  device, more spill to host RAM).
  (`...-tp32-fp8-fast/`: the same without the Q=512 prefill buckets, 69 buckets; `...-tp32-fp8-seg/`: f32-key decode segments.)
* **Process**: `NEURON_RT_VISIBLE_CORES=0-31 NEURON_RT_ROOT_COMM_ID=127.0.0.1:62128 BEHEMOTH_NEURON_NEFF_DIR=<graphs>
  BEHEMOTH_MODEL_NAME=qwen3.5-397b-a17b neuron_server --model <prepared dir> --host 0.0.0.0 --port 8080
  --prefill-chunk 512` (binary from this branch: `neuron/cpp/build.sh OUT server`). Ready in 130-170 s (weights resident in
  62 s from the page cache, 80-130 s from NVMe). Checked on the instance: `/health`, `/v1/models`, three
  `/v1/chat/completions` requests of 200 tokens in 5.3-7.2 s each, identical answers.
* ECS stack (`deploy/ecs-neuron`): as section 11 of the 35B doc with `backend_extra_env = { STREAM_LOAD = "0",
  NEURON_RT_VISIBLE_CORES = "0-31" }`, a trn1.32xlarge capacity provider, the container privileged enough to open
  `/dev/cpu_dma_latency`; not applied or tested here.
* Limits: context 64k tokens per pool (the 4096-page bucket); R=16 decode is slower than R=8 (serve at most 8 rows per
  step: `--max-batch-requests 8`); no prefix cache / speculation (recurrent state), as for the 35B; tool-call output
  parsing for Qwen's format is still missing in the runtime; thinking is on by default in the chat template.

### 8.12 Prefill A/B on the device (8033-token prompt, tok/s after the first chunk; the box was shared with compile jobs)

| prefill path | tok/s | tokens after the prompt |
|---|---|---|
| sub-blocks Q=256, f32 all-reduce | 575-639 | reference |
| segments Q=256, f32 all-reduce | 662-711 | identical (32) |
| segments Q=256, **bf16 all-reduce** (`AR_BF16=1`) | 741 | identical (32); a 233-token prompt + 100 generated tokens: identical to the f32-reduce run |
| segments **Q=512**, f32 all-reduce (unrolled 3 x 512-step recurrence: 41 min compile per bucket) | **897** | identical (32) |

Rounding the 32 partial sums to bf16 before the 120 all-reduces of a chunk did not change a single token in these
tests, so the Q=512 ladder is being built with it (and with `ATT_BF16=1`); result in the status log.
