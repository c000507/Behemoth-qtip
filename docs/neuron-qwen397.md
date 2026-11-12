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

## 5. Collectives (measured on the instance)

See the status log (nccom-test all-reduce over 32 cores, and the in-graph all-reduce of a [1, 4096] f32 row).

## 6. Prepare at this scale

`qwen35_prepare.pack_layer`: the expert tensors of one layer (512 x [2048, 4096] + 512 x [4096, 1024] bf16 = 12.9 GB)
are read in chunks of 32 experts, quantised **once** (row scales are per gate/up output row and per down input channel,
i.e. independent of the rank slice) and then sliced into the 32 ranks' [512, 64, 4096] / [512, 32, 4096] pieces;
`--jobs` packs several layers in parallel worker processes, a `.done/Lnnn.json` marker per layer makes the run
resumable. `tests/test_qwen35_pack.py`: bit-equal to the per-rank `shard_layer` reference (TP=32 and TP=2; dict and
safetensors paths).

## 7. Status log

(appended as gates are passed)
