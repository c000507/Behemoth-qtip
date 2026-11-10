# Reviewer experiments: verify-step cost / speculation value at agentic contexts, and the HBM->SBUF weight-stream wall

Box: i-0adbefb8af607c610 (trn1.2xlarge, libnrt 2.33.10, neuronx-cc 2.26.6360, neuron-profile 2.31.13), 2026-11-10.
Model: Devstral Small 2507 (hidden 5120, 32/8 heads, head_dim 128, ff 32768, 40 layers, vocab 131072), fp8 weights + fp8 KV,
TP=2, the b18 graphs/prepared set on the box (`/home/ubuntu/work/b18/{graphs,prepared,model}`), engine binary
`/home/ubuntu/work/b18/BUILD/neuron_engine_cli` (built there from the repo). Every number below is a measurement run in this
session unless marked "from file". Scripts: `/tmp/fable_exp/` on the workstation (shipped to the box over SSM into
`/home/ubuntu/work/fable/`).

Common env for everything that touches the device: `NEURON_RT_VISIBLE_CORES=0-1 NEURON_RT_ROOT_COMM_ID=127.0.0.1:62128`,
`pkill neurond` first. layer_bench = the repo's current `neuron/cpp/layer_bench.cpp` (md5 1177a848..., the version that
drains with an output read before stopping the clock), rebuilt on the box:
`g++ -O2 -std=c++20 -I/opt/aws/neuron/include layer_bench.cpp -o layer_bench -L/opt/aws/neuron/lib -lnrt -Wl,-rpath,/opt/aws/neuron/lib`.

## Experiment 1: what a verify step costs and what speculation is worth at 1.5k-4.4k context

### 1a. Verify-cost curve (device time, layer_bench, 200 iters, 2 ranks, drained)

Command per bucket: `./layer_bench /home/ubuntu/work/b18/graphs/$b/graph.neff 200 2` (every bucket run at least twice;
all runs listed, the table uses the median).

| bucket | runs (ms/step) | ms/layer used |
|---|---|---|
| dec1_32 | 1.303, 1.293 | 1.30 |
| dec8_32 | 1.687, 1.668 | 1.68 |
| dec1_128 | 1.326, 1.322 | 1.32 |
| dec4_128 | 1.633 | 1.63 |
| dec8_128 | 1.811, 1.798 | 1.80 |
| dec1_512 | 2.476 (first run, outlier), 1.436, 1.426, 1.655 | 1.43 (median of the 4 = 1.55; this bucket is noisy, see caveat) |
| dec4_512 | 2.021, 2.018 | 2.02 |
| dec8_512 | 2.538, 2.536 | 2.54 |
| dec1_1024 | 1.569, 1.549, 1.549 | 1.55 |
| dec8_1024 | 4.040, 4.141, 3.879 | 4.02 |

Derived (arithmetic shown):

| pages | dec8 - dec1 (ms/layer) | per extra row = /7 | verify premium = 40 x (dec8 - dec1) | relative to 40 x dec1 |
|---|---|---|---|---|
| 32 | 1.68 - 1.30 = 0.38 | 0.054 | 15.2 ms/step | +29 % (of 52.0) |
| 128 | 1.80 - 1.32 = 0.48 | 0.069 | 19.2 ms/step | +36 % (of 52.8) |
| 512 | 2.54 - 1.43 = 1.11 | 0.158 | 44.4 ms/step | +78 % (of 57.2) |
| 1024 | 4.02 - 1.55 = 2.47 | 0.353 | 98.8 ms/step | +159 % (of 62.0) |
| dec4: 128 | 1.63 - 1.32 = 0.31 (0.103/row) | | 12.4 ms/step | +23 % |
| dec4: 512 | 2.02 - 1.43 = 0.59 (0.197/row) | | 23.6 ms/step | +41 % |

Check against the doc figures. The "+5 % at 32 pages, +36 % at 128, +57 % at 1024" triple is not in the repo as such; the
nearest claims are PERF_NOTES.md ("+14 / +44 ms per step", "a verify step would cost +57 %" at 14k) and SERVING.md
("dec8_1024 costs 44 ms more per step than dec1_1024"), both written for the 8B Llama model. For Devstral they are wrong by a
lot at the ends: at 32 pages the premium is +29 % (not +5 %), at 128 pages +36 % (matches), at 512 pages +44 ms = +78 %, and at
1024 pages +99 ms = +159 % (not +44 ms / +57 %). The per-row gather cost grows super-linearly with pages (0.054 -> 0.353 ms per
row per layer from 32 to 1024 pages), so the engine's linear `spec_page_cost x drafts x pages` model (0.01 ms) underestimates
the 1024-page bucket by ~2x (model: 0.01 x 7 x 1024 = 72 ms vs measured 99 ms) and overestimates 32-128 pages (model 2.2 / 9.0
ms vs measured 15 / 19 ms the other way: the fixed part of a verify step is not modelled).

Caveat: dec1_512 scattered (1.43-1.66 plus one 2.48 outlier on the first execution of the sweep; dec4_512 / dec8_512 /
dec1_1024 repeated to +-0.02 ms). The 44.4 ms premium uses 1.43; with 1.55 it would be 39.6 ms. The engine measurement
below (same bucket) gives +47.5 ms per verify step, which brackets it.

### 1b. Realistic requests through the engine

Prompts (built on the box with the HF `tokenizer.json`, Tekken control tokens `<s>[SYSTEM_PROMPT]..[/SYSTEM_PROMPT]
[AVAILABLE_TOOLS]..[/AVAILABLE_TOOLS][INST]..[/INST]`): a coding-agent system prompt (operating rules, repo layout, build/test,
conventions) + 8 JSON tool schemas (read_file, write_file, apply_patch, run_command, search_code, list_dir, git, ask_user) +
a task asking for a tool call. "long" adds pre-read repository files (CLI source, SERVING.md excerpt, layer_bench excerpt) =
4398 tokens; "short" = the same without the files = 2243 tokens; "short15" = 4 tools and a trimmed system prompt
(the <= 2048-token / 128-page bucket). Greedy, max_new 128, `BEHEMOTH_PREFILL_CHUNK` default 256. The CLI ran in
`sequential` mode with the request set [long, short, long] (the repeat of long = run-to-run check inside one process), and
[short15, short15]. Buckets are inferred from the engine's pick rule (smallest decode bucket with P >= ceil(ctx/16)):
2243-4526 tokens -> 512 pages (dec1_512 / dec8_512), <= 2048 -> 128 pages.

Env: `BEHEMOTH_NEURON_NEFF_DIR=.../graphs BEHEMOTH_NEURON_PREPARED_DIR=.../prepared BEHEMOTH_NEURON_STEP_LOG=1` plus
(a) `BEHEMOTH_NEURON_SPEC_TOKENS=0`, (b) nothing (adaptive, 7 drafts, page cost 0.01), (c) `BEHEMOTH_NEURON_SPEC_PAGE_COST=0`.
What (c) does (neuron_engine.cpp:423-428): `overhead = page_cost x budget x bucket_pages`; drafting is skipped only when
`overhead > 0.1 x plain_step_ms && alpha x plain_step_ms < 2 x overhead`; page cost 0 makes overhead 0, so the skip never
fires = drafting forced on at every context (prompt-lookup n-gram 3, 7 drafts, dec8 verify). Model load took 53 s per process
(page cache warm); decode ms/token = sum of the step walls after the first-token step / tokens produced by those steps
(the CLI's (last_token_ms - first_token_ms)/(n-1) agrees to 0.03 ms everywhere).

| ctx (prompt tokens) | config | decode steps (verify / plain) | drafts per verify | accepted per verify A | mean verify wall | mean plain wall | ms/token | t/s |
|---|---|---|---|---|---|---|---|---|
| 4398 (512 pages) | (a) plain | 127 (0/127) | - | - | - | 61.44 | **61.44** | **16.3** |
| 4398 | (a) plain, repeat | 127 (0/127) | - | - | - | 62.25 | 62.25 | 16.1 |
| 4398 | (b) adaptive | 127 (0/127) | 0 (never drafted) | - | - | 65.00 | 65.00 | 15.4 |
| 4398 | (b) adaptive, repeat | 127 (0/127) | 0 | - | - | 65.27 | 65.27 | 15.3 |
| 4398 | (c) forced | 78 (60/18) | 7.00 | **0.50** | **107.5** | 60.09 | **69.75** | **14.3** |
| 4398 | (c) forced, repeat | 78 (60/18) | 7.00 | 0.50 | 107.5 | 59.39 | 69.63 | 14.4 |
| 2243 (512 pages) | (a) plain | 52 (0/52) | - | - | - | 62.92 | 62.92 | 15.9 |
| 2243 | (b) adaptive | 52 (0/52) | 0 | - | - | 64.55 | 64.55 | 15.5 |
| 2243 | (c) forced | 32 (25/7) | 7.00 | **0.80** | 107.6 | 58.87 | **59.64** | **16.8** |
| 1364 (128 pages) | (a) plain | 52 (0/52) | - | - | - | 57.05 | **57.05** | **17.5** |
| 1364 | (a) plain, repeat | 52 (0/52) | - | - | - | 56.94 | 56.94 | 17.6 |
| 1364 | (b) adaptive | 48 (7/41) | 7.00 | 0.71 | 80.2 | 56.77 | 54.51 | 18.3 |
| 1364 | (b) adaptive, repeat | 45 (12/33) | 7.00 | 0.58 | 79.2 | 56.49 | 54.12 | 18.5 |
| 1364 | (c) forced | 34 (27/7) | 7.00 | **0.70** | 78.6 | 56.21 | **47.49** | **21.1** |
| 1364 | (c) forced, repeat | 34 (27/7) | 7.00 | 0.70 | 78.5 | 55.26 | 47.28 | 21.1 |

- At 128 pages the engine's verify premium is 78.6 - 56.2 = **+22.4 ms** (layer_bench: +19.2 from the layers). FORCED drafting
  beats the adaptive default by 13 % there (21.1 vs 18.4 t/s, plain 17.5): the gate skips when
  `alpha x plain < 2 x overhead` with overhead = 0.01 x 7 x 128 = 9.0 ms, i.e. whenever the acceptance EMA (0.5-weighted count
  of accepted drafts per verify step, neuron_engine.cpp:569) is below 0.31; two zero-acceptance steps in a row (1.0 -> 0.5 ->
  0.25) switch it off and it recovers at +0.02/step, so only 7-12 of 45-48 steps verified although the per-step mean A was
  0.58-0.71. A page cost of <= 0.006 ms makes the 128-page overhead (<= 5.4 ms) fall under the "within 10 % of a plain step,
  always try" branch while 512 pages stays gated (0.006 x 7 x 512 = 21.5 ms needs alpha >= 0.72): the cheapest real win in this
  report, no kernel work. Not run as a separate config (the forced run IS the 128-page always-on behaviour).

- Verify premium inside the engine at 512 pages: 107.5 - 60.1 = **+47.5 ms per verify step** (layer_bench: +44.4 from the
  layers alone; the rest is the 8-row head + host). SERVING.md's "+44 ms" matches the 512-page bucket of Devstral by
  coincidence, not the 1024-page bucket it names.
- Adaptive = plain at both lengths: with 512 pages the overhead term is 0.01 x 7 x 512 = 35.8 ms > 0.1 x 60 and the gate
  needs alpha x 60 >= 71.7, i.e. alpha > 1, so it never drafts (correct decision given A = 0.5-0.8: forced drafting LOST at
  4.4k, 69.7 vs 61.4 ms/token, and was a wash at 2.2k, 59.6 vs 62.9).
- Process-to-process spread of the plain step: 59.4 / 60.1 / 61.4 / 62.3 / 62.9 / 64.6 / 65.0 / 65.3 ms for the same bucket
  (+-5 %). The adaptive-vs-plain 3.5 ms gap is this spread (adaptive executed exactly the same steps), not a cost of the
  policy. Compare steps within one process when reading the table.
- Output text, first 200 chars (coherent in every run; plain = adaptive byte-identical, forced differs after ~60 chars, i.e.
  the dec8 verify path is not bit-identical to the dec1 path - near-tie top-1 flips, as PERF_NOTES already notes for fp8):
  - long, plain/adaptive: "I'll add the `--sequential-timeout SECONDS` option to neuron_engine_cli.cpp. Here's the plan:\n\n1. Parse the new command-line option and store the timeout value.\n2. In sequential mode, use a separate thread to monitor the" (128 tokens, hit max_new; no tool call within 128 tokens)
  - long, forced: "I'll add the `--sequential-timeout SECONDS` option to the CLI. Here's the plan:\n\n1. Read the CLI source to understand the argument parsing and sequential mode logic.\n2. Add a new command-line option for the timeout.\n3. M" (109 tokens, EOS)
  - short, all configs: "I'll add the `--sequential-timeout` option to the neuron_engine_cli.cpp file. Let's first read the file to understand the current implementation.[TOOL_CALLS]read_file<SPECIAL_32>{\"path\": \"neuron/cpp/neuron_engine_cli.cpp" (53 tokens, EOS: a well-formed tool call; the path copied from the prompt is why A is higher here, 0.80)
- Step-log split at 512 pages, plain: layers_ms 42.9-43.9 (40 async submits), head_ms 15.5 (= drain + head), i.e. the
  device-bound picture of docs/neuron-head-argmax.md holds at 4.4k context too.

### 1c. Arithmetic: a cheaper verify kernel, and grammar-forced positions

Hypothetical shared-page-table verify (verify step = dec1 + 0.12 ms/layer => +4.8 ms/step instead of +47.5):

Per verify step: ms/token = (S_plain + 4.8) / (1 + A); but prompt lookup finds no draft in some steps (18 of 78 at 4.4k, 7 of
32 at 2.2k), so the realistic figure weights the plain steps in, using each run's own measured S_plain:

| ctx | S_plain | A | pure formula (S_plain + 4.8)/(1 + A) | weighted over the measured step mix | current best (plain) | gain |
|---|---|---|---|---|---|---|
| 4398 | 60.1 | 0.50 | 64.9 / 1.5 = 43.3 ms -> 23.1 t/s | (60 x 64.9 + 18 x 60.1) / 108 tok = 4976 / 108 = 46.1 ms -> **21.7 t/s** | 61.4 ms = 16.3 t/s | +33 % |
| 2243 | 58.9 | 0.80 | 63.7 / 1.8 = 35.4 ms -> 28.3 t/s | (25 x 63.7 + 7 x 58.9) / 52 tok = 2005 / 52 = 38.6 ms -> **25.9 t/s** | 62.9 ms = 15.9 t/s | +63 % |
| 1364 | 56.2 | 0.70 | 61.0 / 1.7 = 35.9 ms -> 27.9 t/s | (27 x 61.0 + 7 x 56.2) / 53 tok = 2040 / 53 = 38.5 ms -> **26.0 t/s** | forced 47.4 ms = 21.1 t/s (adaptive 54.3 = 18.4) | +23 % over forced, +41 % over adaptive |

With today's verify kernel the same mix gives 69.7 / 59.6 ms (measured above). A cheaper middle step that needs no new kernel:
4 drafts (dec4_512 = +23.6 ms/step from 1a) - not measured here because A for 3 drafts is unknown (A <= 0.5 at 4.4k puts it at
best at (60.1 + 23.6) / 1.5 = 55.8 ms/token = 17.9 t/s, +10 %).

Grammar-forced positions (local, repo structured-output code: `runtime/` behemoth_core built CPU-only; a harness that
teacher-forces realistic tool-call argument JSONs through `StructuredConstraint(json_schema_grammar(schema))` on the Devstral
Tekken vocabulary (131072 pieces from the public HF tokenizer.json) and counts positions with `allowed().count == 1`):

| tool call (argument JSON, tokens incl. EOS) | forced (count == 1) | count <= 8 |
|---|---|---|
| read_file {"path": ...} (17) | 1 (the final EOS) | 2 |
| read_file {"limit","offset","path"} (31) | 1 | 7 |
| search_code 4 keys (24) | 1 | 8 |
| git {"args","subcommand"} (20) | 1 | 9 |
| apply_patch 243-token diff (244) | 1 | 9 |
| total 337 positions | **5 = 1.5 %** | 36 = 10.7 % |

Grammar-forced tokens are worth nothing as free draft tokens for this vocabulary: Tekken merges `":`, `", `, `{"` into
single tokens, so the structural positions still have >= 2 legal tokens (whitespace variants), and only the terminating EOS is
forced. Side finding while doing this: the repo's `json_schema_grammar` only accepts object members in REVERSE declared order
(optional properties first, the required `path` last: `root-value-rest ::= "," space root-value-path-kv`), so a model that
emits `{"path": "...", "offset": 400}` in schema order is rejected (`token '",' rejected`); `run_command` with `timeout_s`
before `command` was rejected as well. Not fixed here (out of scope), flagged for the structured-output owner.

### Verdict 1

Always-on drafting with the current dec8 verify is a loss at agentic contexts: at 4.4k (512 pages) 14.3 t/s vs 16.3 plain;
the adaptive rule correctly turns it off (and PERF_NOTES's 1.5k/14k conclusions hold). A verify step that costs dec1 + ~0.1
ms/layer would turn the measured acceptance (A = 0.5-0.8 per 7-draft step with the stock prompt-lookup drafter) into
**~22-26 t/s at 2-4.4k context (+33-63 % over plain)**, so it is worth building IF the kernel can really hold the premium
near +5 ms/step; at +20 ms/step (the 128-page premium today) the 4.4k case drops to (60.1 + 19.2 x ... see 1a) ~ (60 x 79.3 +
18 x 60.1) / 108 = 54.1 ms = 18.5 t/s, still +13 %. The acceptance numbers come from two prompts; the planning-text output
(A = 0.5) is the pessimistic case and the tool-call output (A = 0.8) the optimistic one.

## Experiment 2: is ~258-285 GB/s the HBM->SBUF wall for a many-queue weight stream?

### Method

Kernel: `/tmp/fable_exp/stream_kernels.py` (NKI, `nki.jit`, compiled to a NEFF through `torch_neuronx.trace` = the repo's
trace path, neuronx-cc 2.26 on the CPU, no device; 1-4 s per variant). It streams the REAL gate-shard geometry of
`neuron/nki_mlp.py` `make_mlp_tiled`: `gt` tiled = [H/512 x FF/1024 x 128, 4096] uint8 = [20480, 4096] = **83,886,080 B**,
read as 160 tiles of [128 partitions, 4096 B] = one contiguous 512 KB block each, in the kernel's own tile order (`kb`
inner, `nb` outer -> `(kb * NBL + nb) * 128`), into an SBUF ring. Variant knobs: Q = concurrent address streams (stream q walks
tiles [q x NT/Q, (q+1) x NT/Q)), DEPTH = tiles in flight per stream (ring slots; every outer `sequential_range` iteration
issues Q x DEPTH independent DMAs into distinct slots, then consumes them), W = 4096 or 16384 B per partition row (the same
bytes viewed as [5120, 16384], 2 MB tiles), DMA form = `nl.load` (compiler-chosen), `nisa.dma_copy(dge_mode=unknown)` =
explicit DMA with the compiler's default DGE, `dge_mode=swdge`, `dge_mode=none`; consumer = a 64-column bf16 reduce per tile
into a [128,1] accumulator that is stored (keeps the stream live, ~0.3 us/tile of VectorE), or `pe` = one [128,8]x[128,512]
fp8 matmul per tile against a stationary activation tile + a PSUM reduce (the real MLP's consumer shape, 1/8 of a tile).
Q x DEPTH x W <= 128 KB per partition (SBUF), so (16,2),(8,4),(4,8) are the deepest 4 KB rings and (4,2)/(1,8) the deepest
16 KB rings. `hwdge` is trn2-only (not tried).

Timing: `./layer_bench k/<variant>/graph.neff 200 1` (one NeuronCore, 200 drained iterations, deterministic input fill).
Launch floor: the same kernel with a single tile (`floor`, `floor_pe`): 0.059 / 0.063 / 0.063 ms -> **0.061 ms**
(its profile: total_active_time 3.6 us, so the floor is runtime launch/completion, not device work).
GB/s = 83.886 MB / (ms_per_step - 0.061 ms). Profiles: `neuron-profile capture -n graph.neff -s x.ntff` then
`HOME=/root neuron-profile view -n graph.neff -s x.ntff --output-format summary-json > x.json` (the tool needs $HOME and
refuses `--output-file` for the summary formats; the cheap-wins note that "the 2.31.13 tool prints no report" is wrong, it
prints the summary JSON to stdout).

### Raw results

Validity first: the profiler's `hbm_read_bytes` shows that the `nl.load` + partial-consume form is NOT a valid stream probe:
the compiler shrank the loads to what the reduce touches when the ring was shallow.

| variant (DMA form, Q, DEPTH, W) | ms/step | hbm_read_bytes (profile) | DMA active (profile) | valid? |
|---|---|---|---|---|
| q1d2 nl.load 1x2x4K | 0.225 | **42,598,400 (51 %)** | 0.163 ms | NO (half the tensor elided) |
| q16d2 nl.load 16x2x4K | 0.363 / 0.365 | 81,305,600 (97 %) | 0.300 ms | marginal (3 % elided) |
| q1d4 / q1d8 / q4d2 / q4d4 / q4d8 / q8d2 / q8d4 nl.load | 0.301 / 0.337 / 0.334 / 0.353 / 0.364 / 0.353 / 0.363 | not profiled | | NOT used (same elision family) |
| w16 q1d2 / q1d4 / q1d8 / q2d4 / q4d2 nl.load | 0.220 / 0.297 / 0.335 / 0.335 / 0.335 | not profiled | | NOT used |

The explicit `nisa.dma_copy` forms all read the full tensor (profile 83,886,080 B) and are the measurement:

| variant | ms/step (200 iters, drained) | ms - floor | **GB/s (wall)** | profile: DMA active -> GB/s while active | profile notes |
|---|---|---|---|---|---|
| q1d2u  dma_copy dge=unknown 1x2x4K | 0.372 | 0.311 | 270 | 0.3094 ms -> 271 | 23072 sw-dynamic packets (3.6 KB), 18 queues |
| q1d4u  1x4x4K | 0.372 | 0.311 | 270 | 0.3091 ms -> 271 | |
| q4d8u  4x8x4K | 0.374 | 0.313 | 268 | | |
| q8d4u  8x4x4K | 0.371 | 0.310 | 271 | | |
| q16d2u 16x2x4K | 0.371 | 0.310 | 271 | 0.3093 ms -> 271 | |
| q1d4s  swdge 1x4x4K | 0.372 | 0.311 | 270 | | |
| q16d2s swdge 16x2x4K | 0.371 | 0.310 | 271 | | |
| q1d4n  dge=none 1x4x4K | 0.385 | 0.324 | 259 | 0.3505 ms -> 239 | 160 static transfers of 524,288 B, 7717 static packets, 34 queues |
| q4d8n / q8d4n / q16d2n dge=none | 0.386 / 0.385 / 0.386 | 0.325 | 258 | | |
| w16q1d4u / w16q4d2u (16 KB rows, 2 MB tiles) | 0.375 / 0.374 | 0.314 / 0.313 | 267 / 268 | | |
| w16q1d4n / w16q4d2n (16 KB rows, dge none) | 0.384 / 0.384 | 0.323 | 260 | 0.3626 ms -> 231 | 40 transfers of 2,097,152 B |
| pe_q1d4 / pe_q4d8 / pe_q8d4 / pe_q16d2 (nl.load + PE matmul consumer) | 0.373 / 0.375 / 0.374 / 0.374 | 0.312-0.314 | 267-269 | pe_q16d2: 83,887,104 B read, DMA 0.3108 ms -> 270, PE active 52 us | the PE consumer forces the full load |
| pe_q1d4n / pe_q16d2n (dma_copy none + PE) | 0.387 / 0.386 | 0.326 | 257 | | |

Observations:
- Every valid variant lands in **257-271 GB/s** by the drained wall and 239-271 GB/s DMA-active. Q (1 -> 16 address
  streams), DEPTH (2 -> 8 in flight), ring bytes in flight (1 MB -> 16 MB), and 512 KB vs 2 MB tiles change the time by
  <= 0.004 ms (1 %). The spread between forms is the descriptor mode: compiler-default/`swdge` DGE 270 GB/s vs static
  descriptors (`dge_mode.none`) 258 GB/s (the bench_fp8_moving_nki.py docstring's "none was faster" was for a 4096 x 7168
  weight on an older stack; here it is 4 % slower).
- A PE consumer at the real MLP shape costs nothing: pe_* = the pure-DMA time (DMA active 0.311 ms, PE active 0.052 ms,
  fully overlapped).
- The repo's earlier 260-265 GB/s "multi-stream ceiling" (docs/neuron-layer-opt.md section 7, jnp.sum probes) is reproduced
  with a real tiled weight tensor and explicit rings: **~270 GB/s per NeuronCore is the HBM->SBUF read wall**, geometry
  independent once the loads are real.

### The real layer, same session (2-rank neuron-profile capture of dec1_128, `--collectives-workers-per-node 2
--collectives-worker-count 2`, both ranks identical to 0.3 %)

| field | rank 0 |
|---|---|
| hbm_read_bytes | 282,853,256 (277.9 MB weights + KV/page traffic) |
| software-dynamic DMA (the weight tiles) | 280,185,648 B in 1.0971 ms active = **255 GB/s** |
| dma_active_time (all DMA) | 1.1273 ms -> 282.85 MB / 1.1273 = 251 GB/s |
| total_active_time | 1.2606 ms (bench ms/step 1.32: ~60 us idle + the drain) |
| tensor engine active | 0.912 ms (6026 matmuls); vector 0.241 ms |
| from file, b19 dec1_32 summary | hbm_read 278.8 MB, DMA active 1.1035 ms = 253 GB/s, total active 1.2817 ms |

The MLP-phase split (the docs' "1.09 ms for 252 MB = 231 GB/s, 258 GB/s while active") needs the timeline view, which the
summary JSON does not carry; it was not re-derived. The whole-layer numbers I can reproduce are 251-255 GB/s while the DMA is
active (the docs' 258 is the same measurement, rounded), and 282.85 MB / 1.32 ms = **214 GB/s** effective per execute.

### Verdict 2

Best valid streaming variant = 271 GB/s (q8d4u / q16d2u / q1d2u, the decision rule's <= 290 branch): **the weight stream is
at the wall**. The real layer's DMA already runs at 255 GB/s = 94 % of that wall while active; what is left is the 15 % of
the step where the DMA is idle (1.32 - 1.127 = 0.19 ms/layer, ~7.7 ms/token) plus 6 % of rate. Upper bound if the whole 252 MB
MLP streamed at the 270 GB/s wall with perfect overlap: 252 / 270 = 0.933 ms vs the docs' 1.09 ms = -0.157 ms/layer x 40 =
-6.3 ms/token -> 58.2 - 6.3 = 51.9 ms = 19.3 t/s. No queue count, ring depth, tile width or DGE mode reached 320 GB/s, so there
is no geometry to carry into the MLP kernel; stop the weight-stream work. (Only the invalid shrunken-load variants exceed
290 and they read 51-97 % of the bytes.)

## What was validated / not, and failures

- Validated: all layer_bench numbers drained, every bucket >= 2 runs; the kernel bytes via profiler hbm_read_bytes for 10 of
  the 40 kernel runs (the elision finding rests on q1d2 = 51 %, q16d2 = 97 %, all dma_copy/pe variants = 100 %); the
  engine's verify premium by two independent routes (layer_bench 44.4 vs engine 47.5 ms at 512 pages; 19.2 vs 22.4 at 128);
  the engine outputs are coherent in all 15 requests.
- Not validated / not measured: the MLP-phase-only DMA rate (summary JSON only; timeline not parsed); the 7 `nl.load`
  variants that were not profiled (discarded, not reported as rates); a 3-draft (dec4) acceptance; the fp8 head or other
  stacking; the 14k-context case (the 2048-page bucket was not timed; from the 32->1024 curve a dec8_2048 verify step would
  cost well over +200 ms); the proposed SPEC_PAGE_COST=0.006 setting as its own run (its 128-page behaviour equals the
  forced run, its 512-page behaviour equals the adaptive run by the gate arithmetic, both measured).
- Failures and fixes: neuron-profile view needs `HOME` set under SSM ("$HOME is not defined") and rejects `--output-file`
  for summary formats (print to stdout instead); the box's role cannot write to s3://912935854432-neuron-build (AccessDenied
  on PutObject, so the tokenizer for the local grammar harness came from the public HF mirror `unsloth/Devstral-Small-2507`
  tokenizer.json, 131072 pieces, same Tekken vocabulary); the default `nl.load` stream kernels were silently shrunk by the
  compiler (caught by the profiler, replaced with `nisa.dma_copy`); the repo's `Tokenizer` class only loads SentencePiece
  `tokenizer.model` (32768), so the harness feeds the Tekken byte pieces to `Vocabulary` directly.
- Doc claims found wrong: the +44 ms / +57 % verify premiums (8B figures) do not describe Devstral: 1024 pages costs +99 ms
  (+159 %), 32 pages +15 ms (+29 %); the cheap-wins note that neuron-profile 2.31.13 "prints no report" (it prints the
  summary JSON to stdout once HOME is set); the adaptive gate's linear page-cost model is off by ~2x at both ends (1a).
  SERVING.md's "a verify step within 10 % of a plain one" never applies to Devstral with page cost 0.01 above 32 pages.
