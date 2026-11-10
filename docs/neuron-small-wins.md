# Small wins after the shared-table verify kernel: fp8 lm_head, build hardening, the 0.002 gate at 1024 pages

Box: i-0adbefb8af607c610 (trn1.2xlarge, libnrt 2.33.10, neuronx-cc 2.26.6360), 2026-11-10, started for this work and stopped
after it. Model: Devstral Small 2507 (fp8 weights, fp8 KV, TP=2, 40 layers), the b18 graphs / prepared set on the box plus the
dec8s_* buckets of docs/neuron-verify-kernel.md (`/home/ubuntu/work/vk/g`). Engine: `neuron_engine_cli` built on the box from the
merged HEAD sources of this branch's parent (0b8be21: dec8s buckets + page-cost default 0.002 when they are loaded), work dir
`/home/ubuntu/work/sw/` (repo copy `sw/repo`, fp8 heads `sw/g/fhead*`, graph dirs `sw/graphs_bf16` / `sw/graphs_fp8` = symlinks to
every b18 bucket + dec8s_32/128/512/1024, the fp8 dir with `head{1,4,8,16,32,64,256}` pointing at the fp8 heads instead; engine runs
`sw/e`). Nothing in b18, vk or any S3 bucket was modified. Every number below is a measurement run in this session (real engine,
real weights, `BEHEMOTH_NEURON_STEP_LOG=1`, greedy, `sequential` mode), with the arithmetic shown; scripts on the workstation under
`/tmp/fable_vk/s2/` (`build_reqs2.py` builds the prompts, `analyze_agree.py` the agreement tables, the box's
`/home/ubuntu/work/fable/analyze_run.py` the per-run ms/token as in the two earlier docs).

Common env: `NEURON_RT_VISIBLE_CORES=0-1 NEURON_RT_ROOT_COMM_ID=127.0.0.1:62128 BEHEMOTH_NEURON_PREPARED_DIR=.../b18/prepared
BEHEMOTH_NEURON_STEP_LOG=1 BEHEMOTH_NEURON_NEFF_DIR=sw/graphs_<bf16|fp8>`, `pkill neurond` before each run. ms/token = sum of the
decode step walls after the first-token step / tokens produced by them (`analyze_run.py`; the CLI's (last - first)/(n-1) agrees
to 0.03 ms everywhere).

## 1. FP8 lm_head end to end

### Method

Buckets: `HEAD_FP8=1 HEAD_ROWS=T python trace_model.py sw/g/fhead<T> head <S>` for all seven heads of `build_buckets.sh`
(`head1 1 1, head4 4 4, head8 8 8, head16 16 16, head32 32 32, head64 64 1, head256 256 1`), with the bucket env of the box
(`WT=1 MLP=nki KV=fp8 ATTN_DEC=nki UNSAFE_FP8FNCAST=1 MODEL=.../b18/model PAGES=5561 NEURON_PLATFORM_TARGET_OVERRIDE=trn1`).
All seven are needed: `neuron_model.cpp` uploads lm_head in ONE format (E4M3 rows + per-row scale, `quantize_rows`, amax/224 per
vocab row, at load) and refuses a head set that mixes fp8 and bf16. Compile times 28 / 15 / 15 / 15 / 15 / 11 / 11 s
(head1 first, cold), `meta.json` `"fp8": true, "wt": true, "order": ["x","rows","norm","wf","ws"]`. The engine picks `head1` for
plain decode steps, `head8` for 8-row verify steps (rows T = 8, S = 8 sampled), `head256` for prefill chunks.

Prompts (Tekken format, built with the HF `tokenizer.json`, `tok.no_truncation()` - the tokenizer file carries a 4096-token
truncation that silently capped the earlier docs' "long" prompt's system text; their counts are unchanged):

| name | tokens | content | decode bucket |
|---|---|---|---|
| chat | 23 | `[INST] Explain in a short paragraph how speculative decoding works ... [/INST]` | dec1_32 / dec8s_32 |
| tool15 | 1364 | the reviewer doc's `short15`: coding-agent system prompt, 4 tool schemas, the `--sequential-timeout` task | dec1_128 / dec8s_128 |
| code | 796 | `[INST] Complete the following C++ program ... (first 2500 chars of layer_bench.cpp) [/INST]` | dec1_128 / dec8s_128 |

max_new 128, temperature 0. Configs: (a) plain `BEHEMOTH_NEURON_SPEC_TOKENS=0`, (b) adaptive = engine defaults (page cost 0.002
because dec8s buckets are loaded), (c) forced `BEHEMOTH_NEURON_SPEC_PAGE_COST=0`. Each config x {bf16 head, fp8 head} in one
process per (config, head) with the three prompts in sequence. Plain runs also set `NEURON_LOGITS_DUMP` (f32 [vocab] per sampled
position) for the logit comparison.

### 1a. Speed (one process per row; the three prompts of a row share the process)

| head | config | chat: ms/token (t/s) | tool15: ms/token (t/s) | code: ms/token (t/s) | plain step wall (mean, chat / tool15 / code) | head_ms (drain + head, plain steps) |
|---|---|---|---|---|---|---|
| bf16 | plain | 56.44 (17.72) | 57.33 (17.44) | 57.18 (17.49) | 56.44 / 57.33 / 57.18 | 3.45 / 3.46 / 3.45 |
| fp8 | plain | 51.98 (19.24) | 52.77 (18.95) | 52.80 (18.94) | 51.98 / 52.77 / 52.80 | 2.02 / 2.01 / 2.03 |
| bf16 | adaptive (0.002) | 56.96 (17.56) | 41.70 (23.98) | 35.31 (28.32) | 55.53 / 56.19 / 55.97 | |
| fp8 | adaptive (0.002) | 53.54 (18.68) | 39.89 (25.07) | 33.73 (29.65) | 51.79 / 53.15 / 52.67 | |
| bf16 | forced | 56.76 (17.62) | 41.69 (23.99) | 35.35 (28.29) | 55.27 / 55.73 / 56.06 | |
| fp8 | forced | 53.55 (18.68) | 39.94 (25.04) | 33.78 (29.60) | 51.77 / 53.14 / 52.65 | |
| bf16 | plain, repeat 2 | 56.14 (17.81) | 57.06 (17.53) | 56.95 (17.56) | 56.14 / 57.06 / 56.95 | 3.45 / 3.45 / 3.45 |
| fp8 | plain, repeat 2 | 51.73 (19.33) | 52.64 (19.00) | 52.59 (19.01) | 51.73 / 52.64 / 52.59 | 2.02 / 2.01 / 2.01 |
| bf16 | plain, repeat 3 | 56.13 (17.81) | 57.09 (17.52) | 56.95 (17.56) | 56.13 / 57.09 / 56.95 | 3.46 / 3.45 / 3.45 |
| fp8 | plain, repeat 3 | 51.73 (19.33) | 52.70 (18.98) | 52.60 (19.01) | 51.73 / 52.70 / 52.60 | 2.02 / 2.03 / 2.01 |

Arithmetic:

- Head phase (step-log `head_ms`, which is the drain of the 40 async layer launches plus the head launch and logits read):
  3.45 -> 2.02 ms = **-1.43 ms per plain step**, consistent with the standalone head1 measurement (3.414 -> 1.815 = -1.60 ms; the
  engine's `head_ms` overlaps part of the head with the drain).
- Whole plain step, first process pair: tool15 57.33 -> 52.77 = -4.56 ms (-8.0 %), code 57.18 -> 52.80 = -4.38, chat 56.44 ->
  51.98 = -4.46. That is 3 ms more than the head saving; the layer phase also moved (`layers_ms` 50.6 -> 50.0, inside the
  known +-5 % process-to-process spread of the plain step, docs/neuron-reviewer-experiments.md). The repeats (rows "repeat 2/3",
  interleaved fp8 / bf16 / fp8 / bf16) bracket the spread: the three bf16 processes gave tool15 57.33 / 57.06 / 57.09 and the three fp8
  processes 52.77 / 52.64 / 52.70 ms/token (spread 0.27 / 0.13 ms this time, far below the earlier +-5 %), so the fp8 head saves
  **-4.3 to -4.6 ms per plain step on every prompt** (tool15 median 57.09 -> 52.70 = -4.39 ms = -7.7 %, 17.52 -> 18.98 t/s; code
  56.95 -> 52.60 = -4.35; chat 56.14 -> 51.73 = -4.41). The step log attributes 1.43 ms of it to `head_ms` and the layer phase moved
  by only -0.8 ms (`layers_ms` 50.8 -> 50.0; the head read and the 40 async launches overlap, so the per-phase split does not add
  up to the wall); the wall difference is the number that matters and it is reproducible to 0.1 ms across 3 + 3 processes. It is
  larger than the 1.6 ms the standalone head1 measurement implied (58.2 -> 56.6 estimate): the engine saves the bf16 lm_head's
  525 MB HBM read (fp8 262 MB) AND its contention with the layer drain, not only the head launch.
- Verify step with speculation on: tool15 67.28 -> 64.52 = -2.76 ms (head8, 8 rows: the fp8 head saves more on the 8-row head
  than on head1), code 67.09 -> 64.37 = -2.72.
- Adaptive = forced byte for byte at 32 and 128 pages (same steps, same walls within 0.3 ms): with the dec8s buckets the 0.002
  gate's overhead term is 0.002 x 7 x 128 = 1.8 ms < 10 % of a plain step, i.e. "always try", as docs/neuron-verify-kernel.md
  section 7 predicted. The chat prompt (A = 0.16 accepted per verify step) is the case where drafting does not pay: 56.96 vs
  56.44 plain (+0.5 ms/token, 43 verify steps of 68 ms producing 1.16 tokens each).
- Speculation acceptance with the fp8 head: identical to bf16 in every run (tool15 27 verify / 7 plain steps, A = 0.70; code
  51 / 19, A = 1.12; chat 43 / 58, A = 0.16; drafted 842, accepted 83 in both processes): the fp8 head's near-tie flips did not
  change which drafts the prompt-lookup drafter proposed or how many were accepted on these prompts (the token streams are
  compared in 1b).

### 1b. Token identity and logit agreement (real engine, both heads, same prompts)

| prompt | bf16 tokens (finish) | fp8 tokens (finish) | identical positions | first divergence | shared-prefix positions: argmax agree, KL(bf16 || fp8) mean / max | at the divergence |
|---|---|---|---|---|---|---|
| chat (23) | 128 (length) | 119 (EOS) | 14 / 128 = 0.11 | token 14 | 14 / 15, 0.00078 / 0.0064 | bf16 " by" margin 0.125 vs fp8 " in" margin 0.100 (near tie) |
| tool15 (1364) | 53 (EOS) | 53 (EOS) | **53 / 53 = 1.00** | none | 53 / 53, 0.00036 / 0.0057 | - (the tool call is byte-identical) |
| code (796) | 128 (length) | 128 (length) | 55 / 128 = 0.43 | token 51 | 51 / 52, 0.00038 / 0.0061 | bf16 "is" margin **0.000** (exact tie in f32) vs fp8 "type" margin 0.086 |

Run-to-run: bf16 plain vs bf16 plain repeat 2 = 309 / 309 identical tokens, fp8 vs fp8 repeat = 300 / 300: both heads are
deterministic process to process, so every difference above is the head format, not noise. KL(bf16 || fp8) over the full 131072
vocabulary at identical hidden states is 0.0004-0.0008 mean, 0.006 max (PERF_NOTES's 8B numbers of 0.02 were against the CUDA
oracle, i.e. include the layer path). Once the two streams diverge the texts are different continuations of the same prefix, so
"identical positions" after that point is not an accuracy statement (both are greedy samples of nearly the same distribution).

Teacher-forced check (real device, both heads, no CPU emulation): 220 requests = the first 64 .. 283 tokens of the reviewer
doc's 4398-token tool prompt (natural English system prompt text), `max_new 1`, plain config, `NEURON_LOGITS_DUMP`; the engine
runs the prefill of each prefix and samples the next token, so the two heads see IDENTICAL hidden states at 220 positions:

| positions | generated-token agreement | argmax agreement (dumped logits) | KL(bf16 || fp8) mean / median / max | bf16 top1-top2 margin, mean |
|---|---|---|---|---|
| 220 (prefix lengths 64-283) | **213 / 220 = 96.8 %** | 213 / 220 | 0.0016 / 0.0012 / 0.0115 | 1.80 |

The 7 flips, with the bf16 head's own top-1 margin: prefix 64 " able" vs " a" (margin 0.188), 122 "." vs ".\n" (**0.000**),
140 "," vs " or" (**0.000**), 154 "," vs "." (0.125), 170 " repository" vs " workspace" (**0.000**), 241 " just" vs " proceed"
(**0.000**), 276 "," vs " or" (0.125). Four of the seven are exact f32 ties in the bf16 head (the bf16 head's own argmax is
arbitrary there), the other three are margins of 0.13-0.19 logits against a mean margin of 1.8: every flip is a near tie,
none is a confident bf16 prediction overturned (max KL 0.0115 at the 0.125-margin flip). 3.2 % flips per position is consistent
with PERF_NOTES's 8B figure (1 flip in 22 positions = 4.5 %, also at a near tie).

Speculation on (adaptive = forced): the fp8 and bf16 streams are **byte-identical on all three prompts** (109 / 54 / 128 tokens, 291 / 291),
with identical step patterns and acceptance (section 1a). That is a coincidence of these prompts, not a property: with the
8-row verify graph the committed token stream is already a different near-tie realisation than the 1-row path (plain vs
adaptive differ after ~60 chars in both head formats, as in the two earlier docs), and the fp8 head happened to pick the same
side of every tie this path visited. Expect ~3 % of positions to flip in general, as in the teacher-forced table.

Coherence of the diverged continuations (full texts in `sw/e/texts_*.txt`; 220-char prefixes):

- chat, bf16 (128, cut by max_new): "... generation of text by a large language model. It works by generating multiple possible
  next tokens in parallel ... The model first produces a speculative sequence of tokens, which are then verified by the main model.
  If the speculative sequence is accepted, it is used as the output; if not, the process is repeated ..."; fp8 (119, EOS): "...
  generation of text in large language models. It works by generating multiple possible next tokens in parallel ... then verifies
  the likelihood of this sequence using a smaller, faster model or a subset of the full model. If the speculative sequence is
  likely, it is accepted; otherwise, the process is repeated. This method helps when generating long sequences ...". Both coherent;
  the fp8 answer is the more accurate description (mentions the draft model) and finishes.
- tool15: identical well-formed `read_file` tool call in both.
- code, bf16: `if (tensor_info->is_input) { check(nrt_allocate_tensor(model, tensor_info, &inputs, &tensors), ...)`; fp8:
  `if (tensor_info->type == NRT_TENSOR_TYPE_INPUT) { nrt_tensor_t* tensor = nullptr; check(nrt_allocate_tensor(tensor_info,
  &tensor), ...); check(nrt_add_tensor_to_set(inputs, tensor), ...); tensors.push_back(tensor);`. Both are plausible C++ and both
  invent libnrt API details (neither `is_input` nor `nrt_add_tensor_to_set` exists as written); the fp8 one is closer to the real
  API shape (`nrt_tensor_t *` out-parameter, add-to-tensor-set). Equally coherent, neither verifiably more correct.

### 1c. Verdict

**Ship as opt-in now (`HEAD_FP8=1` head set staged as a separate graph prefix / flag), promote to flag-on after one more
agreement sweep on the production prompt mix.** Evidence for: -4.3 to -4.6 ms per plain step on every prompt in 3 + 3 processes
(57.1 -> 52.7 ms/token = 17.5 -> 19.0 t/s at 1.4k, -7.7 %; with drafting on 41.7 -> 39.9 = 24.0 -> 25.1 t/s), -2.7 ms per verify
step, 262 MB less HBM for the head, unchanged speculation acceptance and byte-identical outputs with drafting on, KL 0.0004-0.0016
against the bf16 head at the same hidden states, deterministic process to process, 36 % faster head compile. Evidence against: 3.2 %
of greedy positions flip (7 / 220 teacher-forced, 2 of 3 free-running prompts diverged within 51 tokens), every one at a near tie
(bf16 margin <= 0.19 logits, 4 of 7 exact ties), so greedy outputs are NOT reproducible between a bf16 and an fp8 deployment; a
service that promises bit-stable greedy outputs across image versions must treat the switch as a model change. No case of a
confident prediction overturned and no quality regression was observed in 220 + 309 positions, but the sample is three prompts and
one 220-position text, not a benchmark: run the production eval set once with each head before making it the default. Mechanics
for the promotion: build all seven heads with `HEAD_FP8=1` into the graphs prefix (a mixed set is refused at load), no engine or
env change needed (`meta.json "fp8": true` selects the E4M3 upload), rollback = restore the bf16 head directories.

## 2. Build hardening (repo only, verified on the box)

Implemented (commit b812972; nothing changes by default):

- `neuron/trace_model.py`: `build_info()` writes `meta.json["build"]` = `{timestamp, python, neuronx_cc (venv), neuronx_cc_bin
  (PATH), torch_neuronx, torch_xla, torch, nki, libneuronxla, libnrt (dpkg aws-neuronx-runtime-lib), git_commit, git_dirty,
  env{MODEL CONFIG TP PAGES WEIGHTS KV KVATT ATTN_DEC ATTN_DEC_SHARED ATTN_DEC_MIN_PAGES ATTN_PROFILE WT MLP DOWN_BF16 OUT_F32
  HEAD_ROWS HEAD_FP8 HEAD_ARGMAX UNSAFE_FP8FNCAST NCC_FLAGS NEURON_PLATFORM_TARGET_OVERRIDE}}`, every field best effort.
- `neuron/cpp/neuron_model.cpp`: `parse_build()` reads the key (ignored otherwise; the parser uses `at()`/`value()` on known
  keys, so unknown keys were already ignored); `BEHEMOTH_NEURON_LOG_VERSIONS=1` logs `nrt_get_version()` (`<nrt/nrt_version.h>`,
  passed `sizeof(nrt_version_t)`) and the recorded compiler / libnrt versions, WARNING (stderr) on mixed compilers or a libnrt
  prefix mismatch, never an error. Off by default.
- `neuron/requirements-neuron-build.txt`: the exact `pip list` of `/opt/aws_neuronx_venv_pytorch_2_9` and the dpkg versions
  of the box (collected over SSM, not guessed).
- `neuron/verify_graph_set.py GRAPH_DIR`: table of buckets with versions, exit 1 on mixed compilers / mixed head formats / a
  directory without meta.json or graph.neff / buckets without build info.
- `docs/neuron-build-pinning.md` + a pointer in `neuron/SERVING.md`: rebuild procedure and the libnrt check.

Verified on the box (chain 2):

- `sw/g/v_head1` traced with the new `trace_model.py` (same env as the fp8 heads): `meta.json` gained `"build": {"timestamp":
  "2026-11-10T13:14:21Z", "python": "3.12.3", "neuronx_cc": "2.26.6360.0+6f180f47", "torch_neuronx": "2.9.0.2.15.32035+de43f57c",
  "torch_xla": "2.9.0", "torch": "2.9.1", "nki": "0.5.0+28631259367.ga768afa6", "libneuronxla": "2.2.17544.0+fb9962bf", "libnrt":
  "2.33.10.0-3dcef56f0", "git_dirty": false, "env": {"MODEL": ..., "PAGES": "5561", "KV": "fp8", "ATTN_DEC": "nki", "WT": "1",
  "MLP": "nki", "HEAD_ROWS": "1", "HEAD_FP8": "1", "UNSAFE_FP8FNCAST": "1", "NEURON_PLATFORM_TARGET_OVERRIDE": "trn1"}}`.
  `neuronx_cc_bin` and `git_commit` are absent there: the box copy is a plain file copy (no `.git`, so `rev-parse` fails and
  nothing is recorded - but `git_dirty: false` was still written, a bug), and `neuronx-cc --version` evidently prints its banner
  on stderr (the first version read only stdout). Both fixed in the follow-up commit (banner read from stdout + stderr, `git_dirty`
  only next to a found commit); NOT re-traced on the box (it was stopped by then), so the fixed fields are unverified there. The
  venv `neuronx_cc` field, which `verify_graph_set.py` and the engine fall back to, was correct in the real trace.
- `graphs_fp8v` = the fp8 graph set with `head1 -> v_head1`: the OLD engine (`sw/BUILD`, no provenance code) loaded it and ran
  tool15 plain at 52.6 ms/token = 19.00 t/s, the hardened engine (`sw/BUILD2`, commit b812972 sources, built with
  `ENGINE_BUILD2_EXIT 0`) 52.7 ms/token = 18.99 t/s with the same 53 tokens: unknown meta keys are ignored by both.
- `BEHEMOTH_NEURON_LOG_VERSIONS=1` on BUILD2 printed exactly: `Neuron versions: libnrt loaded 2.33.10.0` (nrt_get_version works),
  `Neuron versions: 1 bucket(s) compiled with neuronx-cc 2.26.6360.0+6f180f47`, `Neuron versions: 36 bucket(s) without build info
  in meta.json (older trace_model.py)`; no WARNING (recorded libnrt `2.33.10.0-3dcef56f0` prefix-matches the loaded 2.33.10.0).
  Without the variable the same binary printed 0 such lines (134 stderr lines, all step log).
- `verify_graph_set.py graphs_fp8v`: lists 37 buckets, `PROBLEM: 36 bucket(s) without build info`, exit 1 (expected: the b18 set
  predates the key); `graphs_bf16`: 37 without build info, exit 1. The checker's mixed-compiler / mixed-head-format paths were
  exercised with a synthetic directory on the workstation only.

Not done: pinning the venv itself on the box (read-only for this task; the pin file is the record), the Dockerfile
`NEURON_RUNTIME_VERSION` default (deploy/ is the supervisor's), and a CI check that calls `verify_graph_set.py` on the uploaded
set.

## 3. The 0.002 gate at 1024 pages (dec8s_1024)

Prompt: the coding-agent system prompt + pre-read `neuron_model.cpp` / `neuron_engine.cpp` / SERVING.md / PERF_NOTES excerpts
+ 8 tool schemas + task = **9501 tokens = 594 pages** -> the 1024-page buckets (`dec1_1024` plain steps, `dec8_1024s` verify
steps in the step log; `pre256_1024` prefill, TTFT 6.0-6.7 s). bf16 head, the staged `vk/g/dec8s_1024` bucket (not rebuilt),
greedy, max_new 128 (all three runs hit EOS first).

| config | decode steps (verify / plain) | A (accepted per verify) | verify wall | plain wall | ms/token | t/s | vs plain |
|---|---|---|---|---|---|---|---|
| plain (SPEC_TOKENS=0) | 74 (0 / 74) | - | - | 65.96 | **65.96** | **15.16** | |
| adaptive, default 0.002 | 45 (10 / 35) | 1.10 | 84.27 | 65.78 | **56.16** | **17.81** | **-14.9 % ms/token, +17 % t/s** |
| forced (PAGE_COST=0) | 31 (24 / 7) | 0.75 | 83.60 | 65.01 | **50.24** | **19.91** | -23.8 %, +31 % |

Arithmetic: verify premium at 1024 pages in the engine = 84.3 - 65.8 = **+18.5 ms per verify step** (layer_bench predicted 15.2
from the layers; the rest is the 8-row head + host, as at 128 / 512 pages where it was +10.9 / +12.7). Break-even acceptance =
18.5 / 65.8 = 0.28 accepted drafts per verify step; measured A = 0.75-1.10. The gate's model: overhead = 0.002 x 7 x 1024 = 14.3
ms > 0.1 x 65.8 = 6.6, so it drafts only while the acceptance EMA alpha >= 2 x 14.3 / 65.8 = 0.44; it verified on 10 of 45
steps (the first steps and after each recovery), skipped 35, and still beat plain by 9.8 ms/token. Forced drafting verified on 24
of 31 steps and beat the gate by a further 5.9 ms/token on this prompt: the 0.002 gate drafts sensibly (never a loss: A = 1.10 on
the steps it took; the plain steps it chose cost exactly a plain step) but is conservative at 1024 pages, as section 7 of the
verify-kernel doc said (alpha >= 0.47 there with a 61 ms step). Caveats: one prompt, one run per config, 49-74 generated tokens
(EOS), and the three configs produced different texts after the first ~60 chars (the known 1-row vs 8-row near-tie flips), so
the token counts differ; the per-step walls (65.0 / 65.8 / 66.0 plain, 83.6 / 84.3 verify) are the robust part.

A page cost of 0.001 would make the gate's 1024-page threshold alpha >= 0.22 (below the measured break-even 0.28 by a little,
above it at 512 pages: 0.001 x 7 x 512 = 3.6 ms < 6.1 = always try). Not changed here: with A = 0.75-1.10 measured on one prompt
the 0.002 default is the safe choice and already wins; recommend revisiting with more 8-16k prompts.

## 4. Validated / not validated / failures

- Validated on the box: 7 fp8 head buckets compile and load with the production layer set; real-engine speed for both heads,
  3 configs x 3 prompts, plain repeated 3x per head (interleaved); token identity and full-vocabulary KL at identical positions
  from the engine's own logits dump; a 220-position teacher-forced argmax comparison on the device (not an emulation); speculation
  acceptance with each head; the 1024-page gate with the staged dec8s_1024 bucket (plain / adaptive / forced); the hardened engine
  builds, ignores and logs the new meta key, and the old engine ignores it; `verify_graph_set.py` on two real graph dirs.
- Not validated: the fp8 head on non-greedy sampling (temperature > 0 was not run; the logits KL suggests no visible effect);
  the 14k / 2048-page bucket (not built); a production-prompt eval of fp8-head quality (3 prompts + 1 text here); the WARNING
  branches of the version log on a real mismatch (no second libnrt / compiler on the box; code-reviewed only); `build.neuronx_cc_bin` / `git_commit` after the
  stderr / no-git fix (missing in the one real trace, see section 2); multi-request steps.
- Failures / fixes: the HF `tokenizer.json` has `truncation.max_length = 4096`, which silently capped the 9.5k prompt at 5299
  tokens on the first attempt (`tok.no_truncation()`; the earlier docs' "long" 4398-token prompt was built through the same
  truncation, its token count is unaffected); the first engine process after the instance start took 213 s (cold page cache for
  the 24 GB prepared set), all later loads ~30 s; the box had 3.2-3.8 GB free throughout, the ~600 MB of logits dumps were deleted
  at the end (`sw/` is 28 MB).
- Instance i-0adbefb8af607c610 was stopped at the end of the session (describe-instances: stopped).
