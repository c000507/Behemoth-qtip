# Reproducible NEFF bucket builds: version pins, provenance in meta.json, and the libnrt check

The production NEFFs are compiled offline (`neuron/build_buckets.sh` -> `neuron/trace_model.py` -> `neuronx-cc`) in a venv
that is not pinned by the repo (`/opt/aws_neuronx_venv_pytorch_2_9` on the build box: neuronx-cc 2.26 today, 2.28 on an
older box), and the serving image installs whatever `aws-neuronx-runtime-lib` is current at image build time
(`deploy/ecs-neuron/backend/Dockerfile`, build arg `NEURON_RUNTIME_VERSION` empty = latest). This note records what the
repo now does about it. Nothing here changes default engine behaviour: the new meta keys are ignored unless asked for.

## What is recorded

`trace_model.py` writes a `"build"` object into every bucket's `meta.json` (`build_info()`, best effort: a missing tool
leaves a key out and never fails the trace):

| key | source | example (build box, 2026-11-10) |
|---|---|---|
| `neuronx_cc` / `neuronx_cc_bin` | `importlib.metadata` of the venv / `neuronx-cc --version` of the binary on PATH | `2.26.6360.0+6f180f47` |
| `torch_neuronx`, `torch_xla`, `torch`, `nki`, `libneuronxla` | `importlib.metadata` | `2.9.0.2.15.32035+de43f57c`, `2.9.0`, `2.9.1`, `0.5.0+28631259367.ga768afa6`, `2.2.17544.0+fb9962bf` |
| `libnrt` | `dpkg-query -W aws-neuronx-runtime-lib` on the build host | `2.33.10.0-3dcef56f0` |
| `git_commit`, `git_dirty` | `git rev-parse HEAD` / `git status --porcelain` of the `neuron/` tree | `0b8be21...`, `false` |
| `env` | the trace flags that shape the graph: `MODEL CONFIG TP PAGES WEIGHTS KV KVATT ATTN_DEC ATTN_DEC_SHARED ATTN_DEC_MIN_PAGES ATTN_PROFILE WT MLP DOWN_BF16 OUT_F32 HEAD_ROWS HEAD_FP8 HEAD_ARGMAX UNSAFE_FP8FNCAST NCC_FLAGS NEURON_PLATFORM_TARGET_OVERRIDE` (only those set) | `{"ATTN_DEC": "nki", "WT": "1", "MLP": "nki", "KV": "fp8", ...}` |
| `timestamp`, `python` | UTC build time, interpreter version | `2026-11-10T13:40:12Z`, `3.12.3` |

The runtime (`neuron/cpp/neuron_model.cpp`) parses `meta.json` with `at()` / `value()` on known keys only, so unknown keys
(including `build`) are ignored by every engine version, old or new; the fp8-head and dec8s buckets of
docs/neuron-small-wins.md were loaded by the current engine with and without the key.

## Version log and mismatch warning (off by default)

`BEHEMOTH_NEURON_LOG_VERSIONS=1` makes the engine print, at bucket discovery (before any device work):

```
Neuron versions: libnrt loaded 2.33.10.0            <- nrt_get_version() of the library this process linked
Neuron versions: 36 bucket(s) compiled with neuronx-cc 2.26.6360.0+6f180f47
Neuron versions: 7 bucket(s) without build info in meta.json (older trace_model.py)
WARNING: the NEFF buckets mix neuronx-cc versions ...                       <- only when they do
WARNING: N bucket(s) were compiled on a host with aws-neuronx-runtime-lib X, this process loaded libnrt Y ...
```

Both WARNINGs are stderr lines only; the engine never refuses a set for version reasons (the NEFF format is the
compatibility contract, and libnrt refuses a NEFF it cannot run with its own error). Without the variable nothing is printed.

## Rebuilding a bucket set reproducibly

1. Build host: a trn1 instance (the compiler needs no device, but `UNSAFE_FP8FNCAST=1` NKI kernels and the final load test do).
   Create the venv from the pin file: `python3.12 -m venv /opt/nb && /opt/nb/bin/pip install --extra-index-url
   https://pip.repos.neuron.amazonaws.com -r neuron/requirements-neuron-build.txt` (the pins are the exact `pip list` of the box
   that compiled the production set; `neuronx-cc --version` must print the pinned version), or use
   `/opt/aws_neuronx_venv_pytorch_2_9` after checking its versions against the file.
2. Compile from a clean checkout of the commit you want recorded: `MODEL=/path/to/checkpoint TP=2 SHARED_VERIFY=1
   neuron/build_buckets.sh OUT 5561` (Devstral: `PAGES=5561`; add `HEAD_FP8=1` only if docs/neuron-small-wins.md's verdict
   makes fp8 heads the default). `build.env` of every bucket then says which flags were in effect.
3. Check the set: `neuron/verify_graph_set.py OUT` lists every bucket with its compiler / libnrt / commit / timestamp and exits 1
   on mixed compiler versions, mixed head formats, a directory without `meta.json` or `graph.neff`, or buckets without build info
   (an old set re-used next to new buckets: rebuild the old ones, or accept knowingly).
4. Upload under the Terraform `graphs_s3_prefix` (one directory per bucket), and build the image with the runtime of the build
   host pinned: `--build-arg NEURON_RUNTIME_VERSION=<dpkg version in build.libnrt> --build-arg NEURON_COLLECTIVES_VERSION=<same
   series>` (`apt-cache madison aws-neuronx-runtime-lib` lists them).
5. Verify in the container: start the server once with `BEHEMOTH_NEURON_LOG_VERSIONS=1` and read the two `Neuron versions` lines:
   the loaded libnrt should be the series the buckets record (`2.33.10.0` here); a WARNING means the image and the buckets were
   built against different runtimes - allowed (libnrt is backward compatible with older NEFFs within a major series), but it is
   the first thing to suspect when a bucket set that worked on the build box fails to load in the image.

## Known gaps

- The `libnrt` field is the build host's dpkg version string, the loaded version is numeric; the comparison is a prefix match
  (`2.33.10.0-3dcef56f0` vs `2.33.10.0`), so a rebuild of the same version with another hash is treated as equal.
- `neuronx-cc --version` is taken from PATH at trace time; `build_buckets.sh` prepends the venv's `bin`, so it is the venv's
  compiler, but a manual `python trace_model.py` with another PATH records that other compiler (both fields are kept for that
  reason).
- The pin file is a snapshot of one box; the Neuron pip index drops old wheels over time, so keep the venv (or an image of it) if
  the exact set must be rebuilt years later.
