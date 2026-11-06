# Experimental Neuron inference path

> The tensor-parallel, device-resident serving path (paged KV, flash attention, cFP8, the server's Engine and an HTTP
> server) is documented in [SERVING.md](SERVING.md); the text below describes the earlier single-sequence NKI runner.

This directory contains a correctness-first Llama runner using NKI 0.7 kernels.
It reads indexed Behemoth QTIP checkpoints, including per-channel FP8 and
adaptive NVFP4 projections, and supports one sequence with GQA, RoPE (including
Llama-3 scaling), causal prefill and cached decode. The token-ID CLI is:

```sh
NEURON_LOGICAL_NC_CONFIG=1 python neuron/run_tokens.py /path/to/checkpoint --tokens 1,2,3 --generate 4
```

It defaults to the NKI CPU simulator. `--device` asks NKI to compile and run
its individual kernels on an attached Neuron device; that path has been exercised on a Trn1 instance with small synthetic
inputs. It is not a production inference server. The experimental `server.py` exposes a serialized greedy-only chat
endpoint using the local Hugging Face tokenizer and chat template. It has no
streaming, tensor parallelism, concurrent execution, or device-resident model
or KV cache. Weights and cache
are held on the host and kernels are dispatched individually; compilation,
transfers and serial loops will make this extremely slow for large models.
The token-ID CLI does not implement EOS handling; the chat endpoint does.

QTIP input/output rotations, packed trellis decode and projection are in
`qtip_kernel.py`. `project_rotated_loop` uses compiled loops so graph size does
not grow by unrolling every weight block. The FP8/NVFP4 projection loops and
attention operations are in `ops.py`; the GQA online-softmax orchestration is
in `backend.py`. Kernels are correctness baselines: they do not exploit the
Tensor Engine for projection, use replicated LUTs and gather-heavy Hadamard
index tables. Factor-12 and factor-28 transforms are split into bounded chunks;
other shapes still need device validation. The
Llama implementation stores BF16-rounded activations and BF16-valued cache
entries on the host; it does not reproduce the ROCm runtime's FP8 KV-cache
arithmetic. Supported dimensions and formats are checked and unsupported
architectures or checkpoint layouts fail explicitly. There is no automatic
ROCm-to-Neuron translator.

On CPU simulation, tests compare all values of four QTIP blocks and a 64x64
projection against the existing C++ reference, FP8/NVFP4 and NKI operations
against NumPy, GQA across a 128-token tile boundary, and three tokens through
an independently implemented tiny Llama model. The 12- and 28-factor Hadamard
variants are checked in both orientations against C++ at small widths. The small-kernel suite and the tiny FP8 Llama test have also passed on a
Trn1 device; they do not establish full-size hardware correctness or
performance.

Standalone kernels have compiled to Trn2 NEFF: a 12,288x12,288 QTIP
projection, 4,096x1,024 FP8 and NVFP4 projections, 12,288- and 28,672-element factored Hadamard transforms,
RMSNorm, SwiGLU, RoPE, and attention tiles. The full model has not been
compiled to one NEFF. A tiny FP8 Llama ran on Trn1; a real checkpoint has not.

To run the tests on Linux x86-64 with Python 3.12 and a C++20 compiler:

```sh
python3.12 -m venv /tmp/qtip-neuron-venv
/tmp/qtip-neuron-venv/bin/pip install 'nki==0.7.0' --extra-index-url https://pip.repos.neuron.amazonaws.com
export NEURON_LOGICAL_NC_CONFIG=1
/tmp/qtip-neuron-venv/bin/python neuron/tests/test_qtip_kernel.py
/tmp/qtip-neuron-venv/bin/python neuron/tests/test_ops.py
/tmp/qtip-neuron-venv/bin/python neuron/tests/test_backend.py
/tmp/qtip-neuron-venv/bin/python neuron/tests/test_checkpoint.py
/tmp/qtip-neuron-venv/bin/python neuron/tests/test_llama.py
/tmp/qtip-neuron-venv/bin/python neuron/tests/test_server.py
```

The QTIP test builds a temporary oracle from `runtime/src/qtip.cpp`. Set `CXX`
to select a compiler or `QTIP_ORACLE` to use an already-built oracle. No model
checkpoint or AWS hardware is needed for those tests.

With a compatible `neuronx-cc` also installed, compile the small projection:

```sh
/tmp/qtip-neuron-venv/bin/pip install 'neuronx-cc==2.28.7631.0' --extra-index-url https://pip.repos.neuron.amazonaws.com
PATH=/tmp/qtip-neuron-venv/bin:$PATH /tmp/qtip-neuron-venv/bin/python neuron/tests/compile_trn2.py /tmp/qtip-project.neff
```

To try the experimental chat endpoint, install a compatible Transformers
version and run `python neuron/server.py /path/to/checkpoint`. It listens on
`127.0.0.1:8000` by default; add `--device` to request device execution. Only
`POST /v1/chat/completions` with `temperature: 0` and no streaming is
supported. Hardware serving is untested.

On Trn1.2xlarge with NKI 0.7.0 and NeuronX Compiler 2.28.7631.0,
`test_device.py` passed ten small simulator-vs-device comparisons and
`NEURON_TEST_DEVICE=1 test_llama.py` passed the tiny FP8 Llama comparison.
The standalone serial 1024x1024 QTIP projection took 6.87 seconds on the first
call and 1.25 seconds on the second, including compilation/dispatch and host
transfers. These are single-call wall times, not inference tokens per second
or isolated kernel timings.

The Anubis Mini 8B base configuration (hidden 4,096, MLP 14,336) also passed
Trn1 and Trn2 compilation for QTIP up/down projections, RMSNorm, SwiGLU and both stages
of the factor-28 Hadamard. These shape checks used synthetic tensors; they do
not validate a model checkpoint or measure performance. To repeat a check, run
`neuron/tests/compile_model_shapes.py` with `qtip-up`, `qtip-down`, `norm`,
`swiglu`, `had-chunks`, or `had-mix`, setting `NKI_ARTIFACTS_DIR` to an empty
directory and `NEURON_PLATFORM_TARGET_OVERRIDE=trn2`.

## Real Anubis Mini checkpoint validation on Trn1

Revision `d34522c3cf326f6e4591d3c543c5e7d5bd633724` has 34 LFS files;
the downloaded files were checked against their SHA-256 metadata. On Trn1,
layer 0 QTIP q_proj (4096→4096) matched the C++ reference with relative L2
2.42e-7, and layer 0 QTIP up_proj (4096→14336) matched at 2.61e-7. Layer 0
FP8 v_proj (4096→1024) matched an independent CPU reference at 1.79e-7, and
layer 2 NVFP4 up_proj (4096→14336) matched at 2.68e-7. The FP8 and NVFP4
comparisons use the checkpoint's per-channel/per-block scales and BF16 input
and NVFP4 weight rounding. These are single-input projection checks, not a
model-level quality evaluation.

Two hardware-specific corrections were necessary. Trn1's native FP8 conversion
interprets E4M3FN values with exponent 15 as infinities or NaNs; byte-indexed
lookup now decodes the checkpoint format explicitly, including NVFP4 block
scales. The power-of-two Hadamard output path also produced wrong results on
Trn1 when the first indexed gather consumed a direct load; the backend now
materializes input with a tensor scale and applies output scale separately.
Direct unsupported output-transform use is rejected rather than silently
producing incorrect results.

Large FP8/NVFP4 projections are split into bounded row launches to avoid the
runtime timeout observed on a full NVFP4 MLP projection. The experimental
NKI 0.7.0 standalone cache (`nki_cache.py`) retains compiled kernels by input
signature and scalar parameters. It uses private NKI interfaces and explicitly
rejects other versions. It transfers inputs and outputs via host memory on each
launch. On the tested real NVFP4 projection, first-run wall time including
compilation dropped from 104 seconds to 39 seconds. Neither number is a
steady-state model throughput measurement.
