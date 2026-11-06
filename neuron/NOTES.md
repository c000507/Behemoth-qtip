# Neuron work log

Environment: trn1.2xlarge i-0d79198d32a0df40f, 1 NeuronDevice (v2/Trn1, /dev/neuron0), 8 vCPU,
32GB RAM. Venv /home/ubuntu/codex-neuron-test/venv: nki 0.7.0+37713544100, neuronx-cc
2.28.7631.0, nrtpy 2.35.30, numpy 2.5.4, ml_dtypes 0.6.0. Model:
/home/ubuntu/models/anubis-mini-8b (Llama: 32 layers, hidden 4096, intermediate 14336, 32/8
heads, dim 128, vocab 128257, theta 500000, llama3 rope scaling; adaptive formats: layers
2-7,9,10 NVFP4 MLP, others QTIP; all K/V projections FP8 E4M3 per-channel; embed/lm_head BF16).
Repo checkout on instance: /home/ubuntu/qtip-neuron (branch rocm-to-neff).

## Device execution env (required)

```
PATH=/home/ubuntu/codex-neuron-test/venv/bin:/opt/aws/neuron/bin:...
LD_LIBRARY_PATH=/opt/aws/neuron/lib       # libnrt.so.1 for nrtpy
NEURON_PLATFORM_TARGET_OVERRIDE=trn1      # else target auto-detect falls back to trn3
NEURON_LOGICAL_NC_CONFIG=1
```
Wrapper: /home/ubuntu/work/nenv.sh. Without PATH, nki's penguin compiler path resolves via
shutil.which("neuronx-cc") to None and ncc dies with `can't open file '.../None'`; without
LD_LIBRARY_PATH `import nki.runtime` fails; without the target override ncc targets trn3
(neuron-ls not on PATH for the SSM shell).

## Model oracle (neuron/tests/model_oracle.cpp)

Full-model fp64 Llama forward over the adaptive checkpoint, reusing runtime/src/qtip.cpp
loaders + reference_forward (device_rounding=true, round_z_bf16 for gate/up), plus NVFP4
reference (BF16-rounded dequant weights, BF16 input, fp64 accumulate — matches ROCm gemv
"E2M1 x E4M3 x global with bf16 rounding"). Mirrors neuron/llama.py BF16 rounding points.
--fp8-kv switches attention to ROCm FP8 KV arithmetic (quantized Q/K/V + FP8 P per 32-token
tile, numeric.hpp encode/decode). Build:
g++ -std=c++20 -O2 -fopenmp -I runtime/include -I third_party neuron/tests/model_oracle.cpp
runtime/src/qtip.cpp runtime/src/checkpoint.cpp runtime/src/attention.cpp -o model_oracle

Bugs found+fixed during bring-up:
- NVFP4 gate/up: oracle first rejected round_z for NVFP4; runner ignores it (no rotated z).
- RoPE pairing bug: must pair (d, d+dim/2) within each 128-wide head row, not across the
  whole 4096-wide q vector. With wrong pairing "The capital of France is" argmax was 45663;
  after fix 12366 (" Paris", max logit 18.42).

Oracle timing: ~5s/token wall (8 vCPU, fp64 decode of every layer per token), RSS ~6.4GB.

## Sessions

### 2026-11-06: first real-checkpoint device run
Prompt 128000,791,6864,315,9822,374 (BOS + "The capital of France is"), --generate 2,
device mode, KernelCache. Layer 0 (QTIP+FP8 KV) ~31s incl. compile, layer 1 ~5s, NVFP4
layers slower. Log: /home/ubuntu/work/dev0.log, logits /home/ubuntu/work/dev0.
