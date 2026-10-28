#!/bin/bash
# Qwen3.5 GPU validation queue — run when the GPUs are free (one GPU process at a time).
# Usage: bash /tmp/qwen35_gpu_validate.sh [step ...]   (default: all, in order)
set -u
cd ~/Development/behemoth-qtip-qwen35
LOG=/tmp/qwen35_gpu.log
exec >> $LOG 2>&1
say() { echo "[$(date +%T)] $*"; }
Q=/mnt/nvme/qwen35-qt
OLD=$Q/Qwen3.5-0.8B-QTIP-2bit
NEW=$Q/Qwen3.5-0.8B-QTIP-2bit-4m
TINY=~/qwen_tiny_moe
TINYQ=$Q/qwen-tiny-moe-QTIP-2bit

step1() { # runtime ppl of the old checkpoint (fixed final norm) vs quantized reference 9.3136
  say "step1: runtime ppl old checkpoint (expect ~9.31)"
  timeout 1200 ./build-runtime/qwen_eval --model $OLD --ppl /tmp/qwen_eval.i32 --chunk 256 --device 0
  timeout 300 ./build-runtime/qwen_eval --model $OLD --prompt /tmp/qwen_prompt.i32 --greedy 12 --device 0
}
step2() { # re-quantize 0.8B with 4M calibration tokens
  say "step2: quantize 0.8B with 4M tokens"
  rm -rf $NEW $Q/work4m
  timeout 14400 ./build/qtip_quant --mode run --model /mnt/nvme/models/Qwen3.5-0.8B \
    --out $NEW --work $Q/work4m --tokens /tmp/qwen_calib.i32 --ntok 4194304 --seq-len 2048 \
    --gpus 0,1 --batch-rows 4096 --backlog 2
  timeout 600 ./build/qtip_quant --mode assemble --model /mnt/nvme/models/Qwen3.5-0.8B --out $NEW --work $Q/work4m
}
step3() { # quantized-reference ppl of the new checkpoint (want closer to 5.7415)
  say "step3: qtref ppl new checkpoint"
  timeout 3600 ./build-runtime/qwen_qt_ref $NEW /tmp/qwen_eval.i32 4096 "" x --ppl
}
step4() { # runtime ppl/greedy of the new checkpoint vs step3
  say "step4: runtime ppl/greedy new checkpoint"
  timeout 1200 ./build-runtime/qwen_eval --model $NEW --ppl /tmp/qwen_eval.i32 --chunk 256 --device 0
  timeout 300 ./build-runtime/qwen_eval --model $NEW --prompt /tmp/qwen_prompt.i32 --greedy 12 --device 0
}
step5() { # tiny MoE: quantize (per-expert Hessians) + runtime parity vs CPU reference
  say "step5: tiny MoE quantize + parity"
  python3 - <<'EOF'
import struct, random
random.seed(9)
t = [random.randrange(0, 2048) for _ in range(4096)]
open("/tmp/tiny_calib.i32", "wb").write(struct.pack("%di" % len(t), *t))
EOF
  rm -rf $TINYQ $Q/worktiny
  timeout 3600 ./build/qtip_quant --mode run --model $TINY --out $TINYQ --work $Q/worktiny \
    --tokens /tmp/tiny_calib.i32 --ntok 4096 --seq-len 64 --gpus 0 --batch-rows 1024
  timeout 600 ./build/qtip_quant --mode assemble --model $TINY --out $TINYQ --work $Q/worktiny
  # CPU reference of the quantized tiny model + GPU runtime ppl, both vs the bf16 baseline
  timeout 1800 ./build-runtime/qwen_qt_ref $TINYQ /tmp/tiny_ppl.bin 1200 "" x --ppl
  timeout 600 ./build-runtime/qwen_eval --model $TINYQ --ppl /tmp/tiny_ppl.bin --chunk 256 --device 0
  # bf16 runtime path vs the HF-validated CPU reference logits
  timeout 600 ./build-runtime/qwen_eval --model $TINY --prompt /tmp/tiny_toks.bin --greedy 8 --device 0
  QWEN_LOGITS_OUT=/tmp/tiny_rt_bf16_logits.bin timeout 600 ./build-runtime/qwen_eval --model $TINY \
    --prompt /tmp/tiny_toks.bin --greedy 1 --device 0
  QWEN_LOGITS_OUT=/tmp/tiny_rt_qt_logits.bin timeout 600 ./build-runtime/qwen_eval --model $TINYQ \
    --prompt /tmp/tiny_toks.bin --greedy 1 --device 0
  timeout 1800 ./build-runtime/qwen_qt_ref $TINYQ /tmp/tiny_toks.bin 16 /tmp/tiny_qtref_logits.bin
  python3 - <<'EOF'
import numpy as np
def b16(a): return (a.astype(np.uint32) << 16).view(np.float32)
cpu = np.fromfile("/tmp/tiny_logits.bin", dtype=np.float32).reshape(16, 2048)
try:
    rt_bf16 = np.fromfile("/tmp/tiny_rt_bf16_logits.bin", dtype=np.float32)[:2048]
    print("bf16 runtime vs CPU-ref logits: max|d| %.5f corr %.6f" %
          (np.abs(rt_bf16 - cpu[15]).max(), np.corrcoef(rt_bf16[::3], cpu[15][::3])[0, 1]))
except Exception as e:
    print("bf16 runtime logits:", e)
try:
    qtref = np.fromfile("/tmp/tiny_qtref_logits.bin", dtype=np.float32).reshape(16, 2048)
    rt_qt = np.fromfile("/tmp/tiny_rt_qt_logits.bin", dtype=np.float32)[:2048]
    print("quantized runtime vs quantized-ref (row15): max|d| %.5f corr %.6f" %
          (np.abs(rt_qt - qtref[15]).max(), np.corrcoef(rt_qt[::3], qtref[15][::3])[0, 1]))
    print("quantized-ref vs original: max|d| %.5f corr %.6f" %
          (np.abs(qtref[15] - cpu[15]).max(), np.corrcoef(qtref[15][::3], cpu[15][::3])[0, 1]))
except Exception as e:
    print("quantized logits:", e)
EOF
}
step6() { # quantizer GPU parity suite
  say "step6: qtip_tests_gpu"
  QWEN35_MODEL=/mnt/nvme/models/Qwen3.5-0.8B timeout 3600 ./build/qtip_tests_gpu 2>&1 | tail -5
}

say "=== qwen35 GPU validation start (steps: ${*:-all}) ==="
for s in "${@:-1 2 3 4 5 6}"; do
  "step$s" || say "step$s FAILED rc=$?"
done
say "=== done ==="
