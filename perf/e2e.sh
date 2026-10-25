#!/bin/bash
# End-to-end decode throughput: baseline binary vs prototype (host / zerocopy transport).
# Usage: e2e.sh  (run from cpp_runtime dir)
set -u
ROOT=/home/charles/Development/Behemoth-R1-123B-v2-aqlm-codebase/cpp_runtime
MODEL=/mnt/nvme/models/Behemoth-R1-123B-v2-aqlm-output
OUT=$ROOT/perf/results/e2e
mkdir -p $OUT
PROMPT=$OUT/prompt.txt
echo "Write a short, vivid paragraph about a lighthouse keeper who discovers that the fog has started to whisper names. Keep it under 150 words." > $PROMPT
# wait for GPU tests to finish
while pgrep -f behemoth_gpu_tests > /dev/null; do sleep 5; done
run_config() {
  local name=$1 bin=$2 transport=$3 layout=${4:-group-major}
  echo "=== $name ($bin --transport $transport)" 
  cd $ROOT
  setsid nohup $bin --model $MODEL --devices 0,1 --transport $transport --code-layout $layout --memory-fraction 0.9 \
      --token-budget 1024 --prefill-chunk 256 --host 127.0.0.1 --port 8080 --threads 4 --profile false \
      > $OUT/server_$name.log 2>&1 < /dev/null &
  local pid=$!
  for i in $(seq 1 240); do curl -sf http://127.0.0.1:8080/health > /dev/null 2>&1 && break; sleep 2; done
  curl -sf http://127.0.0.1:8080/health || { echo "server $name failed to start"; kill $pid; return; }
  for c in $CONCURRENCIES; do
    [ -f $OUT/${name}_c${c}.json ] && [ -s $OUT/${name}_c${c}.json ] && { echo "skip existing $name c$c"; continue; }
    # warm (first-use autotuning for this batch size), then measure; a hang aborts this config
    timeout 600 build-rocm/behemoth_load_bench $PROMPT $c $c 24 > /dev/null 2>&1 || { echo "$name c$c HANG/timeout in warmup"; curl -s http://127.0.0.1:8080/metrics > $OUT/${name}_c${c}_hang_metrics.json; break; }
    timeout 600 build-rocm/behemoth_load_bench $PROMPT $c $c 24 > /dev/null 2>&1 || { echo "$name c$c HANG/timeout in warmup 2"; break; }
    timeout 900 build-rocm/behemoth_load_bench $PROMPT $c $((c*2)) 96 > $OUT/${name}_c${c}.json 2> $OUT/${name}_c${c}.err || { echo "$name c$c HANG/timeout in measurement"; break; }
    curl -s http://127.0.0.1:8080/metrics > $OUT/${name}_c${c}_metrics.json
    python3 - "$OUT/${name}_c${c}.json" $name $c <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
m=d.get("server_rolling_metrics",{})
keys=[k for k in d if "token" in k or "second" in k or "latency" in k or "ms" in k]
m=d.get("server_rolling_metrics",{})
print(sys.argv[2], "concurrency", sys.argv[3], "load_bench tok/s=%.2f"%d["completion_tokens_per_second"], "p50 latency=%.1fs"%d["request_latency_p50_seconds"], "server inter_token p50=%.1f ms p95=%.1f ms ttft p50=%.0f ms"%(m["inter_token_p50_ms"],m["inter_token_p95_ms"],m["ttft_p50_ms"]))
PY
  done
  kill $pid; for i in $(seq 1 30); do kill -0 $pid 2>/dev/null || break; sleep 1; done; kill -9 $pid 2>/dev/null
  sleep 3
}
# Concurrency list: keep <= 8 (both GPUs fell off the PCIe bus during a c=16 run).
CONCURRENCIES=${CONCURRENCIES:-"1 8"}
CONFIGS=${CONFIGS:-"baseline proto_host proto_zerocopy sorted_zerocopy"}
for cfg in $CONFIGS; do
  case $cfg in
    baseline) run_config baseline /home/charles/Development/behemoth-perf/cpp_runtime/build-rocm/behemoth_server host ;;
    proto_host) run_config proto_host $ROOT/build-rocm/behemoth_server host ;;
    proto_zerocopy) run_config proto_zerocopy $ROOT/build-rocm/behemoth_server zerocopy ;;
    sorted_host) run_config sorted_host $ROOT/build-rocm/behemoth_server host sorted ;;
    sorted_zerocopy) run_config sorted_zerocopy $ROOT/build-rocm/behemoth_server zerocopy sorted ;;
  esac
done
echo DONE
