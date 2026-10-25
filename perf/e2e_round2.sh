#!/bin/bash
# End-to-end decode measurement against the running behemoth_server on 127.0.0.1:1337.
# Usage: e2e_round2.sh TAG [CONCURRENCIES="1 4 8"] [TOKENS=96]
# Per concurrency: two warm-up runs (first-use autotuning for that batch size), then
# 2*c requests of TOKENS tokens. Writes perf/results/round2/TAG_c{c}.json and a summary line.
set -u
ROOT=/home/charles/Development/Behemoth-R1-123B-v2-aqlm-codebase/cpp_runtime
OUT=$ROOT/perf/results/round2
mkdir -p $OUT
TAG=$1
CONCURRENCIES=${2:-"1 4 8"}
TOKENS=${3:-96}
PROMPT=$OUT/prompt.txt
echo "Write a short, vivid paragraph about a lighthouse keeper who discovers that the fog has started to whisper names. Keep it under 150 words." > $PROMPT
cd $ROOT
for c in $CONCURRENCIES; do
  timeout 900 build-rocm/behemoth_load_bench $PROMPT $c $c 24 127.0.0.1 1337 > /dev/null 2>&1 || { echo "$TAG c$c warm-up 1 failed"; continue; }
  timeout 900 build-rocm/behemoth_load_bench $PROMPT $c $c 24 127.0.0.1 1337 > /dev/null 2>&1 || { echo "$TAG c$c warm-up 2 failed"; continue; }
  curl -s http://127.0.0.1:1337/metrics > $OUT/${TAG}_c${c}_before.json
  timeout 1800 build-rocm/behemoth_load_bench $PROMPT $c $((c*2)) $TOKENS 127.0.0.1 1337 > $OUT/${TAG}_c${c}.json 2> $OUT/${TAG}_c${c}.err || { echo "$TAG c$c measurement failed"; continue; }
  curl -s http://127.0.0.1:1337/metrics > $OUT/${TAG}_c${c}_after.json
  python3 - $OUT $TAG $c $TOKENS <<'PY'
import json, sys
out, tag, c, tokens = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
d = json.load(open(f"{out}/{tag}_c{c}.json"))
b = json.load(open(f"{out}/{tag}_c{c}_before.json"))
a = json.load(open(f"{out}/{tag}_c{c}_after.json"))
m = d.get("server_rolling_metrics", {})
toks = a["completion_tokens"] - b["completion_tokens"]
secs = a["gpu_step_seconds"] - b["gpu_step_seconds"]
# 2 waves of c requests: each wave = 1 prefill step + (tokens-1) decode steps
steps = 2 * tokens
step_ms = secs / steps * 1000
print(f"{tag} c={c}: load_bench {d['completion_tokens_per_second']:.2f} tok/s, request p50 {d['request_latency_p50_seconds']:.2f} s, "
      f"gpu step mean {step_ms:.1f} ms ({toks} tokens, {secs:.1f} s GPU over {steps} steps), "
      f"steady aggregate {c * 1000 / step_ms:.2f} tok/s, server ttft p50 {m.get('ttft_p50_ms', 0):.0f} ms")
PY
done
echo "$TAG DONE"
