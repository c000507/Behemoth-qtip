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
  jq -nr --arg tag "$TAG" --argjson c "$c" --argjson tokens "$TOKENS" \
    --slurpfile d "$OUT/${TAG}_c${c}.json" \
    --slurpfile b "$OUT/${TAG}_c${c}_before.json" \
    --slurpfile a "$OUT/${TAG}_c${c}_after.json" '
      ($a[0].completion_tokens - $b[0].completion_tokens) as $toks
      | ($a[0].gpu_step_seconds - $b[0].gpu_step_seconds) as $secs
      | (2 * $tokens) as $steps | ($secs / $steps * 1000) as $ms
      | "\($tag) c=\($c): load_bench \($d[0].completion_tokens_per_second) tok/s, request p50 \($d[0].request_latency_p50_seconds) s, gpu step mean \($ms) ms (\($toks) tokens, \($secs) s GPU over \($steps) steps), steady aggregate \(if $ms > 0 then $c * 1000 / $ms else null end) tok/s, server ttft p50 \($d[0].server_rolling_metrics.ttft_p50_ms // 0) ms"
    '
done
echo "$TAG DONE"
