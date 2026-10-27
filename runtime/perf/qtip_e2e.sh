#!/bin/bash
# End-to-end measurements against a running behemoth_server on 127.0.0.1:1337 (QTIP or AQLM).
# Usage: perf/qtip_e2e.sh TAG [EVAL_DIR=/home/charles/qtip_eval]
# Writes perf/results/qtip/TAG_*.json and prints a summary:
#   c=1 short prompt: 8 x 128 tokens -> /metrics inter_token_p50_ms (the 1024-sample window is
#     fully replaced), GPU step mean;  c=4: 8 x 96 tokens -> aggregate tok/s, GPU step mean;
#   TTFT for the ~2k and ~13k prompts (1 output token, latency = prefill + one sample);
#   decode at ~13k context: (latency(128 tokens) - latency(1 token)) / 127.
set -u
ROOT=/home/charles/Development/Behemoth-R1-123B-v2-aqlm-codebase/cpp_runtime
EVAL=${2:-/home/charles/qtip_eval}
OUT=$ROOT/perf/results/qtip
mkdir -p $OUT
TAG=$1
B=$ROOT/build-rocm/behemoth_load_bench
M=http://127.0.0.1:1337/metrics
cd $ROOT
run() { # name prompt conc requests tokens
  timeout 3600 $B $2 $3 $4 $5 127.0.0.1 1337 > $OUT/${TAG}_$1.json 2> $OUT/${TAG}_$1.err || echo "$TAG $1 FAILED"
}
metric() { curl -fsS "$M" | jq -er --arg key "$1" '.[$key]'; }
# warm-up (first-use autotuning of lm_head / attention shapes, first long-prefill effects)
timeout 900 $B $EVAL/prompt_short.txt 1 2 32 127.0.0.1 1337 > /dev/null 2>&1
timeout 900 $B $EVAL/prompt_short.txt 4 4 32 127.0.0.1 1337 > /dev/null 2>&1
timeout 900 $B $EVAL/prompt_2k.txt 1 1 4 127.0.0.1 1337 > /dev/null 2>&1
s0=$(metric gpu_step_seconds); t0=$(metric completion_tokens)
run c1 $EVAL/prompt_short.txt 1 8 128
s1=$(metric gpu_step_seconds); t1=$(metric completion_tokens)
itl=$(metric inter_token_p50_ms)
curl -s $M > $OUT/${TAG}_c1_metrics.json
run c4 $EVAL/prompt_short.txt 4 8 96
s2=$(metric gpu_step_seconds); t2=$(metric completion_tokens)
run ttft2k $EVAL/prompt_2k.txt 1 2 1
run ttft13k $EVAL/prompt_13k.txt 1 1 1
run dec13k $EVAL/prompt_13k.txt 1 1 128
curl -s $M > $OUT/${TAG}_final_metrics.json
jq -n --arg tag "$TAG" \
  --argjson s0 "$s0" --argjson t0 "$t0" --argjson s1 "$s1" --argjson t1 "$t1" \
  --argjson s2 "$s2" --argjson t2 "$t2" --argjson itl "$itl" \
  --slurpfile c1 "$OUT/${TAG}_c1.json" --slurpfile c4 "$OUT/${TAG}_c4.json" \
  --slurpfile a "$OUT/${TAG}_ttft2k.json" --slurpfile b "$OUT/${TAG}_ttft13k.json" \
  --slurpfile d "$OUT/${TAG}_dec13k.json" --slurpfile fm "$OUT/${TAG}_final_metrics.json" '
    {tag: $tag, c1_inter_token_p50_ms: $itl,
     c1_gpu_step_mean_ms: (($s1-$s0) / ([$t1-$t0, 1] | max) * 1000),
     c1_tok_s_load_bench: $c1[0].completion_tokens_per_second,
     c4_aggregate_tok_s_load_bench: $c4[0].completion_tokens_per_second,
     c4_gpu_step_mean_ms: (($s2-$s1) / ([($t2-$t1)/4, 1] | max) * 1000),
     ttft_2k_s: $a[0].request_latency_p50_seconds, prompt_2k_tokens: $a[0].mean_prompt_tokens,
     ttft_13k_s: $b[0].request_latency_p50_seconds, prompt_13k_tokens: $b[0].mean_prompt_tokens,
     decode_13k_ms_per_token: (($d[0].request_latency_p50_seconds - $b[0].request_latency_p50_seconds) / 127 * 1000),
     kv_pages_total: $fm[0].kv_pages_total}
  ' | tee "$OUT/${TAG}_summary.json"
