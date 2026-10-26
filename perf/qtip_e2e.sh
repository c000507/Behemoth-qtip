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
metric() { curl -s $M | python3 -c "import json,sys; print(json.load(sys.stdin)['$1'])"; }
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
python3 - $OUT $TAG $s0 $t0 $s1 $t1 $s2 $t2 $itl <<'PY'
import json, sys
out, tag = sys.argv[1], sys.argv[2]
s0, t0, s1, t1, s2, t2, itl = map(float, sys.argv[3:10])
def j(n):
    try: return json.load(open(f"{out}/{tag}_{n}.json"))
    except Exception: return {}
c1, c4, a, b, d = j("c1"), j("c4"), j("ttft2k"), j("ttft13k"), j("dec13k")
fm = json.load(open(f"{out}/{tag}_final_metrics.json"))
lat = lambda r: r.get("request_latency_p50_seconds", float("nan"))
dec13 = (lat(d) - lat(b)) / 127 * 1000
res = {"tag": tag,
       "c1_inter_token_p50_ms": itl,
       "c1_gpu_step_mean_ms": (s1 - s0) / max(t1 - t0, 1) * 1000,
       "c1_tok_s_load_bench": c1.get("completion_tokens_per_second"),
       "c4_aggregate_tok_s_load_bench": c4.get("completion_tokens_per_second"),
       "c4_gpu_step_mean_ms": (s2 - s1) / max((t2 - t1) / 4, 1) * 1000,
       "ttft_2k_s": lat(a), "prompt_2k_tokens": a.get("mean_prompt_tokens"),
       "ttft_13k_s": lat(b), "prompt_13k_tokens": b.get("mean_prompt_tokens"),
       "decode_13k_ms_per_token": dec13,
       "kv_pages_total": fm.get("kv_pages_total")}
json.dump(res, open(f"{out}/{tag}_summary.json", "w"), indent=1)
print(json.dumps(res))
PY
