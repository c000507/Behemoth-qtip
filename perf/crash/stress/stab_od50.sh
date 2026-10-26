#!/bin/bash
# Stability test at OD -500 MHz / -50 mV / 210 W: alternating long-prefill (~20k tok) and c=4 decode, ~45 min.
cd ~/Development/Behemoth-R1-123B-v2-aqlm-codebase/cpp_runtime
cat *.md perf/crash/*.md 2>/dev/null | head -c 85000 > /tmp/long20k.txt
B=build-rocm/behemoth_load_bench
gpus() { a=$(sudo setpci -s 33:00.0 VENDOR_ID); b=$(sudo setpci -s cc:00.0 VENDOR_ID); echo "$a$b"; }
end=$(( $(date +%s) + 45*60 )); i=0
while [ $(date +%s) -lt $end ]; do
  i=$((i+1))
  echo "=== round $i $(date +%T) gpus=$(gpus)"
  [ "$(gpus)" != "10021002" ] && { echo "GPU LOST $(date +%T)"; exit 2; }
  timeout 900 $B /tmp/long20k.txt 1 1 64 127.0.0.1 1337 | python3 -c "import json,sys; d=json.load(sys.stdin); print('long: prompt_toks',d['mean_prompt_tokens'],'lat_s',round(d['request_latency_p50_seconds'],1))" 2>&1 | tail -1
  [ "$(gpus)" != "10021002" ] && { echo "GPU LOST $(date +%T)"; exit 2; }
  timeout 900 $B ~/stress/prompt.txt 4 16 128 127.0.0.1 1337 | python3 -c "import json,sys; d=json.load(sys.stdin); print('c4: tok/s',round(d['completion_tokens_per_second'],1),'itl',round(d['server_rolling_metrics']['inter_token_p50_ms'],1))" 2>&1 | tail -1
done
echo "STABLE 45min rounds=$i $(date +%T) gpus=$(gpus)"
