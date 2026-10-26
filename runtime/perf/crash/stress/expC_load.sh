#!/bin/bash
# continuous load: c=4, 8 requests, 96 output tokens per batch, until failure
cd /home/charles/Development/Behemoth-R1-123B-v2-aqlm-codebase/cpp_runtime
for i in $(seq 1 240); do curl -sf http://127.0.0.1:1337/health >/dev/null && break; sleep 2; done
echo "$(date +%T) server healthy; start load"
n=0
while true; do
  n=$((n+1)); echo "$(date +%T) batch $n start"
  timeout 900 build-rocm/behemoth_load_bench ~/stress/prompt.txt 4 8 96 127.0.0.1 1337 > ~/stress/expE_batch_$n.json 2> ~/stress/expE_batch_$n.err; rc=$?
  echo "$(date +%T) batch $n rc=$rc tok/s=$(python3 -c "import json;print(json.load(open(\"$HOME/stress/expE_batch_$n.json\")).get(\"completion_tokens_per_second\"))" 2>/dev/null)"
  [ $rc -ne 0 ] && { echo "load failed rc=$rc"; break; }
done
