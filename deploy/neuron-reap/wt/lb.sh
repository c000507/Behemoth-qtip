#!/bin/bash
# lb.sh [-n steps] DIR...: layer_bench of NEFF dirs (states from meta.json), one line each: name ms/launch (best / median of 8 batches)
export PATH=/usr/local/bin:/opt/aws/neuron/bin:$PATH NEURON_RT_VISIBLE_CORES=0-1 NEURON_RT_ROOT_COMM_ID=127.0.0.1:62128
N=64; [ "$1" = -n ] && { N=$2; shift 2; }
for d in "$@"; do
  [ -f $d/graph.neff ] || { echo "$(basename $d) missing"; continue; }
  st=$(python3 -c 'import json,sys; m=json.load(open(sys.argv[1])); print(len(m.get("states",[])))' $d/meta.json)
  o=$(/mnt/scratch/BUILD/layer_bench $d/graph.neff $N 2 $st 2>&1 | tail -1)
  echo "$(basename $(dirname $d))/$(basename $d) $(echo "$o" | python3 -c 'import json,sys
try:
  j=json.loads(sys.stdin.read()); print("%.3f ms (median %.3f, max %.3f) in %.0f MB" % (j["ms_per_step"], j["median_ms"], j["max_ms"], j["input_MB_per_rank"]))
except Exception as e: print("ERR", e)' 2>&1) $(echo "$o" | grep -v '^{' | head -c 200)"
done
