#!/bin/bash
# Final bench sweep: all layouts, rows 1..8, 16, 32, 64, 256 on device 1.
set -u
ROOT=/home/charles/Development/Behemoth-R1-123B-v2-aqlm-codebase/cpp_runtime
OUT=$ROOT/perf/results/round2; cd $ROOT
M=/mnt/nvme/models/Behemoth-R1-123B-v2-aqlm-output
TAG=${1:-final}
for r in 1 2 3 4 5 6 7 8 16 32 64 256; do
  timeout 2400 build-rocm/behemoth_projection_bench $M $r 0 1 all > $OUT/${TAG}_rows$r.json 2> $OUT/${TAG}_rows$r.err; echo "rows $r rc=$?"
  python3 perf/summarize_round2.py $OUT/${TAG}_rows$r.json
done
echo SWEEPDONE
