#!/bin/bash
# build, launch microbench, gpu tests, bench (rows list), policy sweep at rows 1 and 4
set -u
ROOT=/home/charles/Development/Behemoth-R1-123B-v2-aqlm-codebase/cpp_runtime
OUT=$ROOT/perf/results/round2
cd $ROOT/.. && cmake --build cpp_runtime/build-rocm -j 16 2>&1 | grep -E "error|FAILED" | head; echo "BUILD_EXIT=${PIPESTATUS[0]}"
cd $ROOT
HIP_VISIBLE_DEVICES=0 /tmp/micro_rocm10 launch > $OUT/launch_rocm10.txt 2>&1; cat $OUT/launch_rocm10.txt
perf/gpu_check.sh lane3 "1 2 4" sorted-lane 1
M=/mnt/nvme/models/Behemoth-R1-123B-v2-aqlm-output
for pol in 0 1 16 17 24 25; do for r in 1 4; do
  BEHEMOTH_LANE_POLICY=$pol timeout 900 build-rocm/behemoth_projection_bench $M $r 0 1 lane > $OUT/lane3_pol${pol}_rows$r.json 2>/dev/null
  echo "policy $pol rows $r: $(python3 perf/summarize_round2.py $OUT/lane3_pol${pol}_rows$r.json | grep -E '^  lane')"
done; done
echo ALLDONE
