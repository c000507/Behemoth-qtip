#!/bin/bash
# gpu tests + lane/sorted bench at several row counts. Usage: gpu_check.sh TAG "ROWS..." [LAYOUTS=sorted-lane] [DEV=1]
set -u
ROOT=/home/charles/Development/Behemoth-R1-123B-v2-aqlm-codebase/cpp_runtime
OUT=$ROOT/perf/results/round2; mkdir -p $OUT
TAG=$1; ROWS=${2:-"1 2 4 8"}; LAYOUTS=${3:-sorted-lane}; DEV=${4:-1}
MODEL=/mnt/nvme/models/Behemoth-R1-123B-v2-aqlm-output
cd $ROOT
if [ "${SKIP_TESTS:-0}" != 1 ]; then
  echo "$(date +%T) gpu tests"; HIP_VISIBLE_DEVICES=$DEV timeout 1800 build-rocm/behemoth_gpu_tests > $OUT/${TAG}_gpu_tests.log 2>&1; echo "gpu_tests rc=$? $(tail -1 $OUT/${TAG}_gpu_tests.log)"
  timeout 600 build-rocm/behemoth_gpu_tests --two-gpu > $OUT/${TAG}_two_gpu.log 2>&1; echo "two_gpu rc=$? $(tail -1 $OUT/${TAG}_two_gpu.log)"
fi
for r in $ROWS; do
  echo "$(date +%T) bench rows=$r"
  timeout 1800 build-rocm/behemoth_projection_bench $MODEL $r 0 $DEV $LAYOUTS > $OUT/${TAG}_bench_rows$r.json 2> $OUT/${TAG}_bench_rows$r.err; echo "bench rows=$r rc=$?"
  bash perf/summarize_round2.sh $OUT/${TAG}_bench_rows$r.json
done
echo "$TAG DONE"
