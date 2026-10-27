#!/bin/bash
# exp.sh TAG MINUTES SOFF VOFF PCAP : apply OD point, c=1 probe, then continuous c=4 load for MINUTES; detect GPU loss.
TAG=$1; MIN=$2; shift 2
E=~/od_exp; L=$E/$TAG.log; B=/home/charles/Development/Behemoth-R1-123B-v2-aqlm-codebase/cpp_runtime/build-rocm/behemoth_load_bench
cd $E; rm -f $E/$TAG.crash
echo "$(date +%F_%T) START tag=$TAG min=$MIN args=$*" >> $L
if [ "$1" = L1 ]; then ~/gpu_od_apply.sh 0 0 300 >> $L 2>&1; ~/gpu_vf_cap.sh >> $L 2>&1; else ~/gpu_od_apply.sh "$@" >> $L 2>&1; fi
for i in $(seq 1 240); do curl -sf http://127.0.0.1:1337/health >/dev/null && break; sleep 2; done
( while true; do v0=$(sudo setpci -s 33:00.0 VENDOR_ID); v1=$(sudo setpci -s cc:00.0 VENDOR_ID)
  if [ "$v0" != 1002 ] || [ "$v1" != 1002 ]; then echo "$(date +%F_%T) CRASH v0=$v0 v1=$v1" >> $L; date +%F_%T > $E/$TAG.crash
    p=$(cat $E/bench.pid 2>/dev/null); [ -n "$p" ] && kill $p; exit; fi; sleep 1; done ) &
W=$!
bench() { timeout $1 $B ~/stress/prompt.txt $2 $3 96 127.0.0.1 1337 > $4 2>> $E/$TAG.err & echo $! > $E/bench.pid; wait $!; }
t0=$(date +%s); end=$((t0+MIN*60))
bench 300 1 2 $E/${TAG}_c1.json
echo "$(date +%F_%T) c1 rc=$? tok/s,itl_p50=$(jq -r '[.completion_tokens_per_second, .server_rolling_metrics.inter_token_p50_ms] | @tsv' "$E/${TAG}_c1.json" 2>/dev/null)" >> $L
n=0
while [ $(date +%s) -lt $end ] && [ ! -e $E/$TAG.crash ]; do
  n=$((n+1))
  bench 600 4 8 $E/${TAG}_b.json; rc=$?
  echo "$(date +%F_%T) batch $n rc=$rc tok/s=$(jq -r '.completion_tokens_per_second' "$E/${TAG}_b.json" 2>/dev/null)" >> $L
  [ $rc -ne 0 ] && { echo "$(date +%F_%T) load failed rc=$rc" >> $L; break; }
done
if [ ! -e $E/$TAG.crash ]; then bench 300 1 2 $E/${TAG}_c1b.json
  echo "$(date +%F_%T) c1end tok/s,itl_p50=$(jq -r '[.completion_tokens_per_second, .server_rolling_metrics.inter_token_p50_ms] | @tsv' "$E/${TAG}_c1b.json" 2>/dev/null)" >> $L; fi
kill $W 2>/dev/null
echo "$(date +%F_%T) END tag=$TAG elapsed=$(( $(date +%s)-t0 ))s batches=$n crashed=$([ -e $E/$TAG.crash ] && echo yes || echo no)" >> $L
