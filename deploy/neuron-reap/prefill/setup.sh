#!/bin/bash
# setup.sh: instance store, src, prod model + graphs, engine build. Idempotent; log in /mnt/scratch/log/setup.log
set -o pipefail
B=s3://behemoth-neuron-models-912935854432
T=$B/tmp/ornith-prefill
S=/mnt/scratch
export PATH=/usr/local/bin:/opt/aws/neuron/bin:$PATH
if ! mountpoint -q $S; then
  dev=$(lsblk -dnbo NAME,SIZE,TYPE | awk '$3=="disk" && $2>400e9 && $2<600e9 {print "/dev/"$1; exit}')
  [ -n "$dev" ] || { echo "no instance store disk"; lsblk; exit 1; }
  lsblk -no MOUNTPOINT "$dev" | grep -q . && { echo "$dev is mounted"; exit 1; }
  mkfs.xfs -f "$dev" && mkdir -p $S && mount "$dev" $S || exit 1
fi
mkdir -p $S/log $S/model $S/gprod $S/run; cd $S
exec >> log/setup.log 2>&1
echo "== setup $(date -u +%T)"
aws s3 cp --only-show-errors $T/src.tgz src.tgz && mkdir -p src && tar -xzf src.tgz -C src && echo "src ok"
( aws s3 sync --only-show-errors $B/models/ornith-reap20-v3-fp8 model && echo "model ok $(du -sh model | cut -f1)" ) &
aws s3 sync --only-show-errors $B/graphs/ornith-reap20-v3-kv8-seg20-mtp gprod && echo "graphs ok $(ls gprod | wc -l)"
which g++-12 >/dev/null || (apt-get update -qq && apt-get install -y -qq g++-12 libboost-dev libpcre2-dev) > log/apt.log 2>&1
dpkg -l libboost-dev libpcre2-dev >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq libboost-dev libpcre2-dev) >> log/apt.log 2>&1
mkdir -p shim && ln -sf $(which g++-12) shim/g++
(cd src && PATH=$S/shim:$PATH bash neuron/cpp/build.sh $S/BUILD srv) > log/build.log 2>&1; tail -2 log/build.log
$S/shim/g++ -std=c++20 -O2 -pthread -I/opt/aws/neuron/include src/neuron/cpp/layer_bench.cpp -L/opt/aws/neuron/lib -lnrt -Wl,-rpath,/opt/aws/neuron/lib -o BUILD/layer_bench && echo "layer_bench ok"
wait
echo "SETUP_DONE $(date -u +%T)"
