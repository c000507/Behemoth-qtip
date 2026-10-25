#!/bin/bash
# Start behemoth_server with production flags + extra args, wait for health, run e2e_round2 for TAG, stop server.
# Usage: server_cycle.sh TAG "CONCURRENCIES" [extra server args...]   (env vars pass through, e.g. BEHEMOTH_LANE_DIAG)
set -u
ROOT=/home/charles/Development/Behemoth-R1-123B-v2-aqlm-codebase/cpp_runtime
TAG=$1; CONC=$2; shift 2
cd $ROOT
setsid nohup build-rocm/behemoth_server --model /mnt/nvme/models/Behemoth-R1-123B-v2-aqlm-output --devices 0,1 --transport zerocopy --memory-fraction 0.9 --token-budget 1024 --prefill-chunk 256 --host 0.0.0.0 --port 1337 --threads 4 "$@" > /tmp/beh_server_$TAG.log 2>&1 < /dev/null &
pid=$!; echo $pid > /tmp/beh_server.pid
echo "$(date +%T) $TAG server pid $pid args: $*"
for i in $(seq 1 300); do curl -sf http://127.0.0.1:1337/health > /dev/null 2>&1 && break; kill -0 $pid 2>/dev/null || { echo "server died"; tail -5 /tmp/beh_server_$TAG.log; exit 1; }; sleep 2; done
curl -sf http://127.0.0.1:1337/health || { echo "$TAG server failed to start"; exit 1; }
echo "$(date +%T) healthy; $(grep -E 'KV pages' /tmp/beh_server_$TAG.log)"
perf/e2e_round2.sh $TAG "$CONC"
if [ "${KEEP_SERVER:-0}" != 1 ]; then
  kill $pid; for i in $(seq 1 30); do kill -0 $pid 2>/dev/null || break; sleep 1; done; kill -9 $pid 2>/dev/null; echo "$(date +%T) server stopped"
fi
