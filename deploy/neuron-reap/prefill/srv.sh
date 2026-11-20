#!/bin/bash
# srv.sh TAG GRAPHDIR [BUILD] [ENV=V...]: start the server with the prod env on GRAPHDIR; log /mnt/scratch/run/TAG/server.log; waits for /health
S=/mnt/scratch; TAG=$1; G=$2; BLD=${3:-$S/BUILD}; shift 3 2>/dev/null
export PATH=/usr/local/bin:/opt/aws/neuron/bin:$PATH
pkill -f neuron_server; sleep 2; pkill -9 -f neuron_server; sleep 1
mkdir -p $S/run/$TAG
export NEURON_RT_VISIBLE_CORES=0-1 NEURON_RT_ROOT_COMM_ID=127.0.0.1:62128 BEHEMOTH_NEURON_NEFF_DIR=$G BEHEMOTH_MODEL_NAME=ornith-1.5-35b-a3b-reap20-v3
export STREAM_LOAD=0 BEHEMOTH_NEURON_QWEN35_PREFIX_CACHE=1 BEHEMOTH_NEURON_QWEN35_PREFIX_ALIGN=1 BEHEMOTH_NEURON_QWEN35_PREFIX_INTERVAL=2048 BEHEMOTH_NEURON_QWEN35_PREFIX_STATE_GB=4
export BEHEMOTH_NEURON_PREFIX_HOST_GB=4 OMP_WAIT_POLICY=PASSIVE BEHEMOTH_NEURON_MTP_DIR=$S/model/mtp BEHEMOTH_NEURON_SPEC_MTP=1 BEHEMOTH_NEURON_SPEC_MTP_ADAPT=1
export BEHEMOTH_NEURON_STEP_LOG=1
for e in "$@"; do export "$e"; done
nohup $BLD/neuron_server --model $S/model --host 127.0.0.1 --port 1337 --prefill-chunk ${CHUNK:-256} --max-batch-requests 8 --prefix-cache-gb 4 > $S/run/$TAG/server.log 2>&1 &
sp=$!; t0=$(date +%s)
for i in $(seq 1 240); do curl -sf -m 2 http://127.0.0.1:1337/health >/dev/null && break; kill -0 $sp 2>/dev/null || { echo "server died"; tail -20 $S/run/$TAG/server.log; exit 1; }; sleep 3; done
echo "server up after $(( $(date +%s) - t0 ))s pid $sp"
