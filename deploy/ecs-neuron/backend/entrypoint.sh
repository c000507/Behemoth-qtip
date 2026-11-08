#!/bin/bash
# Pull the model + compiled graph buckets from S3 onto local disk (instance store), then exec neuron_server.
# Re-runs on the same instance only fetch files whose size differs (cp -n -s), so a task restart is fast.
set -euo pipefail

: "${S3_BUCKET:?S3_BUCKET not set}"
: "${MODEL_S3_PREFIX:?MODEL_S3_PREFIX not set (e.g. models/cydonia-24b-fp8)}"
: "${GRAPHS_S3_PREFIX:?GRAPHS_S3_PREFIX not set (e.g. graphs/cydonia-24b-fp8-kv)}"
DATA_DIR=${DATA_DIR:-/data}
PORT=${PORT:-1337}
S5CMD_WORKERS=${S5CMD_WORKERS:-64}       # files in flight
S5CMD_CONCURRENCY=${S5CMD_CONCURRENCY:-16} # multipart parts per file
S5CMD_PART_MB=${S5CMD_PART_MB:-64}
MARKER=${STARTUP_MARKER:-/tmp/backend-starting}

MODEL_PREFIX=${MODEL_S3_PREFIX%/}
GRAPHS_PREFIX=${GRAPHS_S3_PREFIX%/}
MODEL_DIR=$DATA_DIR/$MODEL_PREFIX
GRAPH_DIR=$DATA_DIR/$GRAPHS_PREFIX

date +%s > "$MARKER"   # healthcheck.sh reports healthy while this is younger than STARTUP_GRACE_S
mkdir -p "$MODEL_DIR" "$GRAPH_DIR"
df -h "$DATA_DIR" || true

pull() { # s3 prefix -> local dir
  local t0=$SECONDS
  s5cmd --numworkers "$S5CMD_WORKERS" --retry-count 5 \
    cp -n -s --concurrency "$S5CMD_CONCURRENCY" --part-size "$S5CMD_PART_MB" \
    "s3://$S3_BUCKET/$1/*" "$2/"
  echo "pulled s3://$S3_BUCKET/$1 -> $2 ($(du -sh "$2" | cut -f1)) in $((SECONDS - t0)) s"
}
pull "$MODEL_PREFIX" "$MODEL_DIR" &
p1=$!
pull "$GRAPHS_PREFIX" "$GRAPH_DIR" &
p2=$!
wait $p1
wait $p2

test -f "$MODEL_DIR/config.json" || { echo "no config.json in $MODEL_DIR"; exit 1; }
n=$(find "$GRAPH_DIR" -name graph.neff | wc -l)
[ "$n" -gt 0 ] || { echo "no graph.neff under $GRAPH_DIR"; exit 1; }
echo "graph buckets: $n"

export NEURON_RT_VISIBLE_CORES=${NEURON_RT_VISIBLE_CORES:-0-1}
export NEURON_RT_ROOT_COMM_ID=${NEURON_RT_ROOT_COMM_ID:-127.0.0.1:62128}
export BEHEMOTH_NEURON_NEFF_DIR=$GRAPH_DIR
export BEHEMOTH_MODEL_NAME=${BEHEMOTH_MODEL_NAME:-cydonia-24b}

# shellcheck disable=SC2086  # EXTRA_SERVER_ARGS is intentionally word-split
exec neuron_server --model "$MODEL_DIR" --host 0.0.0.0 --port "$PORT" ${EXTRA_SERVER_ARGS:-}
