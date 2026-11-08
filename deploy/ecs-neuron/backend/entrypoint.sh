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

pull() { # s3 prefix -> local dir [extra s5cmd cp args...]
  local t0=$SECONDS src=$1 dst=$2; shift 2
  s5cmd --numworkers "$S5CMD_WORKERS" --retry-count 5 \
    cp -n -s --concurrency "$S5CMD_CONCURRENCY" --part-size "$S5CMD_PART_MB" "$@" \
    "s3://$S3_BUCKET/$src/*" "$dst/"
  echo "pulled s3://$S3_BUCKET/$src -> $dst ($(du -sh "$dst" | cut -f1)) in $((SECONDS - t0)) s"
}

check_graphs() {
  n=$(find "$GRAPH_DIR" -name graph.neff | wc -l)
  [ "$n" -gt 0 ] || { echo "no graph.neff under $GRAPH_DIR"; exit 1; }
  echo "graph buckets: $n"
}

# STREAM_LOAD=1 (default): pull everything but the layer files (config, tokenizer, embeddings, graphs: ~1.5 GB) first, start
# neuron_server at once (it loads the graphs and waits for "<layer>.ready" markers, BEHEMOTH_NEURON_WAIT_READY), and download
# the layer files in index order in the background, at most STREAM_AHEAD layers ahead of the upload (the server writes
# "<layer>.done"). STAGE_DIR (default: the model dir on the instance store) can be a tmpfs; with STREAM_UNLINK=1 the server
# deletes each staged layer once it is on the devices. Download and device load then overlap instead of running back to back.
STREAM_LOAD=${STREAM_LOAD:-1}
STAGE_DIR=${STAGE_DIR:-$MODEL_DIR}
STREAM_PAR=${STREAM_PAR:-4}      # layer files downloading at once
STREAM_AHEAD=${STREAM_AHEAD:-48} # layers downloaded but not yet uploaded (bounds tmpfs use)
# PREPARED_S3_PREFIX: stream prepared shards (neuron_server BEHEMOTH_NEURON_WRITE_PREPARED output: layout.json + one
# model-layer-NNNNN.bin per layer, the device bytes as uploaded) instead of the safetensors layer files: no expansion pass.
PREPARED_PREFIX=${PREPARED_S3_PREFIX%/}

if [ "$STREAM_LOAD" = 1 ]; then
  t0=$SECONDS
  mkdir -p "$STAGE_DIR"
  pull "$MODEL_PREFIX" "$MODEL_DIR" --exclude "model-layer-*" &
  p1=$!
  pull "$GRAPHS_PREFIX" "$GRAPH_DIR" &
  p2=$!
  if [ -n "$PREPARED_PREFIX" ]; then
    SRC_PREFIX=$PREPARED_PREFIX
    MARK_DIR=$DATA_DIR/$PREPARED_PREFIX
    mkdir -p "$MARK_DIR"
    s5cmd cp "s3://$S3_BUCKET/$PREPARED_PREFIX/layout.json" "$MARK_DIR/layout.json" >/dev/null && touch "$MARK_DIR/layout.json.ready"
    layers=$(s5cmd ls "s3://$S3_BUCKET/$PREPARED_PREFIX/model-layer-*.bin" | awk '{print $NF}' | sort)
    export BEHEMOTH_NEURON_PREPARED_DIR=$MARK_DIR
  else
    SRC_PREFIX=$MODEL_PREFIX
    MARK_DIR=$MODEL_DIR
    layers=$(s5cmd ls "s3://$S3_BUCKET/$MODEL_PREFIX/model-layer-*" | awk '{print $NF}' | sort)
  fi
  wait $p1
  wait $p2
  test -f "$MODEL_DIR/config.json" || { echo "no config.json in $MODEL_DIR"; exit 1; }
  check_graphs
  for f in $layers; do
    rm -f "$MARK_DIR/$f.ready" "$MARK_DIR/$f.done" "$MARK_DIR/$f.failed"
    [ "$STAGE_DIR" = "$MARK_DIR" ] || ln -sfn "$STAGE_DIR/$f" "$MARK_DIR/$f"
  done
  echo "small files and graphs in $((SECONDS - t0)) s; streaming $(echo "$layers" | wc -w) layer files from $SRC_PREFIX (par $STREAM_PAR, ahead $STREAM_AHEAD)"
  fetch_layer() { # index name
    local i=$1 f=$2
    while [ $(( i - $(ls "$MARK_DIR"/*.done 2>/dev/null | wc -l) )) -ge "$STREAM_AHEAD" ]; do sleep 0.2; done
    if s5cmd --retry-count 5 cp --concurrency "$S5CMD_CONCURRENCY" --part-size "$S5CMD_PART_MB" \
        "s3://$S3_BUCKET/$SRC_PREFIX/$f" "$STAGE_DIR/$f" >/dev/null; then
      touch "$MARK_DIR/$f.ready"
    else
      echo "download failed: $f"; touch "$MARK_DIR/$f.failed"
    fi
  }
  export -f fetch_layer
  export MARK_DIR STAGE_DIR S3_BUCKET SRC_PREFIX S5CMD_CONCURRENCY S5CMD_PART_MB STREAM_AHEAD
  (
    t1=$SECONDS
    echo "$layers" | nl -v0 -w1 -s' ' | xargs -P "$STREAM_PAR" -n 2 bash -c 'fetch_layer "$0" "$1"'
    echo "layer files pulled in $((SECONDS - t1)) s ($((SECONDS - t0)) s since start)"
  ) &
  export BEHEMOTH_NEURON_WAIT_READY=1
  [ "${STREAM_UNLINK:-0}" = 1 ] && export BEHEMOTH_NEURON_UNLINK_LAYERS=1
else
  pull "$MODEL_PREFIX" "$MODEL_DIR" &
  p1=$!
  pull "$GRAPHS_PREFIX" "$GRAPH_DIR" &
  p2=$!
  wait $p1
  wait $p2
  test -f "$MODEL_DIR/config.json" || { echo "no config.json in $MODEL_DIR"; exit 1; }
  check_graphs
fi

export NEURON_RT_VISIBLE_CORES=${NEURON_RT_VISIBLE_CORES:-0-1}
export NEURON_RT_ROOT_COMM_ID=${NEURON_RT_ROOT_COMM_ID:-127.0.0.1:62128}
export BEHEMOTH_NEURON_NEFF_DIR=$GRAPH_DIR
export BEHEMOTH_MODEL_NAME=${BEHEMOTH_MODEL_NAME:-cydonia-24b}

# shellcheck disable=SC2086  # EXTRA_SERVER_ARGS is intentionally word-split
exec neuron_server --model "$MODEL_DIR" --host 0.0.0.0 --port "$PORT" ${EXTRA_SERVER_ARGS:-}
