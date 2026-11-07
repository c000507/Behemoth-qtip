#!/bin/bash
# Builds the Neuron serving binaries on a trn instance (g++ only; libnrt from /opt/aws/neuron).
#   neuron/cpp/build.sh OUTDIR
set -e
OUT=${1:-build}; mkdir -p "$OUT"
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
R=$ROOT/runtime
CORE="$R/src/qtip.cpp $R/src/checkpoint.cpp $R/src/attention.cpp $R/src/scheduler.cpp $R/src/cache.cpp $R/src/sampling.cpp $R/src/prefix_cache.cpp"
FLAGS="-std=c++20 -O2 -fopenmp -pthread -I$R/include -I$ROOT/third_party -I/opt/aws/neuron/include"
LIBS="-L/opt/aws/neuron/lib -lnrt -Wl,-rpath,/opt/aws/neuron/lib"
g++ $FLAGS -c "$ROOT/neuron/cpp/qtip_expand.cpp" -o "$OUT/qtip_expand.o" &
g++ $FLAGS -c "$ROOT/neuron/cpp/neuron_model.cpp" -o "$OUT/neuron_model.o" &
g++ $FLAGS -c "$ROOT/neuron/cpp/neuron_engine.cpp" -o "$OUT/neuron_engine.o" &
g++ $FLAGS -c "$ROOT/neuron/cpp/neuron_engine_cli.cpp" -o "$OUT/neuron_engine_cli.o" &
for f in $CORE; do g++ $FLAGS -c "$f" -o "$OUT/$(basename "$f" .cpp).o" & done
for job in $(jobs -p); do wait "$job" || { echo "compile failed"; exit 1; }; done   # a failed job must not leave a stale .o linked
g++ $FLAGS "$OUT"/*.o $LIBS -o "$OUT/neuron_engine_cli"
echo built "$OUT/neuron_engine_cli"

# HTTP server: runtime/src/server.cpp unchanged, Neuron Engine, Llama BPE tokenizer + chat template
SERVER=${2:-}
if [ -n "$SERVER" ]; then
  mkdir -p "$OUT/srv"
  SRV="$R/src/server.cpp $R/src/reasoning.cpp $R/src/protocol.cpp $R/src/sorted_layout.cpp $R/src/reference_block.cpp $R/src/gbnf.cpp $R/src/json_schema.cpp $R/src/constraint.cpp $ROOT/glm53/src/tokenizer.cpp $ROOT/glm53/src/config.cpp $ROOT/neuron/cpp/llama_text.cpp"
  SFLAGS="$FLAGS -I$ROOT/glm53/include -I$ROOT/third_party -DBOOST_ERROR_CODE_HEADER_ONLY"
  for f in $SRV; do g++ $SFLAGS -c "$f" -o "$OUT/srv/$(basename "$f" .cpp).o" & done
  for job in $(jobs -p); do wait "$job" || { echo "compile failed"; exit 1; }; done
  g++ $SFLAGS "$OUT/srv"/*.o "$OUT/qtip_expand.o" "$OUT/neuron_model.o" "$OUT/neuron_engine.o" \
      $(for f in $CORE; do echo "$OUT/$(basename "$f" .cpp).o"; done) $LIBS -lpcre2-8 -o "$OUT/neuron_server"
  echo built "$OUT/neuron_server"
fi
