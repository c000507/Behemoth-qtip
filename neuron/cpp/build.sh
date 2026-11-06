#!/bin/bash
# Builds the Neuron serving binaries on a trn instance (g++ only; libnrt from /opt/aws/neuron).
#   neuron/cpp/build.sh OUTDIR
set -e
OUT=${1:-build}; mkdir -p "$OUT"
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
R=$ROOT/runtime
CORE="$R/src/qtip.cpp $R/src/checkpoint.cpp $R/src/attention.cpp $R/src/scheduler.cpp $R/src/cache.cpp $R/src/sampling.cpp"
FLAGS="-std=c++20 -O2 -fopenmp -pthread -I$R/include -I$ROOT/third_party -I/opt/aws/neuron/include"
LIBS="-L/opt/aws/neuron/lib -lnrt -Wl,-rpath,/opt/aws/neuron/lib"
g++ $FLAGS -c "$ROOT/neuron/cpp/qtip_expand.cpp" -o "$OUT/qtip_expand.o" &
g++ $FLAGS -c "$ROOT/neuron/cpp/neuron_model.cpp" -o "$OUT/neuron_model.o" &
g++ $FLAGS -c "$ROOT/neuron/cpp/neuron_engine.cpp" -o "$OUT/neuron_engine.o" &
g++ $FLAGS -c "$ROOT/neuron/cpp/neuron_engine_cli.cpp" -o "$OUT/neuron_engine_cli.o" &
for f in $CORE; do g++ $FLAGS -c "$f" -o "$OUT/$(basename "$f" .cpp).o" & done
wait
g++ $FLAGS "$OUT"/*.o $LIBS -o "$OUT/neuron_engine_cli"
echo built "$OUT/neuron_engine_cli"
