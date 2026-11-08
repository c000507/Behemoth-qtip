#!/bin/bash
# Host-side checks of the Neuron serving code without NeuronCores or the Neuron SDK (g++ only), on the CPU fake model
# (fake_model.cpp) in place of neuron_model.cpp:
#   neuron_engine_test   structured outputs / tool calls through neuron_engine.cpp: validity, speculation, preemption,
#                        prefix cache (runs it)
#   neuron_server_fake   the HTTP server (runtime/src/server.cpp + llama_text.cpp) on the fake model, for curl tests
#                        against a real tokenizer:  BEHEMOTH_NEURON_NEFF_DIR=/tmp neuron_server_fake --model DIR
#   structured_bench     mask exactness and per-token constraint cost on a checkpoint's BPE vocabulary:
#                        structured_bench MODEL_DIR [walks]
#   neuron/cpp/host_test.sh OUTDIR
set -e
OUT=${1:-build-host}; mkdir -p "$OUT/srv"
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
R=$ROOT/runtime
CORE="$R/src/qtip.cpp $R/src/checkpoint.cpp $R/src/attention.cpp $R/src/scheduler.cpp $R/src/cache.cpp $R/src/sampling.cpp $R/src/prefix_cache.cpp"
TEXT="$R/src/reasoning.cpp $R/src/protocol.cpp $R/src/gbnf.cpp $R/src/json_schema.cpp $R/src/constraint.cpp $R/src/tools.cpp"
FLAGS="-std=c++20 -O2 -fopenmp -pthread -Wall -I$R/include -I$ROOT/third_party -I$ROOT/glm53/include -DBOOST_ERROR_CODE_HEADER_ONLY"
for f in $CORE $TEXT "$ROOT/neuron/cpp/neuron_engine.cpp" "$ROOT/neuron/cpp/fake_model.cpp" "$ROOT/neuron/cpp/neuron_engine_test.cpp" \
         "$R/src/server.cpp" "$R/src/sorted_layout.cpp" "$R/src/reference_block.cpp" "$ROOT/glm53/src/tokenizer.cpp" \
         "$ROOT/glm53/src/config.cpp" "$ROOT/neuron/cpp/llama_text.cpp" "$R/tools/structured_bench.cpp"; do
  g++ $FLAGS -c "$f" -o "$OUT/srv/$(basename "$f" .cpp).o" &
done
for job in $(jobs -p); do wait "$job" || { echo "compile failed"; exit 1; }; done
O=$OUT/srv
LIB="$(for f in $CORE $TEXT; do echo "$O/$(basename "$f" .cpp).o"; done)"
g++ $FLAGS $O/neuron_engine_test.o $O/neuron_engine.o $O/fake_model.o $LIB -o "$OUT/neuron_engine_test"
g++ $FLAGS $O/server.o $O/sorted_layout.o $O/reference_block.o $O/tokenizer.o $O/config.o $O/llama_text.o \
    $O/neuron_engine.o $O/fake_model.o $LIB -lpcre2-8 -o "$OUT/neuron_server_fake"
g++ $FLAGS $O/structured_bench.o $O/tokenizer.o $O/config.o $O/llama_text.o $LIB -lpcre2-8 -o "$OUT/structured_bench"
echo built "$OUT/neuron_engine_test" "$OUT/neuron_server_fake" "$OUT/structured_bench"
"$OUT/neuron_engine_test"
