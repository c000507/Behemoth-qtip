#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 BUILD_DIRECTORY MODEL_DIRECTORY" >&2
  exit 2
fi
build=$1
model=$2
for executable in behemoth_gpu_tests behemoth_tokenizer_tests behemoth_inspect behemoth_block_tests behemoth_model_tests; do
  if [[ ! -x "$build/$executable" ]]; then
    echo "Missing $build/$executable; build with BEHEMOTH_ROCM=ON and BEHEMOTH_TOKENIZER=ON" >&2
    exit 1
  fi
done
ctest --test-dir "$build" --output-on-failure
"$build/behemoth_tokenizer_tests" "$model"
"$build/behemoth_inspect" "$model"
"$build/behemoth_block_tests" "$model" 0
"$build/behemoth_model_tests" "$model"
echo 'Hardware/model validation passed. Start the server, then run behemoth_http_tests separately.'
