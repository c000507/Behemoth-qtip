#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 MODEL_DIRECTORY" >&2
  exit 2
fi
destination=$1
mkdir -p "$destination"
base='https://huggingface.co/cgg507/Behemoth-R1-123B-v2-aqlm/resolve/61a69e73fcdbcd8a0b81237c0092aa93518a5c52'
files=(config.json generation_config.json model.safetensors.index.json tokenizer.model tokenizer.json tokenizer_config.json special_tokens_map.json)
for index in {1..7}; do
  printf -v shard 'model-%05d-of-00007.safetensors' "$index"
  files+=("$shard")
done
for name in "${files[@]}"; do
  if [[ -s "$destination/$name" ]]; then
    echo "Already present: $name (the C++ inspector validates tensor shards)"
    continue
  fi
  curl --fail --location --retry 3 --continue-at - \
    --output "$destination/$name.part" "$base/$name"
  mv "$destination/$name.part" "$destination/$name"
done
echo 'Download complete. Run behemoth_inspect before inference.'
