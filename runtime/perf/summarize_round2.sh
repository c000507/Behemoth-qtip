#!/usr/bin/env bash
# Summarize native projection benchmark JSON. Requires jq 1.6+.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [ "$#" -eq 0 ]; then
  echo "Usage: $0 BENCHMARK.json..." >&2
  exit 1
fi
jq -r -f "$SCRIPT_DIR/summarize_round2.jq" "$@"
