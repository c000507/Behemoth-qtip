#!/bin/bash
# Upload the model checkpoint and the compiled graph buckets from the current trn instance to S3.
# Symlinked bucket directories are DEREFERENCED (uploaded as real copies), since S3 has no symlinks and the
# runtime opens <bucket>/graph.neff + meta.json in every entry.
#
#   BUCKET=$(terraform output -raw s3_bucket) deploy/ecs-neuron/scripts/upload_to_s3.sh
#
# Layout (must match model_s3_prefix / graphs_s3_prefix in terraform.tfvars):
#   s3://$BUCKET/models/cydonia-24b-fp8/{config.json,tokenizer*.json,*.safetensors,...}
#   s3://$BUCKET/graphs/cydonia-24b-fp8-kv/<bucket>/{graph.neff,meta.json}
set -euo pipefail
: "${BUCKET:?set BUCKET (terraform output -raw s3_bucket)}"
MODEL_DIR=${MODEL_DIR:-/home/ubuntu/models/cydonia-24b-fp8}
GRAPH_DIR=${GRAPH_DIR:-/home/ubuntu/work/me/cydk}
MODEL_PREFIX=${MODEL_PREFIX:-models/cydonia-24b-fp8}
GRAPHS_PREFIX=${GRAPHS_PREFIX:-graphs/cydonia-24b-fp8-kv}
export AWS_REGION=${AWS_REGION:-us-east-1}

test -f "$MODEL_DIR/config.json" || { echo "no config.json in $MODEL_DIR"; exit 1; }
nneff=$(find -L "$GRAPH_DIR" -name graph.neff | wc -l)
[ "$nneff" -gt 0 ] || { echo "no graph.neff under $GRAPH_DIR"; exit 1; }
echo "model: $(du -shL "$MODEL_DIR" | cut -f1)  graphs: $nneff buckets, $(du -shL "$GRAPH_DIR" | cut -f1) (symlinks followed)"
find "$GRAPH_DIR" -maxdepth 1 -type l -printf '  symlink %f -> %l (uploaded as a copy)\n'

upload() { # local dir -> s3 prefix
  if command -v s5cmd >/dev/null; then
    # s5cmd follows symlinks by default (there is a --no-follow-symlinks flag); sync only uploads changed files
    s5cmd --numworkers 64 sync --size-only "$1/" "s3://$BUCKET/$2/"
  else
    AWS_MAX_ATTEMPTS=10 aws s3 sync --follow-symlinks --only-show-errors --region "$AWS_REGION" \
      "$1/" "s3://$BUCKET/$2/"
  fi
}
upload "$MODEL_DIR" "$MODEL_PREFIX"
upload "$GRAPH_DIR" "$GRAPHS_PREFIX"

# verify: object counts match the dereferenced local trees
lm=$(find -L "$MODEL_DIR" -type f | wc -l)
lg=$(find -L "$GRAPH_DIR" -type f | wc -l)
rm_=$(aws s3 ls --recursive "s3://$BUCKET/$MODEL_PREFIX/" | wc -l)
rg=$(aws s3 ls --recursive "s3://$BUCKET/$GRAPHS_PREFIX/" | wc -l)
rn=$(aws s3 ls --recursive "s3://$BUCKET/$GRAPHS_PREFIX/" | grep -c '/graph.neff$' || true)
echo "model files local=$lm s3=$rm_ | graph files local=$lg s3=$rg | graph.neff local=$nneff s3=$rn"
[ "$lm" -le "$rm_" ] && [ "$lg" -le "$rg" ] && [ "$nneff" -eq "$rn" ] || { echo "MISMATCH"; exit 1; }
echo OK
