#!/bin/bash
# Build the neuron_server image (multi-stage, context = repo root) and push it to the existing ECR repo.
#   deploy/ecs-neuron/backend/build_and_push.sh                 # compile neuron_server in the build stage
#   PREBUILT=OUT/neuron_server deploy/ecs-neuron/backend/build_and_push.sh   # use a binary from neuron/cpp/build.sh OUT srv
# Env: REPO_URI, AWS_REGION, NEURON_RUNTIME_VERSION / NEURON_COLLECTIVES_VERSION (pin to the AMI's driver), EXTRA_BUILD_ARGS.
# Tags: <git short sha> (with -dirty when the tree has changes) and latest.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
REPO_URI=${REPO_URI:-912935854432.dkr.ecr.us-east-1.amazonaws.com/behemoth-neuron-server}
REGION=${AWS_REGION:-us-east-1}
TAG=$(git -C "$ROOT" rev-parse --short HEAD)
git -C "$ROOT" diff --quiet HEAD -- runtime neuron/cpp glm53 third_party deploy/ecs-neuron/backend || TAG="$TAG-dirty"

ARGS=(--platform linux/amd64 -f "$HERE/Dockerfile"
      --build-arg "NEURON_RUNTIME_VERSION=${NEURON_RUNTIME_VERSION:-}"
      --build-arg "NEURON_COLLECTIVES_VERSION=${NEURON_COLLECTIVES_VERSION:-}")
if [ -n "${PREBUILT:-}" ]; then
  mkdir -p "$HERE/bin" && cp "$PREBUILT" "$HERE/bin/neuron_server"
  ARGS+=(--build-arg SERVER_STAGE=server-prebuilt)
fi
# shellcheck disable=SC2206
ARGS+=(${EXTRA_BUILD_ARGS:-})

DOCKER_BUILDKIT=1 docker build "${ARGS[@]}" -t "$REPO_URI:$TAG" -t "$REPO_URI:latest" "$ROOT"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "${REPO_URI%%/*}"
docker push "$REPO_URI:$TAG"
docker push "$REPO_URI:latest"
echo "pushed $REPO_URI:$TAG  (set backend_image_tag = \"$TAG\" in terraform.tfvars and apply)"
