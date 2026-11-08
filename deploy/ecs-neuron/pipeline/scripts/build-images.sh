#!/bin/bash
# Build + push the backend and router images if their tag is not in ECR yet (called by buildspec-images.yml).
#   build-images.sh OUT_ENV        writes BACKEND_TAG / ROUTER_TAG / BACKEND_BUILT / ROUTER_BUILT to OUT_ENV
#
# Tag of an image = short SHA (7) of the LAST COMMIT THAT CHANGED ITS INPUTS (git log -1 -- <paths>), not of HEAD:
#   backend: CMakeLists.txt runtime/ neuron/cpp/ glm53/ third_party/ deploy/ecs-neuron/backend/
#   router:  deploy/ecs-neuron/router/
# so a push that only touches Terraform or the router keeps the backend tag, finds it in ECR and skips the ~10 min
# build. No state about "the previous successful run" is needed, and re-running an old commit yields the same tags.
# If the clone has no history (shallow and unshallow fails) the tag falls back to HEAD: always correct, just slower.
#
# Env: ECR_REGISTRY BACKEND_REPO ROUTER_REPO (required), STACK_DIR, COMMIT_ID (pipeline commit, checked against HEAD),
#      FORCE_IMAGE_BUILD (none | backend | router | all | backend,router), BACKEND_BASE_IMAGE,
#      NEURON_RUNTIME_VERSION, NEURON_COLLECTIVES_VERSION, DOCKERHUB_USERNAME / DOCKERHUB_PASSWORD (optional),
#      AWS_REGION / AWS_DEFAULT_REGION (set by CodeBuild).
set -euo pipefail
OUT_ENV=${1:?usage: build-images.sh OUT_ENV}
: "${ECR_REGISTRY:?}" "${BACKEND_REPO:?}" "${ROUTER_REPO:?}"
STACK_DIR=${STACK_DIR:-deploy/ecs-neuron}
FORCE=${FORCE_IMAGE_BUILD:-none}
REGION=${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"
HEAD=$(git rev-parse HEAD)
if [ -n "${COMMIT_ID:-}" ] && [ "$COMMIT_ID" != "$HEAD" ]; then
  echo "checkout is $HEAD but the pipeline source commit is $COMMIT_ID" >&2
  exit 1
fi

BACKEND_PATHS=(CMakeLists.txt runtime neuron/cpp glm53 third_party "$STACK_DIR/backend")
ROUTER_PATHS=("$STACK_DIR/router")

if [ "$(git rev-parse --is-shallow-repository)" = true ]; then
  git fetch --unshallow --quiet || echo "WARN: shallow clone and unshallow failed: image tags fall back to HEAD" >&2
fi

tag_for() {
  local c=""
  if [ "$(git rev-parse --is-shallow-repository)" != true ]; then
    c=$(git log -1 --format=%H HEAD -- "$@")
  fi
  [ -n "$c" ] || c=$HEAD
  echo "${c:0:7}"
}

forced() { [ "$FORCE" = all ] || [[ ",$FORCE," == *",$1,"* ]]; }

# 0 = tag present, 1 = absent; any other error (permissions, throttling) aborts instead of silently rebuilding
image_exists() {
  local err
  if err=$(aws ecr describe-images --region "$REGION" --repository-name "$1" --image-ids imageTag="$2" 2>&1 >/dev/null); then
    return 0
  fi
  if grep -q ImageNotFoundException <<<"$err"; then
    return 1
  fi
  echo "$err" >&2
  exit 1
}

logged_in=false
login() {
  $logged_in && return 0
  aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR_REGISTRY"
  if [ -n "${DOCKERHUB_USERNAME:-}" ]; then
    printf '%s' "$DOCKERHUB_PASSWORD" | docker login --username "$DOCKERHUB_USERNAME" --password-stdin
  fi
  logged_in=true
}

push() { # repo tag
  docker push "$ECR_REGISTRY/$1:$2"
  docker push "$ECR_REGISTRY/$1:latest"
}

# ------------------------------------------------------------------ backend
BACKEND_TAG=$(tag_for "${BACKEND_PATHS[@]}")
BACKEND_BUILT=false
if forced backend || ! image_exists "$BACKEND_REPO" "$BACKEND_TAG"; then
  echo "== building backend $BACKEND_REPO:$BACKEND_TAG (HEAD $HEAD)"
  login
  DOCKER_BUILDKIT=1 docker build --platform linux/amd64 \
    -f "$STACK_DIR/backend/Dockerfile" \
    --build-arg "BASE_IMAGE=${BACKEND_BASE_IMAGE:-ubuntu:22.04}" \
    --build-arg "NEURON_RUNTIME_VERSION=${NEURON_RUNTIME_VERSION:-}" \
    --build-arg "NEURON_COLLECTIVES_VERSION=${NEURON_COLLECTIVES_VERSION:-}" \
    --label "org.opencontainers.image.revision=$HEAD" \
    -t "$ECR_REGISTRY/$BACKEND_REPO:$BACKEND_TAG" -t "$ECR_REGISTRY/$BACKEND_REPO:latest" .
  push "$BACKEND_REPO" "$BACKEND_TAG"
  BACKEND_BUILT=true
else
  echo "== backend $BACKEND_REPO:$BACKEND_TAG already in ECR, skipping"
fi

# ------------------------------------------------------------------ router (unit tests first, like router/build_and_push.sh)
ROUTER_TAG=$(tag_for "${ROUTER_PATHS[@]}")
ROUTER_BUILT=false
if forced router || ! image_exists "$ROUTER_REPO" "$ROUTER_TAG"; then
  echo "== testing + building router $ROUTER_REPO:$ROUTER_TAG"
  login
  docker run --rm -e PYTHONDONTWRITEBYTECODE=1 -v "$ROOT/$STACK_DIR/router:/src:ro" -w /src \
    public.ecr.aws/docker/library/python:3.12-slim \
    sh -c 'pip install -q --no-cache-dir --root-user-action=ignore -r requirements.txt pytest && python -m pytest -q -p no:cacheprovider'
  docker build --platform linux/amd64 --label "org.opencontainers.image.revision=$HEAD" \
    -t "$ECR_REGISTRY/$ROUTER_REPO:$ROUTER_TAG" -t "$ECR_REGISTRY/$ROUTER_REPO:latest" "$STACK_DIR/router"
  push "$ROUTER_REPO" "$ROUTER_TAG"
  ROUTER_BUILT=true
else
  echo "== router $ROUTER_REPO:$ROUTER_TAG already in ECR, skipping"
fi

printf 'BACKEND_TAG=%s\nROUTER_TAG=%s\nBACKEND_BUILT=%s\nROUTER_BUILT=%s\n' \
  "$BACKEND_TAG" "$ROUTER_TAG" "$BACKEND_BUILT" "$ROUTER_BUILT" >"$OUT_ENV"
cat "$OUT_ENV"
