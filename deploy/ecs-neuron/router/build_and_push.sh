#!/bin/bash
# Build the router image and push it to the Terraform-managed router ECR repo.
#   ./build_and_push.sh [REPO_URI]     (default: terraform output -raw router_ecr_url, run from deploy/ecs-neuron)
# Tags: <git short sha> and latest.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=${1:-$(cd "$HERE/.." && terraform output -raw router_ecr_url)}
REGION=${AWS_REGION:-us-east-1}
TAG=$(git -C "$HERE" rev-parse --short HEAD)
(cd "$HERE" && python3 -m pytest -q -p no:cacheprovider)
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "${REPO%%/*}"
docker build --platform linux/amd64 -t "$REPO:$TAG" -t "$REPO:latest" "$HERE"
docker push "$REPO:$TAG"
docker push "$REPO:latest"
echo "pushed $REPO:$TAG (set router_image_tag = \"$TAG\" in terraform.tfvars, or keep latest)"
