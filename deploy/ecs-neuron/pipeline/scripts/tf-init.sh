#!/bin/bash
# terraform init of the main stack against the S3 backend (shared by buildspec-plan.yml and buildspec-apply.yml).
# -lockfile=readonly: providers must match the committed .terraform.lock.hcl, so plan and apply run the same build.
set -euo pipefail
: "${TF_STATE_BUCKET:?}" "${TF_STATE_KEY:?}" "${TF_STATE_REGION:?}" "${STACK_DIR:?}"
D="${CODEBUILD_SRC_DIR:-$(git rev-parse --show-toplevel)}/$STACK_DIR"

cat >"$D/backend.hcl" <<EOF
bucket       = "$TF_STATE_BUCKET"
key          = "$TF_STATE_KEY"
region       = "$TF_STATE_REGION"
encrypt      = true
use_lockfile = true
EOF
terraform -chdir="$D" init -input=false -lockfile=readonly -backend-config=backend.hcl
