#!/bin/bash
# Install a pinned Terraform from releases.hashicorp.com and verify the zip against a pinned SHA-256.
#   TF_VERSION=1.12.2 TF_SHA256=<sha256 of terraform_1.12.2_linux_amd64.zip> install-terraform.sh [DEST_DIR]
set -euo pipefail
V=${TF_VERSION:?}
SUM=${TF_SHA256:?}
DEST=${1:-/usr/local/bin}

if command -v terraform >/dev/null && terraform version | head -1 | grep -qx "Terraform v$V"; then
  terraform version
  exit 0
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
zip="terraform_${V}_linux_amd64.zip"
curl -fsSL --retry 3 -o "$tmp/$zip" "https://releases.hashicorp.com/terraform/$V/$zip"
echo "$SUM  $tmp/$zip" | sha256sum -c -
unzip -o -q "$tmp/$zip" terraform -d "$DEST"
terraform version
