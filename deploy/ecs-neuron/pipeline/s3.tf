locals {
  state_bucket    = var.state_bucket_name != "" ? var.state_bucket_name : "${var.name}-tfstate-${var.account_id}"
  artifact_bucket = var.artifact_bucket_name != "" ? var.artifact_bucket_name : "${var.name}-pipeline-artifacts-${var.account_id}"

  state_key  = "${var.state_key_prefix}/terraform.tfstate"
  config_key = "${var.state_key_prefix}/terraform.tfvars"
  lock_key   = "${local.state_key}.tflock" # S3 native locking (use_lockfile = true)

  buckets = {
    state     = local.state_bucket
    artifacts = local.artifact_bucket
  }
}

# ------------------------------------------------------------------ state + config bucket, artifact bucket (same baseline)
resource "aws_s3_bucket" "this" {
  for_each = local.buckets
  bucket   = each.value

  lifecycle {
    prevent_destroy = true # holds the only copy of the main stack's state
  }
}

resource "aws_s3_bucket_ownership_controls" "this" {
  for_each = aws_s3_bucket.this
  bucket   = each.value.id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_public_access_block" "this" {
  for_each                = aws_s3_bucket.this
  bucket                  = each.value.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "this" {
  for_each = aws_s3_bucket.this
  bucket   = each.value.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "this" {
  for_each = aws_s3_bucket.this
  bucket   = each.value.id
  versioning_configuration {
    status = "Enabled" # state: rollback of a bad write; artifacts: required by CodePipeline
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "state" {
  bucket     = aws_s3_bucket.this["state"].id
  depends_on = [aws_s3_bucket_versioning.this]

  rule {
    id     = "expire-old-state-versions"
    status = "Enabled"
    filter {}
    noncurrent_version_expiration {
      noncurrent_days           = var.state_noncurrent_retention_days
      newer_noncurrent_versions = 50 # always keep the 50 most recent old versions
    }
    abort_incomplete_multipart_upload {
      days_after_initiation = 3
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket     = aws_s3_bucket.this["artifacts"].id
  depends_on = [aws_s3_bucket_versioning.this]

  rule {
    id     = "expire-artifacts"
    status = "Enabled"
    filter {}
    expiration {
      days = var.artifact_retention_days
    }
    noncurrent_version_expiration {
      noncurrent_days = 1
    }
    abort_incomplete_multipart_upload {
      days_after_initiation = 3
    }
  }
}

data "aws_iam_policy_document" "tls_only" {
  for_each = aws_s3_bucket.this

  statement {
    sid       = "DenyInsecureTransport"
    effect    = "Deny"
    actions   = ["s3:*"]
    resources = [each.value.arn, "${each.value.arn}/*"]
    principals {
      type        = "*"
      identifiers = ["*"]
    }
    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "this" {
  for_each   = aws_s3_bucket.this
  bucket     = each.value.id
  policy     = data.aws_iam_policy_document.tls_only[each.key].json
  depends_on = [aws_s3_bucket_public_access_block.this]
}
