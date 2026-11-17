# MiniMax H3 (video generation) runs on a hand-run trn2.3xlarge Capacity Block in sa-east-1 (the only region the
# Capacity Block is bought in: H3's licence excludes the US, EU, UK and South Korea, so its weights, compiled graphs
# and outputs stay out of var.region). This file creates what that host needs:
#   - a private S3 bucket in sa-east-1 (weights are downloaded from Hugging Face straight onto the instance; the bucket
#     holds prepared shards, NEFFs, goldens and generated outputs)
#   - an instance role/profile <name>-h3-neuron: SSM Session Manager plus read/write on that bucket only. It has no
#     access to the models bucket in var.region, so a bug cannot copy H3 data into the US.
# The security group (no ingress, all egress for Hugging Face and SSM) comes from remote_regions.tf. The Capacity Block
# itself is purchased by hand; launch with the outputs, for example:
#   aws ec2 run-instances --region sa-east-1 --instance-type trn2.3xlarge --image-id <Neuron DLAMI> \
#     --instance-market-options MarketType=capacity-block --capacity-reservation-specification \
#     CapacityReservationTarget={CapacityReservationId=cr-...} --iam-instance-profile Name=<h3_instance_profile> \
#     --security-group-ids <remote_security_groups[sa-east-1]> --subnet-id <a sa-east-1b subnet> \
#     --block-device-mappings DeviceName=/dev/sda1,Ebs={VolumeSize=1000,VolumeType=gp3,Throughput=1000,Iops=6000}
locals {
  h3_enabled     = contains(var.remote_regions, "sa-east-1")
  h3_bucket_name = var.h3_bucket_name != "" ? var.h3_bucket_name : "${var.name}-h3-sa-east-1-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket" "h3" {
  count    = local.h3_enabled ? 1 : 0
  provider = aws.sa_east_1
  bucket   = local.h3_bucket_name
}

resource "aws_s3_bucket_ownership_controls" "h3" {
  count    = length(aws_s3_bucket.h3)
  provider = aws.sa_east_1
  bucket   = aws_s3_bucket.h3[0].id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_public_access_block" "h3" {
  count                   = length(aws_s3_bucket.h3)
  provider                = aws.sa_east_1
  bucket                  = aws_s3_bucket.h3[0].id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "h3" {
  count    = length(aws_s3_bucket.h3)
  provider = aws.sa_east_1
  bucket   = aws_s3_bucket.h3[0].id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256" # var.s3_kms_key_arn is a key of var.region: a KMS key cannot be used across regions
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "h3" {
  count    = length(aws_s3_bucket.h3)
  provider = aws.sa_east_1
  bucket   = aws_s3_bucket.h3[0].id
  rule {
    id     = "abort-incomplete-multipart"
    status = "Enabled"
    filter {}
    abort_incomplete_multipart_upload {
      days_after_initiation = 3
    }
  }
}

data "aws_iam_policy_document" "h3_bucket" {
  count = length(aws_s3_bucket.h3)
  statement {
    sid       = "DenyInsecureTransport"
    effect    = "Deny"
    actions   = ["s3:*"]
    resources = [aws_s3_bucket.h3[0].arn, "${aws_s3_bucket.h3[0].arn}/*"]
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

resource "aws_s3_bucket_policy" "h3" {
  count      = length(aws_s3_bucket.h3)
  provider   = aws.sa_east_1
  bucket     = aws_s3_bucket.h3[0].id
  policy     = data.aws_iam_policy_document.h3_bucket[0].json
  depends_on = [aws_s3_bucket_public_access_block.h3]
}

resource "aws_iam_role" "h3_neuron" {
  count                = length(aws_s3_bucket.h3)
  name                 = "${var.name}-h3-neuron"
  assume_role_policy   = data.aws_iam_policy_document.ec2_assume.json
  permissions_boundary = var.role_permissions_boundary_arn != "" ? var.role_permissions_boundary_arn : null
}

resource "aws_iam_role_policy_attachment" "h3_neuron_ssm" {
  count      = length(aws_iam_role.h3_neuron)
  role       = aws_iam_role.h3_neuron[0].name
  policy_arn = "arn:${local.partition}:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

data "aws_iam_policy_document" "h3_neuron" {
  count = length(aws_s3_bucket.h3)
  statement {
    sid       = "H3BucketList"
    actions   = ["s3:ListBucket", "s3:GetBucketLocation"]
    resources = [aws_s3_bucket.h3[0].arn]
  }
  statement {
    sid       = "H3BucketObjects" # prepared shards, NEFFs, goldens, outputs
    actions   = ["s3:GetObject", "s3:PutObject", "s3:AbortMultipartUpload", "s3:ListBucketMultipartUploads"]
    resources = ["${aws_s3_bucket.h3[0].arn}/*"]
  }
  statement {
    sid       = "Logs"
    actions   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "h3_neuron" {
  count  = length(aws_iam_role.h3_neuron)
  name   = "h3-bucket-logs"
  role   = aws_iam_role.h3_neuron[0].id
  policy = data.aws_iam_policy_document.h3_neuron[0].json
}

resource "aws_iam_instance_profile" "h3_neuron" {
  count = length(aws_iam_role.h3_neuron)
  name  = "${var.name}-h3-neuron"
  role  = aws_iam_role.h3_neuron[0].name
}

# S3 gateway endpoint for the default VPC of sa-east-1 (where the H3 host runs): keeps the host's S3 traffic (weights,
# NEFFs, outputs, SSM-staged tarballs) on the AWS network instead of the internet gateway. Gateway endpoints are free
# and add one prefix-list route to the VPC's main route table (the host's subnet has no explicit association, so it
# uses the main table). The endpoint policy is the default (full access): the H3 bucket policy and the instance role
# stay the access controls. The host keeps its public IP and internet route for Hugging Face downloads.
data "aws_route_table" "h3_main" {
  count    = local.h3_enabled ? 1 : 0
  provider = aws.sa_east_1
  vpc_id   = module.remote_sg_sa_east_1[0].vpc_id
  filter {
    name   = "association.main"
    values = ["true"]
  }
}

resource "aws_vpc_endpoint" "h3_s3" {
  count             = local.h3_enabled ? 1 : 0
  provider          = aws.sa_east_1
  vpc_id            = module.remote_sg_sa_east_1[0].vpc_id
  service_name      = "com.amazonaws.sa-east-1.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = [data.aws_route_table.h3_main[0].id]
  tags              = { Name = "${var.name}-h3-s3" }
}
