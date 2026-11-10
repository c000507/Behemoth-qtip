# Trainium hosts outside var.region (e.g. Capacity Blocks in Sydney or Mumbai, run by hand rather than by ECS): one
# instance profile (IAM is global) and a no-ingress security group in each listed region's default VPC. The hosts are
# reached through SSM and read the model bucket and images in var.region.
locals {
  remote_tags = merge({ Project = var.name, ManagedBy = "terraform" }, var.tags)
}

provider "aws" {
  alias  = "ap_southeast_2"
  region = "ap-southeast-2"
  default_tags {
    tags = local.remote_tags
  }
}

provider "aws" {
  alias  = "ap_south_1"
  region = "ap-south-1"
  default_tags {
    tags = local.remote_tags
  }
}

module "remote_sg_ap_southeast_2" {
  source    = "./modules/remote_sg"
  count     = contains(var.remote_regions, "ap-southeast-2") ? 1 : 0
  providers = { aws = aws.ap_southeast_2 }
  name      = var.name
}

module "remote_sg_ap_south_1" {
  source    = "./modules/remote_sg"
  count     = contains(var.remote_regions, "ap-south-1") ? 1 : 0
  providers = { aws = aws.ap_south_1 }
  name      = var.name
}

resource "aws_iam_role" "remote_neuron" {
  count                = length(var.remote_regions) > 0 ? 1 : 0
  name                 = "${var.name}-remote-neuron"
  assume_role_policy   = data.aws_iam_policy_document.ec2_assume.json
  permissions_boundary = var.role_permissions_boundary_arn != "" ? var.role_permissions_boundary_arn : null
}

resource "aws_iam_role_policy_attachment" "remote_neuron_ssm" {
  count      = length(aws_iam_role.remote_neuron)
  role       = aws_iam_role.remote_neuron[0].name
  policy_arn = "arn:${local.partition}:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

data "aws_iam_policy_document" "remote_neuron" {
  statement {
    sid       = "ModelBucket" # weights, graphs and prepared shards
    actions   = ["s3:ListBucket", "s3:GetObject"]
    resources = [aws_s3_bucket.models.arn, "${aws_s3_bucket.models.arn}/*"]
  }
  statement {
    sid = "ImagesAndLogs"
    actions = [
      "ecr:GetAuthorizationToken", "ecr:BatchCheckLayerAvailability", "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage",
      "logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams",
    ]
    resources = ["*"]
  }
  dynamic "statement" {
    for_each = var.s3_kms_key_arn != "" ? [1] : []
    content {
      actions   = ["kms:Decrypt"]
      resources = [var.s3_kms_key_arn]
    }
  }
}

resource "aws_iam_role_policy" "remote_neuron" {
  count  = length(aws_iam_role.remote_neuron)
  name   = "models-images-logs"
  role   = aws_iam_role.remote_neuron[0].id
  policy = data.aws_iam_policy_document.remote_neuron.json
}

resource "aws_iam_instance_profile" "remote_neuron" {
  count = length(aws_iam_role.remote_neuron)
  name  = "${var.name}-remote-neuron"
  role  = aws_iam_role.remote_neuron[0].name
}
