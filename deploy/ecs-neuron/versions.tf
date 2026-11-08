terraform {
  required_version = ">= 1.10.0" # S3 backend use_lockfile

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.60, < 7.0"
    }
  }

  # Remote state in S3 with native locking (no DynamoDB), partial config: the values come from backend.hcl
  # (see backend.hcl.example; the bucket is created by pipeline/). One-time migration from the local state:
  #   terraform init -migrate-state -backend-config=backend.hcl
  backend "s3" {}
}

provider "aws" {
  region = var.region

  default_tags {
    tags = merge({ Project = var.name, ManagedBy = "terraform" }, var.tags)
  }
}

data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}
