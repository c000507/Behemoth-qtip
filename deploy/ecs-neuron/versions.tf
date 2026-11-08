terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.60, < 7.0"
    }
  }

  # Optional remote state (create the bucket/table first, then `terraform init -migrate-state`):
  # backend "s3" {
  #   bucket         = "behemoth-tfstate-912935854432"
  #   key            = "ecs-neuron/terraform.tfstate"
  #   region         = "us-east-1"
  #   encrypt        = true
  #   use_lockfile   = true            # Terraform >= 1.10; or dynamodb_table = "terraform-locks"
  # }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = merge({ Project = var.name, ManagedBy = "terraform" }, var.tags)
  }
}

data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}
