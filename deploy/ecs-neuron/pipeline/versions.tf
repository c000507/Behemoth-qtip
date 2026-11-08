# Bootstrap root for the CI/CD of deploy/ecs-neuron. It keeps LOCAL state on purpose (pipeline/terraform.tfstate,
# gitignored by deploy/ecs-neuron/.gitignore): it creates the bucket the main stack's remote state lives in, so it cannot
# store its own state there before it exists. See README.md ("Bootstrap state") to move it into the bucket afterwards.
terraform {
  required_version = ">= 1.10.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 6.0, < 7.0"
    }
  }
}

provider "aws" {
  region              = var.region
  allowed_account_ids = [var.account_id] # refuse to run against any other account

  default_tags {
    tags = merge({ Project = var.name, ManagedBy = "terraform", Stack = "ecs-neuron-pipeline" }, var.tags)
  }
}

data "aws_partition" "current" {}
