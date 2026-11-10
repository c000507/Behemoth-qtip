# No-ingress security group in the region's default VPC for a Trainium host managed through SSM; all egress (S3 and ECR
# in the stack's region, SSM, the Hugging Face hub).
terraform {
  required_providers {
    aws = { source = "hashicorp/aws" }
  }
}

variable "name" {
  type = string
}

data "aws_vpc" "default" {
  default = true
}

resource "aws_security_group" "this" {
  name        = "${var.name}-remote-neuron"
  description = "Trainium host outside the stack region: no ingress (SSM), all egress"
  vpc_id      = data.aws_vpc.default.id

  egress {
    description = "All egress"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

output "security_group_id" {
  value = aws_security_group.this.id
}

output "vpc_id" {
  value = data.aws_vpc.default.id
}
