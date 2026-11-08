# The backend repository already exists (created outside Terraform: scan on push, AES256, mutable tags).
# It is only READ here so `terraform destroy` can never delete the pushed images. To manage it in Terraform instead,
# replace this data source by an aws_ecr_repository "backend" resource plus
#   import { to = aws_ecr_repository.backend, id = "behemoth-neuron-server" }
data "aws_ecr_repository" "backend" {
  name = var.backend_ecr_repository_name
}

resource "aws_ecr_repository" "router" {
  name                 = "${var.name}-router"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

resource "aws_ecr_lifecycle_policy" "router" {
  repository = aws_ecr_repository.router.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "keep the last 10 images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 10 }
      action       = { type = "expire" }
    }]
  })
}
