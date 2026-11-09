output "s3_bucket" {
  description = "Model/graph bucket (BUCKET for scripts/upload_to_s3.sh)."
  value       = aws_s3_bucket.models.bucket
}

output "model_s3_uri" {
  description = "Where the checkpoint must be uploaded."
  value       = "s3://${aws_s3_bucket.models.bucket}/${var.model_s3_prefix}/"
}

output "graphs_s3_uri" {
  description = "Where the compiled graph buckets must be uploaded."
  value       = "s3://${aws_s3_bucket.models.bucket}/${var.graphs_s3_prefix}/"
}

output "backend_ecr_url" {
  description = "Existing neuron_server ECR repository."
  value       = data.aws_ecr_repository.backend.repository_url
}

output "router_ecr_url" {
  description = "Router ECR repository."
  value       = aws_ecr_repository.router.repository_url
}

output "cluster_name" {
  value = aws_ecs_cluster.this.name
}

output "backend_service_name" {
  value = aws_ecs_service.backend.name
}

output "router_service_name" {
  value = aws_ecs_service.router.name
}

output "backend_ami_id" {
  description = "AMI the trn launch template resolved to."
  value       = local.backend_ami_id
}

output "alb_dns_name" {
  description = "ALB hostname (when enable_alb)."
  value       = var.enable_alb ? aws_lb.router[0].dns_name : null
}

output "api_keys_table" {
  description = "DynamoDB table of approved router API keys (router/api_keys.py --table)"
  value       = aws_dynamodb_table.api_keys.name
}
