output "s3_bucket" {
  description = "Model/graph bucket (BUCKET for scripts/upload_to_s3.sh)."
  value       = aws_s3_bucket.models.bucket
}

output "model_s3_uri" {
  description = "Per model: where the checkpoint must be uploaded."
  value       = { for k, m in local.models : k => "s3://${aws_s3_bucket.models.bucket}/${m.model_s3_prefix}/" }
}

output "graphs_s3_uri" {
  description = "Per model: where the compiled graph buckets must be uploaded."
  value       = { for k, m in local.models : k => "s3://${aws_s3_bucket.models.bucket}/${m.graphs_s3_prefix}/" }
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
  description = "Per model: its ECS service."
  value       = { for k, s in aws_ecs_service.backend : k => s.name }
}

output "backend_asg_name" {
  description = "Per model: its primary ASG, and the on-demand fallback ASG of a spot model (null without one)."
  value = { for k, _ in local.models : k => {
    primary  = local.primary_asg[k]
    fallback = lookup(local.fallback_asg, k, null)
  } }
}

output "served_models" {
  description = "Per model: the ids the router serves, capacity, idle timeout."
  value = { for k, m in local.models : k => {
    model_ids     = m.model_ids
    capacity      = m.use_spot ? (m.spot_fallback ? "spot, on-demand fallback" : "spot") : "on-demand"
    idle_timeout  = m.idle_timeout_seconds
    default_model = k == local.default_model
  } }
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

output "remote_instance_profile" {
  description = "Instance profile for hand-run Trainium hosts outside the stack region (empty without remote_regions)."
  value       = one(aws_iam_instance_profile.remote_neuron[*].name)
}

output "remote_security_groups" {
  description = "Per remote region: the no-ingress security group and the default VPC it is in."
  value = merge(
    { for m in module.remote_sg_ap_southeast_2 : "ap-southeast-2" => { security_group_id = m.security_group_id, vpc_id = m.vpc_id } },
    { for m in module.remote_sg_ap_south_1 : "ap-south-1" => { security_group_id = m.security_group_id, vpc_id = m.vpc_id } },
    { for m in module.remote_sg_sa_east_1 : "sa-east-1" => { security_group_id = m.security_group_id, vpc_id = m.vpc_id } },
  )
}

output "h3_bucket" {
  description = "sa-east-1 bucket for H3 weights, NEFFs and outputs (empty without sa-east-1 in remote_regions)."
  value       = one(aws_s3_bucket.h3[*].bucket)
}

output "h3_instance_profile" {
  description = "Instance profile for the H3 trn2 Capacity Block host in sa-east-1: SSM, read/write of the H3 bucket only (no access to the models bucket in the stack region)."
  value       = one(aws_iam_instance_profile.h3_neuron[*].name)
}
