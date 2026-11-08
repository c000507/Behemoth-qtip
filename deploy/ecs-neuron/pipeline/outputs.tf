output "state_bucket" {
  description = "Bucket of the main stack's state and config (bucket in backend.hcl)."
  value       = aws_s3_bucket.this["state"].bucket
}

output "state_key" {
  value = local.state_key
}

output "config_s3_uri" {
  description = "Where the main stack's terraform.tfvars must be uploaded."
  value       = "s3://${aws_s3_bucket.this["state"].bucket}/${local.config_key}"
}

output "artifact_bucket" {
  value = aws_s3_bucket.this["artifacts"].bucket
}

output "github_connection_arn" {
  description = "Created PENDING: complete it in the console (Developer Tools > Settings > Connections)."
  value       = aws_codestarconnections_connection.github.arn
}

output "github_connection_status" {
  value = aws_codestarconnections_connection.github.connection_status
}

output "pipeline_name" {
  value = aws_codepipeline.this.name
}

output "pipeline_console_url" {
  value = "https://${var.region}.console.aws.amazon.com/codesuite/codepipeline/pipelines/${aws_codepipeline.this.name}/view?region=${var.region}"
}

output "codebuild_role_arns" {
  value = { for k, v in aws_iam_role.codebuild : k => v.arn }
}
