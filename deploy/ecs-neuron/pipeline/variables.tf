# Every default here is a public, non-secret value (the account id already appears in the stack's README/examples), so
# the bootstrap applies without a tfvars file. Override in a gitignored pipeline/terraform.tfvars if needed.

# ------------------------------------------------------------------ account
variable "account_id" {
  description = "AWS account the pipeline and the stack live in (provider allowed_account_ids)."
  type        = string
  default     = "912935854432"
}

variable "region" {
  description = "AWS region of the pipeline and of the stack."
  type        = string
  default     = "us-east-1"
}

variable "remote_regions" {
  description = "Regions where the stack's remote_regions.tf may manage security groups (keep in step with the stack's remote_regions)."
  type        = list(string)
  default     = ["ap-southeast-2", "ap-south-1"]
}

variable "name" {
  description = "Name prefix of the MAIN stack (its var.name): used to scope the apply role to the stack's resources."
  type        = string
  default     = "behemoth"
}

variable "tags" {
  description = "Extra tags on every resource of this bootstrap stack."
  type        = map(string)
  default     = {}
}

# ------------------------------------------------------------------ source
variable "github_repository" {
  description = "GitHub owner/repo the pipeline builds."
  type        = string
  default     = "c000507/Behemoth-qtip"
}

variable "github_branch" {
  description = "Branch whose pushes trigger the pipeline."
  type        = string
  default     = "rocm-to-neff"
}

variable "trigger_file_paths" {
  description = "Glob patterns (repo-relative) of pushed files that start the pipeline. Empty list = every push to the branch."
  type        = list(string)
  default = [
    "deploy/ecs-neuron/**",
    "runtime/**",
    "neuron/cpp/**",
    "glm53/**",
    "third_party/**",
    "CMakeLists.txt", # COPY'd by backend/Dockerfile
  ]
}

variable "stack_dir" {
  description = "Repo-relative directory of the main Terraform stack."
  type        = string
  default     = "deploy/ecs-neuron"
}

# ------------------------------------------------------------------ buckets
variable "state_bucket_name" {
  description = "Bucket for the main stack's state (ecs-neuron/terraform.tfstate) and config (ecs-neuron/terraform.tfvars). Empty = <name>-tfstate-<account>."
  type        = string
  default     = ""
}

variable "state_key_prefix" {
  description = "Key prefix of the main stack's objects in the state bucket."
  type        = string
  default     = "ecs-neuron"
}

variable "artifact_bucket_name" {
  description = "CodePipeline artifact bucket. Empty = <name>-pipeline-artifacts-<account>."
  type        = string
  default     = ""
}

variable "artifact_retention_days" {
  description = "Days pipeline artifacts (source bundles, plans) are kept."
  type        = number
  default     = 30
}

variable "state_noncurrent_retention_days" {
  description = "Days old versions of the state/config objects are kept (state rollback window)."
  type        = number
  default     = 180
}

# ------------------------------------------------------------------ builds
variable "terraform_version" {
  description = "Terraform version the plan/apply builds download."
  type        = string
  default     = "1.12.2"
}

variable "terraform_sha256" {
  description = "SHA-256 of terraform_<version>_linux_amd64.zip (from releases.hashicorp.com/terraform/<version>/terraform_<version>_SHA256SUMS). Must match terraform_version."
  type        = string
  default     = "1eaed12ca41fcfe094da3d76a7e9aa0639ad3409c43be0103ee9f5a1ff4b7437"
}

variable "codebuild_image" {
  description = "CodeBuild managed image (x86_64, ships Docker, git, jq, unzip, AWS CLI v2)."
  type        = string
  default     = "aws/codebuild/standard:7.0"
}

variable "image_build_compute_type" {
  description = "Compute type of the image build (the backend compiles neuron_server; LARGE = 8 vCPU / 15 GB)."
  type        = string
  default     = "BUILD_GENERAL1_LARGE"
}

variable "terraform_compute_type" {
  description = "Compute type of the plan / apply builds."
  type        = string
  default     = "BUILD_GENERAL1_SMALL"
}

variable "image_build_timeout_minutes" {
  description = "Timeout of the image build."
  type        = number
  default     = 60
}

variable "backend_ecr_repository_name" {
  description = "Existing ECR repo of the backend image (main stack var.backend_ecr_repository_name)."
  type        = string
  default     = "behemoth-neuron-server"
}

variable "router_ecr_repository_name" {
  description = "Router ECR repo created by the main stack. Empty = <name>-router."
  type        = string
  default     = ""
}

variable "backend_base_image" {
  description = "BASE_IMAGE build arg of backend/Dockerfile. The ECR Public mirror of the Docker official image avoids Docker Hub's anonymous pull limit on shared CodeBuild IPs."
  type        = string
  default     = "public.ecr.aws/docker/library/ubuntu:22.04"
}

variable "neuron_runtime_version" {
  description = "NEURON_RUNTIME_VERSION build arg (pin to the AMI's driver generation; empty = latest). Changing it does NOT change the image tag: release with FORCE_IMAGE_BUILD=backend."
  type        = string
  default     = ""
}

variable "neuron_collectives_version" {
  description = "NEURON_COLLECTIVES_VERSION build arg (see neuron_runtime_version)."
  type        = string
  default     = ""
}

variable "dockerhub_secret_arn" {
  description = "Optional Secrets Manager secret (JSON with keys username, password) for an authenticated Docker Hub login before builds (the router Dockerfile pulls python:3.12-slim from Docker Hub). Empty = anonymous pulls."
  type        = string
  default     = ""
}

variable "log_retention_days" {
  description = "Retention of the CodeBuild log groups."
  type        = number
  default     = 30
}

# ------------------------------------------------------------------ deploy
variable "apply_role_policy" {
  description = "\"scoped\" (default): the apply role gets ReadOnlyAccess plus write access limited to the stack's services and <name>-* resources. \"admin\": AdministratorAccess (use if a scoped apply keeps failing on a missing permission; see README)."
  type        = string
  default     = "scoped"
  validation {
    condition     = contains(["scoped", "admin"], var.apply_role_policy)
    error_message = "apply_role_policy must be scoped or admin."
  }
}

variable "models_bucket_name" {
  description = "Model bucket of the main stack (its s3_bucket_name). Empty = <name>-neuron-models-<account> (the stack's default)."
  type        = string
  default     = ""
}

variable "approval_sns_topic_arn" {
  description = "Optional SNS topic notified when a plan waits for approval."
  type        = string
  default     = ""
}

variable "execution_mode" {
  description = "Pipeline execution mode. SUPERSEDED: a newer run replaces one still waiting for approval, so a stale plan is never applied."
  type        = string
  default     = "SUPERSEDED"
}
