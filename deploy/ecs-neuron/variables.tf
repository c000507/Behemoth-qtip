# ------------------------------------------------------------------ general
variable "region" {
  description = "AWS region."
  type        = string
  default     = "us-east-1"
}

variable "name" {
  description = "Name prefix for all resources (cluster, services, roles, log groups)."
  type        = string
  default     = "behemoth"
}

variable "tags" {
  description = "Extra tags added to every resource."
  type        = map(string)
  default     = {}
}

variable "log_retention_days" {
  description = "CloudWatch log retention for the backend and router log groups."
  type        = number
  default     = 14
}

# ------------------------------------------------------------------ network
variable "vpc_id" {
  description = "VPC holding the backend instance, the router and (optionally) the ALB."
  type        = string
}

variable "backend_subnet_ids" {
  description = "Subnets for the spot ASG and the backend task ENI. Trn1 is only offered in some AZs (us-east-1f today)."
  type        = list(string)
  default     = ["subnet-05a70c02d594e2500"]
}

variable "router_subnet_ids" {
  description = "Subnets for the Fargate router task. Defaults to backend_subnet_ids when empty."
  type        = list(string)
  default     = []
}

variable "alb_subnet_ids" {
  description = "Subnets (>= 2 AZs) for the optional ALB. Required when enable_alb = true."
  type        = list(string)
  default     = []
}

variable "backend_instance_security_group_ids" {
  description = "Security groups for the trn EC2 host itself (ECS agent, image pulls). Needs egress to ECS/ECR/CloudWatch."
  type        = list(string)
  default     = ["sg-0264b44fbe050547a"]
}

variable "backend_associate_public_ip" {
  description = "Give the trn host a public IP (needed when the subnet has an internet gateway but no NAT, for the ECS agent, ECR and CloudWatch)."
  type        = bool
  default     = true
}

variable "router_assign_public_ip" {
  description = "Assign a public IP to the Fargate router task (needed in a public subnet without NAT, for cloudflared, ECS API, ECR)."
  type        = bool
  default     = true
}

variable "extra_backend_ingress_cidrs" {
  description = "Additional CIDRs allowed to reach the backend task on its port (e.g. your VPN for debugging). The router SG is always allowed."
  type        = list(string)
  default     = []
}

variable "create_s3_gateway_endpoint" {
  description = "Create an S3 gateway VPC endpoint. awsvpc tasks on EC2 get no public IP, so without a NAT gateway this endpoint is the backend task's only path to S3. Set false if the VPC already has one."
  type        = bool
  default     = true
}

variable "nat_security_group_id" {
  description = "Security group of a NAT instance the subnets route 0.0.0.0/0 through (empty: NAT gateway or public IPs, nothing to open). The router and backend task groups are admitted to it."
  type        = string
  default     = ""
}

variable "s3_endpoint_route_table_ids" {
  description = "Route tables (of the backend subnets) to attach the S3 gateway endpoint to. Required when create_s3_gateway_endpoint = true."
  type        = list(string)
  default     = []
}

# ------------------------------------------------------------------ storage
variable "s3_bucket_name" {
  description = "Bucket for models and compiled graphs. Empty = \"<name>-neuron-models-<account id>\"."
  type        = string
  default     = ""
}

variable "s3_versioning" {
  description = "Enable versioning on the model bucket (doubles storage cost on re-uploads)."
  type        = bool
  default     = false
}

variable "s3_kms_key_arn" {
  description = "KMS key for bucket SSE. Empty = SSE-S3 (AES256). With a key the backend task role also gets kms:Decrypt on it."
  type        = string
  default     = ""
}

variable "model_s3_prefix" {
  description = "Key prefix of the checkpoint directory (no leading/trailing slash)."
  type        = string
  default     = "models/cydonia-24b-fp8"
}

variable "graphs_s3_prefix" {
  description = "Key prefix of the compiled graph bucket directory (one subdirectory per bucket with graph.neff + meta.json)."
  type        = string
  default     = "graphs/cydonia-24b-fp8-kv"
}

# ------------------------------------------------------------------ images
variable "backend_ecr_repository_name" {
  description = "Existing ECR repository of the neuron_server image (created outside Terraform, read via data source)."
  type        = string
  default     = "behemoth-neuron-server"
}

variable "backend_image_tag" {
  description = "Tag of the neuron_server image to run (git short sha from build_and_push.sh, or latest)."
  type        = string
  default     = "latest"
}

variable "router_image_tag" {
  description = "Tag of the router image in the Terraform-managed router ECR repo."
  type        = string
  default     = "latest"
}

variable "cloudflared_image" {
  description = "cloudflared sidecar image (pin a version)."
  type        = string
  default     = "cloudflare/cloudflared:2024.12.2"
}

# ------------------------------------------------------------------ backend (spot trn)
variable "backend_instance_types" {
  description = "Instance types for the spot ASG (first = preferred). Must all expose /dev/neuron0 with >= 2 NeuronCores."
  type        = list(string)
  default     = ["trn1.2xlarge"]
}

variable "backend_ami_ssm_parameter" {
  description = "SSM parameter holding the ECS-optimized Neuron AMI id."
  type        = string
  default     = "/aws/service/ecs/optimized-ami/amazon-linux-2023/neuron/recommended/image_id"
}

variable "backend_ami_id" {
  description = "Explicit AMI id overriding backend_ami_ssm_parameter (pin it so a new AMI/driver cannot break the container's runtime version)."
  type        = string
  default     = ""
}

variable "backend_root_volume_gb" {
  description = "Root gp3 volume size (docker images; also the fallback for model files if no instance store is found)."
  type        = number
  default     = 100
}

variable "backend_use_spot" {
  description = "Run the trn backend on spot (true) or on-demand (false). Spot needs the 'All Trn Spot Instance Requests' vCPU quota; on-demand uses 'Running On-Demand Trn instances'. Switching only changes the ASG's instances distribution (in place)."
  type        = bool
  default     = true
}

variable "spot_max_price" {
  description = "Optional spot max price per hour (empty = on-demand price cap)."
  type        = string
  default     = ""
}

variable "backend_port" {
  description = "Port neuron_server listens on."
  type        = number
  default     = 1337
}

variable "backend_model_name" {
  description = "BEHEMOTH_MODEL_NAME reported by the backend."
  type        = string
  default     = "cydonia-24b"
}

variable "backend_cpu" {
  description = "Task CPU units reserved for the backend (trn1.2xlarge registers 8192)."
  type        = number
  default     = 7168
}

variable "backend_memory_reservation_mib" {
  description = "Soft memory reservation of the backend container (trn1.2xlarge has 32 GiB; ECS registers ~31 GiB)."
  type        = number
  default     = 24576
}

variable "backend_memory_hard_limit_mib" {
  description = "Optional hard memory limit (0 = none; the task is alone on the instance)."
  type        = number
  default     = 0
}

variable "backend_shm_size_mib" {
  description = "/dev/shm size for the backend container."
  type        = number
  default     = 4096
}

variable "backend_startup_grace_seconds" {
  description = "How long healthcheck.sh reports healthy while the entrypoint is still pulling/loading (ECS startPeriod is capped at 300 s)."
  type        = number
  default     = 1200
}

variable "backend_extra_env" {
  description = "Extra environment for neuron_server (e.g. BEHEMOTH_NEURON_LOAD_THREADS)."
  type        = map(string)
  default     = {}
}

variable "backend_extra_server_args" {
  description = "Extra neuron_server flags (e.g. \"--prefix-cache-slots 64\")."
  type        = string
  default     = ""
}

variable "backend_disable_ipv6" {
  description = "Disable IPv6 in the backend task's network namespace so the S3 pull uses IPv4 and the S3 gateway endpoint (DNS64 subnets otherwise route it through NAT64)."
  type        = bool
  default     = true
}

variable "s5cmd_workers" {
  description = "Parallel files for the S3 pull."
  type        = number
  default     = 64
}

# ------------------------------------------------------------------ router
variable "router_port" {
  description = "Router listen port. 1337 keeps an existing tunnel ingress rule (http://localhost:1337) valid."
  type        = number
  default     = 1337
}

variable "router_cpu" {
  description = "Fargate task CPU (router + cloudflared)."
  type        = number
  default     = 256
}

variable "router_memory" {
  description = "Fargate task memory MiB (router + cloudflared)."
  type        = number
  default     = 512
}

variable "model_ids" {
  description = "Model ids returned by the router's GET /v1/models."
  type        = list(string)
  default     = ["cydonia-24b"]
}

variable "model_owned_by" {
  description = "owned_by field of /v1/models."
  type        = string
  default     = "cgg507"
}

variable "proxy_paths" {
  description = "POST paths proxied to the backend (neuron_server only implements /v1/chat/completions today)."
  type        = list(string)
  default     = ["/v1/chat/completions"]
}

variable "idle_timeout_seconds" {
  description = "Scale the backend to 0 after this long without requests (and none in flight)."
  type        = number
  default     = 300
}

variable "max_wait_seconds" {
  description = "Max time a request waits for a cold backend before 503 + Retry-After."
  type        = number
  default     = 900
}

variable "cold_nonstream_mode" {
  description = "Non-streaming request on a cold backend: \"wait\" (hold the request up to max_wait_seconds) or \"reject\" (503 + Retry-After immediately, warm up in the background)."
  type        = string
  default     = "wait"
  validation {
    condition     = contains(["wait", "reject"], var.cold_nonstream_mode)
    error_message = "cold_nonstream_mode must be wait or reject."
  }
}

variable "enable_cloudflared" {
  description = "Run the cloudflared sidecar in the router task."
  type        = bool
  default     = true
}

variable "cloudflare_tunnel_token_secret_arn" {
  description = "ARN of a Secrets Manager secret (plain string) or SSM SecureString parameter holding the tunnel token. Required when enable_cloudflared."
  type        = string
  default     = ""
}

variable "secrets_kms_key_arn" {
  description = "Customer KMS key encrypting the tunnel token / API key secrets (empty = AWS managed key)."
  type        = string
  default     = ""
}

variable "router_api_key_secret_arn" {
  description = "Optional Secrets Manager / SSM ARN of a bearer key the router requires (useful with the ALB). Empty = no auth at the router."
  type        = string
  default     = ""
}

variable "enable_alb" {
  description = "Also expose the router through an internet-facing ALB."
  type        = bool
  default     = false
}

variable "alb_certificate_arn" {
  description = "ACM certificate for an HTTPS listener. Empty = HTTP listener on port 80 only."
  type        = string
  default     = ""
}

variable "alb_ingress_cidrs" {
  description = "CIDRs allowed to reach the ALB."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "alb_idle_timeout" {
  description = "ALB idle timeout (s); must exceed max_wait_seconds for cold non-streaming requests."
  type        = number
  default     = 1200
}

variable "container_insights" {
  description = "Enable ECS Container Insights on the cluster (extra CloudWatch cost)."
  type        = bool
  default     = false
}
