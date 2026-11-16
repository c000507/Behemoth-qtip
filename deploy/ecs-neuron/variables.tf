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
  description = "Single-model mode only (models = {}): Key prefix of the checkpoint directory (no leading/trailing slash)."
  type        = string
  default     = "models/cydonia-24b-fp8"
}

variable "graphs_s3_prefix" {
  description = "Single-model mode only (models = {}): Key prefix of the compiled graph bucket directory (one subdirectory per bucket with graph.neff + meta.json)."
  type        = string
  default     = "graphs/cydonia-24b-fp8-kv"
}

# ------------------------------------------------------------------ images
variable "prepared_s3_prefix" {
  description = "Single-model mode only (models = {}): Optional prefix of prepared shards (neuron_server BEHEMOTH_NEURON_WRITE_PREPARED output: layout.json + model-layer-NNNNN.bin): the backend streams these instead of the safetensors layer files and skips the expansion pass. Must match the graphs' layout (the server checks layout.json)."
  type        = string
  default     = ""
}

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

# Tags are mutable: the pipeline's plan resolves each tag to its digest and the task definitions name the image by
# digest, so the approved plan runs exactly the image that was planned, whatever is later pushed under the tag.
variable "backend_image_digest" {
  description = "sha256:... digest of backend_image_tag; set, the task definition pins it. Empty = the tag alone."
  type        = string
  default     = ""
  validation {
    condition     = var.backend_image_digest == "" || can(regex("^sha256:[0-9a-f]{64}$", var.backend_image_digest))
    error_message = "backend_image_digest must be sha256:<64 hex>."
  }
}

variable "router_image_digest" {
  description = "sha256:... digest of router_image_tag; set, the task definition pins it. Empty = the tag alone."
  type        = string
  default     = ""
  validation {
    condition     = var.router_image_digest == "" || can(regex("^sha256:[0-9a-f]{64}$", var.router_image_digest))
    error_message = "router_image_digest must be sha256:<64 hex>."
  }
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
  description = "Default of models[*].use_spot: run a model's trn instance on spot (true; a spot ASG, plus an on-demand fallback ASG with backend_spot_fallback) or on-demand (false). Spot needs the 'All Trn Spot Instance Requests' vCPU quota; on-demand uses 'Running On-Demand Trn instances'. Switching creates/destroys the model's spot ASG; an existing ASG is never converted."
  type        = bool
  default     = true
}

variable "backend_warm_pool" {
  description = "Default of models[*].warm_pool for on-demand models: keep one stopped, pre-initialised backend instance in an ASG warm pool (cold start skips launch, boot and the image pull; costs the stopped instance's root volume, ~$8/month for 100 GB gp3). Spot models use backend_fallback_warm_pool for their on-demand fallback ASG."
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
  description = "Single-model mode only (models = {}): BEHEMOTH_MODEL_NAME reported by the backend."
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
  description = "Single-model mode only (models = {}): Extra environment for neuron_server (e.g. BEHEMOTH_NEURON_LOAD_THREADS)."
  type        = map(string)
  default     = {}
}

variable "backend_extra_server_args" {
  description = "Single-model mode only (models = {}): Extra neuron_server flags (e.g. \"--prefix-cache-slots 64\")."
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

# ------------------------------------------------------------------ models (MULTI_MODEL.md)
variable "models" {
  description = <<-EOT
    Models to serve, keyed by a short name ([a-z0-9-], <= 20 chars). Each gets its own ECS service, task definition,
    launch template, ASG(s), capacity provider(s) and S3 read policy; the router wakes only the model a request names.
    Empty = one model built from the single-model variables (model_s3_prefix, model_ids, backend_extra_env, ...) under
    the key "ornith" with the pre-multi-model resource names. Unset optional fields fall back to the global backend_* /
    router variables. The entry keyed "ornith" with legacy_names = true keeps the resources (and names) that existed
    before the refactor (moved blocks in models.tf); any other key creates new resources named <name>-...-<key>.
  EOT
  type = map(object({
    model_s3_prefix    = string
    graphs_s3_prefix   = string
    prepared_s3_prefix = optional(string, "")
    model_ids          = list(string)     # OpenAI `model` values served (first = BEHEMOTH_MODEL_NAME by default)
    model_name         = optional(string) # BEHEMOTH_MODEL_NAME
    extra_server_args  = optional(string, "")
    extra_env          = optional(map(string), {}) # replaces backend_extra_env for this model
    image              = optional(string)          # full image reference; default the stack's backend image
    legacy_names       = optional(bool, false)

    instance_types              = optional(list(string))
    use_spot                    = optional(bool)
    spot_fallback               = optional(bool) # spot only: an on-demand fallback ASG + capacity provider
    spot_fallback_after_seconds = optional(number)
    spot_max_price              = optional(string)
    warm_pool                   = optional(bool) # on-demand ASG (primary, or the spot fallback): stopped warm instance
    cpu                         = optional(number)
    memory_reservation_mib      = optional(number)
    memory_hard_limit_mib       = optional(number)
    shm_size_mib                = optional(number)
    startup_grace_seconds       = optional(number)

    idle_timeout_seconds = optional(number)
    max_wait_seconds     = optional(number)
    cold_nonstream_mode  = optional(string)
    min_count            = optional(number, 0) # 1 = always running (never idle-stopped or evicted)
  }))
  default = {}

  validation {
    condition     = alltrue([for k, _ in var.models : can(regex("^[a-z0-9][a-z0-9-]{0,19}$", k))])
    error_message = "models keys must match ^[a-z0-9][a-z0-9-]{0,19}$ (they become parts of AWS resource names)."
  }
  validation {
    condition     = alltrue([for k, m in var.models : !m.legacy_names || k == "ornith"])
    error_message = "legacy_names is only valid on the key \"ornith\" (the moved blocks in models.tf point there)."
  }
  validation {
    condition     = length(flatten([for m in var.models : m.model_ids])) == length(distinct(flatten([for m in var.models : m.model_ids])))
    error_message = "model_ids must be unique across models."
  }
  validation {
    condition     = alltrue([for m in var.models : length(m.model_ids) > 0 && contains([0, 1], m.min_count)])
    error_message = "every model needs at least one model id; min_count is 0 or 1."
  }
  validation {
    condition     = alltrue([for m in var.models : m.cold_nonstream_mode == null ? true : contains(["keepalive", "wait", "reject"], m.cold_nonstream_mode)])
    error_message = "cold_nonstream_mode must be keepalive, wait or reject."
  }
}

variable "default_model" {
  description = "Model key used for requests without a `model` field (and listed first by /v1/models). Empty = \"ornith\" if present, else the first key."
  type        = string
  default     = ""
}

variable "max_active_models" {
  description = "Most models starting or running at once (each holds one trn instance: bounds spot/on-demand vCPU quota use and cost)."
  type        = number
  default     = 2
  validation {
    condition     = var.max_active_models >= 1
    error_message = "max_active_models must be >= 1."
  }
}

variable "cap_policy" {
  description = "Cold request for a model while max_active_models are active: \"evict\" stops the longest-idle active model (nothing in flight, idle >= cap_min_idle_seconds, min_count 0) and starts the new one, else 503; \"reject\" always answers 503 + Retry-After."
  type        = string
  default     = "evict"
  validation {
    condition     = contains(["evict", "reject"], var.cap_policy)
    error_message = "cap_policy must be evict or reject."
  }
}

variable "cap_min_idle_seconds" {
  description = "An active model must have been idle this long before cap_policy = evict may stop it."
  type        = number
  default     = 30
}

variable "backend_spot_fallback" {
  description = "With spot: also create an on-demand ASG + capacity provider per model; the router moves a model's service there when its spot ASG cannot launch (failed launch activity, or no instance after spot_fallback_after_seconds). false = spot only (a spot shortage leaves the model down until capacity returns)."
  type        = bool
  default     = true
}

variable "spot_fallback_after_seconds" {
  description = "Spot fallback: seconds into a startup without any spot instance (pending or in service) before the router falls back to on-demand. A failed spot launch falls back at once."
  type        = number
  default     = 240
}

variable "backend_fallback_warm_pool" {
  description = "Warm pool (one stopped, pre-initialised instance) on each on-demand fallback ASG: a fallback start skips launch + boot + image pull. Costs the stopped root volume (~$8/month for 100 GB gp3) per model."
  type        = bool
  default     = true
}

variable "router_lost_retries" {
  description = "Times the router re-wakes a model and re-sends a request whose backend vanished (spot interruption, crash) before any byte was returned. 0 = 503 at once."
  type        = number
  default     = 1
}

variable "router_stream_loss_error" {
  description = "Backend lost mid-stream: end the SSE stream with an error event + [DONE] (true) instead of a bare EOF."
  type        = bool
  default     = true
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
  description = "Single-model mode only (models = {}): Model ids returned by the router's GET /v1/models."
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
  description = "Non-streaming request on a cold backend: \"keepalive\" (200 at once, a space every few seconds so Cloudflare's 100 s idle limit never trips, then the JSON body; errors arrive as a 200 with an error body), \"wait\" (hold silently up to max_wait_seconds) or \"reject\" (503 + Retry-After immediately, warm up in the background)."
  type        = string
  default     = "keepalive"
  validation {
    condition     = contains(["keepalive", "wait", "reject"], var.cold_nonstream_mode)
    error_message = "cold_nonstream_mode must be keepalive, wait or reject."
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
  description = "Optional Secrets Manager / SSM ARN of one extra bearer key the router accepts besides the approved keys in the <name>-api-keys DynamoDB table."
  type        = string
  default     = ""
}

variable "role_permissions_boundary_arn" {
  description = "Permissions boundary set on every role the stack creates. The pipeline passes its workload boundary (TF_VAR_role_permissions_boundary_arn), which its scoped apply role requires; empty = none (local applies with admin credentials)."
  type        = string
  default     = ""
}

variable "router_allow_anonymous" {
  description = "Accept requests without any key (the api-keys table is ignored). Only for deployments whose ingress authenticates callers itself (Cloudflare Access, an authenticating gateway); a tunnel token authenticates the connector, not callers."
  type        = bool
  default     = false
}

variable "enable_alb" {
  description = "Also expose the router through an internet-facing ALB."
  type        = bool
  default     = false
}

variable "alb_certificate_arn" {
  description = "ACM certificate for the ALB's HTTPS listener; required with enable_alb (port 80 only redirects)."
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

variable "remote_regions" {
  description = "Regions outside var.region that get a no-ingress security group (default VPC) for hand-run Trainium hosts, e.g. Capacity Blocks; with any, the stack also creates the <name>-remote-neuron instance profile. Supported: ap-southeast-2, ap-south-1, sa-east-1 (H3 video; see h3_sa_east_1.tf)."
  type        = list(string)
  default     = ["ap-southeast-2", "ap-south-1", "sa-east-1"]
  validation {
    condition     = alltrue([for r in var.remote_regions : contains(["ap-southeast-2", "ap-south-1", "sa-east-1"], r)])
    error_message = "remote_regions supports ap-southeast-2, ap-south-1 and sa-east-1 (each needs a provider alias in remote_regions.tf)."
  }
}

variable "h3_break_glass_principal_arns" {
  description = "IAM principal ARNs (besides the <name>-h3-neuron role) that the H3 bucket policy lets get and put objects, e.g. a named admin role for break-glass access from sa-east-1. Empty = the instance role only (the account root user is denied too). Wildcards allowed (ArnNotLike)."
  type        = list(string)
  default     = []
}

variable "h3_bucket_name" {
  description = "S3 bucket in sa-east-1 for MiniMax H3 (video model) weights, prepared shards, NEFFs and outputs. Empty = <name>-h3-sa-east-1-<account id>. Created only when sa-east-1 is in remote_regions. H3's licence excludes the US, EU, UK and South Korea, so nothing H3 may be stored in or copied to var.region buckets."
  type        = string
  default     = ""
}
