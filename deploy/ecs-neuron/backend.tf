resource "aws_cloudwatch_log_group" "backend" {
  name              = "/ecs/${var.name}/backend"
  retention_in_days = var.log_retention_days
}

locals {
  backend_image = "${data.aws_ecr_repository.backend.repository_url}:${var.backend_image_tag}${var.backend_image_digest != "" ? "@${var.backend_image_digest}" : ""}"

  # IAM needs some prefix even when prepared shards are off; a name nothing is ever written under
  prepared_prefix = var.prepared_s3_prefix != "" ? var.prepared_s3_prefix : "prepared/unused"
  backend_env = merge({
    AWS_REGION              = var.region
    S3_BUCKET               = aws_s3_bucket.models.bucket
    MODEL_S3_PREFIX         = var.model_s3_prefix
    GRAPHS_S3_PREFIX        = var.graphs_s3_prefix
    PREPARED_S3_PREFIX      = var.prepared_s3_prefix
    DATA_DIR                = "/data"
    PORT                    = tostring(var.backend_port)
    S5CMD_WORKERS           = tostring(var.s5cmd_workers)
    STARTUP_GRACE_S         = tostring(var.backend_startup_grace_seconds)
    NEURON_RT_VISIBLE_CORES = "0-1"
    NEURON_RT_ROOT_COMM_ID  = "127.0.0.1:62128"
    BEHEMOTH_MODEL_NAME     = var.backend_model_name
    EXTRA_SERVER_ARGS       = var.backend_extra_server_args
  }, var.backend_extra_env)
}

resource "aws_ecs_task_definition" "backend" {
  family                   = "${var.name}-backend"
  requires_compatibilities = ["EC2"]
  network_mode             = "awsvpc" # task gets its own ENI; DescribeTasks reports its private IP to the router
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.backend_task.arn

  volume {
    name                = "data"
    host_path           = "${local.backend_data_mount}/behemoth" # NVMe instance store (see templates/backend_user_data.sh.tftpl)
    configure_at_launch = false
  }

  # null-valued keys (e.g. no hard memory limit) are dropped so the provider sees no perpetual diff
  container_definitions = jsonencode([{ for k, v in {
    name              = "neuron-server"
    image             = local.backend_image
    essential         = true
    cpu               = var.backend_cpu
    memoryReservation = var.backend_memory_reservation_mib
    memory            = var.backend_memory_hard_limit_mib > 0 ? var.backend_memory_hard_limit_mib : null
    # hostPort / volumesFrom / capabilities.drop / configure_at_launch spelled out as ECS stores them: otherwise every plan
    # replaces the task definition, and each apply restarts a running backend
    portMappings = [{ containerPort = var.backend_port, hostPort = var.backend_port, protocol = "tcp" }]
    volumesFrom  = []
    environment  = [for k, v in local.backend_env : { name = k, value = v }]
    mountPoints  = [{ sourceVolume = "data", containerPath = "/data", readOnly = false }]
    linuxParameters = {
      devices            = [{ hostPath = "/dev/neuron0", containerPath = "/dev/neuron0", permissions = ["read", "write"] }]
      sharedMemorySize   = var.backend_shm_size_mib
      initProcessEnabled = true
      capabilities       = { add = ["IPC_LOCK"], drop = [] } # pinned host memory for Neuron DMA (with memlock ulimit)
    }
    ulimits = [{ name = "memlock", softLimit = -1, hardLimit = -1 }, { name = "nofile", softLimit = 65536, hardLimit = 65536 }]
    # IPv4 only inside the task: the subnets resolve through DNS64, so S3 (IPv4-only) comes back as a 64:ff9b:: address
    # that Go clients (s5cmd) try first; that path is NAT64 on the NAT instance (~115 MB/s, and past the free S3 gateway
    # endpoint). Without IPv6 the pull takes the gateway endpoint: 5.3 GB in 5 s instead of 46 s.
    systemControls = var.backend_disable_ipv6 ? [{ namespace = "net.ipv6.conf.all.disable_ipv6", value = "1" }] : null
    healthCheck = {
      command     = ["CMD", "/usr/local/bin/healthcheck.sh"]
      interval    = 30
      timeout     = 10
      retries     = 3
      startPeriod = 300 # ECS maximum; healthcheck.sh extends it to STARTUP_GRACE_S while pulling/loading
    }
    stopTimeout = 60
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.backend.name
        awslogs-region        = var.region
        awslogs-stream-prefix = "backend"
      }
    }
  } : k => v if v != null }])
}

resource "aws_ecs_service" "backend" {
  name            = "${var.name}-backend"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.backend.arn
  desired_count   = 0 # the router owns desiredCount (0 <-> 1)

  capacity_provider_strategy {
    capacity_provider = aws_ecs_capacity_provider.trn_spot.name
    weight            = 1
    base              = 0
  }

  network_configuration {
    subnets          = var.backend_subnet_ids
    security_groups  = [aws_security_group.backend_task.id]
    assign_public_ip = false # not supported for awsvpc on EC2; S3 goes through the gateway endpoint
  }

  # one Neuron device per instance and max 1 instance: never run old + new side by side
  deployment_minimum_healthy_percent = 0
  deployment_maximum_percent         = 100

  deployment_circuit_breaker {
    enable   = false # a spot capacity shortage must not roll the service back
    rollback = false
  }

  enable_ecs_managed_tags = true
  propagate_tags          = "SERVICE"
  wait_for_steady_state   = false

  lifecycle {
    ignore_changes = [desired_count]
  }

  depends_on = [aws_ecs_cluster_capacity_providers.this, aws_iam_role_policy.backend_task]
}
