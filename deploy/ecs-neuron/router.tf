resource "aws_cloudwatch_log_group" "router" {
  name              = "/ecs/${var.name}/router"
  retention_in_days = var.log_retention_days
}

locals {
  router_image = "${aws_ecr_repository.router.repository_url}:${var.router_image_tag}${var.router_image_digest != "" ? "@${var.router_image_digest}" : ""}"

  router_env = {
    AWS_REGION          = var.region
    ECS_CLUSTER         = aws_ecs_cluster.this.name
    ECS_SERVICE         = aws_ecs_service.backend.name
    BACKEND_ASG         = aws_autoscaling_group.backend.name
    BACKEND_PORT        = tostring(var.backend_port)
    LISTEN_PORT         = tostring(var.router_port)
    MODEL_IDS           = join(",", var.model_ids)
    MODEL_OWNED_BY      = var.model_owned_by
    PROXY_PATHS         = join(",", var.proxy_paths)
    IDLE_TIMEOUT_S      = tostring(var.idle_timeout_seconds)
    MAX_WAIT_S          = tostring(var.max_wait_seconds)
    COLD_NONSTREAM_MODE = var.cold_nonstream_mode
    ALLOW_ANONYMOUS     = tostring(var.router_allow_anonymous)
  }

  log_options = {
    awslogs-group  = aws_cloudwatch_log_group.router.name
    awslogs-region = var.region
  }

  router_container = {
    name         = "router"
    image        = local.router_image
    essential    = true
    portMappings = [{ containerPort = var.router_port, protocol = "tcp" }]
    environment  = [for k, v in local.router_env : { name = k, value = v }]
    secrets      = var.router_api_key_secret_arn != "" ? [{ name = "ROUTER_API_KEY", valueFrom = var.router_api_key_secret_arn }] : []
    healthCheck = {
      command     = ["CMD", "python", "-c", "import urllib.request;urllib.request.urlopen('http://127.0.0.1:${var.router_port}/health',timeout=4)"]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 10
    }
    logConfiguration = { logDriver = "awslogs", options = merge(local.log_options, { awslogs-stream-prefix = "router" }) }
  }

  # cloudflared connects out to Cloudflare; the tunnel's public hostname must point at http://localhost:<router_port>
  # (containers of a Fargate task share localhost).
  cloudflared_container = {
    name             = "cloudflared"
    image            = var.cloudflared_image
    essential        = true
    command          = ["tunnel", "--no-autoupdate", "--metrics", "127.0.0.1:2000", "run"]
    secrets          = [{ name = "TUNNEL_TOKEN", valueFrom = var.cloudflare_tunnel_token_secret_arn }]
    dependsOn        = [{ containerName = "router", condition = "HEALTHY" }]
    logConfiguration = { logDriver = "awslogs", options = merge(local.log_options, { awslogs-stream-prefix = "cloudflared" }) }
  }
}

resource "aws_ecs_task_definition" "router" {
  family                   = "${var.name}-router"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.router_cpu
  memory                   = var.router_memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.router_task.arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  container_definitions = jsonencode(concat(
    [local.router_container],
    var.enable_cloudflared ? [local.cloudflared_container] : []
  ))

  lifecycle {
    precondition {
      condition     = !var.enable_cloudflared || var.cloudflare_tunnel_token_secret_arn != ""
      error_message = "enable_cloudflared requires cloudflare_tunnel_token_secret_arn."
    }
  }
}

resource "aws_ecs_service" "router" {
  name             = "${var.name}-router"
  cluster          = aws_ecs_cluster.this.id
  task_definition  = aws_ecs_task_definition.router.arn
  desired_count    = 1
  launch_type      = "FARGATE"
  platform_version = "LATEST"

  network_configuration {
    subnets          = local.router_subnet_ids
    security_groups  = [aws_security_group.router.id]
    assign_public_ip = var.router_assign_public_ip
  }

  lifecycle {
    precondition {
      condition     = var.router_api_key_secret_arn != "" || var.router_allow_anonymous
      error_message = "Set router_api_key_secret_arn, or router_allow_anonymous = true when an outer gate (Cloudflare Access, a gateway) authenticates callers: the router refuses to start without either."
    }
  }

  # single router: its in-memory state (in-flight count, idle timer) must not be split across replicas
  deployment_minimum_healthy_percent = 0
  deployment_maximum_percent         = 100

  dynamic "load_balancer" {
    for_each = var.enable_alb ? [1] : []
    content {
      target_group_arn = aws_lb_target_group.router[0].arn
      container_name   = "router"
      container_port   = var.router_port
    }
  }

  depends_on = [aws_iam_role_policy.router_task, aws_lb_listener.router_http, aws_lb_listener.router_https]
}

# ------------------------------------------------------------------ optional ALB
resource "aws_security_group" "alb" {
  count       = var.enable_alb ? 1 : 0
  name        = "${var.name}-alb"
  description = "Behemoth router ALB"
  vpc_id      = var.vpc_id

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = var.alb_ingress_cidrs
  }

  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = var.alb_ingress_cidrs
  }

  egress {
    description = "to router"
    from_port   = var.router_port
    to_port     = var.router_port
    protocol    = "tcp"
    cidr_blocks = [data.aws_vpc.this.cidr_block]
  }
}

resource "aws_lb" "router" {
  count              = var.enable_alb ? 1 : 0
  name               = "${var.name}-router"
  load_balancer_type = "application"
  internal           = false
  subnets            = var.alb_subnet_ids
  security_groups    = [aws_security_group.alb[0].id]
  idle_timeout       = var.alb_idle_timeout

  lifecycle {
    precondition {
      condition     = length(var.alb_subnet_ids) >= 2
      error_message = "enable_alb requires alb_subnet_ids in at least two AZs."
    }
    precondition {
      condition     = var.alb_certificate_arn != ""
      error_message = "enable_alb requires alb_certificate_arn: inference (prompts, bearer keys) is served over HTTPS only."
    }
  }
}

resource "aws_lb_target_group" "router" {
  count       = var.enable_alb ? 1 : 0
  name        = "${var.name}-router"
  port        = var.router_port
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = var.vpc_id

  health_check {
    path    = "/health" # router-local, never wakes the backend
    matcher = "200"
  }
  deregistration_delay = 30
}

resource "aws_lb_listener" "router_http" {
  count             = var.enable_alb ? 1 : 0
  load_balancer_arn = aws_lb.router[0].arn
  port              = 80
  protocol          = "HTTP"

  # never forwards: plain HTTP only redirects (a key already sent over it should be treated as exposed)
  default_action {
    type = "redirect"
    redirect {
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }
}

resource "aws_lb_listener" "router_https" {
  count             = var.enable_alb ? 1 : 0
  load_balancer_arn = aws_lb.router[0].arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.alb_certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.router[0].arn
  }
}
