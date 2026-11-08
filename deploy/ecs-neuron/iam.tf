locals {
  account_id = data.aws_caller_identity.current.account_id
  partition  = data.aws_partition.current.partition

  # secrets referenced by the router task: Secrets Manager or SSM parameter ARNs
  router_secret_arns = compact([
    var.enable_cloudflared ? var.cloudflare_tunnel_token_secret_arn : "",
    var.router_api_key_secret_arn,
  ])
  router_sm_secret_arns  = [for a in local.router_secret_arns : a if can(regex("^arn:[^:]+:secretsmanager:", a))]
  router_ssm_secret_arns = [for a in local.router_secret_arns : a if can(regex("^arn:[^:]+:ssm:", a))]
}

data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "ecs_tasks_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.account_id]
    }
  }
}

# ------------------------------------------------------------------ trn host (ECS container instance)
resource "aws_iam_role" "instance" {
  name               = "${var.name}-ecs-instance"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
}

resource "aws_iam_role_policy_attachment" "instance_ecs" {
  role       = aws_iam_role.instance.name
  policy_arn = "arn:${local.partition}:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_role_policy_attachment" "instance_ssm" {
  role       = aws_iam_role.instance.name
  policy_arn = "arn:${local.partition}:iam::aws:policy/AmazonSSMManagedInstanceCore" # Session Manager shell for debugging
}

resource "aws_iam_instance_profile" "instance" {
  name = "${var.name}-ecs-instance"
  role = aws_iam_role.instance.name
}

# ------------------------------------------------------------------ task execution role (image pull, logs, secrets)
resource "aws_iam_role" "execution" {
  name               = "${var.name}-ecs-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:${local.partition}:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

data "aws_iam_policy_document" "execution_secrets" {
  count = length(local.router_secret_arns) > 0 ? 1 : 0

  dynamic "statement" {
    for_each = length(local.router_sm_secret_arns) > 0 ? [1] : []
    content {
      actions   = ["secretsmanager:GetSecretValue"]
      resources = local.router_sm_secret_arns
    }
  }
  dynamic "statement" {
    for_each = length(local.router_ssm_secret_arns) > 0 ? [1] : []
    content {
      actions   = ["ssm:GetParameters"]
      resources = local.router_ssm_secret_arns
    }
  }
  dynamic "statement" {
    for_each = var.secrets_kms_key_arn != "" ? [1] : []
    content {
      actions   = ["kms:Decrypt"]
      resources = [var.secrets_kms_key_arn]
    }
  }
}

resource "aws_iam_role_policy" "execution_secrets" {
  count  = length(local.router_secret_arns) > 0 ? 1 : 0
  name   = "router-secrets"
  role   = aws_iam_role.execution.id
  policy = data.aws_iam_policy_document.execution_secrets[0].json
}

# ------------------------------------------------------------------ backend task role: read-only on the model bucket
resource "aws_iam_role" "backend_task" {
  name               = "${var.name}-backend-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

data "aws_iam_policy_document" "backend_task" {
  statement {
    sid       = "ListPrefixes"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.models.arn]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values = [
        var.model_s3_prefix, "${var.model_s3_prefix}/*",
        var.graphs_s3_prefix, "${var.graphs_s3_prefix}/*",
        local.prepared_prefix, "${local.prepared_prefix}/*",
      ]
    }
  }
  statement {
    sid     = "ReadObjects"
    actions = ["s3:GetObject"]
    resources = [
      "${aws_s3_bucket.models.arn}/${var.model_s3_prefix}/*",
      "${aws_s3_bucket.models.arn}/${var.graphs_s3_prefix}/*",
      "${aws_s3_bucket.models.arn}/${local.prepared_prefix}/*",
    ]
  }
  dynamic "statement" {
    for_each = var.s3_kms_key_arn != "" ? [1] : []
    content {
      actions   = ["kms:Decrypt"]
      resources = [var.s3_kms_key_arn]
    }
  }
}

resource "aws_iam_role_policy" "backend_task" {
  name   = "s3-read-models"
  role   = aws_iam_role.backend_task.id
  policy = data.aws_iam_policy_document.backend_task.json
}

# ------------------------------------------------------------------ router task role: control the backend service only
resource "aws_iam_role" "router_task" {
  name               = "${var.name}-router-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

data "aws_iam_policy_document" "router_task" {
  statement {
    sid       = "BackendService"
    actions   = ["ecs:UpdateService", "ecs:DescribeServices"]
    resources = [aws_ecs_service.backend.id] # service ARN
  }
  statement {
    sid       = "ListBackendTasks"
    actions   = ["ecs:ListTasks"]
    resources = ["*"] # ListTasks is authorized on the container instance; scope by cluster
    condition {
      test     = "ArnEquals"
      variable = "ecs:cluster"
      values   = [aws_ecs_cluster.this.arn]
    }
  }
  # Scale the backend ASG directly: the capacity provider's own scale-out from 0 takes ~2-3 min to react.
  statement {
    sid       = "ScaleBackendAsg"
    actions   = ["autoscaling:SetDesiredCapacity"]
    resources = [aws_autoscaling_group.backend.arn]
  }
  statement {
    sid       = "DescribeAsgs"
    actions   = ["autoscaling:DescribeAutoScalingGroups"]
    resources = ["*"] # Describe* takes no resource scoping
  }
  statement {
    sid       = "DescribeBackendTasks"
    actions   = ["ecs:DescribeTasks"]
    resources = ["arn:${local.partition}:ecs:${var.region}:${local.account_id}:task/${aws_ecs_cluster.this.name}/*"]
  }
}

resource "aws_iam_role_policy" "router_task" {
  name   = "control-backend"
  role   = aws_iam_role.router_task.id
  policy = data.aws_iam_policy_document.router_task.json
}
