locals {
  ecr_registry = "${local.acct}.dkr.ecr.${var.region}.amazonaws.com"

  terraform_env = {
    TF_VERSION         = var.terraform_version
    TF_SHA256          = var.terraform_sha256
    TF_STATE_BUCKET    = local.state_bucket
    TF_STATE_KEY       = local.state_key
    TF_CONFIG_KEY      = local.config_key
    TF_STATE_REGION    = var.region
    STACK_DIR          = var.stack_dir
    TF_IN_AUTOMATION   = "1"
    TF_INPUT           = "0"
    CHECKPOINT_DISABLE = "1"
  }

  # project-level (static) environment; per-run values (commit, tags) are passed by the pipeline actions
  build_env = {
    images = {
      ECR_REGISTRY               = local.ecr_registry
      BACKEND_REPO               = var.backend_ecr_repository_name
      ROUTER_REPO                = local.router_repo
      STACK_DIR                  = var.stack_dir
      BACKEND_BASE_IMAGE         = var.backend_base_image
      NEURON_RUNTIME_VERSION     = var.neuron_runtime_version
      NEURON_COLLECTIVES_VERSION = var.neuron_collectives_version
    }
    plan  = local.terraform_env
    apply = local.terraform_env
  }

  build_secrets = {
    images = var.dockerhub_secret_arn != "" ? {
      DOCKERHUB_USERNAME = "${var.dockerhub_secret_arn}:username"
      DOCKERHUB_PASSWORD = "${var.dockerhub_secret_arn}:password"
    } : {}
    plan  = {}
    apply = {}
  }
}

resource "aws_cloudwatch_log_group" "codebuild" {
  for_each          = local.builds
  name              = local.log_group_names[each.key]
  retention_in_days = var.log_retention_days
}

resource "aws_codebuild_project" "this" {
  for_each      = local.builds
  name          = "${var.name}-pipeline-${each.key}"
  description   = each.value.description
  service_role  = aws_iam_role.codebuild[each.key].arn
  build_timeout = each.key == "images" ? var.image_build_timeout_minutes : 30

  artifacts {
    type = "CODEPIPELINE"
  }

  source {
    type      = "CODEPIPELINE"
    buildspec = "${var.stack_dir}/pipeline/buildspec-${each.key}.yml" # read from the commit being built
  }

  environment {
    type                        = "LINUX_CONTAINER"
    image                       = var.codebuild_image
    image_pull_credentials_type = "CODEBUILD"
    compute_type                = each.key == "images" ? var.image_build_compute_type : var.terraform_compute_type
    privileged_mode             = each.key == "images" # Docker daemon for docker build

    dynamic "environment_variable" {
      for_each = local.build_env[each.key]
      content {
        name  = environment_variable.key
        value = environment_variable.value
        type  = "PLAINTEXT"
      }
    }
    dynamic "environment_variable" {
      for_each = local.build_secrets[each.key]
      content {
        name  = environment_variable.key
        value = environment_variable.value
        type  = "SECRETS_MANAGER"
      }
    }
  }

  # best-effort Docker layer cache on the build host (only hit when builds run close together)
  cache {
    type  = each.key == "images" ? "LOCAL" : "NO_CACHE"
    modes = each.key == "images" ? ["LOCAL_DOCKER_LAYER_CACHE"] : []
  }

  logs_config {
    cloudwatch_logs {
      status     = "ENABLED"
      group_name = aws_cloudwatch_log_group.codebuild[each.key].name
    }
  }

  depends_on = [aws_iam_role_policy.codebuild_base]
}
