# Per-model configuration (MULTI_MODEL.md). var.models, or one legacy model from the single-model variables, is
# normalised here into local.models: every field concrete, AWS names derived. ecs_cluster.tf, backend.tf, iam.tf and
# router.tf iterate over it.

locals {
  legacy_model = {
    ornith = {
      model_s3_prefix             = var.model_s3_prefix
      graphs_s3_prefix            = var.graphs_s3_prefix
      prepared_s3_prefix          = var.prepared_s3_prefix
      model_ids                   = var.model_ids
      model_name                  = var.backend_model_name
      extra_server_args           = var.backend_extra_server_args
      extra_env                   = var.backend_extra_env
      image                       = null
      legacy_names                = true
      instance_types              = null
      use_spot                    = null
      spot_fallback               = null
      spot_fallback_after_seconds = null
      spot_max_price              = null
      warm_pool                   = null
      cpu                         = null
      memory_reservation_mib      = null
      memory_hard_limit_mib       = null
      shm_size_mib                = null
      startup_grace_seconds       = null
      idle_timeout_seconds        = null
      max_wait_seconds            = null
      cold_nonstream_mode         = null
      min_count                   = 0
    }
  }

  models_raw = length(var.models) > 0 ? var.models : local.legacy_model

  models = { for k, m in local.models_raw : k => {
    legacy = m.legacy_names

    # AWS names: the legacy entry keeps exactly the names the single-model stack created
    # (the single-model stack's on-demand ASG / capacity provider were named "...-trn-spot" from before the warm pool)
    service_name    = m.legacy_names ? "${var.name}-backend" : "${var.name}-backend-${k}"
    od_cp_name      = m.legacy_names ? "${var.name}-trn-spot" : "${var.name}-trn-${k}-od"
    od_asg_prefix   = m.legacy_names ? "${var.name}-trn-spot-" : "${var.name}-trn-${k}-od-"
    spot_cp_name    = "${var.name}-trn-${k}-spot"
    spot_asg_prefix = "${var.name}-trn-${k}-spot-"
    lt_prefix       = m.legacy_names ? "${var.name}-trn-" : "${var.name}-trn-${k}-"
    instance_tag    = m.legacy_names ? "${var.name}-trn-spot" : "${var.name}-trn-${k}"
    policy_name     = m.legacy_names ? "s3-read-models" : "s3-read-models-${k}"
    log_prefix      = m.legacy_names ? "backend" : "backend-${k}"

    model_s3_prefix    = m.model_s3_prefix
    graphs_s3_prefix   = m.graphs_s3_prefix
    prepared_s3_prefix = m.prepared_s3_prefix
    # IAM needs some prefix even when prepared shards are off; a name nothing is ever written under
    prepared_iam_prefix = m.prepared_s3_prefix != "" ? m.prepared_s3_prefix : "prepared/unused"
    model_ids           = m.model_ids
    model_name          = m.model_name != null ? m.model_name : m.model_ids[0]
    extra_server_args   = m.extra_server_args
    extra_env           = m.extra_env
    image               = m.image != null ? m.image : local.backend_image

    instance_types = m.instance_types != null ? m.instance_types : var.backend_instance_types
    use_spot       = m.use_spot != null ? m.use_spot : var.backend_use_spot
    spot_fallback  = (m.use_spot != null ? m.use_spot : var.backend_use_spot) && (m.spot_fallback != null ? m.spot_fallback : var.backend_spot_fallback)
    fallback_after = m.spot_fallback_after_seconds != null ? m.spot_fallback_after_seconds : var.spot_fallback_after_seconds
    spot_max_price = m.spot_max_price != null ? m.spot_max_price : var.spot_max_price
    # warm pool of the on-demand ASG: backend_warm_pool when it is the primary, backend_fallback_warm_pool as fallback
    od_warm_pool    = m.warm_pool != null ? m.warm_pool : ((m.use_spot != null ? m.use_spot : var.backend_use_spot) ? var.backend_fallback_warm_pool : var.backend_warm_pool)
    cpu             = m.cpu != null ? m.cpu : var.backend_cpu
    mem_reservation = m.memory_reservation_mib != null ? m.memory_reservation_mib : var.backend_memory_reservation_mib
    mem_hard_limit  = m.memory_hard_limit_mib != null ? m.memory_hard_limit_mib : var.backend_memory_hard_limit_mib
    shm_size_mib    = m.shm_size_mib != null ? m.shm_size_mib : var.backend_shm_size_mib
    startup_grace   = m.startup_grace_seconds != null ? m.startup_grace_seconds : var.backend_startup_grace_seconds

    idle_timeout_seconds = m.idle_timeout_seconds != null ? m.idle_timeout_seconds : var.idle_timeout_seconds
    max_wait_seconds     = m.max_wait_seconds != null ? m.max_wait_seconds : var.max_wait_seconds
    cold_nonstream_mode  = m.cold_nonstream_mode != null ? m.cold_nonstream_mode : var.cold_nonstream_mode
    min_count            = m.min_count
  } }

  ondemand_models = { for k, m in local.models : k => m if !m.use_spot || m.spot_fallback }
  spot_models     = { for k, m in local.models : k => m if m.use_spot }

  default_model = var.default_model != "" ? var.default_model : (contains(keys(local.models), "ornith") ? "ornith" : keys(local.models)[0])
}

# ------------------------------------------------------------------ migration from the single-model stack
# The single-model resources become the "ornith" instances: a state move, no replacement (names are unchanged through
# legacy_names). The old ASG was on-demand (backend_use_spot = false, warm pool: production since the spot quota wait),
# so it becomes ornith's on-demand ASG unchanged; spot gets a new ASG. Keep these blocks until every state that predates
# the refactor has been applied once.
moved {
  from = aws_launch_template.backend
  to   = aws_launch_template.backend["ornith"]
}

moved {
  from = aws_autoscaling_group.backend
  to   = aws_autoscaling_group.ondemand["ornith"]
}

moved {
  from = aws_ecs_capacity_provider.trn_spot
  to   = aws_ecs_capacity_provider.ondemand["ornith"]
}

moved {
  from = aws_ecs_task_definition.backend
  to   = aws_ecs_task_definition.backend["ornith"]
}

moved {
  from = aws_ecs_service.backend
  to   = aws_ecs_service.backend["ornith"]
}

moved {
  from = aws_iam_role_policy.backend_task
  to   = aws_iam_role_policy.backend_task["ornith"]
}
