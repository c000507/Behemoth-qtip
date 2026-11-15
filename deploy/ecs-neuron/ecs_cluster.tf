locals {
  backend_data_mount = "/mnt/instance-store"
}

resource "aws_ecs_cluster" "this" {
  name = var.name

  setting {
    name  = "containerInsights"
    value = var.container_insights ? "enabled" : "disabled"
  }
}

# ECS-optimized AL2023 Neuron AMI (ships aws-neuronx-dkms + the ECS agent).
data "aws_ssm_parameter" "neuron_ami" {
  count = var.backend_ami_id == "" ? 1 : 0
  name  = var.backend_ami_ssm_parameter
}

locals {
  backend_ami_id = var.backend_ami_id != "" ? var.backend_ami_id : nonsensitive(data.aws_ssm_parameter.neuron_ami[0].value)
}

# ------------------------------------------------------------------ per model (local.models, models.tf)
# A launch template per model and, by purchase type:
#   ondemand[k]: on-demand ASG launched from the template (optional warm pool) + capacity provider, when the model is
#                on-demand, or spot with backend_spot_fallback (the router's fallback when spot cannot launch);
#   spot[k]:     spot ASG through a mixed instances policy (no warm pool: not supported with spot) + capacity provider.
# Switching a model between spot and on-demand creates/destroys the spot ASG; an existing ASG is never converted.
resource "aws_launch_template" "backend" {
  for_each = local.models

  name_prefix   = each.value.lt_prefix
  image_id      = local.backend_ami_id
  instance_type = each.value.instance_types[0]
  user_data = base64encode(templatefile("${path.module}/templates/backend_user_data.sh.tftpl", {
    cluster_name = aws_ecs_cluster.this.name
    data_mount   = local.backend_data_mount
  }))

  iam_instance_profile {
    arn = aws_iam_instance_profile.instance.arn
  }

  network_interfaces {
    device_index                = 0
    associate_public_ip_address = var.backend_associate_public_ip
    security_groups             = var.backend_instance_security_group_ids
    delete_on_termination       = true
  }

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size           = var.backend_root_volume_gb
      volume_type           = "gp3"
      encrypted             = true
      delete_on_termination = true
    }
  }

  metadata_options {
    http_tokens                 = "required"
    http_put_response_hop_limit = 1 # containers use task-role credentials, not the instance profile
  }

  tag_specifications {
    resource_type = "instance"
    tags          = { Name = each.value.instance_tag }
  }

  update_default_version = true
}

resource "aws_autoscaling_group" "ondemand" {
  for_each = local.ondemand_models

  name_prefix           = each.value.od_asg_prefix
  min_size              = 0
  max_size              = 1
  desired_capacity      = 0
  vpc_zone_identifier   = var.backend_subnet_ids
  protect_from_scale_in = true # required by managed termination protection
  capacity_rebalance    = false
  health_check_type     = "EC2"

  launch_template {
    id      = aws_launch_template.backend[each.key].id
    version = "$Latest"
  }

  # Warm pool (on-demand only): one stopped, already-initialised instance (cluster join done, image cached on the root
  # volume once used) that a scale-out starts instead of launching and booting a new one; scale-in returns it to the pool.
  dynamic "warm_pool" {
    for_each = each.value.od_warm_pool ? [1] : []
    content {
      pool_state                  = "Stopped"
      min_size                    = 1
      max_group_prepared_capacity = 1
      instance_reuse_policy {
        reuse_on_scale_in = true
      }
    }
  }

  tag {
    key                 = "AmazonECSManaged"
    value               = "true"
    propagate_at_launch = true
  }

  lifecycle {
    ignore_changes = [desired_capacity] # owned by ECS managed scaling and the router
  }
}

resource "aws_ecs_capacity_provider" "ondemand" {
  for_each = local.ondemand_models

  name = each.value.od_cp_name

  auto_scaling_group_provider {
    auto_scaling_group_arn         = aws_autoscaling_group.ondemand[each.key].arn
    managed_termination_protection = "ENABLED"
    managed_draining               = "ENABLED"

    managed_scaling {
      status                    = "ENABLED"
      target_capacity           = 100
      minimum_scaling_step_size = 1
      maximum_scaling_step_size = 1
      instance_warmup_period    = 180
    }
  }
}

resource "aws_autoscaling_group" "spot" {
  for_each = local.spot_models

  name_prefix           = each.value.spot_asg_prefix
  min_size              = 0
  max_size              = 1
  desired_capacity      = 0
  vpc_zone_identifier   = var.backend_subnet_ids
  protect_from_scale_in = true
  # no capacity rebalancing: with max_size 1 a rebalance would launch a second instance next to the one running the
  # (scale-in protected) task; an interruption is handled by ECS spot draining + the router instead
  capacity_rebalance = false
  health_check_type  = "EC2"

  mixed_instances_policy {
    instances_distribution {
      on_demand_base_capacity                  = 0
      on_demand_percentage_above_base_capacity = 0 # spot only; on-demand capacity is the ondemand ASG
      spot_allocation_strategy                 = "price-capacity-optimized"
      spot_max_price                           = each.value.spot_max_price != "" ? each.value.spot_max_price : null
    }
    launch_template {
      launch_template_specification {
        launch_template_id = aws_launch_template.backend[each.key].id
        version            = "$Latest"
      }
      dynamic "override" {
        for_each = each.value.instance_types
        content {
          instance_type = override.value
        }
      }
    }
  }

  tag {
    key                 = "AmazonECSManaged"
    value               = "true"
    propagate_at_launch = true
  }

  lifecycle {
    ignore_changes = [desired_capacity]
  }
}

resource "aws_ecs_capacity_provider" "spot" {
  for_each = local.spot_models

  name = each.value.spot_cp_name

  auto_scaling_group_provider {
    auto_scaling_group_arn         = aws_autoscaling_group.spot[each.key].arn
    managed_termination_protection = "ENABLED"
    managed_draining               = "ENABLED"

    managed_scaling {
      status                    = "ENABLED"
      target_capacity           = 100
      minimum_scaling_step_size = 1
      maximum_scaling_step_size = 1
      instance_warmup_period    = 180
    }
  }
}

locals {
  # per model: the capacity provider / ASG its service normally runs on, and the on-demand fallback of a spot model
  primary_cp = merge(
    { for k, cp in aws_ecs_capacity_provider.ondemand : k => cp.name if !local.models[k].use_spot },
    { for k, cp in aws_ecs_capacity_provider.spot : k => cp.name },
  )
  primary_asg = merge(
    { for k, g in aws_autoscaling_group.ondemand : k => g.name if !local.models[k].use_spot },
    { for k, g in aws_autoscaling_group.spot : k => g.name },
  )
  fallback_cp      = { for k, cp in aws_ecs_capacity_provider.ondemand : k => cp.name if local.models[k].spot_fallback }
  fallback_asg     = { for k, g in aws_autoscaling_group.ondemand : k => g.name if local.models[k].spot_fallback }
  backend_asg_arns = concat([for g in aws_autoscaling_group.ondemand : g.arn], [for g in aws_autoscaling_group.spot : g.arn])
}

resource "aws_ecs_cluster_capacity_providers" "this" {
  cluster_name = aws_ecs_cluster.this.name
  capacity_providers = concat(
    [for cp in aws_ecs_capacity_provider.ondemand : cp.name],
    [for cp in aws_ecs_capacity_provider.spot : cp.name],
    ["FARGATE"],
  )
}
