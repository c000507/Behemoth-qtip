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

resource "aws_launch_template" "backend" {
  name_prefix   = "${var.name}-trn-"
  image_id      = local.backend_ami_id
  instance_type = var.backend_instance_types[0]
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
    tags          = { Name = "${var.name}-trn-spot" }
  }

  update_default_version = true
}

resource "aws_autoscaling_group" "backend" {
  name_prefix           = "${var.name}-trn-spot-"
  min_size              = 0
  max_size              = 1
  desired_capacity      = 0
  vpc_zone_identifier   = var.backend_subnet_ids
  protect_from_scale_in = true # required by managed termination protection
  capacity_rebalance    = false
  health_check_type     = "EC2"

  mixed_instances_policy {
    instances_distribution {
      on_demand_base_capacity                  = 0
      on_demand_percentage_above_base_capacity = 0 # spot only
      spot_allocation_strategy                 = "price-capacity-optimized"
      spot_max_price                           = var.spot_max_price != "" ? var.spot_max_price : null
    }
    launch_template {
      launch_template_specification {
        launch_template_id = aws_launch_template.backend.id
        version            = "$Latest"
      }
      dynamic "override" {
        for_each = var.backend_instance_types
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
    ignore_changes = [desired_capacity] # owned by ECS managed scaling
  }
}

resource "aws_ecs_capacity_provider" "trn_spot" {
  name = "${var.name}-trn-spot"

  auto_scaling_group_provider {
    auto_scaling_group_arn         = aws_autoscaling_group.backend.arn
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

resource "aws_ecs_cluster_capacity_providers" "this" {
  cluster_name       = aws_ecs_cluster.this.name
  capacity_providers = [aws_ecs_capacity_provider.trn_spot.name, "FARGATE"]
}
