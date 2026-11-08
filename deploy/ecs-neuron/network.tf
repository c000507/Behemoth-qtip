locals {
  router_subnet_ids = length(var.router_subnet_ids) > 0 ? var.router_subnet_ids : var.backend_subnet_ids
}

data "aws_vpc" "this" {
  id = var.vpc_id
}

# Router task ENI: outbound only (cloudflared, ECS API, ECR); inbound from the ALB when enabled.
resource "aws_security_group" "router" {
  name        = "${var.name}-router"
  description = "Behemoth router (Fargate)"
  vpc_id      = var.vpc_id

  egress {
    description = "all outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    # the subnets assign IPv6 addresses and resolve through DNS64: IPv4-only AWS endpoints come back as 64:ff9b:: (NAT64)
    # addresses tried first, so without IPv6 egress every API call hangs until the client's connect timeout
    ipv6_cidr_blocks = ["::/0"]
  }
}

resource "aws_vpc_security_group_ingress_rule" "router_from_alb" {
  count                        = var.enable_alb ? 1 : 0
  security_group_id            = aws_security_group.router.id
  referenced_security_group_id = aws_security_group.alb[0].id
  from_port                    = var.router_port
  to_port                      = var.router_port
  ip_protocol                  = "tcp"
  description                  = "ALB to router"
}

# Backend task ENI (awsvpc): inbound only from the router on the server port.
resource "aws_security_group" "backend_task" {
  name        = "${var.name}-backend-task"
  description = "Behemoth neuron_server task ENI"
  vpc_id      = var.vpc_id

  egress {
    description = "all outbound (S3 via the gateway endpoint)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    # the subnets assign IPv6 addresses and resolve through DNS64: IPv4-only AWS endpoints come back as 64:ff9b:: (NAT64)
    # addresses tried first, so without IPv6 egress every API call hangs until the client's connect timeout
    ipv6_cidr_blocks = ["::/0"]
  }
}

resource "aws_vpc_security_group_ingress_rule" "backend_from_router" {
  security_group_id            = aws_security_group.backend_task.id
  referenced_security_group_id = aws_security_group.router.id
  from_port                    = var.backend_port
  to_port                      = var.backend_port
  ip_protocol                  = "tcp"
  description                  = "router to neuron_server"
}

resource "aws_vpc_security_group_ingress_rule" "backend_extra" {
  for_each          = toset(var.extra_backend_ingress_cidrs)
  security_group_id = aws_security_group.backend_task.id
  cidr_ipv4         = each.value
  from_port         = var.backend_port
  to_port           = var.backend_port
  ip_protocol       = "tcp"
  description       = "extra access to neuron_server"
}

# S3 gateway endpoint: free, and the only S3 path for an awsvpc task on EC2 when the VPC has no NAT gateway.
resource "aws_vpc_endpoint" "s3" {
  count             = var.create_s3_gateway_endpoint ? 1 : 0
  vpc_id            = var.vpc_id
  service_name      = "com.amazonaws.${var.region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = var.s3_endpoint_route_table_ids
  tags              = { Name = "${var.name}-s3" }

  lifecycle {
    precondition {
      condition     = length(var.s3_endpoint_route_table_ids) > 0
      error_message = "Set s3_endpoint_route_table_ids (route tables of backend_subnet_ids) or create_s3_gateway_endpoint = false."
    }
  }
}

# A NAT instance (instead of a NAT gateway) only forwards traffic its own security group admits: let the router and
# backend task ENIs reach it (ECR, Secrets Manager, CloudWatch Logs, the Cloudflare edge go out through it).
resource "aws_vpc_security_group_ingress_rule" "nat_from_router" {
  count                        = var.nat_security_group_id != "" ? 1 : 0
  security_group_id            = var.nat_security_group_id
  referenced_security_group_id = aws_security_group.router.id
  ip_protocol                  = "-1"
  description                  = "${var.name} router egress via the NAT instance"
}

resource "aws_vpc_security_group_ingress_rule" "nat_from_backend_task" {
  count                        = var.nat_security_group_id != "" ? 1 : 0
  security_group_id            = var.nat_security_group_id
  referenced_security_group_id = aws_security_group.backend_task.id
  ip_protocol                  = "-1"
  description                  = "${var.name} backend task egress via the NAT instance"
}
