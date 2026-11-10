locals {
  partition = data.aws_partition.current.partition
  acct      = var.account_id
  p         = var.name

  router_repo   = var.router_ecr_repository_name != "" ? var.router_ecr_repository_name : "${var.name}-router"
  models_bucket = var.models_bucket_name != "" ? var.models_bucket_name : "${var.name}-neuron-models-${var.account_id}"

  ecr_repo_arns = [
    "arn:${local.partition}:ecr:${var.region}:${local.acct}:repository/${var.backend_ecr_repository_name}",
    "arn:${local.partition}:ecr:${var.region}:${local.acct}:repository/${local.router_repo}",
  ]

  # the stack's own roles (deploy/ecs-neuron/iam.tf): the only roles the apply build may pass, all under the boundary
  stack_role_arns = [for r in ["ecs-instance", "ecs-execution", "backend-task", "router-task", "remote-neuron"] :
  "arn:${local.partition}:iam::${local.acct}:role/${local.p}-${r}"]

  state_bucket_arn    = aws_s3_bucket.this["state"].arn
  artifact_bucket_arn = aws_s3_bucket.this["artifacts"].arn

  # one CodeBuild project (and role) per pipeline step
  builds = {
    images = { description = "Build and push the backend / router images (skips tags already in ECR)" }
    plan   = { description = "terraform plan of ${var.stack_dir} with the built image tags" }
    apply  = { description = "terraform apply of the approved saved plan" }
  }

  log_group_names = { for k, _ in local.builds : k => "/aws/codebuild/${var.name}-pipeline-${k}" }
}

data "aws_iam_policy_document" "codebuild_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["codebuild.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.acct]
    }
  }
}

data "aws_iam_policy_document" "codepipeline_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["codepipeline.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.acct]
    }
  }
}

# ------------------------------------------------------------------ baseline shared by the three CodeBuild roles
data "aws_iam_policy_document" "codebuild_base" {
  for_each = local.builds

  statement {
    sid     = "Logs"
    actions = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = [
      "arn:${local.partition}:logs:${var.region}:${local.acct}:log-group:${local.log_group_names[each.key]}",
      "arn:${local.partition}:logs:${var.region}:${local.acct}:log-group:${local.log_group_names[each.key]}:*",
    ]
  }
  statement {
    sid       = "PipelineArtifacts"
    actions   = ["s3:GetObject", "s3:GetObjectVersion", "s3:PutObject"]
    resources = ["${local.artifact_bucket_arn}/*"]
  }
  statement {
    sid       = "PipelineArtifactsBucket"
    actions   = ["s3:GetBucketLocation", "s3:GetBucketVersioning", "s3:ListBucket"]
    resources = [local.artifact_bucket_arn]
  }
  # OutputArtifactFormat = CODEBUILD_CLONE_REF: every build git-clones the repo through the connection
  statement {
    sid = "GitClone"
    actions = [
      "codestar-connections:UseConnection", "codestar-connections:GetConnection", "codestar-connections:GetConnectionToken",
      "codeconnections:UseConnection", "codeconnections:GetConnection", "codeconnections:GetConnectionToken",
    ]
    resources = [aws_codestarconnections_connection.github.arn]
  }
}

resource "aws_iam_role" "codebuild" {
  for_each           = local.builds
  name               = "${var.name}-pipeline-${each.key}"
  description        = "CodeBuild: ${each.value.description}"
  assume_role_policy = data.aws_iam_policy_document.codebuild_assume.json
}

resource "aws_iam_role_policy" "codebuild_base" {
  for_each = local.builds
  name     = "codebuild-base"
  role     = aws_iam_role.codebuild[each.key].id
  policy   = data.aws_iam_policy_document.codebuild_base[each.key].json
}

# ------------------------------------------------------------------ images: push to the two repos only
data "aws_iam_policy_document" "images" {
  statement {
    sid       = "EcrLogin"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  statement {
    sid = "EcrPushPull"
    actions = [
      "ecr:BatchCheckLayerAvailability", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer",
      "ecr:DescribeImages", "ecr:ListImages",
      "ecr:InitiateLayerUpload", "ecr:UploadLayerPart", "ecr:CompleteLayerUpload", "ecr:PutImage",
    ]
    resources = local.ecr_repo_arns
  }
  dynamic "statement" {
    for_each = var.dockerhub_secret_arn != "" ? [1] : []
    content {
      sid       = "DockerHubCredentials"
      actions   = ["secretsmanager:GetSecretValue"]
      resources = [var.dockerhub_secret_arn]
    }
  }
}

resource "aws_iam_role_policy" "images" {
  name   = "ecr-push"
  role   = aws_iam_role.codebuild["images"].id
  policy = data.aws_iam_policy_document.images.json
}

# ------------------------------------------------------------------ plan: read everything, never write the state
resource "aws_iam_role_policy_attachment" "readonly" {
  for_each   = toset(["plan", "apply"])
  role       = aws_iam_role.codebuild[each.key].name
  policy_arn = "arn:${local.partition}:iam::aws:policy/ReadOnlyAccess"
}

data "aws_iam_policy_document" "state_plan" {
  statement {
    sid       = "StateBucket"
    actions   = ["s3:ListBucket"]
    resources = [local.state_bucket_arn]
  }
  statement {
    sid       = "ReadStateAndConfig"
    actions   = ["s3:GetObject"]
    resources = ["${local.state_bucket_arn}/${local.state_key}", "${local.state_bucket_arn}/${local.config_key}"]
  }
  statement {
    sid       = "Lock"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = ["${local.state_bucket_arn}/${local.lock_key}"]
  }
}

resource "aws_iam_role_policy" "state_plan" {
  name   = "tfstate-read-lock"
  role   = aws_iam_role.codebuild["plan"].id
  policy = data.aws_iam_policy_document.state_plan.json
}

# ------------------------------------------------------------------ apply: write the state + manage the stack
data "aws_iam_policy_document" "state_apply" {
  statement {
    sid       = "StateBucket"
    actions   = ["s3:ListBucket"]
    resources = [local.state_bucket_arn]
  }
  statement {
    sid       = "ReadWriteState"
    actions   = ["s3:GetObject", "s3:PutObject"]
    resources = ["${local.state_bucket_arn}/${local.state_key}"]
  }
  statement {
    sid       = "Lock"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = ["${local.state_bucket_arn}/${local.lock_key}"]
  }
}

resource "aws_iam_role_policy" "state_apply" {
  name   = "tfstate-read-write"
  role   = aws_iam_role.codebuild["apply"].id
  policy = data.aws_iam_policy_document.state_apply.json
}

# Scoped write access for the resource types in deploy/ecs-neuron/*.tf. Reads (refresh, data sources, the AMI SSM
# parameter) come from ReadOnlyAccess. Services whose create calls take no resource ARN (ECS task definitions,
# security groups, launch templates, ALB) are limited to the region; IAM, S3, ECR, logs and the ASG to <name>-* names.
data "aws_iam_policy_document" "apply_scoped" {
  statement {
    sid = "Ecs"
    actions = [
      "ecs:CreateCluster", "ecs:DeleteCluster", "ecs:UpdateCluster", "ecs:UpdateClusterSettings",
      "ecs:PutClusterCapacityProviders",
      "ecs:CreateCapacityProvider", "ecs:UpdateCapacityProvider", "ecs:DeleteCapacityProvider",
      "ecs:RegisterTaskDefinition", "ecs:DeregisterTaskDefinition", "ecs:DeleteTaskDefinitions",
      "ecs:CreateService", "ecs:UpdateService", "ecs:DeleteService",
      "ecs:TagResource", "ecs:UntagResource",
    ]
    resources = ["*"]
    condition {
      test     = "StringEquals"
      variable = "aws:RequestedRegion"
      values   = [var.region]
    }
  }
  statement {
    sid = "Ec2"
    actions = [
      "ec2:CreateLaunchTemplate", "ec2:CreateLaunchTemplateVersion", "ec2:ModifyLaunchTemplate",
      "ec2:DeleteLaunchTemplate", "ec2:DeleteLaunchTemplateVersions",
      "ec2:CreateSecurityGroup", "ec2:DeleteSecurityGroup",
      "ec2:AuthorizeSecurityGroupIngress", "ec2:AuthorizeSecurityGroupEgress",
      "ec2:RevokeSecurityGroupIngress", "ec2:RevokeSecurityGroupEgress", "ec2:ModifySecurityGroupRules",
      "ec2:UpdateSecurityGroupRuleDescriptionsIngress", "ec2:UpdateSecurityGroupRuleDescriptionsEgress",
      "ec2:CreateVpcEndpoint", "ec2:ModifyVpcEndpoint", "ec2:DeleteVpcEndpoints",
      "ec2:CreateTags", "ec2:DeleteTags",
      "ec2:RunInstances", # Auto Scaling checks the caller may launch from the launch template on create/update
    ]
    resources = ["*"]
    condition {
      test     = "StringEquals"
      variable = "aws:RequestedRegion"
      values   = [var.region]
    }
  }
  statement {
    sid = "RemoteSecurityGroups" # remote_regions.tf: the no-ingress groups for hand-run Trainium hosts in other regions
    actions = [
      "ec2:CreateSecurityGroup", "ec2:DeleteSecurityGroup",
      "ec2:AuthorizeSecurityGroupEgress", "ec2:RevokeSecurityGroupEgress", "ec2:RevokeSecurityGroupIngress",
      "ec2:ModifySecurityGroupRules", "ec2:UpdateSecurityGroupRuleDescriptionsEgress",
      "ec2:CreateTags", "ec2:DeleteTags",
    ]
    resources = ["*"]
    condition {
      test     = "StringEquals"
      variable = "aws:RequestedRegion"
      values   = var.remote_regions
    }
  }
  statement {
    sid = "AutoScaling"
    actions = [
      "autoscaling:CreateAutoScalingGroup", "autoscaling:UpdateAutoScalingGroup", "autoscaling:DeleteAutoScalingGroup",
      "autoscaling:CreateOrUpdateTags", "autoscaling:DeleteTags",
      "autoscaling:PutWarmPool", "autoscaling:DeleteWarmPool",
      "autoscaling:SetInstanceProtection", "autoscaling:TerminateInstanceInAutoScalingGroup",
    ]
    resources = ["arn:${local.partition}:autoscaling:${var.region}:${local.acct}:autoScalingGroup:*:autoScalingGroupName/${local.p}-*"]
  }
  statement {
    sid = "LoadBalancer" # only used with enable_alb = true
    actions = [
      "elasticloadbalancing:CreateLoadBalancer", "elasticloadbalancing:DeleteLoadBalancer",
      "elasticloadbalancing:ModifyLoadBalancerAttributes", "elasticloadbalancing:SetSecurityGroups",
      "elasticloadbalancing:SetSubnets",
      "elasticloadbalancing:CreateTargetGroup", "elasticloadbalancing:DeleteTargetGroup",
      "elasticloadbalancing:ModifyTargetGroup", "elasticloadbalancing:ModifyTargetGroupAttributes",
      "elasticloadbalancing:CreateListener", "elasticloadbalancing:DeleteListener", "elasticloadbalancing:ModifyListener",
      "elasticloadbalancing:AddTags", "elasticloadbalancing:RemoveTags",
    ]
    resources = ["*"]
    condition {
      test     = "StringEquals"
      variable = "aws:RequestedRegion"
      values   = [var.region]
    }
  }
  # Roles the apply build creates or edits always carry the workload boundary (below), so no inline policy it writes
  # can grant more than the boundary does: the build cannot mint an administrator role and run a workload as it.
  statement {
    sid = "IamStackRolesBounded"
    actions = [
      "iam:CreateRole", "iam:PutRolePolicy", "iam:DeleteRolePolicy", "iam:DetachRolePolicy",
      "iam:PutRolePermissionsBoundary",
    ]
    resources = ["arn:${local.partition}:iam::${local.acct}:role/${local.p}-*"]
    condition {
      test     = "StringEquals"
      variable = "iam:PermissionsBoundary"
      values   = [aws_iam_policy.workload_boundary.arn]
    }
  }
  statement {
    sid = "IamStackRoles"
    actions = [
      "iam:DeleteRole", "iam:UpdateRole", "iam:UpdateRoleDescription", "iam:UpdateAssumeRolePolicy",
      "iam:TagRole", "iam:UntagRole",
    ]
    resources = ["arn:${local.partition}:iam::${local.acct}:role/${local.p}-*"]
  }
  statement {
    sid       = "PassStackRoles" # task/execution roles to ECS, the instance role to EC2 / its instance profile
    actions   = ["iam:PassRole"]
    resources = local.stack_role_arns
    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["ecs-tasks.amazonaws.com", "ec2.amazonaws.com"]
    }
  }
  statement {
    sid       = "KeepBoundaries"
    effect    = "Deny"
    actions   = ["iam:DeleteRolePermissionsBoundary", "iam:CreatePolicyVersion", "iam:SetDefaultPolicyVersion"]
    resources = ["*"]
  }
  # the pipeline's own roles also match <name>-*: never let the apply build edit them (or itself)
  statement {
    sid         = "DenyPipelineRoles"
    effect      = "Deny"
    not_actions = ["iam:Get*", "iam:List*"]
    resources   = ["arn:${local.partition}:iam::${local.acct}:role/${local.p}-pipeline*"]
  }
  statement {
    sid       = "IamAttachKnownManagedPolicies"
    actions   = ["iam:AttachRolePolicy"]
    resources = ["arn:${local.partition}:iam::${local.acct}:role/${local.p}-*"]
    condition {
      test     = "StringEquals"
      variable = "iam:PermissionsBoundary"
      values   = [aws_iam_policy.workload_boundary.arn]
    }
    condition {
      test     = "ArnEquals"
      variable = "iam:PolicyARN"
      values = [
        "arn:${local.partition}:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role",
        "arn:${local.partition}:iam::aws:policy/AmazonSSMManagedInstanceCore",
        "arn:${local.partition}:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy",
      ]
    }
  }
  statement {
    sid = "IamInstanceProfile"
    actions = [
      "iam:CreateInstanceProfile", "iam:DeleteInstanceProfile",
      "iam:AddRoleToInstanceProfile", "iam:RemoveRoleFromInstanceProfile",
      "iam:TagInstanceProfile", "iam:UntagInstanceProfile",
    ]
    resources = ["arn:${local.partition}:iam::${local.acct}:instance-profile/${local.p}-*"]
  }
  statement {
    sid       = "ServiceLinkedRoles"
    actions   = ["iam:CreateServiceLinkedRole"]
    resources = ["*"]
    condition {
      test     = "StringEquals"
      variable = "iam:AWSServiceName"
      values = [
        "ecs.amazonaws.com", "autoscaling.amazonaws.com", "elasticloadbalancing.amazonaws.com", "spot.amazonaws.com",
      ]
    }
  }
  statement {
    sid = "ModelBucket"
    actions = [
      "s3:CreateBucket", "s3:DeleteBucket",
      "s3:PutBucketPolicy", "s3:DeleteBucketPolicy", "s3:PutBucketPublicAccessBlock", "s3:PutBucketOwnershipControls",
      "s3:PutBucketVersioning", "s3:PutEncryptionConfiguration", "s3:PutLifecycleConfiguration", "s3:PutBucketTagging",
    ]
    resources = ["arn:${local.partition}:s3:::${local.models_bucket}"]
  }
  statement {
    sid = "RouterRepo"
    actions = [
      "ecr:CreateRepository", "ecr:DeleteRepository", "ecr:PutLifecyclePolicy", "ecr:DeleteLifecyclePolicy",
      "ecr:PutImageScanningConfiguration", "ecr:PutImageTagMutability", "ecr:SetRepositoryPolicy",
      "ecr:DeleteRepositoryPolicy", "ecr:TagResource", "ecr:UntagResource",
    ]
    resources = ["arn:${local.partition}:ecr:${var.region}:${local.acct}:repository/${local.router_repo}"]
  }
  statement {
    sid = "ApiKeysTable"
    actions = [
      "dynamodb:CreateTable", "dynamodb:DeleteTable", "dynamodb:UpdateTable", "dynamodb:UpdateContinuousBackups",
      "dynamodb:TagResource", "dynamodb:UntagResource", "dynamodb:UpdateTimeToLive",
    ]
    resources = ["arn:${local.partition}:dynamodb:${var.region}:${local.acct}:table/${local.p}-*"]
  }
  statement {
    sid = "StackLogGroups"
    actions = [
      "logs:CreateLogGroup", "logs:DeleteLogGroup", "logs:PutRetentionPolicy", "logs:DeleteRetentionPolicy",
      "logs:TagResource", "logs:UntagResource", "logs:TagLogGroup", "logs:UntagLogGroup",
    ]
    resources = [
      "arn:${local.partition}:logs:${var.region}:${local.acct}:log-group:/ecs/${local.p}/*",
      "arn:${local.partition}:logs:${var.region}:${local.acct}:log-group:/ecs/${local.p}/*:*",
    ]
  }
}

# ------------------------------------------------------------------ workload boundary for the stack's roles
# The most any role the stack manages may do: what the trn host (ECS agent, SSM), the task execution role (image pull,
# logs, secrets) and the backend / router tasks need. No IAM, no STS, no instance launches, reads only from the model
# bucket and the stack's secrets.
data "aws_iam_policy_document" "workload_boundary" {
  statement {
    sid       = "Ecs"
    actions   = ["ecs:*"]
    resources = ["*"]
    condition {
      test     = "StringEquals"
      variable = "aws:RequestedRegion"
      values   = [var.region]
    }
  }
  statement {
    sid = "ImagesAndLogs"
    actions = [
      "ecr:GetAuthorizationToken", "ecr:BatchCheckLayerAvailability", "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage",
      "logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams",
    ]
    resources = ["*"]
  }
  statement {
    sid = "HostAgents" # AmazonEC2ContainerServiceforEC2Role / AmazonSSMManagedInstanceCore
    actions = [
      "ec2:DescribeTags", "ec2:DescribeInstances", "ec2messages:*", "ssmmessages:*",
      "ssm:DescribeAssociation", "ssm:GetDeployablePatchSnapshotForInstance", "ssm:GetDocument", "ssm:DescribeDocument",
      "ssm:GetManifest", "ssm:ListAssociations", "ssm:ListInstanceAssociations", "ssm:PutInventory",
      "ssm:PutComplianceItems", "ssm:PutConfigurePackageResult", "ssm:UpdateAssociationStatus",
      "ssm:UpdateInstanceAssociationStatus", "ssm:UpdateInstanceInformation",
    ]
    resources = ["*"]
  }
  statement {
    sid = "BackendAsg" # the router scales the backend ASG directly
    actions = [
      "autoscaling:DescribeAutoScalingGroups", "autoscaling:SetDesiredCapacity", "autoscaling:SetInstanceProtection",
    ]
    resources = ["*"]
  }
  statement {
    sid       = "ApiKeys" # the router reads approved keys
    actions   = ["dynamodb:GetItem"]
    resources = ["arn:${local.partition}:dynamodb:${var.region}:${local.acct}:table/${local.p}-*"]
  }
  statement {
    sid       = "ModelBucket"
    actions   = ["s3:ListBucket", "s3:GetObject"]
    resources = ["arn:${local.partition}:s3:::${local.models_bucket}", "arn:${local.partition}:s3:::${local.models_bucket}/*"]
  }
  statement {
    sid     = "StackSecrets" # the tunnel token / router key, named <name>/... or <name>-...
    actions = ["secretsmanager:GetSecretValue", "ssm:GetParameters", "ssm:GetParameter"]
    resources = [
      "arn:${local.partition}:secretsmanager:${var.region}:${local.acct}:secret:${local.p}/*",
      "arn:${local.partition}:secretsmanager:${var.region}:${local.acct}:secret:${local.p}-*",
      "arn:${local.partition}:ssm:${var.region}:${local.acct}:parameter/${local.p}/*",
    ]
  }
  statement {
    sid       = "DecryptViaServices"
    actions   = ["kms:Decrypt"]
    resources = ["*"]
    condition {
      test     = "StringEquals"
      variable = "kms:ViaService"
      values   = ["secretsmanager.${var.region}.amazonaws.com", "ssm.${var.region}.amazonaws.com", "s3.${var.region}.amazonaws.com"]
    }
  }
}

resource "aws_iam_policy" "workload_boundary" {
  name   = "${var.name}-workload-boundary"
  policy = data.aws_iam_policy_document.workload_boundary.json
}

resource "aws_iam_role_policy" "apply_scoped" {
  count  = var.apply_role_policy == "scoped" ? 1 : 0
  name   = "manage-ecs-neuron-stack"
  role   = aws_iam_role.codebuild["apply"].id
  policy = data.aws_iam_policy_document.apply_scoped.json
}

resource "aws_iam_role_policy_attachment" "apply_admin" {
  count      = var.apply_role_policy == "admin" ? 1 : 0
  role       = aws_iam_role.codebuild["apply"].name
  policy_arn = "arn:${local.partition}:iam::aws:policy/AdministratorAccess"
}

# ------------------------------------------------------------------ CodePipeline service role
resource "aws_iam_role" "codepipeline" {
  name               = "${var.name}-pipeline"
  assume_role_policy = data.aws_iam_policy_document.codepipeline_assume.json
}

data "aws_iam_policy_document" "codepipeline" {
  statement {
    sid       = "Artifacts"
    actions   = ["s3:GetObject", "s3:GetObjectVersion", "s3:PutObject", "s3:PutObjectAcl"]
    resources = ["${local.artifact_bucket_arn}/*"]
  }
  statement {
    sid       = "ArtifactsBucket"
    actions   = ["s3:GetBucketLocation", "s3:GetBucketVersioning", "s3:ListBucket"]
    resources = [local.artifact_bucket_arn]
  }
  statement {
    sid       = "Source"
    actions   = ["codestar-connections:UseConnection", "codeconnections:UseConnection"]
    resources = [aws_codestarconnections_connection.github.arn]
  }
  statement {
    sid       = "Builds"
    actions   = ["codebuild:StartBuild", "codebuild:BatchGetBuilds", "codebuild:StopBuild"]
    resources = [for k, v in aws_codebuild_project.this : v.arn]
  }
  dynamic "statement" {
    for_each = var.approval_sns_topic_arn != "" ? [1] : []
    content {
      sid       = "ApprovalNotification"
      actions   = ["sns:Publish"]
      resources = [var.approval_sns_topic_arn]
    }
  }
}

resource "aws_iam_role_policy" "codepipeline" {
  name   = "pipeline"
  role   = aws_iam_role.codepipeline.id
  policy = data.aws_iam_policy_document.codepipeline.json
}
