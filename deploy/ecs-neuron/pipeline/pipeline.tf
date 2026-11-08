# Created in PENDING state: complete the GitHub handshake once in the console (README.md, "Bootstrap" step 2).
resource "aws_codestarconnections_connection" "github" {
  name          = "${var.name}-github"
  provider_type = "GitHub"
}

resource "aws_codepipeline" "this" {
  name           = "${var.name}-ecs-neuron"
  role_arn       = aws_iam_role.codepipeline.arn
  pipeline_type  = "V2"
  execution_mode = var.execution_mode

  artifact_store {
    location = aws_s3_bucket.this["artifacts"].bucket
    type     = "S3"
  }

  # Override at "Release change" time to rebuild an image whose tag already exists (e.g. after changing the Neuron pins).
  variable {
    name          = "FORCE_IMAGE_BUILD"
    default_value = "none"
    description   = "none | backend | router | all: rebuild and re-push these images even if their tag is already in ECR."
  }

  trigger {
    provider_type = "CodeStarSourceConnection"
    git_configuration {
      source_action_name = "Source"
      push {
        branches {
          includes = [var.github_branch]
        }
        dynamic "file_paths" {
          for_each = length(var.trigger_file_paths) > 0 ? [1] : []
          content {
            includes = var.trigger_file_paths
          }
        }
      }
    }
  }

  stage {
    name = "Source"
    action {
      name             = "Source"
      category         = "Source"
      owner            = "AWS"
      provider         = "CodeStarSourceConnection"
      version          = "1"
      namespace        = "SourceVariables"
      output_artifacts = ["SourceOutput"]
      configuration = {
        ConnectionArn    = aws_codestarconnections_connection.github.arn
        FullRepositoryId = var.github_repository
        BranchName       = var.github_branch
        # full git clone in CodeBuild (instead of a zip): the image build needs history to find the last commit that
        # touched the backend / router inputs
        OutputArtifactFormat = "CODEBUILD_CLONE_REF"
      }
    }
  }

  stage {
    name = "BuildImages"
    action {
      name            = "Images"
      category        = "Build"
      owner           = "AWS"
      provider        = "CodeBuild"
      version         = "1"
      namespace       = "BuildVariables" # BACKEND_TAG, ROUTER_TAG, BACKEND_BUILT, ROUTER_BUILT
      input_artifacts = ["SourceOutput"]
      configuration = {
        ProjectName = aws_codebuild_project.this["images"].name
        EnvironmentVariables = jsonencode([
          { name = "COMMIT_ID", value = "#{SourceVariables.CommitId}", type = "PLAINTEXT" },
          { name = "FORCE_IMAGE_BUILD", value = "#{variables.FORCE_IMAGE_BUILD}", type = "PLAINTEXT" },
        ])
      }
    }
  }

  stage {
    name = "Plan"
    action {
      name             = "Plan"
      category         = "Build"
      owner            = "AWS"
      provider         = "CodeBuild"
      version          = "1"
      namespace        = "PlanVariables" # PLAN_SUMMARY, PLAN_LOG_URL
      input_artifacts  = ["SourceOutput"]
      output_artifacts = ["PlanOutput"]
      configuration = {
        ProjectName = aws_codebuild_project.this["plan"].name
        EnvironmentVariables = jsonencode([
          { name = "COMMIT_ID", value = "#{SourceVariables.CommitId}", type = "PLAINTEXT" },
          { name = "BACKEND_TAG", value = "#{BuildVariables.BACKEND_TAG}", type = "PLAINTEXT" },
          { name = "ROUTER_TAG", value = "#{BuildVariables.ROUTER_TAG}", type = "PLAINTEXT" },
        ])
      }
    }
  }

  stage {
    name = "Approve"
    action {
      name     = "ApprovePlan"
      category = "Approval"
      owner    = "AWS"
      provider = "Manual"
      version  = "1"
      configuration = merge({
        CustomData         = "Commit #{SourceVariables.CommitId}: backend=#{BuildVariables.BACKEND_TAG} router=#{BuildVariables.ROUTER_TAG}. #{PlanVariables.PLAN_SUMMARY} Full plan: the Plan build log (link) or plan.txt in the PlanOutput artifact."
        ExternalEntityLink = "#{PlanVariables.PLAN_LOG_URL}"
      }, var.approval_sns_topic_arn != "" ? { NotificationArn = var.approval_sns_topic_arn } : {})
    }
  }

  stage {
    name = "Apply"
    action {
      name            = "Apply"
      category        = "Build"
      owner           = "AWS"
      provider        = "CodeBuild"
      version         = "1"
      input_artifacts = ["SourceOutput", "PlanOutput"] # the plan is at $CODEBUILD_SRC_DIR_PlanOutput
      configuration = {
        ProjectName   = aws_codebuild_project.this["apply"].name
        PrimarySource = "SourceOutput"
        EnvironmentVariables = jsonencode([
          { name = "COMMIT_ID", value = "#{SourceVariables.CommitId}", type = "PLAINTEXT" },
        ])
      }
    }
  }

  depends_on = [aws_iam_role_policy.codepipeline]
}
