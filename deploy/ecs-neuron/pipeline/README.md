# CI/CD for deploy/ecs-neuron (CodePipeline V2 + CodeBuild)

Written, `terraform validate`d, **not applied**. No AWS API call was made while writing it.

This directory is a separate Terraform root (the "bootstrap" stack). It has its own local state
(`pipeline/terraform.tfstate`, gitignored). It creates:

* `behemoth-tfstate-912935854432`, the state and config bucket. It is versioned, uses SSE-S3, blocks public access,
  only accepts TLS and has `prevent_destroy`. It holds two keys:
  * `ecs-neuron/terraform.tfstate` is the main stack's state. Locking uses the S3 lock file
    `ecs-neuron/terraform.tfstate.tflock` (`use_lockfile = true`, no DynamoDB).
  * `ecs-neuron/terraform.tfvars` is the main stack's config. It is gitignored locally and never goes into the public repo.
* `behemoth-pipeline-artifacts-912935854432`, the CodePipeline artifact bucket. It has the same baseline, and objects
  expire after 30 days.
* The GitHub connection `behemoth-github` (CodeStar Connections). It is created **PENDING**.
* The pipeline `behemoth-ecs-neuron` (V2), three CodeBuild projects, their log groups and four IAM roles.

```
push to rocm-to-neff touching deploy/ecs-neuron/** runtime/** neuron/cpp/** glm53/** third_party/** CMakeLists.txt
  |
  v
Source (GitHub connection, full clone: CODEBUILD_CLONE_REF)          namespace SourceVariables (CommitId)
  |
  v
BuildImages  CodeBuild behemoth-pipeline-images  (privileged, LARGE)  buildspec-images.yml -> scripts/build-images.sh
  |   backend tag = 7-char SHA of the last commit touching backend inputs; skipped if the tag is in ECR
  |   router  tag = 7-char SHA of the last commit touching deploy/ecs-neuron/router; tests run, then skip/build
  |   exports BACKEND_TAG ROUTER_TAG BACKEND_BUILT ROUTER_BUILT    namespace BuildVariables
  v
Plan  CodeBuild behemoth-pipeline-plan (ReadOnlyAccess + state read/lock)   buildspec-plan.yml
  |   init (S3 backend, lockfile=readonly), fetch tfvars from S3, plan -out=tfplan -var *_image_tag=...
  |   artifact PlanOutput: tfplan, plan.txt, plan-meta.env       namespace PlanVariables (PLAN_SUMMARY, PLAN_LOG_URL)
  v
Approve  manual approval (summary + link to the plan log; optional SNS)
  |
  v
Apply  CodeBuild behemoth-pipeline-apply (scoped or admin)   buildspec-apply.yml: terraform apply <saved tfplan>
```

## Files

| path | what |
|---|---|
| `versions.tf`, `variables.tf`, `outputs.tf` | provider (`allowed_account_ids = [912935854432]`) and inputs. Every default is public and non-secret, so no tfvars file is needed |
| `s3.tf` | state/config bucket and artifact bucket |
| `iam.tf` | roles for the pipeline, images, plan and apply, plus the scoped apply policy |
| `codebuild.tf` | the three projects and their log groups |
| `pipeline.tf` | GitHub connection and pipeline (trigger, stages, pipeline variable `FORCE_IMAGE_BUILD`) |
| `buildspec-images.yml`, `buildspec-plan.yml`, `buildspec-apply.yml` | buildspecs, read from the commit being built |
| `scripts/build-images.sh` | tag computation, ECR check, router tests, build and push |
| `scripts/install-terraform.sh` | downloads Terraform 1.12.2 and checks it against the pinned SHA-256 (`terraform_sha256`) |
| `scripts/tf-init.sh` | writes `backend.hcl` from the environment, then runs `terraform init -lockfile=readonly` |
| `../backend.hcl.example` | backend config for local use |

## Image tags and when an image is rebuilt

An image's tag is the short SHA of the last commit that changed its inputs (`git log -1 -- <paths>`). It is not
the SHA of HEAD.

* backend inputs: `CMakeLists.txt runtime/ neuron/cpp/ glm53/ third_party/ deploy/ecs-neuron/backend/`
* router inputs: `deploy/ecs-neuron/router/`

The build is skipped when the tag already exists in ECR. A push that only changes Terraform therefore re-uses both
images, and the plan shows no image change. No record of "the last successful run" is needed. Re-running an old
commit gives the same tags. A build that failed halfway pushed nothing, so the next run builds again.

When an input is not in git, the tag does not change: the Neuron pins (`neuron_runtime_version` /
`neuron_collectives_version`), `backend_base_image`, or a new upstream package behind an unpinned apt version. To
rebuild in that case, start the pipeline with **Release change** and set the pipeline variable
`FORCE_IMAGE_BUILD=backend` (also `router`, `all` or `backend,router`). The existing tag is overwritten. Both repos
are MUTABLE.

`latest` is moved on every build. The stack itself always runs the explicit SHA tags passed as `-var`, which override
the `*_image_tag = "latest"` in tfvars.

## Bootstrap (one time, in this order)

Your own admin credentials run every step below. The pipeline never runs these.

1. **Apply the bootstrap stack**
   ```sh
   cd deploy/ecs-neuron/pipeline
   terraform init && terraform apply
   ```
   CodePipeline starts one execution as soon as the pipeline is created. That execution fails at Source because the
   connection is still pending. This is expected.
2. **Complete the GitHub connection.** Open the console in us-east-1: CodePipeline -> left menu **Settings ->
   Connections**
   (`https://us-east-1.console.aws.amazon.com/codesuite/settings/connections?region=us-east-1`). Select
   `behemoth-github` (status *Pending*) and click **Update pending connection**. In the GitHub pop-up, authorize *AWS
   Connector for GitHub*, then click **Install a new app**. Choose the `c000507` account, select *Only select
   repositories* -> `Behemoth-qtip`, and click Install. Back in the console, click **Connect**. The status must now be
   *Available* (`terraform output github_connection_status` shows it after a refresh).
3. **Migrate the main stack's state** (local -> S3):
   ```sh
   cd deploy/ecs-neuron
   cp terraform.tfstate ~/behemoth-ecs-neuron.tfstate.bak-$(date +%F)    # keep a copy outside the repo
   cp backend.hcl.example backend.hcl                                     # gitignored
   terraform init -migrate-state -backend-config=backend.hcl              # answer "yes" to copy the state
   aws s3 ls s3://behemoth-tfstate-912935854432/ecs-neuron/                # terraform.tfstate is there
   terraform plan                                                         # same result as before the migration
   mv terraform.tfstate terraform.tfstate.migrated                        # so nothing reads the stale copy
   ```
4. **Upload the config**:
   ```sh
   aws s3 cp terraform.tfvars s3://behemoth-tfstate-912935854432/ecs-neuron/terraform.tfvars --sse AES256
   ```
5. **Release a change.** Commit and push this directory, the `versions.tf` backend block and `backend.hcl.example`
   to `rocm-to-neff`. The push matches `deploy/ecs-neuron/**` and starts the pipeline. Steps 3 and 4 must come
   before this push. The Plan build refuses to run when the state object is missing, because an empty state would
   plan to re-create about 40 resources.

   On the first run, BuildImages builds the backend (tag `36fb795` today, ~10-15 min) and the router (`aa9bc4c` or
   newer), unless those tags already exist in ECR. Plan then shows the two task definitions being replaced, with the
   image changing from `:latest` to the SHA tag, and in-place updates of the services. Approve the plan. The router
   redeploys, which takes about 1 min of downtime because it runs as one replica. The backend stays at desired 0.

### Day-to-day

* **Code or Terraform change.** Push to `rocm-to-neff`. The pipeline runs Build, then Plan, then waits for your
  approval, then runs Apply.
* **Config change.** Edit `terraform.tfvars`, upload it with `aws s3 cp` (step 4), then click **Release change** in
  the console. A tfvars change is not a git change, so it does not trigger the pipeline.
* **Local terraform.** After the migration, every checkout needs `terraform init -backend-config=backend.hcl`.
  Locally the tfvars tags (`latest`) apply, so pass the deployed tags. They are shown as `BuildVariables` in the last
  execution's details:
  `terraform plan -var backend_image_tag=<sha> -var router_image_tag=<sha>`. The S3 lock prevents local runs and the
  pipeline from overlapping.
* **A stuck lock** (a build killed mid-apply): run `terraform force-unlock <id>` locally. The id is in the error
  message, or delete `ecs-neuron/terraform.tfstate.tflock`.

### Optional: bootstrap state

The bootstrap stack's local state can move into the same bucket once it exists. Add
`backend "s3" { bucket = "behemoth-tfstate-912935854432" key = "pipeline/terraform.tfstate" region = "us-east-1" use_lockfile = true }`
to `versions.tf`, then run `terraform init -migrate-state`. The plan/apply roles have no access to that key.

## Rollback

* **Preferred: re-run the pipeline at an older commit.** In the console, click **Release change** and set *Source
  revision override* to the commit ID. That commit's images are still in ECR, so BuildImages skips them. Plan shows
  the old config and old tags. Approve. Keep in mind:
  * The commit must contain the `backend "s3" {}` block and the pipeline buildspecs, so this only works for commits
    after the migration.
  * The tfvars come from S3 and are always the current version. To roll back config too, restore an older version
    of `ecs-neuron/terraform.tfvars` first (the bucket is versioned).
  * The next push rolls forward again, so revert in git (`git revert`) if the rollback should stick.
* **Tags only (emergency, local):**
  `terraform plan -out=rb.tfplan -var backend_image_tag=<old> -var router_image_tag=<old>`, check it, then
  `terraform apply rb.tfplan`. The router repo keeps only the last 10 images (lifecycle policy), so a very old router
  tag may be gone.
* **State.** Every state write is a new S3 object version, kept for 180 days and at least 50 versions. Restoring an
  old state version is a last resort, because the infrastructure no longer matches it. Prefer re-applying.

## Costs (us-east-1, rough)

| item | price | per release |
|---|---|---|
| CodePipeline V2 | $0.002 per action-execution minute, first 100 min/month free, no fixed monthly fee | ~15-25 min, $0.03-0.05 |
| CodeBuild images, `BUILD_GENERAL1_LARGE` | ~$0.02/min | backend+router ~15 min, $0.30. Skipped images ~1-2 min, $0.03 |
| CodeBuild plan / apply, `BUILD_GENERAL1_SMALL` | ~$0.005/min | ~2 + ~3 min, $0.03 |
| S3 (state, config, artifacts), CloudWatch Logs, connection | cents / free | - |
| ECR storage of backend images | $0.10/GB-month, ~2 GB per backend build | the backend repo has no lifecycle policy (it is outside Terraform). Add one, e.g. keep 20 |

Example: 30 releases/month, 8 of them backend rebuilds, comes to about **$4-6/month**. With no pushes the cost is
~$0. Time spent waiting at the manual approval may count as action minutes (not verified). Even a 24 h wait costs at
most ~$2.9.

## Apply role: scoped vs AdministratorAccess

`apply_role_policy = "scoped"` is the default. It grants ReadOnlyAccess for refresh, data sources and the AMI SSM
parameter, plus write access to the following:

* **IAM**: roles `behemoth-*`, instance profiles `behemoth-*`, and PassRole on `behemoth-*`. AttachRolePolicy only
  works for the three AWS managed policies the stack uses. The pipeline's own roles (`behemoth-pipeline*`) are
  explicitly denied.
* **S3**: only the models bucket.
* **ECR**: only the router repo.
* **Logs**: only `/ecs/behemoth/*`.
* **Auto Scaling**: only groups named `behemoth-*`.
* **ECS, EC2 (launch templates, SGs and rules, VPC endpoint, RunInstances for the ASG check) and ELB**: region-wide,
  because their create calls cannot be scoped by name.

If an apply fails with `AccessDenied` on an action the stack gained later (KMS keys when `s3_kms_key_arn` /
`secrets_kms_key_arn` are set, a new resource type), add the action to `iam.tf`. You can also switch to
`apply_role_policy = "admin"`. Admin never breaks on a new resource type, but whoever controls the branch plus the
approval then controls the whole account. Scoped does not make privilege escalation impossible: the role can still
write inline policies on `behemoth-*` roles and pass them to ECS. It does limit what a bad plan, or a
misconfiguration, can touch. **The manual approval is the real gate**, so read every plan.

## Risks

* **Branch = deploy rights.** Anyone who can push to `rocm-to-neff` runs code in CodeBuild:
  * The image build has ECR push rights.
  * The plan has account-wide read access (ReadOnlyAccess can read S3 objects and SSM parameters).
  * Nothing reaches Apply without approval.

  Pushes from forks or PRs do not trigger the pipeline, which only reacts to pushes to the branch. Protect the
  branch on GitHub.
* **Public repo.** The buildspecs, bucket names and account ID are public. They are not secrets. The plan file
  contains the tfvars values and stays in the private artifact bucket. The tunnel token is only referenced by ARN.
* **Docker Hub rate limits.** Shared CodeBuild IPs hit the anonymous pull limit. The backend base image comes from
  ECR Public (`backend_base_image`), but the router Dockerfile (`python:3.12-slim`) and the backend's
  `# syntax=docker/dockerfile:1.7` line still pull from Docker Hub. If builds fail with `429 Too Many Requests`, set
  `dockerhub_secret_arn` to a secret shaped `{"username": "...", "password": "<access token>"}`.
* **Stale plans.** Terraform refuses to apply a plan made before the state changed ("Saved plan is stale"), so re-run
  the pipeline. In SUPERSEDED mode, a newer execution replaces one still waiting for approval.
* **Provider pinning.** `-lockfile=readonly` makes CI use exactly the committed `.terraform.lock.hcl`. After changing
  provider constraints, run `terraform init -upgrade` locally and commit the lock file, or Plan fails.
* **Tag collisions.** Seven hex characters can, in principle, collide. A collision would skip a needed build.
  Lengthen `${c:0:7}` in `build-images.sh` if that ever matters.
* **Neuron driver/runtime drift.** This is unchanged by the pipeline (see the main README). Pin
  `neuron_runtime_version` / `neuron_collectives_version` here and `backend_ami_id` in tfvars.

## Unverified (no AWS calls were made)

* CodePipeline V2 trigger `file_paths` globs (`deploy/ecs-neuron/**`) are matched as documented, and the default
  change detection of the source action does not also fire on every push once `trigger` is set. If it does, set
  `DetectChanges = "false"` in the Source action configuration.
* Variables (`#{PlanVariables.*}`) are resolved in the manual approval `CustomData` / `ExternalEntityLink`.
* `CODEBUILD_CLONE_REF` produces a full (non-shallow) clone. If it is shallow, the script tries `git fetch --unshallow`,
  and if that fails it falls back to HEAD tags, which means a rebuild on every run. Also unverified: that the
  `codestar-connections:*` / `codeconnections:*` permissions granted are enough for the clone.
* `aws/codebuild/standard:7.0` provides Docker with BuildKit (for `# syntax=`), git, jq, unzip and AWS CLI v2 in
  privileged mode.
* The scoped apply policy covers every call the stack makes (e.g. Auto Scaling resource-level ARNs for
  `PutWarmPool`, `ec2:RunInstances` for the ASG launch template check). If not, see "Apply role" above.
* The CodeBuild per-minute prices and whether approval wait time is billed.
