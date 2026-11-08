# Scale-to-zero Neuron serving on ECS (spot trn1.2xlarge + Fargate router)

Nothing here has been applied. `terraform validate` and the router unit tests pass locally; every AWS-side
assumption still has to be checked against the account (see "Unverified").

```
 client / net.beaverai.org  (lists Cgg507/cydonia-24b)
            |
            v  HTTPS
   Cloudflare edge --- tunnel (token from Secrets Manager/SSM) ---+
                                                                   |
 +------------------------- ECS cluster "behemoth" ----------------|-------------------------------+
 |                                                                 v                               |
 |  Fargate service behemoth-router (desired 1, always on, ~$10-15/mo)                             |
 |  +-----------------------------------------------------------------------------+               |
 |  | cloudflared sidecar --localhost:1337--> router (aiohttp)                    |   [optional   |
 |  |   GET /v1/models, /health: answered locally                                 |<-- ALB]       |
 |  |   POST /v1/chat/completions: ensure backend up, then proxy (SSE passthrough)|               |
 |  +---------+--------------------------------------------+----------------------+               |
 |            | ecs:UpdateService desiredCount 0/1          | http://<task ENI IP>:1337           |
 |            | ecs:ListTasks/DescribeTasks (ENI IP)        |                                     |
 |            v                                             v                                     |
 |  EC2 service behemoth-backend (desired 0..1) --- capacity provider behemoth-trn-spot            |
 |  (managed scaling) --> ASG min 0 / max 1, SPOT only, trn1.2xlarge, ECS AL2023 Neuron AMI        |
 |  +-----------------------------------------------------------------------------+               |
 |  | task (awsvpc ENI, /dev/neuron0, task role: s3:GetObject on 2 prefixes)       |               |
 |  |   entrypoint.sh: s5cmd pull  s3://<bucket>/models/cydonia-24b-fp8/          |               |
 |  |                              s3://<bucket>/graphs/cydonia-24b-fp8-kv/       |               |
 |  |                  -> /data = NVMe instance store (474 GB)                    |               |
 |  |                  exec neuron_server --model ... --port 1337 (~70 s load)    |               |
 |  +-------------------------------------+---------------------------------------+               |
 +----------------------------------------|------------------------------------------------------+
                                          | S3 gateway VPC endpoint (free, no NAT needed)
                                          v
                            S3 bucket <name>-neuron-models-<acct> (private, SSE, TLS-only)
```

## Files

| path | what |
|---|---|
| `versions.tf` | provider, default tags, `backend "s3" {}` (partial config: `backend.hcl`, see `backend.hcl.example`) |
| `variables.tf` | all inputs with descriptions |
| `network.tf` | router / backend-task SGs (router -> backend:1337), S3 gateway endpoint |
| `s3.tf` | model bucket: public access blocked, SSE-S3 or KMS, TLS-only policy, optional versioning |
| `ecr.tf` | existing backend repo via `data "aws_ecr_repository"`; router repo created |
| `iam.tf` | instance role/profile, execution role (+ secret read), backend task role (S3 read), router task role (ECS control) |
| `ecs_cluster.tf` | cluster, Neuron AMI lookup, launch template, spot ASG, capacity provider |
| `backend.tf` | backend task definition + service (desired 0, router-owned) |
| `router.tf` | router task (router + cloudflared), Fargate service, optional ALB |
| `outputs.tf` | bucket / S3 URIs / repo URLs / names / AMI |
| `templates/backend_user_data.sh.tftpl` | mounts the instance store, joins the cluster, enables spot draining |
| `backend/Dockerfile` | multi-stage: compile `neuron_server` with `neuron/cpp/build.sh OUT srv`, runtime = binary + Neuron runtime libs + s5cmd |
| `backend/entrypoint.sh`, `backend/healthcheck.sh` | S3 pull then exec; health check covering the long startup |
| `backend/build_and_push.sh` | build, ECR login, push `<sha>` + `latest` |
| `router/router.py`, `router/test_router.py` | router service and its tests |
| `scripts/upload_to_s3.sh` | upload model + graphs from the current instance (symlinks dereferenced) |
| `pipeline/` | separate bootstrap root: state/config bucket, GitHub connection, CodePipeline V2 + CodeBuild (build images, plan, approve, apply). See `pipeline/README.md` |

## Prerequisites

* **Spot quota.** Service Quotas -> EC2 -> "All Trn Spot Instance Requests" (L-6B0D517C at the time of writing,
  check it) must be at least 8 vCPUs (one trn1.2xlarge). The account has 64 on-demand Trn vCPUs; the spot quota is
  unknown and new accounts often have 0. Without it the ASG never launches and every cold request ends in a 503.
* Terraform >= 1.5, Docker with BuildKit, AWS CLI v2. Push the images from x86_64.
* The VPC id and the route table of `subnet-05a70c02d594e2500`, for the S3 gateway endpoint.
* The tunnel token, stored once outside Terraform:
  `aws secretsmanager create-secret --name behemoth/cloudflared-token --secret-string "$TOKEN"`.
  Only its ARN goes into `terraform.tfvars`.
* In the Cloudflare dashboard (Zero Trust -> Tunnels -> the tunnel -> Public hostname) the service must be
  `http://localhost:1337`. The router listens on 1337 (`router_port`), so a rule already pointing there needs no change.
  **Stop the cloudflared on the hand-run instance** before the Fargate one starts. Two connectors on the same token
  are load-balanced replicas, so half the requests would go to the old box.

## 1. Build and push the images

```sh
cd /path/to/Behemoth-qtip-neuron
# backend: compiles neuron_server inside the build stage (context = repo root)
deploy/ecs-neuron/backend/build_and_push.sh
# ...or reuse a binary built on the trn instance (neuron/cpp/build.sh OUT srv) on the same Ubuntu release:
PREBUILT=OUT/neuron_server deploy/ecs-neuron/backend/build_and_push.sh
# pin Neuron libs to the AMI's driver generation (recommended):
NEURON_RUNTIME_VERSION=... NEURON_COLLECTIVES_VERSION=... deploy/ecs-neuron/backend/build_and_push.sh
```
The script pushes to `912935854432.dkr.ecr.us-east-1.amazonaws.com/behemoth-neuron-server` (override with `REPO_URI`).
That repo already exists, so Terraform only reads it through a data source and `terraform destroy` cannot delete it.
The router repo is created by Terraform, so build the router after the first apply. Its service then retries until
the image exists:
```sh
cd deploy/ecs-neuron && router/build_and_push.sh        # runs the unit tests first
```

## 2. Upload the model and graphs (on the current trn instance)

```sh
BUCKET=<terraform output -raw s3_bucket> \
MODEL_DIR=/home/ubuntu/models/cydonia-24b-fp8 GRAPH_DIR=/home/ubuntu/work/me/cydk \
  deploy/ecs-neuron/scripts/upload_to_s3.sh
```
The script uploads with s5cmd if it is installed, otherwise with `aws s3 sync --follow-symlinks`. The ~33 bucket
directories include symlinks to sibling directories; these are uploaded as real copies, because S3 has no symlinks.
It then compares file and `graph.neff` counts between the local trees and S3. The instance profile needs
`s3:PutObject`/`ListBucket` on the bucket, and `neuron-inf2-profile` may not have it. Otherwise run the script with
user credentials.

## 3. Apply

```sh
cd deploy/ecs-neuron
cp terraform.tfvars.example terraform.tfvars    # fill vpc_id, s3_endpoint_route_table_ids, secret ARN
cp backend.hcl.example backend.hcl                # remote state in S3 (bucket created by pipeline/)
terraform init -backend-config=backend.hcl && terraform plan && terraform apply
```
Once `pipeline/` is bootstrapped, changes go through CodePipeline (push to `rocm-to-neff`, approve the plan), and
`terraform.tfvars` lives in `s3://behemoth-tfstate-912935854432/ecs-neuron/terraform.tfvars`. See
[pipeline/README.md](pipeline/README.md) for the one-time state migration, rollback and costs.
To get the bucket name for step 2 before anything else exists, apply the bucket first with
`-target=aws_s3_bucket.models`, or set `s3_bucket_name`. The backend service is created with `desiredCount=0`, and
Terraform ignores later changes to it.

## How scale-up and idle-down work

* **Wake.** A POST to a proxied path calls `Backend.ensure()`. With no cached healthy address it starts one startup
  coroutine, and every concurrent request awaits that same coroutine (single-flight). The coroutine sets
  `desiredCount=1` if it is 0. The task goes to PROVISIONING, managed scaling raises the ASG to 1, the spot instance
  boots and registers, the task is placed, the image is pulled, and the entrypoint pulls S3 and loads the model. The
  router polls ListTasks/DescribeTasks every 5 s. Once the task is RUNNING it reads the task ENI's
  `privateIPv4Address`, then polls `GET http://ip:1337/health` until it returns 200, and then proxies the request.
* **Address discovery: DescribeTasks, not Cloud Map.** The router already has to poll ECS for task state, so the ENI
  IP comes from the same call. There is no DNS TTL or stale record after a spot replacement, and no namespace to run.
  The awsvpc network mode puts the IP straight into the task attachment, so `ec2:DescribeNetworkInterfaces` is not
  needed.
* **Waiting.** Streaming requests (`"stream": true`) get `200 text/event-stream` immediately, then an SSE comment
  (`: waiting for backend to start`) every 15 s until the backend is ready, then the backend's stream unchanged.
  Non-streaming requests are held silently by default (`cold_nonstream_mode = "wait"`) for up to `max_wait_seconds`
  (900). Any proxy in between has to tolerate that silence. Cloudflare closes an origin that sends nothing for 100 s
  (HTTP 524) on most plans, so behind the tunnel a cold non-streaming request fails at the edge while the warm-up
  continues, and a client retry a few minutes later succeeds. Set `cold_nonstream_mode = "reject"` to return
  `503 Retry-After: 120` immediately and warm up in the background instead. The ALB's idle timeout defaults to 1200 s
  for this reason.
* **Idle.** Every `min(30 s, idle/5)` the router checks for no request in flight, no startup running and
  `idle_timeout_seconds` (300) since the last request ended. When all hold it sets `desiredCount=0`. The task stops,
  managed scaling scales the ASG to 0 once its scale-in alarm fires (about 15 one-minute datapoints, so the instance
  lives about 15 more minutes), and managed termination protection guarantees an instance is never terminated under a
  running task.
* **Backend loss (spot interruption).** `ECS_ENABLE_SPOT_INSTANCE_DRAINING` drains the task on the 2-minute notice.
  The service keeps `desiredCount=1`, so ECS and the capacity provider try to replace it by themselves. In the router,
  a failed connect or health check clears the cached address. The next request waits for a new task, and a request
  whose connect failed before any byte is retried once against the new task. A stream cut mid-response ends with EOF
  (the client sees no `[DONE]`). If the backend is not healthy within `max_wait_seconds`, the router returns
  `503 Retry-After` (or an SSE error event followed by `[DONE]` when the 200 has already been sent). `desiredCount`
  stays at 1 so the next attempt continues, and the idle timer scales it back to 0 if nobody asks again.
* `GET /v1/models` and `GET /health` are answered by the router itself and never wake the backend. `/health` returns
  `{"healthy":true,"backend":"ready|starting|stopped|..."}`.

## Storage choice: NVMe instance store

trn1.2xlarge has a 474 GB NVMe instance store. The user data formats it (xfs) and mounts it at
`/mnt/instance-store`, and the task bind-mounts `/mnt/instance-store/behemoth` as `/data`. The reasons:

* It costs nothing. A gp3 volume fast enough to absorb a 25 GB burst needs provisioned throughput, and any EBS volume
  would be deleted with the scaled-in instance anyway.
* Local NVMe writes faster than the network delivers, so the S3 pull is network-bound.
* Persistence buys nothing here. Scale-to-zero terminates the instance, so every cold start pulls from S3 either way.
  A task restart on the same instance only re-fetches files whose size changed (`s5cmd cp -n -s`).

If no instance store is found (another instance type), the files go onto the root gp3 volume
(`backend_root_volume_gb`, default 100).

## Expected cold-start latency

| step | estimate |
|---|---|
| managed scaling reacts to the PROVISIONING task | 1-2 min |
| spot fulfilment + boot + ECS agent registration | 1.5-3 min |
| image pull from ECR (~2 GB) | 0.5-1 min |
| S3 pull ~25 GB (model + graphs in parallel, 64 workers; trn1.2xlarge "up to 12.5 Gbps") | 0.5-2 min |
| neuron_server load onto both cores | ~70 s |
| **total** | **~5-9 min typical, 15 min `max_wait_seconds` ceiling** |

If spot capacity is short in us-east-1f, the wait has no upper bound and the request ends in a 503.

## Cost estimate (us-east-1)

| item | cost |
|---|---|
| trn1.2xlarge spot | ~$0.22/h while up (on-demand would be $1.34/h) |
| per wake | busy time + 5 min idle + ~15 min scale-in lag: ~$0.07 of spot overhead per wake |
| Fargate router 0.25 vCPU / 0.5 GB, 24x7 | ~$9/mo + public IPv4 $3.6/mo + logs: **~$10-15/mo** |
| S3 ~30 GB standard | ~$0.70/mo; EC2<->S3 same-region transfer through the gateway endpoint is free |
| ECR images | ~$0.20-0.50/mo |
| 4 h/day of use | ~$30/mo spot + ~$13 router, against ~$980/mo for 24x7 on-demand |

## Known risks and limits

* **Spot interruptions.** In-flight requests are lost; the stream ends without `[DONE]`. trn1 spot is offered only in
  a few AZs (here just us-east-1f), so a capacity shortage means no backend at all. Add more instance types or
  subnets if more become available, or temporarily switch the ASG to on-demand
  (`on_demand_percentage_above_base_capacity = 100`).
* **Timeouts during cold start.** See "Waiting". Upstream clients and the beaverai router have their own timeouts,
  and those are unknown. Streaming plus keepalive is the only path that reliably survives a 5-9 min cold start.
* **Neuron driver vs container runtime.** The host's `aws-neuronx-dkms` comes from the AMI, and `libnrt` /
  collectives come from the image. A new `recommended` AMI can break an older image, and the other way round. Pin
  `backend_ami_id` after the first good run, and pin `NEURON_RUNTIME_VERSION` / `NEURON_COLLECTIVES_VERSION` to the
  same Neuron SDK release. The NEFFs were compiled with neuronx-cc 2.26, so the runtime must accept them.
* **Instance store is ephemeral.** It is wiped when the instance stops or terminates, so every scale-from-zero pulls
  the full ~25 GB again.
* **One router replica by design.** The in-flight count and idle timer live in memory. If the router restarts, the
  backend is scaled to 0 after `idle_timeout_seconds` without traffic.
* **Endpoints.** `neuron_server` only implements `POST /v1/chat/completions`, so `/v1/completions` is not proxied
  (add it to `proxy_paths` once the backend supports it). Backend `/metrics` is not exposed through the router.
* **No NAT.** The awsvpc task has no public IP, so its only egress is the S3 gateway endpoint. Image pulls and logs go
  through the host's public IP. Any other egress the task needs requires a NAT gateway or more VPC endpoints.
* **Startup health.** ECS caps `startPeriod` at 300 s, so `healthcheck.sh` reports healthy while the startup marker
  is younger than `backend_startup_grace_seconds` (1200 s). A load that hangs is only detected after that.

## Unverified (no AWS calls were made)

* The SSM parameter name `/aws/service/ecs/optimized-ami/amazon-linux-2023/neuron/recommended/image_id`, from the
  ECS docs' list of AL2023 Neuron AMI parameters. Check it with
  `aws ssm get-parameter --name ... --query Parameter.Value`. The AL2 variant was `.../amazon-linux-2/inf/recommended`.
* That `ecs.service` on that AMI starts after cloud-init, so the instance-store mount exists before the first task is
  placed, and that the instance store appears as `/dev/disk/by-id/nvme-Amazon_EC2_NVMe_Instance_Storage_*`.
* That `aws-neuronx-runtime-lib` installs `/opt/aws/neuron/include/nrt/*.h`, which the build stage checks and fails
  without, and that Ubuntu 22.04 with g++-12 compiles `build.sh srv` (boost and pcre2 from apt).
* The s5cmd release asset name `s5cmd_<ver>_Linux-64bit.tar.gz` (set `S5CMD_SHA256` to pin it), and that the `cp`
  flags `-n -s --concurrency --part-size` behave as documented.
* Whether IPC_LOCK, a memlock ulimit or `/dev/shm` size are actually required by libnrt in a container. They are set
  defensively.
* The ENI limit of trn1.2xlarge leaves room for the awsvpc task ENI (it should, at one task).
* The current `cloudflared` ingress configuration of the existing tunnel.
