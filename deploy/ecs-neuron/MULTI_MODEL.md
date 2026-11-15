# Multi-model serving: N models, router-controlled activation, spot with on-demand fallback

Status: designed, implemented and tested offline (branch `multi-model`). Nothing has been applied. The pipeline Plan
is the first real check.

## 1. Design

```
                 router (Fargate, 1 task)  MODELS_JSON: [{name, ids, service, asg_name, capacity_provider,
                        |                                fallback_asg_name, fallback_capacity_provider, ...}, ...]
       request.model -> model entry (unknown -> 404 model_not_found)
                        |
     +------------------+---------------------+
     v                                        v
 behemoth-backend          (ornith)       behemoth-backend-cydonia     (cydonia)
   strategy: behemoth-trn-ornith-spot       strategy: behemoth-trn-cydonia-spot
   spot ASG   behemoth-trn-ornith-spot-*    spot ASG   behemoth-trn-cydonia-spot-*
   od ASG     behemoth-trn-spot-* (legacy,  od ASG     behemoth-trn-cydonia-od-*  (warm pool)
              warm pool) = fallback CP      = fallback CP behemoth-trn-cydonia-od
              behemoth-trn-spot
```

Each entry of `var.models` gets its own resources:

| resource | address | name: legacy entry (`ornith`) | name: new entry `<k>` |
|---|---|---|---|
| launch template | `aws_launch_template.backend[k]` | `behemoth-trn-*` | `behemoth-trn-<k>-*` |
| on-demand ASG (on-demand model, or spot fallback) | `aws_autoscaling_group.ondemand[k]` | `behemoth-trn-spot-*` (historic name) | `behemoth-trn-<k>-od-*` |
| on-demand capacity provider | `aws_ecs_capacity_provider.ondemand[k]` | `behemoth-trn-spot` | `behemoth-trn-<k>-od` |
| spot ASG (mixed instances, price-capacity-optimized) | `aws_autoscaling_group.spot[k]` | `behemoth-trn-ornith-spot-*` | `behemoth-trn-<k>-spot-*` |
| spot capacity provider | `aws_ecs_capacity_provider.spot[k]` | `behemoth-trn-ornith-spot` | `behemoth-trn-<k>-spot` |
| task definition | `aws_ecs_task_definition.backend[k]` | family `behemoth-backend` | `behemoth-backend-<k>` |
| ECS service (desired 0) | `aws_ecs_service.backend[k]` | `behemoth-backend` | `behemoth-backend-<k>` |
| S3 read policy (on the shared task role) | `aws_iam_role_policy.backend_task[k]` | `s3-read-models` | `s3-read-models-<k>` |
| log stream prefix (shared log group) | | `backend` | `backend-<k>` |

Shared resources: cluster, instance role/profile, execution role, backend task role, security groups, log group,
router. The task role is shared because the pipeline's apply role may pass only the fixed stack role names
(`pipeline/iam.tf` `stack_role_arns`). Each model gets its own inline policy scoped to its own prefixes. As a result,
any model's task can read every model's prefixes. A role per model would need `stack_role_arns` widened in the
pipeline stack.

The spot and on-demand ASGs are separate resources keyed by purchase type. Turning spot on or off for a model creates
or destroys its spot ASG. An existing ASG is never converted between launch template and mixed instances policy,
which would also need its warm pool deleted in the same call.

### Router

* **Routing.** The `model` field picks the entry (any of its `model_ids`). With no `model` field, the request goes to
  `default_model`. An unknown model gets `404 {"error": {"type": "model_not_found", ...}}` and wakes nothing.
  `/v1/models` lists every configured id, default model first, active or not, and never calls AWS.
* **Activation.** The router wakes only that model's service (`UpdateService desiredCount 1`) and scales its active ASG
  to 1 directly. It does not wait for the capacity provider's scale-out. Each model has its own single-flight startup.
  Everything the router did before is kept, per model: keepalive/wait/reject cold modes (`cold_nonstream_mode`,
  configurable per model), the stream keepalive, `max_wait` (per model), DRAINING warm-pool reactivation, the
  scale-in-protection release, DynamoDB API keys, and per-caller logging (the log line now names the model entry too).
* **DRAINING reactivation** is scoped to the instances of the model's own ASGs. Another model's instance may be
  draining on purpose.
* **Idle.** Each model scales to 0 after its own `idle_timeout_seconds`, then its ASGs go to 0. `min_count = 1` keeps a
  model running: it is never idle-stopped or evicted.
* **Cap.** At most `max_active_models` (default 2) models can be starting or running at once. "Running" includes a
  model the router finds at `desiredCount > 0` after its own restart. The cap check and the slot reservation take one
  lock, so concurrent cold requests cannot overshoot. When the cap is hit:
  * `cap_policy = "evict"` (default): the router stops the longest-idle active model and starts the requested one. A
    model is evictable only if it has nothing in flight (waiting requests count as in flight), is not starting, has
    `min_count` 0, and has been idle for at least `cap_min_idle_seconds` (30). If none is evictable, the request
    gets the reject answer below.
  * `cap_policy = "reject"`: `503 {"error": {"type": "capacity_exceeded"}}` with `Retry-After` set to the shortest time
    until an active model's idle timeout. It is at least 5 s and at most `RETRY_AFTER_S` (120).

  The evicted model's instance terminates in the background, about 1 min after the stop (ECS task stop, then ASG 0).
  The two instances can overlap for that minute, which the 32 vCPU spot quota covers (section 4).
* **Health.** `/health` keeps `backend` (the default model's state), `inflight` and `idle_s`. It adds `active`,
  `max_active` and `models: {name: {state, inflight, idle_s, on_demand_fallback}}`.
* **Compatibility.** Without `MODELS_JSON`, the router builds one model from `ECS_SERVICE` / `BACKEND_ASG` /
  `MODEL_IDS`, exactly as before.

## 2. Variables

`models` (map, key = model name `^[a-z0-9][a-z0-9-]{0,19}$`). Per entry:

| field | default | notes |
|---|---|---|
| `model_s3_prefix`, `graphs_s3_prefix` | required | |
| `prepared_s3_prefix` | `""` | Devstral/Cydonia prepared shards |
| `model_ids` | required | OpenAI `model` values; unique across models |
| `model_name` | `model_ids[0]` | `BEHEMOTH_MODEL_NAME` |
| `extra_server_args`, `extra_env` | `""`, `{}` | per model (Ornith STREAM_LOAD=0 vs Cydonia staging) |
| `image` | stack backend image (pinned digest) | full image ref, e.g. to pin an older engine for one model |
| `legacy_names` | false | only on key `ornith`: keeps the pre-refactor names (moved blocks) |
| `instance_types` | `backend_instance_types` | first = launch template type, all = spot overrides |
| `use_spot` | `backend_use_spot` | |
| `spot_fallback` | `backend_spot_fallback` (true) | on-demand fallback ASG + capacity provider |
| `spot_fallback_after_seconds` | `spot_fallback_after_seconds` (240) | |
| `spot_max_price` | `spot_max_price` (`""` = on-demand cap) | |
| `warm_pool` | on-demand model: `backend_warm_pool`; spot model's fallback: `backend_fallback_warm_pool` (true) | |
| `cpu`, `memory_reservation_mib`, `memory_hard_limit_mib`, `shm_size_mib`, `startup_grace_seconds` | the `backend_*` globals | |
| `idle_timeout_seconds`, `max_wait_seconds`, `cold_nonstream_mode` | router globals | |
| `min_count` | 0 | 0 or 1 |

Globals: `default_model` (`""` = `ornith` if present, else the first key), `max_active_models` (2), `cap_policy`
(`evict`), `cap_min_idle_seconds` (30), `backend_spot_fallback` (true), `spot_fallback_after_seconds` (240),
`backend_fallback_warm_pool` (true), `router_lost_retries` (1), `router_stream_loss_error` (true).

`models = {}` (the default) builds one model, `ornith` with `legacy_names`, from the old single-model variables
(`model_s3_prefix`, `model_ids`, `backend_extra_env`, `backend_use_spot`, ...). The single-model stack is the special
case.

There is no `max_count`. The router routes each model to one task, and a TP=2 model holds the whole Neuron device of
a trn1.2xlarge, so each ASG has `max_size = 1`.

## 3. The tfvars (in git: `models.auto.tfvars`, replacing `ornith.auto.tfvars`)

The Ornith entry reproduces exactly what runs today: the S3 `terraform.tfvars` values (`backend_shm_size_mib = 6144`,
cpu/memory defaults) plus `ornith.auto.tfvars`. The offline plan confirms it: the Ornith task definition is a pure
move with no diff.

```hcl
default_model     = "ornith"
max_active_models = 2
cap_policy        = "evict"

models = {
  ornith = {
    legacy_names       = true
    model_s3_prefix    = "models/ornith-reap20-v3-fp8"
    graphs_s3_prefix   = "graphs/ornith-reap20-v3-kv8-bound"
    prepared_s3_prefix = ""
    model_ids          = ["ornith-1.5-35b-a3b-reap20-v3"]
    model_name         = "ornith-1.5-35b-a3b-reap20-v3"
    shm_size_mib       = 6144
    extra_server_args  = "--prefill-chunk 256 --max-batch-requests 8 --prefix-cache-gb 4"
    extra_env = {
      STREAM_LOAD                            = "0"
      BEHEMOTH_NEURON_QWEN35_PREFIX_CACHE    = "1"
      BEHEMOTH_NEURON_QWEN35_PREFIX_ALIGN    = "1"
      BEHEMOTH_NEURON_QWEN35_PREFIX_INTERVAL = "2048"
      BEHEMOTH_NEURON_QWEN35_PREFIX_STATE_GB = "4"
      BEHEMOTH_NEURON_PREFIX_HOST_GB         = "4"
    }
    use_spot      = true
    spot_fallback = true
  }
  cydonia = {
    model_s3_prefix    = "models/cydonia-24b-fp8"
    graphs_s3_prefix   = "graphs/cydonia-24b-fp8-kv"
    prepared_s3_prefix = "prepared/cydonia-24b-fp8-tp2-wt-mlptiled"
    model_ids          = ["cydonia-24b"]
    shm_size_mib       = 6144
    extra_server_args  = "--mixed-prefill-tokens 256 --prefill-interval 2"
    extra_env = {
      STREAM_LOAD                    = "1"
      STAGE_DIR                      = "/dev/shm/layers"
      STREAM_UNLINK                  = "1"
      STREAM_AHEAD                   = "8"
      BEHEMOTH_NEURON_EMBED_RESIDENT = "1"
    }
    use_spot      = true
    spot_fallback = true
  }
}
```

Nothing has to change in the S3 `terraform.tfvars`. Its single-model values (Devstral prefixes, `backend_use_spot =
false`) are ignored once `models` is set, because the per-entry `use_spot` wins. Spot settings at a glance:

* To keep a model on-demand only: `use_spot = false` (the warm pool stays).
* Spot without a fallback: `spot_fallback = false`. A spot shortage then leaves that model down until capacity
  returns: requests wait up to `max_wait_seconds`, then get 503.

Cydonia's model id is `cydonia-24b`. It is the id the old config, the `terraform.tfvars.example`, the entrypoint
default and the router default used, and the one the beaverai client lists. The "Cgg507/" prefix is the client side.

## 4. Spot

Account and region facts, read-only, checked 2026-11-15:

| | |
|---|---|
| `All Trn Spot Instance Requests` (L-6B0D517C) | **32 vCPU** = 4 x trn1.2xlarge. Cap 2 x 8 = 16 vCPU, plus 8 during an eviction overlap = 24, within the quota. Not a blocker. |
| `Running On-Demand Trn instances` (L-2C3B7624) | 64 vCPU (fallback: 2 x 8 = 16) |
| trn1.2xlarge on-demand | $1.34375/h |
| trn1.2xlarge spot, us-east-1f (use1-az5), last 30 days | $0.185-0.246/h, $0.189 now (about 86% off) |
| trn1.2xlarge spot, us-east-1d (use1-az6) | $0.165-0.186/h |
| spot placement score, trn1.2xlarge, 1 or 2 instances, single AZ | **1 of 10 in both az5 and az6**. This is a weak capacity signal: spot launches may often fail. That is why the on-demand fallback defaults to on. |
| trn1.2xlarge offered in | us-east-1d and us-east-1f. The stack's only backend subnet is in 1f. The VPC also has subnet-04c4ce5b11c4d1de2 in 1d, which is not added here: check its routes, NAT and S3 endpoint first. Adding it widens the spot pools. |
| other types with >= 2 NeuronCores (listed only, not added) | inf2.xlarge: 4 vCPU, 16 GiB RAM, too little host RAM. inf2.8xlarge: 32 vCPU, 128 GiB, 1 Inferentia2 chip / 2 NeuronCore-v2, 32 GiB device memory, spot $0.32-0.58/h, in 1b/c/d/f. It would need NEFFs compiled for inf2 (untested) and uses the separate Inf quotas (spot 32, on-demand 4 vCPU, which is below one inf2.8xlarge). trn1.32xlarge / trn1n.32xlarge are excluded per the brief. |

How the spot path works:

* **Spot ASG.** A mixed instances policy, `on_demand_percentage_above_base_capacity = 0`, `price-capacity-optimized`,
  `spot_max_price` (empty = capped at the on-demand price), overrides = `instance_types`. There is no warm pool
  (warm pools do not support spot or mixed instances policies) and no capacity rebalance (with `max_size 1`).
* **Circuit breaker.** The deployment circuit breaker stays disabled with `rollback = false`, so a spot capacity
  shortage never rolls the service back. The service just keeps a PROVISIONING task.
* **Fallback.** During a startup, the router falls back when either:
  * the spot ASG reports a Failed or Cancelled scaling activity since the startup began (for example "We currently do
    not have sufficient trn1.2xlarge capacity"), or
  * `spot_fallback_after_seconds` (240) have passed without any Pending or InService spot instance.

  Then it calls `UpdateService` with the strategy set to the model's on-demand provider and `forceNewDeployment`, sets
  the spot ASG to 0 and the on-demand ASG to 1. A warm-pool instance usually starts within about 1 min. On the next
  stop (idle or evict), the service goes back to the spot provider. A Terraform apply while a model is on its fallback
  also resets it to spot: the Plan shows that drift, and if the model is running its task is replaced.
* **Boundary caveat.** Reading the scaling activities needs `autoscaling:DescribeScalingActivities` in the workload
  permissions boundary. It is added in `pipeline/iam.tf`, but it only takes effect when the pipeline stack is applied
  (separately, by you). Until then the router logs one warning and falls back on the 240 s timer only. Everything
  else works.
* **Interruption (2-minute notice).** `ECS_ENABLE_SPOT_INSTANCE_DRAINING=true` makes ECS set the instance to DRAINING
  and stop the task: SIGTERM, then `stopTimeout` 60 s. The service, still at desired 1, places a replacement task. The
  spot ASG, still at desired 1, launches a replacement instance. If spot has no capacity, the next startup falls back
  to on-demand. What the router does and what the client sees:
  * **Before any byte of the answer** (connect refused or reset, health failing): the router marks the model lost,
    re-wakes it, and re-sends the request up to `router_lost_retries` (1) times. While it waits, the client is held
    like a cold start:
    * streams get keepalive chunks;
    * non-streaming requests in `keepalive` mode get a 200 with whitespace keepalives;
    * `reject` mode answers 503 + Retry-After at once and re-wakes in the background;
    * `wait` mode holds silently, which can hit Cloudflare's 100 s limit.

    With `router_lost_retries = 0`: 503 + Retry-After.
  * **Mid-stream (SSE)**: the client keeps the tokens already streamed, then gets
    `data: {"error": {"message": "backend lost mid-response (spot interruption or crash); the response is incomplete, retry the request", "type": "backend_unavailable"}}`
    and `data: [DONE]` (`router_stream_loss_error`). Before this change it got a bare EOF with no `[DONE]`. The request
    is never re-sent once bytes went out: tokens would duplicate. The next request re-wakes the model.
  * **Mid-body (non-streaming)**: neuron_server sends the JSON body in one piece at the end, so a loss almost always
    falls in the "before any byte" case. If it happens mid-body, the client gets a truncated body: a JSON parse error.

### Cold-start estimates

Measured today: Ornith on the on-demand warm pool took **204-214 s** from scale-up to `backend ready` (router logs,
2026-11-15, 8 starts; one took 112 s). For Devstral/Cydonia the README gives 149 s from a fresh launch and 107 s
with the warm pool. Spot removes the warm pool and adds spot fulfilment.

| path | Ornith (28.9 GB, STREAM_LOAD=0) | Cydonia (25 GB model + 22 GB prepared, streamed) |
|---|---|---|
| spot, capacity available (fresh launch, boot, ECS join, image pull, S3 pull, load) | ~4-5.5 min (estimate: +40-60 s launch/boot over the warm-pool 3.5 min, +0-30 s fulfilment) | ~2.5-3.5 min |
| spot launch fails fast (Failed activity) -> on-demand warm pool | ~4.5-5 min | ~2.5-3 min |
| no activity signal (boundary not yet updated) -> 240 s timer -> warm pool | ~7.5-8 min | ~6 min |
| on-demand warm pool (spot off) | ~3.5 min (measured) | ~2 min |

All of these are within `max_wait_seconds = 900`. Cold requests are held with keepalives, as before. The beaverai
client's own timeout must exceed these.

## 5. Cydonia artifacts and engine compatibility (read-only check)

* **Artifacts in `s3://behemoth-neuron-models-912935854432`:**
  * `models/cydonia-24b-fp8/`: 48 objects, 24.95 GB. `config.json` (LlamaForCausalLM, from Mistral), tokenizer
    files, chat template, `model-embed.safetensors` (2.68 GB), 40 `model-layer-*.safetensors`.
  * `graphs/cydonia-24b-fp8-kv/`: 304 objects, 50.6 MB, 37 buckets (dec*, dec8s_*, head*, pre256_*, pre64_32).
  * `prepared/cydonia-24b-fp8-tp2-wt-mlptiled/`: `layout.json` + 40 x 556 MB `model-layer-*.bin`, 22.25 GB.
* **Verdict: likely works, nothing to rebuild.** Cydonia has no `prepared.json`, so `NeuronModel` dispatches it to the
  Llama/Mistral path, not Qwen3.5 or GLM. The recent Ornith work (qwen35_model.cpp, the GDN/prefix-cache paths)
  is behind that dispatch. The Llama loader only gained optional meta keys with safe defaults (`shared_table`,
  `argmax`, `build`). Geometry is still checked only against `config.json`, and it matches. `layout.json` (format 1,
  tp 2, wt, mlp_tiled, geometry) matches what the engine requires exactly. The scheduler changes are additive. The
  entrypoint is unchanged since the Devstral image a74e679.
* **Risks:**
  * This engine has not been run on a device with a Llama-path model since those changes.
  * The dec8s_* shared-table buckets (added 2026-11-10) are now used for single-request speculation verify, a kernel
    never served in production. If output looks wrong, set `BEHEMOTH_NEURON_SPEC_PAGE_COST = "0.01"` in Cydonia's
    `extra_env` (the old default). Or pin an older image for Cydonia alone with the per-model `image` field.
  * `layout.json` carries no model hash, so the engine cannot tell Cydonia's prepared dir from Devstral's. ETags and
    timestamps show `prepared/cydonia-*` was written from `models/cydonia-24b-fp8`, but that is not provable from
    the files.
  * Max context is 32k (largest decode bucket); keep `--prefill-chunk` at or below 256.

First functional check after apply: one request with `"model": "cydonia-24b"`. Look for `backend ready` in the router
log and for the load lines in `/ecs/behemoth/backend` stream prefix `backend-cydonia`.

## 6. Migration (moved blocks) and the expected pipeline Plan

`models.tf` moves the single-model resources to the `ornith` instances:

```
aws_launch_template.backend          -> aws_launch_template.backend["ornith"]
aws_autoscaling_group.backend        -> aws_autoscaling_group.ondemand["ornith"]   (it is on-demand + warm pool today)
aws_ecs_capacity_provider.trn_spot   -> aws_ecs_capacity_provider.ondemand["ornith"]
aws_ecs_task_definition.backend      -> aws_ecs_task_definition.backend["ornith"]
aws_ecs_service.backend              -> aws_ecs_service.backend["ornith"]
aws_iam_role_policy.backend_task     -> aws_iam_role_policy.backend_task["ornith"]
```

The moved blocks assume the pre-refactor ASG was on-demand, which is what production runs. A state that still had
`backend_use_spot = true` would move its spot ASG into the `ondemand` slot. Such a state needs a manual
`terraform state mv` to `aws_autoscaling_group.spot["ornith"]` instead.

How this was checked: `terraform plan -lock=false -refresh=false` against a local copy of the S3 state (local
backend override, so the remote state and lock were never touched; nothing applied). Inputs: the S3 `terraform.tfvars`,
this branch's `models.auto.tfvars`, the deployed image digests, and the pipeline's
`TF_VAR_role_permissions_boundary_arn`.

**`Plan: 11 to add, 4 to change, 1 to destroy.`**

* moved, **no change**: the Ornith launch template, on-demand ASG (+ warm pool), on-demand capacity provider
  `behemoth-trn-spot`, task definition `behemoth-backend:23`, S3 policy `s3-read-models`
* create (11):
  * Ornith: spot ASG `behemoth-trn-ornith-spot-*` + capacity provider `behemoth-trn-ornith-spot`
  * Cydonia: launch template, spot ASG + capacity provider, on-demand ASG `behemoth-trn-cydonia-od-*` (warm pool) +
    capacity provider, task definition `behemoth-backend-cydonia`, service `behemoth-backend-cydonia` (desired 0),
    policy `s3-read-models-cydonia`
  * the new router task definition revision
* update in place (4):
  * `aws_ecs_service.backend["ornith"]`: strategy `behemoth-trn-spot` -> `behemoth-trn-ornith-spot`, and
    `force_new_deployment = true`. The provider requires the latter to change a strategy.
  * `aws_ecs_cluster_capacity_providers.this`: +3 providers
  * `aws_iam_role_policy.router_task`: all services and ASGs, plus `DescribeScalingActivities`
  * `aws_ecs_service.router`: the new task definition
* replace/destroy (1): the router task definition (new revision; the old revision is deregistered)
* **No replacement of any Ornith resource.**

The real pipeline Plan should match this count:

* The backend image tag comes from the last commit touching the backend paths. This branch does not touch them, so
  the tag stays `0d1d8ca` and the Ornith task definition stays unchanged.
* The router tag changes because `router/` changed. The router is rebuilt, and its unit tests run in the image build.
* A refresh can add drift that the offline plan (`-refresh=false`) does not show.

With `models = {}` (old single-model tfvars), the plan is moves plus `force_new_deployment` and the router changes
(1 add / 3 change / 1 destroy), which confirms the special case.

Apply side effects:

* Cydonia's new warm pool launches one on-demand trn1.2xlarge to initialise it and then stops it: a few minutes at
  $1.34/h.
* The router task restarts once, about 1 min. Requests in flight through the router are dropped, and cold-start
  state is relearned.
* If Ornith is running when the apply reaches its service, the forced deployment replaces its task onto spot, which
  costs one cold start. **Approve while Ornith is idle** (`/health`: `models.ornith.state` = `stopped`). While the
  old router is still up during the apply, a request may start Ornith on spot through the capacity provider, slowly;
  the new router cleans up both ASGs on the next stop.

## 7. Cost

* Active hours: spot about $0.19/h per model vs $1.34/h on-demand. Fallback hours are billed on-demand.
* Idle: $0 per model, plus the stopped warm-pool root volumes: about $8/month each, Ornith's existing one + Cydonia's
  new one = about $16/month.
* With the cap of 2, the worst case is 2 x $1.34/h (both on fallback), or about $0.38/h on spot.
* Router (Fargate 0.25 vCPU) unchanged.

## 8. Risks

1. **Spot capacity.** Placement score 1/10: spot launches may often fail. The fallback covers it, at on-demand price,
   at the cost of about +0.5 min of cold start (or +4 min until the boundary update).
2. **Boundary.** `DescribeScalingActivities` stays denied until the pipeline stack is applied. That only slows the
   fallback (section 4).
3. **Cydonia on the current engine.** Untested on device, and the new speculation verify kernel is involved
   (section 5).
4. **Shared task role.** Each model's task can read every model's S3 prefixes (section 1).
5. **Router state is in memory.** It is still one router task (`desired 1`, min healthy 0). After a router restart,
   models found at `desiredCount > 0` count against the cap and are idle-stopped after their timeout, as before.
6. **The Terraform code cannot simply be reverted** once applied. See the rollback recipe.
7. **The legacy name `behemoth-trn-spot`** is now Ornith's on-demand provider. It was already on-demand before this
   change; the name is historic.

## 9. Rollback

Do these through the pipeline. All are config-only and keep the moved Ornith resources.

* **Spot off (everything back on-demand, warm pools):** set `use_spot = false` in both entries. The spot ASGs and
  providers are destroyed, and the services move back to their on-demand providers (forced deployment, harmless at
  desired 0). Cydonia keeps its on-demand ASG (with a warm pool by default).
* **Cydonia off:** delete the `cydonia` entry. Its 8 resources are destroyed, and Ornith is untouched.
* **Back to the single-model setup:** replace `models.auto.tfvars` with the old `ornith.auto.tfvars`:
  `git show 0d1d8ca:deploy/ecs-neuron/ornith.auto.tfvars`. `models = {}` then means Ornith only. Spot follows the S3
  `backend_use_spot = false`, so Ornith's spot ASG and provider and all Cydonia resources are destroyed.
* **Reverting the code itself after an apply:** the old code has no `for_each` addresses. Without reverse moves,
  Terraform would destroy and recreate Ornith's ASG, provider and service, and the provider name would clash. First
  go back to the single-model config (previous bullet) and apply. Then add reverse `moved` blocks to the reverted
  code:

  ```hcl
  moved { from = aws_launch_template.backend["ornith"]        to = aws_launch_template.backend }
  moved { from = aws_autoscaling_group.ondemand["ornith"]     to = aws_autoscaling_group.backend }
  moved { from = aws_ecs_capacity_provider.ondemand["ornith"] to = aws_ecs_capacity_provider.trn_spot }
  moved { from = aws_ecs_task_definition.backend["ornith"]    to = aws_ecs_task_definition.backend }
  moved { from = aws_ecs_service.backend["ornith"]            to = aws_ecs_service.backend }
  moved { from = aws_iam_role_policy.backend_task["ornith"]   to = aws_iam_role_policy.backend_task }
  ```

  The alternative is `terraform state mv` with the pipeline paused.

## 10. Tests

`cd deploy/ecs-neuron/router && python3 -m pytest -q`: 49 tests. The 28 existing single-model tests are unchanged.
The new ones cover:

* `/v1/models` lists all ids;
* per-model routing, including a second id of the same model, and several active at once;
* unknown model -> 404;
* cap reject, cap evict (longest idle), evict never takes a busy or recently used model, the cap counts models found
  running after a router restart, concurrent cold requests respect the cap;
* per-model idle scale-down, `min_count`;
* spot fallback on a failed launch and back to spot on stop, fallback by timeout, fallback by timeout when activities
  are denied, no fallback while a spot instance boots;
* spot interruption: re-wake + retry, a held non-stream retry with keepalives, reject mode 503, `lost_retries = 0`
  -> 503, mid-stream loss -> SSE error + `[DONE]`;
* DRAINING reactivation scoped to the model's ASG;
* `MODELS_JSON` parsing and the config checks.

`terraform fmt -check -recursive` and `terraform validate` (`init -backend=false`) are clean for both
`deploy/ecs-neuron` and `deploy/ecs-neuron/pipeline`.
