# Repository security audit — 2026-11-09

Audited revision: `a74e679a15532e193172f0ab427daece4b0fb29e`, branch
`rocm-to-neff`, pulled with `git pull --ff-only` from `origin`. The previously
untracked binary-decoder experiment was preserved. Production source was not
modified by this audit. This report and the local reproducers are new files.

**Result: six high-severity and four medium-severity findings.** Do not expose
the inference API to untrusted clients until the parser crash, vulnerable HTTP
dependency, authentication, and admission limits are addressed. This is a
repository audit, not a claim that a live deployment has been penetrated or
that every possible defect has been found.

## Scope and method

Reviewed the HTTP servers and ECS router; structured output/tool request
parsing; request scheduling; safetensors and prepared-model loading; Neuron
host integration; Docker, shell download/startup paths, Terraform networking,
IAM and deployment pipeline; dependency declarations and recognizable secrets.
GPU kernels were considered at their host/input-validation boundary; no
GPU/Neuron device was available for malformed-input testing on hardware.

- Enumerated 626 tracked files. Secret-prefix/private-key scanning covered all
  of them and 1,387,994 diff lines in all locally available branch histories.
  No matching credentials found. This is a pattern scan, not proof that no
  unrecognized or encoded secrets exist.
- Bandit scanned 8,277 Python lines and returned 12 medium/high-confidence,
  medium/high-severity items. Manually triaged them; a Jinja autoescape warning
  in a tokenizer parity test is not a browser XSS finding, and the tokenizer
  loader uses `local_files_only=True`. These are not counted as vulnerabilities.
- `pip-audit` resolved 17 router dependencies. Eight advisories affect pinned
  `aiohttp==3.14.3`; all have a reported fix in `3.14.4`. OSV was queried
  independently and the advisories reviewed for applicability.
- Existing router tests: **21 passed**. Added local fake-AWS router probes and
  compiled the actual C++ parsers for adversarial inputs; no cloud APIs called
  by the probes. Native safetensors reproduction used ASan/UBSan. The grammar
  crash was reproduced with sanitizers and an ordinary optimized build.
- Existing native `core_correctness` and `structured_outputs` suites both
  passed under ASan/UBSan (179.47 seconds total; leak detection disabled).
- See `security-audit-2026-11-09-evidence.json` for machine-readable evidence
  and final native regression-test results.

No running AWS/ECS resources, real public endpoint, container image contents,
live IAM policy simulator, external Cloudflare Access configuration, or private
model storage was inspected. There are no configured cloud credentials in this
workspace. Terraform conclusions concern the checked-in policy/configuration;
SCPs and externally imposed permissions boundaries may change actual exposure.
Dependency results cover the resolved Python router environment, not a complete
SBOM of CUDA, ROCm, Neuron, OS packages, or vendored C++ components.

## Findings

| ID | Severity | Finding | Evidence |
|---|---|---|---|
| SEC-01 | High | Client GBNF nesting crashes the native server | Reproduced on production parser, optimized and sanitized builds |
| SEC-02 | High | Vulnerable aiohttp version pinned in router | pip-audit + OSV |
| SEC-03 | High | Public routing can fail open and provision compute without authentication | Reproduced with fake AWS clients; deployment-dependent exposure |
| SEC-04 | High | Unbounded admission and oversized retained request bodies | Scheduler and local router reproductions |
| SEC-05 | High | Scoped Terraform apply role permits privilege escalation | Static IAM permission chain |
| SEC-06 | High | Native model loaders trust unchecked offsets/lengths | Safetensors sanitizer reproduction; GLM static validation review |
| SEC-07 | Medium | Quantizer shard index can escape model directory | Reproduced with synthetic sibling safetensors file |
| SEC-08 | Medium | Internet-facing ALB permits plaintext inference and bearer credentials | Static configuration; conditional on ALB without certificate |
| SEC-09 | Medium | Standalone GLM/CUDA server can be stalled by one idle connection | Static blocking-I/O review |
| SEC-10 | Medium | Mutable deployment tags weaken saved-plan approval | Static ECR, pipeline and task-definition review |

### SEC-01 — Unbounded recursive GBNF parsing

**Locations:** `runtime/src/gbnf.cpp:267–274`,
`runtime/src/protocol.cpp:57–65`, `runtime/src/server.cpp:350–353`.

The parser increments `group_depth` and recursively calls `ParseAlternates`
without checking nesting first. The subsequent rule-chain validation cannot
protect the parse itself. The API allows up to 1 MiB of grammar text.

**Reproduction:** a grammar consisting of `root ::= `, 50,000 opening
parentheses, a one-character literal, and 50,000 closing parentheses is about
100 KB. An optimized build of the actual parser exits on SIGSEGV (`-11`). A
10,000-level variant produced an ASan stack-overflow report. A 100-level
control completed normally. Both fit within the grammar and HTTP body limits.

**Impact:** a client allowed to POST a grammar can crash the entire native
serving process, terminating other requests. With SEC-03, this can be reachable
without credentials. Authentication alone does not protect against a malicious
authorized client. No code execution was demonstrated.

**Fix:** enforce a small parse-time nesting bound before every recursive
descent, or use an explicit stack. Apply an overall parse/expansion work budget
as well. Add an HTTP regression test that expects 400 and verifies another
request still succeeds; test optimized and sanitizer builds.

### SEC-02 — Router HTTP dependency has known DoS advisories

**Location:** `deploy/ecs-neuron/router/requirements.txt:1` pins `aiohttp==3.14.3`.

OSV and pip-audit identify eight advisories, all fixed in `3.14.4`. The two
particularly relevant server-side advisories are:

- [GHSA-2g87-pp6c-29x5](https://osv.dev/vulnerability/GHSA-2g87-pp6c-29x5):
  excessive/unbounded memory and CPU on attacker-controlled requests.
- [GHSA-5wr8-gcqf-gqf3](https://osv.dev/vulnerability/GHSA-5wr8-gcqf-gqf3):
  no initial request/header timeout; idle connections can persist indefinitely.

Do not interpret all eight as demonstrated exploits of this application.
The multipart descriptor advisory requires `Request.post()`, which this router
does not call. Scheme spoofing requires security decisions based on
`Request.scheme`, absent here. Request smuggling is conditional on the
pure-Python parser; the standard wheel normally uses the C parser. Cookie
advisories relate to client-side behavior and need further conditions.

**Fix:** upgrade to at least `3.14.4`, run router tests, rebuild and deploy the
image, and scan the actual image/transitive lock. Edge timeouts reduce exposure
but do not replace the dependency update.

### SEC-03 — Missing API key is treated as public access; invalid requests wake compute

**Locations:** `deploy/ecs-neuron/router/router.py:328–331,343–356,365–378`,
`deploy/ecs-neuron/router.tf:35`, `deploy/ecs-neuron/variables.tf:340–343`.

The API-key secret is optional and an empty key causes `_authorized()` to return
true. The router supports public Cloudflare tunnel and ALB ingress, but this
repository does not require a separate identity gate. Invalid JSON is caught
and treated as a non-streaming inference request, reaching `ensure()` and the
ECS/ASG scale-out path before backend validation.

**Reproduction:** POST `not-json` without Authorization using the default
empty key and cold `reject` mode: HTTP 503, fake ECS `desiredCount` updates
`[1]`. With a configured key and the same unauthenticated request: HTTP 401,
no AWS update. This is a fail-open configuration, not a bypass of a correctly
configured bearer key.

**Impact:** where no external authentication exists, strangers can consume
inference, trigger paid infrastructure starts, and keep infrastructure active.
Even with authentication, malformed requests should not trigger provisioning.

**Fix:** fail startup for public exposure without an explicit authentication
mechanism; make anonymous operation an explicit opt-in. Validate JSON, route,
model and basic request shape before waking compute. Add identity-based rate
limits/quotas. Verify Cloudflare Access separately if used; a tunnel token
authenticates the connector, not the inference caller.

### SEC-04 — Batch size is not an admission limit

**Locations:** `deploy/ecs-neuron/router/router.py:347–356,453`,
`runtime/src/scheduler.cpp:12–31`, `runtime/src/server.cpp:439`,
`neuron/cpp/neuron_engine.cpp:578–590`.

The router buffers up to 64 MiB per request and retains it while waiting for a
backend. There is no cap on active/waiting requests. The backend accepts at
most 4 MiB. The scheduler's `max_requests` limits execution batches, but
`submit()` still copies every new prompt into an unbounded request map.

**Reproduction:** a 5,242,911-byte body was admitted by the router, caused
fake backend scale-out, and only then received 413 from a backend with the
native 4-MiB limit. The actual scheduler accepted 1,000 pending requests with
`max_requests=1`; no model or accelerator is needed to demonstrate this.

**Impact:** concurrent requests can exhaust router/backend RAM or occupy the
queue for long periods. This remains relevant with a shared API key or a
malicious authorized tenant. No high-volume test was sent to a live endpoint.

**Fix:** reject oversized input before buffering/waking; align limits across
layers; bound queued request count, bytes, prompt tokens, compilation work,
and wait duration. Return 429/503 at capacity, cancel disconnected waiters,
and implement per-identity budgets. Bound grammar state growth during active
requests, not only when returning a grammar to the cache.

### SEC-05 — “Scoped” apply role can grant administrator-like privileges

**Location:** `deploy/ecs-neuron/pipeline/iam.tf:261–298`, also the regional
EC2/ECS write statements at `195–232`.

The apply role can create/update non-pipeline `${name}-*` roles, write arbitrary
inline policies (`iam:PutRolePolicy`), and pass those roles. It can also create
instance profiles and launch EC2 instances/update ECS services. No mandatory
permissions boundary constrains the roles it manages. Restricting managed
policy attachments does not restrict inline policies; denying edits to the
pipeline roles does not stop escalation through another stack role.

**Impact:** an actor already able to execute code as the apply role can attach
an unrestricted inline policy to an eligible role and arrange for an EC2/ECS
workload to use it, escaping the intended stack-only scope. This is not an
unauthenticated internet attack. The manual approval stage helps gate normal
changes but does not enforce an IAM privilege ceiling after compromise.
Actual account SCPs/external boundaries were not available to test.

**Fix:** provision application roles in a separate privileged bootstrap, or
require an approved permissions boundary for all managed roles and prohibit
boundary removal/change. Restrict `iam:PassRole` to exact roles and intended
services; reduce regional wildcard mutation rights. Verify the revised chain
in an isolated AWS account or policy simulator.

### SEC-06 — Malformed checkpoint metadata reaches native memory unsafely

**Locations:** `src/safetensors.cpp:34–50`,
`neuron/cpp/glm_model.cpp:174–187,360–366`.

The quantizer/training reader allocates from an unbounded header length and
indexes `data_offsets[0]` and `[1]` without checking the vector length. It does
not validate shape/dtype byte counts, checked arithmetic, or file bounds.

**Reproduction:** a synthetic tensor with `data_offsets: []` causes UBSan to
report a null reference/load at `src/safetensors.cpp:49`, followed by ASan
SIGSEGV. This occurs when opening the model, before inference.

The new GLM prepared-weight loader similarly passes `data.data() + offset`
and attacker-described `bytes` to `nrt_tensor_write` without verifying
`offset <= data.size()` and `bytes <= data.size() - offset`. Its embedding
mapping checks dtype/shape but not whether the offset and matrix extent fit
the mapped file. These GLM paths were statically verified; no Neuron runtime
was available to execute malformed prepared weights.

**Impact:** a malicious or corrupt checkpoint can crash loading; the unchecked
GLM source span can read host memory outside the loaded buffer into device
memory, or fault. No arbitrary write, code execution, or remotely observable
memory disclosure was demonstrated. Exploitation requires control over the
checkpoint/prepared artifact, not merely a chat request.

**Fix:** centralize a bounded, checked tensor reader; validate header limits,
offset vector length/order, checked addition/multiplication, file extents,
tensor shape/dtype/byte consistency and allocation budgets before any
pointer arithmetic/allocation/upload. The stricter runtime `SafeTensorFile`
implementation is a useful starting point. Test empty, reversed, overflowing,
overlapping and truncated offsets and malformed GLM metadata.

### SEC-07 — Quantizer index allows shard-directory traversal

**Locations:** `src/safetensors.cpp:17–23,34–35,69–74`.

Shard filenames from `weight_map` are joined with the model directory without
rejecting absolute paths or `..`, or resolving symlinks within the intended
root. This differs from the stricter runtime and Python checkpoint loaders.

**Reproduction:** an index referencing `../outside.safetensors` successfully
read four bytes from a synthetic sibling file outside its model directory.

**Impact:** a supplied model can make the quantizer consume other accessible
tensor files. This does not establish arbitrary plaintext-file disclosure:
the current loader still expects a safetensors header, and no network
exfiltration path was demonstrated.

**Fix:** restrict shard paths to permitted relative names, canonicalize beneath
the model root and address symlink escapes. Prefer descriptor-relative opens
with appropriate no-follow/beneath constraints where hostile local races matter.

### SEC-08 — Optional ALB forwards inference over HTTP without a certificate

**Locations:** `deploy/ecs-neuron/router.tf:182–207`,
`deploy/ecs-neuron/variables.tf:352–361`.

When ALB exposure is enabled with the default empty certificate ARN, port 80
forwards to the router. Allowed CIDRs default to the internet. A configured
bearer key does not encrypt the Authorization header or conversation bodies.

**Impact:** plaintext traffic can expose credentials/prompts/responses to an
on-path observer. This finding is conditional on the HTTP-only ALB deployment;
it does not describe the TLS-protected Cloudflare edge path.

**Fix:** require a certificate for internet-facing ALB use and serve inference
only over HTTPS. Restrict or remove HTTP ingress. An HTTP redirect cannot
protect a bearer token that a client already sent in its initial HTTP request.

### SEC-09 — Standalone GLM/CUDA server has no socket deadline

**Location:** `glm53/src/server.cpp:133–141`.

The server accepts a socket and synchronously calls `http::read` in its sole
request loop, without a socket read deadline. A connected client that never
finishes its headers/body prevents the loop from serving other clients,
including health checks. Blocking writes also permit a non-reading client
to stall the loop. The 4-MiB body cap is not a timeout.

The experimental Python server (`neuron/server.py:75,98`) has a related
unbounded-thread/read-wait problem. The shared native Neuron/ROCm server has
an initial 30-second deadline and is not the same implementation.

**Impact:** denial of service if the standalone servers are network-exposed.
Their localhost default reduces default reachability. Static evidence only;
no live GPU server was available for a socket-level reproduction.

**Fix:** implement actual enforced asynchronous read/write deadlines and a
bounded connection pool. Keep reference servers local-only unless these
controls are added. A reverse proxy is useful defense in depth.

### SEC-10 — Approved plans reference mutable image tags

**Locations:** `deploy/ecs-neuron/ecr.tf:1–11`, `router.tf:6`,
`backend.tf:7`, `pipeline/iam.tf:107–118`,
`pipeline/scripts/build-images.sh:71–114`.

The managed router ECR repository explicitly uses mutable tags; the backend
repository is documented as mutable but is externally managed. Task
definitions use `repository:tag`, not a digest. The image-build role can push
to both repositories before the manual approval stage. A hash-looking tag is
still mutable, and the build script trusts an already-existing tag.

**Impact:** an actor with the image-push role can replace an approved tag's
contents. A subsequent task/host pull can execute a different image without
a changed Terraform plan. Existing cached/running tasks are not necessarily
affected immediately. This is a CI/image-publisher trust issue, not a public
request exploit; it was not exercised against ECR.

**Fix:** resolve digests in BuildImages, carry them through plan metadata and
approve/apply task definitions referencing `@sha256:...`. Enforce immutable
release tags and separate any mutable `latest` alias from deployment inputs.
Restrict promotion permissions separately from unapproved builds.

## Additional hardening and controls already present

The model/state/artifact buckets have public-access blocking, encryption and
TLS-only policies; instance metadata requires IMDSv2 with hop limit 1; backend
ingress defaults to the router security group; router containers run as a
non-root user; AWS secrets are supplied by ARN rather than embedded credentials.
The pipeline applies a saved plan and checks its source commit. These controls
are useful but do not fix the findings above.

Remaining hardening: pin base images and download checksums (s5cmd verification
is optional); keep a transitive dependency lock; scan produced images; run the
backend with reduced Linux capabilities/non-root where Neuron permits; bound
diagnostic retention and use restrictive permissions because enabled captures
contain full prompts, responses and reasoning. Do not count tool-call output
as command execution: this server emits tool descriptions/calls and does not
itself execute arbitrary named tools.

## Reproduction and remediation order

Local probes intentionally exercise vulnerable behavior in child processes:

```sh
python tools/security_audit/reproduce_native.py --out /tmp/behemoth-audit-native
python tools/security_audit/reproduce_router.py
```

The first command needs a C++20 compiler and sanitizers; the second needs
aiohttp. They use synthetic files, localhost servers and fake AWS clients.

1. Before public serving: fix SEC-01, upgrade aiohttp, require caller
   authentication and HTTPS, add admission/body/queue limits.
2. Before consuming new checkpoints: fix checked metadata loading and shard
   containment across all loaders, including the GLM prepared path.
3. Before relying on deployment approval as a security boundary: constrain
   apply-role delegation and deploy immutable image digests.
4. Add negative HTTP/parser tests and dependency/image scans to CI; validate
   the deployed edge controls and IAM policy in the actual account.

Audit-only changes: findings remain unfixed in production source at this
revision. The earlier binary-decoder prototype remains uncommitted and separate.
