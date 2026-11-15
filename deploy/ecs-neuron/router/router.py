"""Scale-to-zero router for N Neuron LLM backends on ECS (one ECS service + ASG + capacity provider per model).

* GET  /v1/models            answered locally: every configured model id, active or not (needs a key; wakes nothing)
* GET  /health               router liveness + per-model state (no key: the ECS health check; wakes nothing)
* POST <PROXY_PATHS>         the request's `model` picks the model entry (unknown -> 404 model_not_found); wake ONLY
                             that model's ECS service if needed (single-flight per model), wait until its task is
                             RUNNING and its /health is OK, then proxy (SSE streaming passes through chunk by chunk)
* active cap                 at most MAX_ACTIVE_MODELS models starting/running at once; a cold request beyond it either
                             evicts the longest-idle active model (CAP_POLICY=evict, the default; only a model with nothing
                             in flight, idle >= CAP_MIN_IDLE_S, min_count 0) or gets 503 + Retry-After (CAP_POLICY=reject,
                             or nothing evictable)
* idle reaper                per model: desiredCount=0 after its own idle timeout without requests and none in flight
* spot fallback              a model with a fallback (on-demand) ASG + capacity provider: when its spot ASG reports a
                             failed launch, or has no instance spot_fallback_after_s into a startup, the service is moved
                             to the on-demand capacity provider (UpdateService strategy + forceNewDeployment); it goes back
                             to the spot provider when the model is stopped
* backend loss               (spot interruption, crash) before any byte: re-wake the model and retry up to LOST_RETRIES
                             times; after bytes were streamed: an SSE error event + [DONE] (STREAM_LOSS_ERROR)

Models come from MODELS_JSON (a JSON list of ModelSpec fields, written by router.tf); without it the legacy single-model
variables (ECS_SERVICE, BACKEND_ASG, MODEL_IDS) describe one model.

Callers need `Authorization: Bearer <key>`: a key approved in the API_KEYS_TABLE DynamoDB table (item per key: its
SHA-256 as key_hash, the owner's email and name, enabled; see api_keys.py), or the single ROUTER_API_KEY. With neither
configured the router refuses to start unless ALLOW_ANONYMOUS opts in. Each request is logged with its key's owner. Unauthorized, malformed (not a JSON object, unknown model, non-boolean stream), oversized (MAX_BODY_BYTES)
and excess (MAX_INFLIGHT) requests are answered before anything can wake the backend.

Backend address: ECS DescribeTasks -> awsvpc ENI attachment -> privateIPv4Address.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
from dataclasses import dataclass, field

from aiohttp import ClientError, ClientSession, ClientTimeout, web

log = logging.getLogger("router")

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
    "transfer-encoding", "upgrade", "host", "content-length",
}


@dataclass
class ModelSpec:
    """One model entry: its served ids, its ECS service and its capacity. None fields take the Config default."""
    name: str
    ids: list[str]
    service: str
    asg_name: str | None = None  # primary ASG (spot or on-demand): scaled directly
    capacity_provider: str | None = None  # the service's normal capacity provider (needed with a fallback)
    fallback_asg_name: str | None = None  # on-demand ASG used when spot capacity is short
    fallback_capacity_provider: str | None = None
    spot_fallback_after_s: float = 300.0  # no spot instance this long into a startup -> fall back
    idle_timeout_s: float | None = None
    max_wait_s: float | None = None
    cold_nonstream_mode: str | None = None
    min_count: int = 0  # 1 = keep running (never idle-stopped or evicted)
    port: int | None = None  # backend port (default Config.backend_port)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelSpec":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known and v is not None})


@dataclass
class Config:
    cluster: str = "behemoth"
    service: str = "behemoth-backend"
    asg_name: str | None = None  # backend ASG: scaled directly (the capacity provider alone takes minutes to react from 0)
    region: str | None = None
    backend_port: int = 1337
    listen_port: int = 1337
    model_ids: list[str] = field(default_factory=lambda: ["cydonia-24b"])
    owned_by: str = "cgg507"
    proxy_paths: list[str] = field(default_factory=lambda: ["/v1/chat/completions"])
    idle_timeout_s: float = 300.0
    max_wait_s: float = 900.0
    poll_interval_s: float = 5.0
    keepalive_interval_s: float = 5.0
    stream_keepalive: bool = True
    stream_keepalive_mode: str = "data"  # data (empty chat.completion.chunk) | comment (": ..." SSE comment)
    # keepalive: 200 right away, a space every keepalive_interval_s, then the backend's body (errors keep 200)
    # wait: hold silently up to max_wait_s | reject: 503 + Retry-After at once, warm up in the background
    cold_nonstream_mode: str = "keepalive"
    retry_after_s: int = 120
    request_timeout_s: float = 1800.0
    health_ttl_s: float = 10.0
    api_key: str | None = None
    api_keys_table: str | None = None  # DynamoDB table of approved keys (key_hash -> email, name, enabled)
    key_cache_s: float = 60.0  # how long a key lookup (approved or not) is trusted; revocation takes effect within this
    allow_anonymous: bool = False  # no ROUTER_API_KEY: refuse to start (and answer 401) unless this opts in
    max_body_bytes: int = 4 * 1024 * 1024  # the backend's own body limit: larger bodies get 413 before any wake-up
    max_inflight: int = 256  # requests proxied or waiting for the backend at once; more get 429
    models: list[ModelSpec] = field(default_factory=list)  # empty: one model from service / asg_name / model_ids
    default_model: str | None = None  # model name used when a request has no `model` (default: the first model)
    max_active_models: int = 2  # models starting or running at once (spot vCPU quota / cost bound)
    cap_policy: str = "evict"  # evict (longest-idle evictable model) | reject (503 + Retry-After)
    cap_min_idle_s: float = 30.0  # an active model must be idle this long to be evicted
    lost_retries: int = 1  # re-wake + retry a request whose backend vanished before any byte was sent
    stream_loss_error: bool = True  # backend lost mid-stream: send an SSE error event + [DONE] instead of a bare EOF

    def resolved_models(self) -> list[ModelSpec]:
        ms = self.models or [ModelSpec(name="default", ids=list(self.model_ids), service=self.service,
                                       asg_name=self.asg_name)]
        out = []
        for m in ms:
            out.append(ModelSpec(**{**m.__dict__,
                                    "idle_timeout_s": self.idle_timeout_s if m.idle_timeout_s is None else m.idle_timeout_s,
                                    "max_wait_s": self.max_wait_s if m.max_wait_s is None else m.max_wait_s,
                                    "cold_nonstream_mode": m.cold_nonstream_mode or self.cold_nonstream_mode,
                                    "port": m.port or self.backend_port}))
        return out

    @classmethod
    def from_env(cls, env=os.environ) -> "Config":
        def lst(name, default):
            v = env.get(name)
            return [x.strip() for x in v.split(",") if x.strip()] if v else default

        c = cls()
        c.cluster = env.get("ECS_CLUSTER", c.cluster)
        c.service = env.get("ECS_SERVICE", c.service)
        c.asg_name = env.get("BACKEND_ASG") or None
        c.region = env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION")
        c.backend_port = int(env.get("BACKEND_PORT", c.backend_port))
        c.listen_port = int(env.get("LISTEN_PORT", c.listen_port))
        c.model_ids = lst("MODEL_IDS", c.model_ids)
        c.owned_by = env.get("MODEL_OWNED_BY", c.owned_by)
        c.proxy_paths = lst("PROXY_PATHS", c.proxy_paths)
        c.idle_timeout_s = float(env.get("IDLE_TIMEOUT_S", c.idle_timeout_s))
        c.max_wait_s = float(env.get("MAX_WAIT_S", c.max_wait_s))
        c.poll_interval_s = float(env.get("POLL_INTERVAL_S", c.poll_interval_s))
        c.keepalive_interval_s = float(env.get("KEEPALIVE_INTERVAL_S", c.keepalive_interval_s))
        c.stream_keepalive = env.get("STREAM_KEEPALIVE", "true").lower() in ("1", "true", "yes")
        c.stream_keepalive_mode = env.get("STREAM_KEEPALIVE_MODE", c.stream_keepalive_mode)
        c.cold_nonstream_mode = env.get("COLD_NONSTREAM_MODE", c.cold_nonstream_mode)
        c.retry_after_s = int(env.get("RETRY_AFTER_S", c.retry_after_s))
        c.request_timeout_s = float(env.get("REQUEST_TIMEOUT_S", c.request_timeout_s))
        c.api_key = env.get("ROUTER_API_KEY") or None
        c.api_keys_table = env.get("API_KEYS_TABLE") or None
        c.key_cache_s = float(env.get("KEY_CACHE_S", c.key_cache_s))
        c.allow_anonymous = env.get("ALLOW_ANONYMOUS", "false").lower() in ("1", "true", "yes")
        c.max_body_bytes = int(env.get("MAX_BODY_BYTES", c.max_body_bytes))
        c.max_inflight = int(env.get("MAX_INFLIGHT", c.max_inflight))
        if env.get("MODELS_JSON"):
            c.models = [ModelSpec.from_dict(d) for d in json.loads(env["MODELS_JSON"])]
        c.default_model = env.get("DEFAULT_MODEL") or None
        c.max_active_models = int(env.get("MAX_ACTIVE_MODELS", c.max_active_models))
        c.cap_policy = env.get("CAP_POLICY", c.cap_policy)
        c.cap_min_idle_s = float(env.get("CAP_MIN_IDLE_S", c.cap_min_idle_s))
        c.lost_retries = int(env.get("LOST_RETRIES", c.lost_retries))
        c.stream_loss_error = env.get("STREAM_LOSS_ERROR", "true").lower() in ("1", "true", "yes")
        return c

    def check(self):
        if not self.api_key and not self.api_keys_table and not self.allow_anonymous:
            raise SystemExit("no API_KEYS_TABLE or ROUTER_API_KEY: set one, or ALLOW_ANONYMOUS=true to serve without "
                             "authentication")
        ms = self.resolved_models()
        ids = [i for m in ms for i in m.ids]
        if not ms or not all(m.ids for m in ms):
            raise SystemExit("every model needs at least one id")
        if len(ids) != len(set(ids)) or len({m.name for m in ms}) != len(ms) or len({m.service for m in ms}) != len(ms):
            raise SystemExit("model names, model ids and ECS services must be unique across models")
        if self.default_model and self.default_model not in {m.name for m in ms}:
            raise SystemExit(f"DEFAULT_MODEL {self.default_model!r} is not a configured model")
        if self.cap_policy not in ("evict", "reject"):
            raise SystemExit("CAP_POLICY must be evict or reject")
        if self.max_active_models < 1:
            raise SystemExit("MAX_ACTIVE_MODELS must be >= 1")
        if any(m.cold_nonstream_mode not in ("keepalive", "wait", "reject") for m in ms):
            raise SystemExit("cold_nonstream_mode must be keepalive, wait or reject")
        if any(bool(m.fallback_asg_name) != bool(m.fallback_capacity_provider) or
               (m.fallback_capacity_provider and not m.capacity_provider) for m in ms):
            raise SystemExit("a spot fallback needs fallback_asg_name, fallback_capacity_provider and capacity_provider")


def key_hash(key: str) -> str:
    """How a key is stored: only its SHA-256, so the table never holds a usable key."""
    return hashlib.sha256(key.encode()).hexdigest()


@dataclass
class Caller:
    email: str
    name: str


class KeyStore:
    """Approved keys from DynamoDB, each lookup (hit or miss) cached for key_cache_s; one lookup in flight per key."""

    MAX_CACHED = 10000  # unknown keys are cached too (a flood of guesses costs one lookup each, then nothing)

    def __init__(self, cfg: Config, dynamodb):
        self.cfg, self.db = cfg, dynamodb
        self._cache: dict[str, tuple[float, Caller | None]] = {}
        self._pending: dict[str, asyncio.Future] = {}

    async def lookup(self, key: str) -> Caller | None:
        h = key_hash(key)
        now = time.monotonic()
        hit = self._cache.get(h)
        if hit and now - hit[0] < self.cfg.key_cache_s:
            return hit[1]
        if h not in self._pending:
            self._pending[h] = asyncio.ensure_future(self._fetch(h))
        try:
            caller = await asyncio.shield(self._pending[h])
        finally:
            self._pending.pop(h, None)
        if len(self._cache) >= self.MAX_CACHED:
            self._cache.clear()
        self._cache[h] = (time.monotonic(), caller)
        return caller

    async def _fetch(self, h: str) -> Caller | None:
        item = (await asyncio.to_thread(self.db.get_item, TableName=self.cfg.api_keys_table,
                                        Key={"key_hash": {"S": h}}, ConsistentRead=True)).get("Item")
        if not item or not item.get("enabled", {}).get("BOOL", False):
            return None
        return Caller(email=item.get("email", {}).get("S", ""), name=item.get("name", {}).get("S", ""))


class StartupError(Exception):
    pass


class Backend:
    """One model: owns its ECS service's desiredCount (and capacity provider strategy) and its backend address."""

    def __init__(self, cfg: Config, ecs, http: ClientSession, asg=None, spec: ModelSpec | None = None):
        self.cfg, self.ecs, self.http, self.asg = cfg, ecs, http, asg
        self.m = spec or cfg.resolved_models()[0]
        self.address: str | None = None  # "ip:port"
        self.healthy_at = 0.0
        self.state = "unknown"  # unknown | stopped | starting | ready | failed | stopping
        self.desired: int | None = None  # last desiredCount seen / set (None: not looked at yet)
        self.on_fallback = False  # the service runs on the on-demand fallback capacity provider
        self._activities_warned = False
        self._startup: asyncio.Task | None = None
        self.inflight = 0
        self.last_activity = time.monotonic()

    @property
    def name(self) -> str:
        return self.m.name

    async def _call(self, name, **kw):
        return await asyncio.to_thread(getattr(self.ecs, name), **kw)

    async def _health(self, address: str) -> bool:
        try:
            async with self.http.get(f"http://{address}/health", timeout=ClientTimeout(total=5)) as r:
                return r.status == 200
        except (ClientError, asyncio.TimeoutError, OSError):
            return False

    async def _running_address(self) -> str | None:
        arns = (await self._call("list_tasks", cluster=self.cfg.cluster, serviceName=self.m.service,
                                 desiredStatus="RUNNING")).get("taskArns", [])
        if not arns:
            return None
        tasks = (await self._call("describe_tasks", cluster=self.cfg.cluster, tasks=arns)).get("tasks", [])
        for t in tasks:
            if t.get("lastStatus") != "RUNNING":
                continue
            for att in t.get("attachments", []):
                if att.get("type") != "ElasticNetworkInterface":
                    continue
                for d in att.get("details", []):
                    if d.get("name") == "privateIPv4Address":
                        return f"{d['value']}:{self.m.port}"
        return None

    async def _describe_service(self) -> dict | None:
        svcs = (await self._call("describe_services", cluster=self.cfg.cluster,
                                 services=[self.m.service])).get("services", [])
        if svcs:
            self.desired = svcs[0].get("desiredCount", 0)
        return svcs[0] if svcs else None

    async def _set_desired(self, n: int, **kw):
        await self._call("update_service", cluster=self.cfg.cluster, service=self.m.service, desiredCount=n, **kw)
        self.desired = n

    def _asgs(self) -> list[str]:
        return [a for a in (self.m.asg_name, self.m.fallback_asg_name) if a]

    def _active_asg(self) -> str | None:
        return self.m.fallback_asg_name if self.on_fallback else self.m.asg_name

    async def _describe_asg(self, name: str) -> dict | None:
        groups = (await asyncio.to_thread(self.asg.describe_auto_scaling_groups,
                                          AutoScalingGroupNames=[name])).get("AutoScalingGroups", [])
        return groups[0] if groups else None

    async def _asg_desired(self, name: str | None = None) -> int | None:
        name = name or self._active_asg()
        if not (self.asg and name):
            return None
        g = await self._describe_asg(name)
        return g.get("DesiredCapacity", 0) if g else None

    async def _set_asg(self, n: int, name: str | None = None):
        await asyncio.to_thread(self.asg.set_desired_capacity, AutoScalingGroupName=name or self._active_asg(),
                                DesiredCapacity=n, HonorCooldown=False)

    async def _asg_scale_out(self):
        """Launch the instance now instead of waiting for the capacity provider's scale-out (~2-3 min from 0)."""
        name = self._active_asg()
        try:
            cur = await self._asg_desired(name)
            if cur is not None and cur < 1:
                log.info("[%s] scaling ASG %s to 1", self.name, name)
                await self._set_asg(1, name)
        except Exception as e:  # the capacity provider still scales out on its own
            log.warning("[%s] ASG scale-out failed (capacity provider will catch up): %s", self.name, e)

    async def _own_instance_ids(self) -> set[str] | None:
        """EC2 instances of this model's ASGs (None: no ASG known, so no filter)."""
        if not (self.asg and self._asgs()):
            return None
        ids: set[str] = set()
        for name in self._asgs():
            g = await self._describe_asg(name)
            ids |= {i["InstanceId"] for i in (g or {}).get("Instances", [])}
        return ids

    async def reactivate_draining(self) -> int:
        """Warm-pool reuse: an instance scaled into the warm pool is drained by ECS and keeps status DRAINING when the pool
        starts it again, so the pending task is never placed (seen: 22 min stuck). Set connected, idle DRAINING container
        instances back to ACTIVE while a startup is waiting for a task. With several models in the cluster only instances
        of this model's own ASGs are touched (another model's instance may be draining on purpose)."""
        arns = (await self._call("list_container_instances", cluster=self.cfg.cluster, status="DRAINING")).get("containerInstanceArns", [])
        if not arns:
            return 0
        cis = (await self._call("describe_container_instances", cluster=self.cfg.cluster, containerInstances=arns)).get("containerInstances", [])
        own = await self._own_instance_ids()
        idle = [c["containerInstanceArn"] for c in cis
                if c.get("agentConnected") and not c.get("runningTasksCount") and not c.get("pendingTasksCount")
                and (own is None or c.get("ec2InstanceId") in own)]
        if idle:
            log.info("[%s] reactivating %d DRAINING container instance(s) left over from the warm pool", self.name, len(idle))
            await self._call("update_container_instances_state", cluster=self.cfg.cluster, containerInstances=idle, status="ACTIVE")
        return len(idle)

    async def maybe_asg_scale_in(self) -> bool:
        """After an idle stop: once no backend task is left, drop the ASG(s) to 0 now (managed scaling would wait ~15 min).
        Runs every reaper cycle while stopped: ECS managed termination protection can leave the idle instance protected
        from scale-in, which cancels the ASG's scale-in ("all remaining instances are protected") with no retry, so an
        in-service instance with no task gets its protection cleared here and the ASG set to 0 again."""
        if self.state != "stopped" or not (self.asg and self._asgs()):
            return False
        if (await self._call("list_tasks", cluster=self.cfg.cluster, serviceName=self.m.service)).get("taskArns"):
            return False
        acted = False
        for name in self._asgs():
            g = await self._describe_asg(name)
            if not g:
                continue
            in_service = [i for i in g.get("Instances", []) if i.get("LifecycleState") == "InService"]
            protected = [i["InstanceId"] for i in in_service if i.get("ProtectedFromScaleIn")]
            if protected:
                log.info("[%s] stopped: clearing scale-in protection on %s", self.name, ",".join(protected))
                await asyncio.to_thread(self.asg.set_instance_protection, AutoScalingGroupName=name,
                                        InstanceIds=protected, ProtectedFromScaleIn=False)
                acted = True
            if g.get("DesiredCapacity", 0) > 0 or (in_service and protected):
                log.info("[%s] stopped: scaling ASG %s to 0", self.name, name)
                await self._set_asg(0, name)
                acted = True
        return acted

    # ---------------------------------------------------------------- spot -> on-demand fallback
    async def _spot_short(self, t0_wall: float, t0: float) -> bool:
        """The spot ASG cannot deliver: a launch failed/cancelled since the startup began, or no instance (pending or in
        service) spot_fallback_after_s into it."""
        name = self.m.asg_name
        try:
            acts = (await asyncio.to_thread(self.asg.describe_scaling_activities, AutoScalingGroupName=name,
                                            MaxRecords=10)).get("Activities", [])
        except Exception as e:  # e.g. not allowed by the role's permissions boundary: the timeout signal still works
            if not self._activities_warned:
                log.warning("[%s] cannot read ASG scaling activities (%s); spot fallback by timeout only", self.name, e)
                self._activities_warned = True
            acts = []
        for a in acts:
            st = a.get("StartTime")
            ts = st.timestamp() if hasattr(st, "timestamp") else (float(st) if st is not None else t0_wall)
            if ts + 1 < t0_wall:
                continue
            if a.get("StatusCode") in ("Failed", "Cancelled"):
                log.warning("[%s] spot launch failed: %s", self.name, (a.get("StatusMessage") or "")[:300])
                return True
        if time.monotonic() - t0 < self.m.spot_fallback_after_s:
            return False
        g = await self._describe_asg(name)
        live = [i for i in (g or {}).get("Instances", [])
                if str(i.get("LifecycleState", "")).startswith(("Pending", "InService"))]
        return not live

    async def _fall_back(self):
        log.warning("[%s] spot capacity short: moving %s to the on-demand capacity provider %s", self.name,
                    self.m.service, self.m.fallback_capacity_provider)
        await self._set_desired(1, capacityProviderStrategy=[
            {"capacityProvider": self.m.fallback_capacity_provider, "weight": 1, "base": 0}], forceNewDeployment=True)
        self.on_fallback = True
        try:
            await self._set_asg(0, self.m.asg_name)  # a late spot instance would only sit idle
        except Exception as e:
            log.warning("[%s] spot ASG scale-in failed: %s", self.name, e)
        await self._asg_scale_out()

    # ---------------------------------------------------------------- start / stop
    async def _start(self) -> str:
        self.state = "starting"
        t0, t0_wall = time.monotonic(), time.time()
        deadline = t0 + self.m.max_wait_s
        svc = await self._describe_service()
        if svc is None:
            raise StartupError(f"service {self.m.service} not found")
        strategy = [s.get("capacityProvider") for s in svc.get("capacityProviderStrategy") or []]
        self.on_fallback = bool(self.m.fallback_capacity_provider) and self.m.fallback_capacity_provider in strategy
        if svc.get("desiredCount", 0) < 1:
            log.info("[%s] scaling %s to 1", self.name, self.m.service)
            await self._set_desired(1)
        await self._asg_scale_out()
        fallback_armed = bool(self.m.fallback_capacity_provider and self.asg and not self.on_fallback)
        while time.monotonic() < deadline:
            try:
                addr = await self._running_address()
                if not addr:
                    await self.reactivate_draining()
                    if fallback_armed and await self._spot_short(t0_wall, t0):
                        fallback_armed = False
                        await self._fall_back()
            except Exception as e:  # transient API errors: keep polling until the deadline
                log.warning("[%s] ECS poll failed: %s", self.name, e)
                addr = None
            if addr and await self._health(addr):
                log.info("[%s] backend ready at %s%s", self.name, addr, " (on-demand fallback)" if self.on_fallback else "")
                return addr
            await asyncio.sleep(self.cfg.poll_interval_s)
        raise StartupError(f"backend not healthy after {self.m.max_wait_s:.0f}s")

    def is_warm(self) -> bool:
        return self.address is not None and self.state == "ready"

    def is_starting(self) -> bool:
        return self.state == "starting" or bool(self._startup and not self._startup.done())

    def is_active(self) -> bool:
        """Counts against the active-model cap: starting, running, or found running (desiredCount > 0) by a router
        that has not served it yet (e.g. after a router restart)."""
        if self.state in ("starting", "ready") or self.is_starting():
            return True
        return self.state in ("unknown", "failed") and bool(self.desired)

    def idle_for(self) -> float:
        return time.monotonic() - self.last_activity

    async def sync(self):
        """Learn desiredCount of a model this router has not looked at yet (a stopped one becomes 'stopped')."""
        if self.state == "unknown" and self.desired is None and not self.is_starting():
            await self._describe_service()
            if self.desired == 0:
                self.state = "stopped"

    async def ensure(self) -> str:
        """Return a healthy backend address, starting the service if needed. Concurrent callers share one startup."""
        if self.address and time.monotonic() - self.healthy_at < self.cfg.health_ttl_s:
            return self.address
        if self.address and await self._health(self.address):
            self.healthy_at = time.monotonic()
            return self.address
        self.address = None
        if self._startup is None or self._startup.done():
            self._startup = asyncio.create_task(self._start())
        try:
            addr = await asyncio.shield(self._startup)
        except StartupError:
            self.state = "failed"
            raise
        except Exception as e:
            self.state = "failed"
            raise StartupError(str(e)) from e
        self.address, self.healthy_at, self.state = addr, time.monotonic(), "ready"
        return addr

    def lost(self):
        """Backend vanished (spot interruption, crash): forget it; the next ensure() re-wakes it."""
        log.warning("[%s] backend %s lost", self.name, self.address)
        self.address, self.state = None, "unknown"

    async def stop(self, reason: str, svc: dict | None = None):
        """desiredCount 0; a service left on the on-demand fallback goes back to its spot capacity provider."""
        svc = svc if svc is not None else await self._describe_service()
        kw = {}
        strategy = [s.get("capacityProvider") for s in (svc or {}).get("capacityProviderStrategy") or []]
        if self.m.capacity_provider and (self.on_fallback or (strategy and strategy != [self.m.capacity_provider])):
            kw = {"capacityProviderStrategy": [{"capacityProvider": self.m.capacity_provider, "weight": 1, "base": 0}],
                  "forceNewDeployment": True}
        if (svc and svc.get("desiredCount", 0) > 0) or kw:
            log.info("[%s] %s: scaling %s to 0%s", self.name, reason, self.m.service,
                     " and back to the spot capacity provider" if kw else "")
            await self._set_desired(0, **kw)
        self.on_fallback = False
        self.address, self.state, self.desired = None, "stopped", 0

    async def maybe_idle_stop(self) -> bool:
        if self.inflight or self.m.min_count > 0 or self.is_starting():
            return False
        if self.idle_for() < self.m.idle_timeout_s:
            return False
        if self.state == "stopped":
            return False
        await self.stop(f"idle for {self.m.idle_timeout_s:.0f}s")
        return True


def _error(status: int, msg: str, retry_after: int | None = None, kind: str | None = None) -> web.Response:
    headers = {"Retry-After": str(retry_after)} if retry_after else None
    kind = kind or ("backend_unavailable" if status >= 500 else "rate_limited" if status == 429 else "invalid_request_error")
    return web.json_response({"error": {"message": msg, "type": kind}}, status=status, headers=headers)


def _held_error(msg: str, stream: bool) -> bytes:
    if stream:
        return _sse_error(msg)
    return json.dumps({"error": {"message": msg, "type": "backend_unavailable"}}).encode()


def _sse_error(msg: str) -> bytes:
    return (b"data: " + json.dumps({"error": {"message": msg, "type": "backend_unavailable"}}).encode()
            + b"\n\ndata: [DONE]\n\n")


def _keepalive_bytes(stream: bool, mode: str, model: str) -> bytes:
    if not stream:
        return b" "  # leading whitespace before the JSON body
    if mode == "comment":
        return b": waiting for backend to start\n\n"
    # a real (empty) chunk: clients that time out on the first data event, not on bytes, keep waiting
    chunk = {"id": "chatcmpl-wait", "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
             "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}
    return b"data: " + json.dumps(chunk).encode() + b"\n\n"


async def _keepalive_until(resp: web.StreamResponse, fut: asyncio.Future, interval: float, payload: bytes):
    while not fut.done():
        done, _ = await asyncio.wait({fut}, timeout=interval)
        if not done:
            await resp.write(payload)


class Router:
    def __init__(self, cfg: Config, ecs, http: ClientSession | None = None, asg=None, dynamodb=None):
        self.cfg, self.ecs, self._http, self.asg = cfg, ecs, http, asg
        self.keys = KeyStore(cfg, dynamodb) if cfg.api_keys_table else None
        self.specs = cfg.resolved_models()
        self.backends: dict[str, Backend] = {}  # model name -> Backend
        self.by_id: dict[str, str] = {i: m.name for m in self.specs for i in m.ids}  # served id -> model name
        names = [m.name for m in self.specs]
        self.default = cfg.default_model if cfg.default_model in names else names[0]
        self._admit = asyncio.Lock()  # cap check + reservation of a cold model are one step
        self._reaper: asyncio.Task | None = None

    @property
    def backend(self) -> Backend:
        """The default model's backend (single-model deployments: the only one)."""
        return self.backends[self.default]

    def model_ids(self) -> list[str]:
        order = [self.default] + [m.name for m in self.specs if m.name != self.default]
        spec = {m.name: m for m in self.specs}
        return [i for n in order for i in spec[n].ids]

    async def on_startup(self, app):
        if self._http is None:
            self._http = ClientSession(timeout=ClientTimeout(total=None, sock_connect=10))
        self.backends = {m.name: Backend(self.cfg, self.ecs, self._http, self.asg, m) for m in self.specs}
        self._reaper = asyncio.create_task(self._reap())

    async def on_cleanup(self, app):
        if self._reaper:
            self._reaper.cancel()
        if self._http:
            await self._http.close()

    async def _reap(self):
        interval = max(1.0, min(30.0, min(m.idle_timeout_s for m in self.specs) / 5))
        while True:
            await asyncio.sleep(interval)
            for b in self.backends.values():
                try:
                    await b.maybe_idle_stop()
                    await b.maybe_asg_scale_in()
                except Exception as e:
                    log.warning("[%s] idle check failed: %s", b.name, e)

    async def _admit_cold(self, b: Backend) -> web.Response | None:
        """Active-model cap. None = b may run (already active, or a slot was free / freed); else the 503 to send."""
        cap = self.cfg.max_active_models
        if b.is_active() or len(self.backends) <= cap:
            return None
        async with self._admit:
            if b.is_active():
                return None
            others = [x for x in self.backends.values() if x is not b]
            for x in others:
                try:
                    await x.sync()
                except Exception as e:  # unknown stays unknown (counted only if seen running)
                    log.warning("[%s] state sync failed: %s", x.name, e)
            active = [x for x in others if x.is_active()]
            if len(active) >= cap and self.cfg.cap_policy == "evict":
                cands = [x for x in active if not x.inflight and not x.is_starting() and x.m.min_count == 0
                         and x.idle_for() >= self.cfg.cap_min_idle_s]
                if cands:
                    victim = max(cands, key=lambda x: x.idle_for())
                    try:
                        await victim.stop(f"evicted (idle {victim.idle_for():.0f}s) to start {b.name}")
                        active.remove(victim)
                    except Exception as e:
                        log.warning("[%s] eviction failed: %s", victim.name, e)
            if len(active) >= cap:
                wait = [max(0.0, x.m.idle_timeout_s - x.idle_for()) for x in active if x.m.min_count == 0]
                retry = int(min(max(5.0, min(wait)) if wait else self.cfg.retry_after_s, self.cfg.retry_after_s))
                log.info("cap: %d/%d models active (%s), %s rejected", len(active), cap,
                         ",".join(x.name for x in active), b.name)
                return _error(503, f"{len(active)} of {cap} models are active ({', '.join(x.name for x in active)}); "
                                   f"model {b.name} cannot start now, retry later", retry, kind="capacity_exceeded")
            b.state = "starting"  # reserve the slot until ensure() takes over
            return None

    async def _caller(self, request: web.Request) -> Caller | None:
        """Who the request's bearer key belongs to; None = not authorized."""
        if not self.cfg.api_key and not self.keys:
            return Caller("", "anonymous") if self.cfg.allow_anonymous else None
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or len(auth) > 512:
            return None
        key = auth[7:].strip()
        if self.cfg.api_key and hmac.compare_digest(key.encode(), self.cfg.api_key.encode()):
            return Caller("", "router key")
        if self.keys:
            try:
                return await self.keys.lookup(key)
            except Exception as e:  # DynamoDB unreachable: fail closed
                log.warning("API key lookup failed: %s", e)
        return None

    async def models(self, request):
        if await self._caller(request) is None:
            return web.json_response({"error": {"message": "unauthorized", "type": "invalid_api_key"}}, status=401)
        now = int(time.time())
        return web.json_response({"object": "list", "data": [
            {"id": m, "object": "model", "created": now, "owned_by": self.cfg.owned_by} for m in self.model_ids()]})

    async def health(self, request):
        b = self.backend
        bs = self.backends.values()
        return web.json_response({
            "healthy": True, "backend": b.state, "inflight": sum(x.inflight for x in bs),
            "idle_s": round(b.idle_for(), 1),
            "active": sum(x.is_active() for x in bs), "max_active": self.cfg.max_active_models,
            "models": {x.name: {"state": x.state, "inflight": x.inflight, "idle_s": round(x.idle_for(), 1),
                                "on_demand_fallback": x.on_fallback} for x in bs}})

    def inflight(self) -> int:
        return sum(x.inflight for x in self.backends.values())

    async def proxy(self, request: web.Request):
        caller = await self._caller(request)
        if caller is None:
            return web.json_response({"error": {"message": "unauthorized", "type": "invalid_api_key"}}, status=401)
        if self.inflight() >= self.cfg.max_inflight:
            return _error(429, "too many requests in flight, retry later", 5)
        # everything that can be rejected is rejected here, before a request can wake (and pay for) the backend
        try:
            body = await request.read()
        except web.HTTPRequestEntityTooLarge:
            return _error(413, f"request body exceeds {self.cfg.max_body_bytes} bytes")
        try:
            req = json.loads(body)
        except ValueError:
            return _error(400, "request body is not valid JSON")
        if not isinstance(req, dict):
            return _error(400, "request body must be a JSON object")
        model = req.get("model", self.backends[self.default].m.ids[0])
        if not isinstance(model, str) or model not in self.by_id:
            return _error(404, f"The model {model!r} does not exist", kind="model_not_found")
        stream = req.get("stream", False)
        if not isinstance(stream, bool):
            return _error(400, "'stream' must be a boolean")
        b = self.backends[self.by_id[model]]
        log.info("request %s from %s <%s> model=%s (%s) stream=%s bytes=%d", request.path, caller.name, caller.email,
                 model, b.name, stream, len(body))
        if not b.is_warm():
            refused = await self._admit_cold(b)
            if refused is not None:
                return refused
        b.inflight += 1
        b.last_activity = time.monotonic()
        try:
            return await self._proxy(request, body, stream, model, b)
        except _ClientGone as e:
            log.info("client disconnected mid-response")
            return e.resp
        finally:
            b.inflight -= 1
            b.last_activity = time.monotonic()

    async def _proxy(self, request, body, stream, model, b: Backend):
        cfg = self.cfg
        resp: web.StreamResponse | None = None
        keepalive = _keepalive_bytes(stream, cfg.stream_keepalive_mode, model)
        cold_mode = b.m.cold_nonstream_mode
        if not b.is_warm() and not stream and cold_mode == "reject":
            fut = asyncio.ensure_future(b.ensure())  # warm up in the background, tell the client to come back
            fut.add_done_callback(lambda f: f.cancelled() or f.exception())
            return _error(503, "backend is starting, retry later", cfg.retry_after_s)
        ensure = asyncio.ensure_future(b.ensure())
        hold = cfg.stream_keepalive if stream else cold_mode == "keepalive"
        if hold and not b.is_warm():
            # commit to a 200 now so clients/proxies (Cloudflare: 100 s without bytes) see bytes while the instance boots
            ctype = "text/event-stream" if stream else "application/json"
            resp = web.StreamResponse(status=200, headers={"Content-Type": ctype, "Cache-Control": "no-cache"})
            await resp.prepare(request)
            try:
                await _keepalive_until(resp, ensure, cfg.keepalive_interval_s, keepalive)
            except (ConnectionResetError, ClientError):
                ensure.cancel()
                raise
        retries = max(0, cfg.lost_retries)
        for attempt in range(retries + 1):
            try:
                addr = await ensure
            except StartupError as e:
                if resp is not None:
                    await resp.write(_held_error(f"backend startup failed: {e}", stream))
                    await resp.write_eof()
                    return resp
                return _error(503, f"backend startup failed: {e}", cfg.retry_after_s)
            try:
                return await self._forward(request, body, addr, resp, stream)
            except _BackendGone as e:
                b.lost()
                if e.resp is not None:
                    resp = e.resp
                if e.started or attempt == retries:
                    if resp is not None:
                        if not e.started:
                            await resp.write(_held_error("backend lost", stream))
                        elif stream and cfg.stream_loss_error:
                            # bytes already went out: tell an SSE client the answer is cut short instead of a bare EOF
                            # (the model is re-woken by the next request)
                            await resp.write(_sse_error("backend lost mid-response (spot interruption or crash); "
                                                        "the response is incomplete, retry the request"))
                        await resp.write_eof()
                        return resp
                    return _error(503, "backend lost", cfg.retry_after_s)
                # nothing was sent yet: re-wake the model (a spot interruption reschedules the task on a new
                # instance, or on the on-demand fallback) and send the request again
                log.info("[%s] retrying request after backend loss (%d/%d)", b.name, attempt + 1, retries)
                ensure = asyncio.ensure_future(b.ensure())
                if resp is None and not stream and cold_mode == "reject":
                    ensure.add_done_callback(lambda f: f.cancelled() or f.exception())
                    return _error(503, "backend lost, restarting, retry later", cfg.retry_after_s)
                if resp is None and hold:
                    # the re-wake can take minutes (a new instance): hold the client like a cold start
                    ctype = "text/event-stream" if stream else "application/json"
                    resp = web.StreamResponse(status=200, headers={"Content-Type": ctype, "Cache-Control": "no-cache"})
                    await resp.prepare(request)
                if resp is not None:
                    try:
                        await _keepalive_until(resp, ensure, cfg.keepalive_interval_s, keepalive)
                    except (ConnectionResetError, ClientError):
                        ensure.cancel()
                        raise

    async def _forward(self, request, body, addr, resp, stream):
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP | {"authorization"}}
        url = f"http://{addr}{request.path_qs}"
        started = False
        try:
            async with self._http.request(request.method, url, data=body, headers=headers,
                                          timeout=ClientTimeout(total=self.cfg.request_timeout_s,
                                                                sock_connect=10)) as up:
                if resp is None:
                    out_headers = {k: v for k, v in up.headers.items() if k.lower() not in HOP_BY_HOP}
                    resp = web.StreamResponse(status=up.status, headers=out_headers)
                    await resp.prepare(request)
                elif up.status != 200 and stream:
                    started = True
                    text = (await up.read()).decode(errors="replace")
                    await resp.write(_sse_error(f"backend returned {up.status}: {text[:500]}"))
                    await resp.write_eof()
                    return resp
                # held non-stream response: the status is already 200, the backend's JSON body (or error) follows
                started = True
                async for chunk in up.content.iter_any():
                    try:
                        await resp.write(chunk)
                    except (ClientError, OSError) as e:  # the client went away; not a backend failure
                        raise _ClientGone(resp) from e
                await resp.write_eof()
                return resp
        except (ClientError, asyncio.TimeoutError, OSError) as e:
            raise _BackendGone(started, resp if started else None) from e


class _ClientGone(Exception):
    def __init__(self, resp):
        super().__init__("client gone")
        self.resp = resp


class _BackendGone(Exception):
    def __init__(self, started: bool, resp: web.StreamResponse | None = None):
        super().__init__("backend gone")
        self.started = started
        self.resp = resp  # the client response already prepared from the backend's headers, if any


def make_app(cfg: Config, ecs, http: ClientSession | None = None, asg=None, dynamodb=None) -> web.Application:
    r = Router(cfg, ecs, http, asg, dynamodb)
    app = web.Application(client_max_size=cfg.max_body_bytes)
    app.on_startup.append(r.on_startup)
    app.on_cleanup.append(r.on_cleanup)
    app.router.add_get("/v1/models", r.models)
    app.router.add_get("/health", r.health)
    for p in cfg.proxy_paths:
        app.router.add_post(p, r.proxy)
    return app


def main():
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    import boto3  # only needed in production

    cfg = Config.from_env()
    cfg.check()
    ecs = boto3.client("ecs", region_name=cfg.region)
    asg = boto3.client("autoscaling", region_name=cfg.region) \
        if any(m.asg_name or m.fallback_asg_name for m in cfg.resolved_models()) else None
    dynamodb = boto3.client("dynamodb", region_name=cfg.region) if cfg.api_keys_table else None
    web.run_app(make_app(cfg, ecs, asg=asg, dynamodb=dynamodb), host="0.0.0.0", port=cfg.listen_port, access_log=None)


if __name__ == "__main__":
    main()
