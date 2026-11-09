"""Scale-to-zero router for the Neuron LLM backend on ECS.

* GET  /v1/models            answered locally from MODEL_IDS (never wakes the backend)
* GET  /health               router liveness + backend state (never wakes the backend)
* POST <PROXY_PATHS>         wake the backend ECS service if needed (single-flight), wait until the task is RUNNING
                             and its /health is OK, then proxy (SSE streaming passes through chunk by chunk)
* idle reaper                desiredCount=0 after IDLE_TIMEOUT_S without requests and none in flight

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
        return c

    def check(self):
        if not self.api_key and not self.api_keys_table and not self.allow_anonymous:
            raise SystemExit("no API_KEYS_TABLE or ROUTER_API_KEY: set one, or ALLOW_ANONYMOUS=true to serve without "
                             "authentication")


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
    """Owns the ECS service's desiredCount and the backend address."""

    def __init__(self, cfg: Config, ecs, http: ClientSession, asg=None):
        self.cfg, self.ecs, self.http, self.asg = cfg, ecs, http, asg
        self.address: str | None = None  # "ip:port"
        self.healthy_at = 0.0
        self.state = "unknown"  # unknown | stopped | starting | ready | failed | stopping
        self._startup: asyncio.Task | None = None
        self.inflight = 0
        self.last_activity = time.monotonic()

    async def _call(self, name, **kw):
        return await asyncio.to_thread(getattr(self.ecs, name), **kw)

    async def _health(self, address: str) -> bool:
        try:
            async with self.http.get(f"http://{address}/health", timeout=ClientTimeout(total=5)) as r:
                return r.status == 200
        except (ClientError, asyncio.TimeoutError, OSError):
            return False

    async def _running_address(self) -> str | None:
        arns = (await self._call("list_tasks", cluster=self.cfg.cluster, serviceName=self.cfg.service,
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
                        return f"{d['value']}:{self.cfg.backend_port}"
        return None

    async def _set_desired(self, n: int):
        await self._call("update_service", cluster=self.cfg.cluster, service=self.cfg.service, desiredCount=n)

    async def _asg_desired(self) -> int | None:
        if not (self.asg and self.cfg.asg_name):
            return None
        groups = (await asyncio.to_thread(self.asg.describe_auto_scaling_groups,
                                          AutoScalingGroupNames=[self.cfg.asg_name])).get("AutoScalingGroups", [])
        return groups[0].get("DesiredCapacity", 0) if groups else None

    async def _set_asg(self, n: int):
        await asyncio.to_thread(self.asg.set_desired_capacity, AutoScalingGroupName=self.cfg.asg_name,
                                DesiredCapacity=n, HonorCooldown=False)

    async def _asg_scale_out(self):
        """Launch the instance now instead of waiting for the capacity provider's scale-out (~2-3 min from 0)."""
        try:
            cur = await self._asg_desired()
            if cur is not None and cur < 1:
                log.info("scaling ASG %s to 1", self.cfg.asg_name)
                await self._set_asg(1)
        except Exception as e:  # the capacity provider still scales out on its own
            log.warning("ASG scale-out failed (capacity provider will catch up): %s", e)

    async def reactivate_draining(self) -> int:
        """Warm-pool reuse: an instance scaled into the warm pool is drained by ECS and keeps status DRAINING when the pool
        starts it again, so the pending task is never placed (seen: 22 min stuck). Set connected, idle DRAINING container
        instances of the cluster back to ACTIVE while a startup is waiting for a task."""
        arns = (await self._call("list_container_instances", cluster=self.cfg.cluster, status="DRAINING")).get("containerInstanceArns", [])
        if not arns:
            return 0
        cis = (await self._call("describe_container_instances", cluster=self.cfg.cluster, containerInstances=arns)).get("containerInstances", [])
        idle = [c["containerInstanceArn"] for c in cis
                if c.get("agentConnected") and not c.get("runningTasksCount") and not c.get("pendingTasksCount")]
        if idle:
            log.info("reactivating %d DRAINING container instance(s) left over from the warm pool", len(idle))
            await self._call("update_container_instances_state", cluster=self.cfg.cluster, containerInstances=idle, status="ACTIVE")
        return len(idle)

    async def maybe_asg_scale_in(self) -> bool:
        """After an idle stop: once no backend task is left, drop the ASG to 0 now (managed scaling would wait ~15 min).
        Runs every reaper cycle while stopped: ECS managed termination protection can leave the idle instance protected
        from scale-in, which cancels the ASG's scale-in ("all remaining instances are protected") with no retry, so an
        in-service instance with no task gets its protection cleared here and the ASG set to 0 again."""
        if self.state != "stopped" or not (self.asg and self.cfg.asg_name):
            return False
        if (await self._call("list_tasks", cluster=self.cfg.cluster, serviceName=self.cfg.service)).get("taskArns"):
            return False
        groups = (await asyncio.to_thread(self.asg.describe_auto_scaling_groups,
                                          AutoScalingGroupNames=[self.cfg.asg_name])).get("AutoScalingGroups", [])
        if not groups:
            return False
        g = groups[0]
        in_service = [i for i in g.get("Instances", []) if i.get("LifecycleState") == "InService"]
        protected = [i["InstanceId"] for i in in_service if i.get("ProtectedFromScaleIn")]
        acted = False
        if protected:
            log.info("backend stopped: clearing scale-in protection on %s", ",".join(protected))
            await asyncio.to_thread(self.asg.set_instance_protection, AutoScalingGroupName=self.cfg.asg_name,
                                    InstanceIds=protected, ProtectedFromScaleIn=False)
            acted = True
        if g.get("DesiredCapacity", 0) > 0 or (in_service and protected):
            log.info("backend stopped: scaling ASG %s to 0", self.cfg.asg_name)
            await self._set_asg(0)
            acted = True
        return acted

    async def _start(self) -> str:
        self.state = "starting"
        deadline = time.monotonic() + self.cfg.max_wait_s
        svc = (await self._call("describe_services", cluster=self.cfg.cluster, services=[self.cfg.service]))
        svcs = svc.get("services", [])
        if not svcs:
            raise StartupError(f"service {self.cfg.service} not found")
        if svcs[0].get("desiredCount", 0) < 1:
            log.info("scaling %s to 1", self.cfg.service)
            await self._set_desired(1)
        await self._asg_scale_out()
        while time.monotonic() < deadline:
            try:
                addr = await self._running_address()
                if not addr:
                    await self.reactivate_draining()
            except Exception as e:  # transient API errors: keep polling until the deadline
                log.warning("ECS poll failed: %s", e)
                addr = None
            if addr and await self._health(addr):
                log.info("backend ready at %s", addr)
                return addr
            await asyncio.sleep(self.cfg.poll_interval_s)
        raise StartupError(f"backend not healthy after {self.cfg.max_wait_s:.0f}s")

    def is_warm(self) -> bool:
        return self.address is not None and self.state == "ready"

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
        """Backend vanished (spot interruption, crash): forget it; the next request restarts it."""
        log.warning("backend %s lost", self.address)
        self.address, self.state = None, "unknown"

    async def maybe_idle_stop(self) -> bool:
        if self.inflight or (self._startup and not self._startup.done()):
            return False
        if time.monotonic() - self.last_activity < self.cfg.idle_timeout_s:
            return False
        if self.state == "stopped":
            return False
        svcs = (await self._call("describe_services", cluster=self.cfg.cluster,
                                 services=[self.cfg.service])).get("services", [])
        if svcs and svcs[0].get("desiredCount", 0) > 0:
            log.info("idle for %.0fs: scaling %s to 0", self.cfg.idle_timeout_s, self.cfg.service)
            await self._set_desired(0)
        self.address, self.state = None, "stopped"
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
        self.backend: Backend | None = None
        self._reaper: asyncio.Task | None = None

    async def on_startup(self, app):
        if self._http is None:
            self._http = ClientSession(timeout=ClientTimeout(total=None, sock_connect=10))
        self.backend = Backend(self.cfg, self.ecs, self._http, self.asg)
        self._reaper = asyncio.create_task(self._reap())

    async def on_cleanup(self, app):
        if self._reaper:
            self._reaper.cancel()
        if self._http:
            await self._http.close()

    async def _reap(self):
        interval = max(1.0, min(30.0, self.cfg.idle_timeout_s / 5))
        while True:
            await asyncio.sleep(interval)
            try:
                await self.backend.maybe_idle_stop()
                await self.backend.maybe_asg_scale_in()
            except Exception as e:
                log.warning("idle check failed: %s", e)

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
        now = int(time.time())
        return web.json_response({"object": "list", "data": [
            {"id": m, "object": "model", "created": now, "owned_by": self.cfg.owned_by} for m in self.cfg.model_ids]})

    async def health(self, request):
        b = self.backend
        return web.json_response({"healthy": True, "backend": b.state, "inflight": b.inflight,
                                  "idle_s": round(time.monotonic() - b.last_activity, 1)})

    async def proxy(self, request: web.Request):
        caller = await self._caller(request)
        if caller is None:
            return web.json_response({"error": {"message": "unauthorized", "type": "invalid_api_key"}}, status=401)
        b = self.backend
        if b.inflight >= self.cfg.max_inflight:
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
        model = req.get("model", self.cfg.model_ids[0])
        if not isinstance(model, str) or model not in self.cfg.model_ids:
            return _error(404, f"model {model!r} is not served here", kind="model_not_found")
        stream = req.get("stream", False)
        if not isinstance(stream, bool):
            return _error(400, "'stream' must be a boolean")
        log.info("request %s from %s <%s> model=%s stream=%s bytes=%d", request.path, caller.name, caller.email,
                 model, stream, len(body))
        b.inflight += 1
        b.last_activity = time.monotonic()
        try:
            return await self._proxy(request, body, stream, model)
        except _ClientGone as e:
            log.info("client disconnected mid-response")
            return e.resp
        finally:
            b.inflight -= 1
            b.last_activity = time.monotonic()

    async def _proxy(self, request, body, stream, model):
        b, cfg = self.backend, self.cfg
        resp: web.StreamResponse | None = None
        keepalive = _keepalive_bytes(stream, cfg.stream_keepalive_mode, model)
        if not b.is_warm() and not stream and cfg.cold_nonstream_mode == "reject":
            fut = asyncio.ensure_future(b.ensure())  # warm up in the background, tell the client to come back
            fut.add_done_callback(lambda f: f.cancelled() or f.exception())
            return _error(503, "backend is starting, retry later", cfg.retry_after_s)
        ensure = asyncio.ensure_future(b.ensure())
        hold = cfg.stream_keepalive if stream else cfg.cold_nonstream_mode == "keepalive"
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
        for attempt in range(2):
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
                if e.started or attempt == 1:
                    if resp is not None:
                        if not e.started:
                            await resp.write(_held_error("backend lost", stream))
                        await resp.write_eof()
                        return resp
                    return _error(503, "backend lost", cfg.retry_after_s)
                ensure = asyncio.ensure_future(b.ensure())  # connect failed before any byte: retry once
                if resp is not None:
                    await _keepalive_until(resp, ensure, cfg.keepalive_interval_s, keepalive)

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
            raise _BackendGone(started) from e


class _ClientGone(Exception):
    def __init__(self, resp):
        super().__init__("client gone")
        self.resp = resp


class _BackendGone(Exception):
    def __init__(self, started: bool):
        super().__init__("backend gone")
        self.started = started


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
    asg = boto3.client("autoscaling", region_name=cfg.region) if cfg.asg_name else None
    dynamodb = boto3.client("dynamodb", region_name=cfg.region) if cfg.api_keys_table else None
    web.run_app(make_app(cfg, ecs, asg=asg, dynamodb=dynamodb), host="0.0.0.0", port=cfg.listen_port, access_log=None)


if __name__ == "__main__":
    main()
