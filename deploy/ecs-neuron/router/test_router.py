"""Router unit tests: fake ECS client + a fake backend HTTP server on localhost. Run: python3 -m pytest -q"""

import asyncio
import json
import threading
import time
import unittest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from router import Config, ModelSpec, key_hash, make_app


class FakeDynamo:
    """The router's one DynamoDB call: get_item on the key table."""

    def __init__(self, keys: dict[str, tuple[str, str, bool]] | None = None, fail=False):
        self.items = {key_hash(k): {"key_hash": {"S": key_hash(k)}, "email": {"S": e}, "name": {"S": n},
                                    "enabled": {"BOOL": on}} for k, (e, n, on) in (keys or {}).items()}
        self.gets, self.fail = 0, fail

    def get_item(self, TableName, Key, ConsistentRead=False):
        self.gets += 1
        if self.fail:
            raise RuntimeError("dynamodb down")
        item = self.items.get(Key["key_hash"]["S"])
        return {"Item": item} if item else {}


class FakeECS:
    """Mimics the boto3 ECS calls the router uses. A task becomes RUNNING `boot_polls` list_tasks calls after
    desiredCount goes to 1; `kill()` simulates a spot interruption (task disappears, service reschedules)."""

    def __init__(self, boot_polls=2, never_start=False):
        self.lock = threading.Lock()
        self.desired = 0
        self.updates = []
        self.polls_since_up = 0
        self.boot_polls = boot_polls
        self.never_start = never_start
        self.calls = []

    def describe_services(self, cluster, services):
        self.calls.append("describe_services")
        with self.lock:
            return {"services": [{"serviceName": services[0], "desiredCount": self.desired}]}

    def update_service(self, cluster, service, desiredCount, **kw):
        self.calls.append("update_service")
        with self.lock:
            self.updates.append(desiredCount)
            self.desired = desiredCount
            self.polls_since_up = 0

    def list_tasks(self, cluster, serviceName, desiredStatus="RUNNING"):
        self.calls.append("list_tasks")
        with self.lock:
            if self.desired < 1 or self.never_start:
                return {"taskArns": []}
            self.polls_since_up += 1
            return {"taskArns": ["arn:task/1"]} if self.polls_since_up >= 1 else {"taskArns": []}

    def describe_tasks(self, cluster, tasks):
        self.calls.append("describe_tasks")
        with self.lock:
            status = "RUNNING" if self.polls_since_up >= self.boot_polls else "PROVISIONING"
        return {"tasks": [{"taskArn": tasks[0], "lastStatus": status, "attachments": [
            {"type": "ElasticNetworkInterface", "details": [
                {"name": "subnetId", "value": "subnet-1"}, {"name": "privateIPv4Address", "value": "127.0.0.1"}]}]}]}

    def list_container_instances(self, cluster, status):
        self.calls.append("list_container_instances")
        return {"containerInstanceArns": [c["containerInstanceArn"] for c in getattr(self, "cis", []) if c["status"] == status]}

    def describe_container_instances(self, cluster, containerInstances):
        return {"containerInstances": [c for c in getattr(self, "cis", []) if c["containerInstanceArn"] in containerInstances]}

    def update_container_instances_state(self, cluster, containerInstances, status):
        for c in getattr(self, "cis", []):
            if c["containerInstanceArn"] in containerInstances:
                c["status"] = status
        self.reactivated = list(containerInstances)

    def kill(self):
        with self.lock:
            self.polls_since_up = 0


class FakeASG:
    """Mimics describe_auto_scaling_groups / set_desired_capacity."""

    def __init__(self, desired=0, instances=None):
        self.desired, self.sets, self.unprotected = desired, [], []
        self.instances = instances if instances is not None else []

    def describe_auto_scaling_groups(self, AutoScalingGroupNames):
        return {"AutoScalingGroups": [{"AutoScalingGroupName": AutoScalingGroupNames[0], "DesiredCapacity": self.desired,
                                       "Instances": self.instances}]}

    def set_desired_capacity(self, AutoScalingGroupName, DesiredCapacity, HonorCooldown):
        self.sets.append(DesiredCapacity)
        self.desired = DesiredCapacity

    def set_instance_protection(self, AutoScalingGroupName, InstanceIds, ProtectedFromScaleIn):
        self.unprotected += InstanceIds
        for i in self.instances:
            if i["InstanceId"] in InstanceIds:
                i["ProtectedFromScaleIn"] = ProtectedFromScaleIn


def fake_backend(state):
    async def health(request):
        return web.json_response({"healthy": state["healthy"]}, status=200 if state["healthy"] else 503)

    async def chat(request):
        body = await request.json()
        state["requests"] += 1
        if body.get("stream"):
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            for tok in ["Hel", "lo"]:
                await resp.write(f"data: {json.dumps({'choices': [{'delta': {'content': tok}}]})}\n\n".encode())
                await asyncio.sleep(0.01)
            await resp.write(b"data: [DONE]\n\n")
            return resp
        return web.json_response({"choices": [{"message": {"role": "assistant", "content": "Hello"}}],
                                  "model": body.get("model")})

    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_post("/v1/chat/completions", chat)
    return app


class RouterTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.state = {"healthy": True, "requests": 0}
        self.backend = TestServer(fake_backend(self.state))
        await self.backend.start_server()

    async def asyncTearDown(self):
        if getattr(self, "client", None):
            await self.client.close()
        await self.backend.close()

    async def start(self, ecs, asg=None, **kw):
        kw.setdefault("cold_nonstream_mode", "wait")
        kw.setdefault("allow_anonymous", True)  # most tests check scaling state right after the response
        cfg = Config(backend_port=self.backend.port, poll_interval_s=0.02, keepalive_interval_s=0.02,
                     max_wait_s=kw.pop("max_wait_s", 5), idle_timeout_s=kw.pop("idle_timeout_s", 60),
                     health_ttl_s=kw.pop("health_ttl_s", 10), **kw)
        self.ecs = ecs
        self.client = TestClient(TestServer(make_app(cfg, ecs, asg=asg, dynamodb=getattr(self, "dynamo", None))))
        await self.client.start_server()
        return self.client

    async def test_models_local(self):
        c = await self.start(FakeECS(), model_ids=["cydonia-24b", "other"])
        r = await c.get("/v1/models")
        data = await r.json()
        self.assertEqual([m["id"] for m in data["data"]], ["cydonia-24b", "other"])
        self.assertEqual(self.ecs.calls, [])
        r = await c.get("/health")
        self.assertEqual((await r.json())["healthy"], True)
        self.assertEqual(self.ecs.calls, [])

    async def test_cold_nonstream_scales_up_and_proxies(self):
        c = await self.start(FakeECS())
        r = await c.post("/v1/chat/completions", json={"model": "cydonia-24b", "messages": []})
        self.assertEqual(r.status, 200)
        self.assertEqual((await r.json())["choices"][0]["message"]["content"], "Hello")
        self.assertEqual(self.ecs.updates, [1])
        # warm: no new scale-up, no ECS polling
        n = len(self.ecs.calls)
        r = await c.post("/v1/chat/completions", json={"messages": []})
        self.assertEqual(r.status, 200)
        self.assertEqual(self.ecs.calls[n:], [])

    async def test_single_flight(self):
        c = await self.start(FakeECS(boot_polls=5))
        rs = await asyncio.gather(*[c.post("/v1/chat/completions", json={"messages": []}) for _ in range(8)])
        self.assertTrue(all(r.status == 200 for r in rs))
        self.assertEqual(self.ecs.updates, [1])
        self.assertEqual(self.ecs.calls.count("describe_services"), 1)
        self.assertEqual(self.state["requests"], 8)

    async def test_stream_keepalive_then_passthrough(self):
        c = await self.start(FakeECS(boot_polls=4))
        r = await c.post("/v1/chat/completions", json={"stream": True, "messages": []})
        self.assertEqual(r.status, 200)
        self.assertEqual(r.headers["Content-Type"], "text/event-stream")
        text = (await r.read()).decode()
        # keepalives are empty chunks, so clients waiting for a first data event don't give up
        self.assertIn('"chatcmpl-wait"', text)
        self.assertIn('"delta": {"role": "assistant", "content": ""}', text)
        self.assertNotIn(": waiting", text)
        self.assertIn('"Hel"', text)
        self.assertTrue(text.rstrip().endswith("data: [DONE]"))

    async def test_stream_keepalive_comment_mode(self):
        c = await self.start(FakeECS(boot_polls=4), stream_keepalive_mode="comment")
        r = await c.post("/v1/chat/completions", json={"stream": True, "messages": []})
        text = (await r.read()).decode()
        self.assertIn(": waiting for backend to start", text)
        self.assertNotIn("chatcmpl-wait", text)
        self.assertIn('"Hel"', text)

    async def test_cold_nonstream_keepalive(self):
        c = await self.start(FakeECS(boot_polls=4), cold_nonstream_mode="keepalive")
        r = await c.post("/v1/chat/completions", json={"model": "cydonia-24b", "messages": []})
        self.assertEqual(r.status, 200)
        self.assertEqual(r.headers["Content-Type"], "application/json")
        raw = await r.read()
        self.assertTrue(raw.startswith(b" "))  # whitespace keepalives before the body
        self.assertEqual(json.loads(raw)["choices"][0]["message"]["content"], "Hello")
        self.assertEqual(self.ecs.updates, [1])
        # warm: plain passthrough, no leading whitespace
        r = await c.post("/v1/chat/completions", json={"messages": []})
        self.assertFalse((await r.read()).startswith(b" "))

    async def test_cold_nonstream_keepalive_startup_failure(self):
        c = await self.start(FakeECS(never_start=True), cold_nonstream_mode="keepalive", max_wait_s=0.2)
        r = await c.post("/v1/chat/completions", json={"messages": []})
        self.assertEqual(r.status, 200)  # already committed; the error is in the body
        self.assertIn("backend startup failed", json.loads(await r.read())["error"]["message"])

    async def test_warm_stream_passthrough_headers(self):
        c = await self.start(FakeECS())
        await c.post("/v1/chat/completions", json={"messages": []})
        r = await c.post("/v1/chat/completions", json={"stream": True, "messages": []})
        text = (await r.read()).decode()
        self.assertNotIn(": waiting", text)
        self.assertIn('"lo"', text)

    async def test_startup_timeout_503(self):
        c = await self.start(FakeECS(never_start=True), max_wait_s=0.2, retry_after_s=60)
        r = await c.post("/v1/chat/completions", json={"messages": []})
        self.assertEqual(r.status, 503)
        self.assertEqual(r.headers["Retry-After"], "60")
        r = await c.post("/v1/chat/completions", json={"stream": True, "messages": []})
        text = (await r.read()).decode()
        self.assertIn("backend startup failed", text)
        self.assertIn("[DONE]", text)

    async def test_unhealthy_backend_times_out(self):
        self.state["healthy"] = False
        c = await self.start(FakeECS(), max_wait_s=0.2)
        r = await c.post("/v1/chat/completions", json={"messages": []})
        self.assertEqual(r.status, 503)
        self.assertEqual(self.state["requests"], 0)

    async def test_asg_scaled_directly(self):
        asg = FakeASG()
        c = await self.start(FakeECS(), asg=asg, asg_name="behemoth-asg", idle_timeout_s=0.3)
        r = await c.post("/v1/chat/completions", json={"model": "cydonia-24b", "messages": []})
        self.assertEqual(r.status, 200)
        self.assertEqual(asg.sets, [1])               # scale-out without waiting for the capacity provider
        for _ in range(60):                           # idle: service to 0, then ASG to 0 once no task is left
            await asyncio.sleep(0.1)
            if asg.sets == [1, 0]:
                break
        self.assertEqual(self.ecs.updates, [1, 0])
        self.assertEqual(asg.sets, [1, 0])

    async def test_stuck_protected_instance_released(self):
        # desired already 0 but the idle instance is still in service and protected (the scale-in was cancelled)
        asg = FakeASG(desired=0, instances=[{"InstanceId": "i-1", "LifecycleState": "InService", "ProtectedFromScaleIn": True}])
        c = await self.start(FakeECS(), asg=asg, asg_name="behemoth-asg", idle_timeout_s=0.2)
        self.client.app  # started; the reaper sees state "unknown" first, then idle-stops (nothing to stop) and releases
        for _ in range(60):
            await asyncio.sleep(0.1)
            if asg.unprotected:
                break
        self.assertEqual(asg.unprotected, ["i-1"])
        self.assertEqual(asg.sets[-1], 0)

    async def test_draining_warm_pool_instance_reactivated(self):
        ecs = FakeECS(boot_polls=3)
        ecs.cis = [{"containerInstanceArn": "ci-1", "status": "DRAINING", "agentConnected": True, "runningTasksCount": 0, "pendingTasksCount": 0},
                   {"containerInstanceArn": "ci-2", "status": "DRAINING", "agentConnected": True, "runningTasksCount": 1, "pendingTasksCount": 0}]
        c = await self.start(ecs)
        r = await c.post("/v1/chat/completions", json={"model": "cydonia-24b", "messages": []})
        self.assertEqual(r.status, 200)
        self.assertEqual(ecs.reactivated, ["ci-1"])  # only the idle one; never touch an instance running tasks

    async def test_asg_failure_is_not_fatal(self):
        class Broken(FakeASG):
            def set_desired_capacity(self, **kw):
                raise RuntimeError("AccessDenied")
        c = await self.start(FakeECS(), asg=Broken(), asg_name="behemoth-asg")
        r = await c.post("/v1/chat/completions", json={"model": "cydonia-24b", "messages": []})
        self.assertEqual(r.status, 200)

    async def test_idle_shutdown(self):
        c = await self.start(FakeECS(), idle_timeout_s=0.3)
        r = await c.post("/v1/chat/completions", json={"messages": []})
        self.assertEqual(r.status, 200)
        for _ in range(100):
            await asyncio.sleep(0.05)
            if self.ecs.updates[-1] == 0:
                break
        self.assertEqual(self.ecs.updates, [1, 0])
        r = await c.get("/health")
        self.assertEqual((await r.json())["backend"], "stopped")
        # next request wakes it again
        r = await c.post("/v1/chat/completions", json={"messages": []})
        self.assertEqual(r.status, 200)
        self.assertEqual(self.ecs.updates, [1, 0, 1])

    async def test_no_idle_shutdown_while_inflight(self):
        c = await self.start(FakeECS(boot_polls=40), idle_timeout_s=0.1)  # startup takes ~0.8 s > idle timeout
        r = await c.post("/v1/chat/completions", json={"messages": []})
        self.assertEqual(r.status, 200)
        self.assertEqual(self.ecs.updates[0], 1)
        self.assertNotIn(0, self.ecs.updates[:1])

    async def test_backend_loss_restarts(self):
        c = await self.start(FakeECS(), health_ttl_s=0)
        r = await c.post("/v1/chat/completions", json={"messages": []})
        self.assertEqual(r.status, 200)
        # spot interruption: health fails, task disappears and comes back after a few polls
        self.state["healthy"] = False
        self.ecs.kill()

        async def recover():
            await asyncio.sleep(0.15)
            self.state["healthy"] = True

        asyncio.ensure_future(recover())
        r = await c.post("/v1/chat/completions", json={"messages": []})
        self.assertEqual(r.status, 200)
        self.assertGreater(self.ecs.calls.count("list_tasks"), 1)

    async def test_backend_connection_refused_then_503(self):
        c = await self.start(FakeECS(), max_wait_s=0.3)
        r = await c.post("/v1/chat/completions", json={"messages": []})
        self.assertEqual(r.status, 200)
        await self.backend.close()  # backend process dies
        r = await c.post("/v1/chat/completions", json={"messages": []})
        self.assertEqual(r.status, 503)
        self.assertIn("Retry-After", r.headers)

    async def test_reject_mode(self):
        c = await self.start(FakeECS(), cold_nonstream_mode="reject", retry_after_s=30)
        r = await c.post("/v1/chat/completions", json={"messages": []})
        self.assertEqual(r.status, 503)
        self.assertEqual(r.headers["Retry-After"], "30")
        for _ in range(100):
            await asyncio.sleep(0.02)
            if self.ecs.updates:
                break
        self.assertEqual(self.ecs.updates, [1])

    async def test_api_key(self):
        c = await self.start(FakeECS(), api_key="k")
        r = await c.post("/v1/chat/completions", json={"messages": []})
        self.assertEqual(r.status, 401)
        r = await c.post("/v1/chat/completions", json={"messages": []}, headers={"Authorization": "Bearer k"})
        self.assertEqual(r.status, 200)

    async def test_dynamo_keys(self):
        self.dynamo = FakeDynamo({"good": ("ada@example.com", "Ada", True), "old": ("bob@example.com", "Bob", False)})
        c = await self.start(FakeECS(), allow_anonymous=False, api_keys_table="keys")
        with self.assertLogs("router", "INFO") as logs:
            r = await c.post("/v1/chat/completions", json={"messages": []}, headers={"Authorization": "Bearer good"})
        self.assertEqual(r.status, 200)
        self.assertTrue(any("Ada <ada@example.com>" in m for m in logs.output))
        for auth in ["Bearer old", "Bearer nope", "Basic good", ""]:
            r = await c.post("/v1/chat/completions", json={"messages": []}, headers={"Authorization": auth})
            self.assertEqual(r.status, 401, auth)
        self.assertEqual(self.ecs.updates, [1])  # only the approved request woke the backend
        gets = self.dynamo.gets
        for _ in range(3):  # cached: no further lookups for known keys, approved or not
            await c.post("/v1/chat/completions", json={"messages": []}, headers={"Authorization": "Bearer good"})
            await c.post("/v1/chat/completions", json={"messages": []}, headers={"Authorization": "Bearer nope"})
        self.assertEqual(self.dynamo.gets, gets)

    async def test_dynamo_down_fails_closed(self):
        self.dynamo = FakeDynamo(fail=True)
        c = await self.start(FakeECS(), allow_anonymous=False, api_keys_table="keys")
        r = await c.post("/v1/chat/completions", json={"messages": []}, headers={"Authorization": "Bearer x"})
        self.assertEqual(r.status, 401)
        self.assertEqual(self.ecs.updates, [])

    async def test_key_cache_expiry_revokes(self):
        self.dynamo = FakeDynamo({"k": ("a@example.com", "A", True)})
        c = await self.start(FakeECS(), allow_anonymous=False, api_keys_table="keys", key_cache_s=0.05)
        h = {"Authorization": "Bearer k"}
        self.assertEqual((await c.post("/v1/chat/completions", json={"messages": []}, headers=h)).status, 200)
        self.dynamo.items[key_hash("k")]["enabled"] = {"BOOL": False}
        await asyncio.sleep(0.1)
        self.assertEqual((await c.post("/v1/chat/completions", json={"messages": []}, headers=h)).status, 401)

    async def test_models_need_key(self):
        self.dynamo = FakeDynamo({"good": ("ada@example.com", "Ada", True)})
        c = await self.start(FakeECS(), allow_anonymous=False, api_keys_table="keys")
        self.assertEqual((await c.get("/v1/models")).status, 401)
        self.assertEqual((await c.get("/v1/models", headers={"Authorization": "Bearer bad"})).status, 401)
        r = await c.get("/v1/models", headers={"Authorization": "Bearer good"})
        self.assertEqual(r.status, 200)
        self.assertEqual([m["id"] for m in (await r.json())["data"]], ["cydonia-24b"])
        self.assertEqual((await c.get("/health")).status, 200)  # the ECS health check stays open
        self.assertEqual(self.ecs.calls, [])

    async def test_no_key_fails_closed(self):
        c = await self.start(FakeECS(), allow_anonymous=False)
        r = await c.post("/v1/chat/completions", json={"messages": []})
        self.assertEqual(r.status, 401)
        self.assertEqual(self.ecs.updates, [])
        with self.assertRaises(SystemExit):
            Config().check()
        Config(api_key="k").check()
        Config(allow_anonymous=True).check()
        Config(api_keys_table="t").check()

    async def test_invalid_requests_do_not_wake(self):
        c = await self.start(FakeECS(), cold_nonstream_mode="reject", max_body_bytes=1024)
        for body, status in [(b"not-json", 400), (b"[1]", 400), (b'{"model": "other"}', 404),
                             (b'{"model": 5}', 404), (b'{"stream": "yes"}', 400),
                             (b'{"messages": "' + b"x" * 2048 + b'"}', 413)]:
            r = await c.post("/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
            self.assertEqual(r.status, status, body[:40])
        await asyncio.sleep(0.1)
        self.assertEqual(self.ecs.updates, [])

    async def test_inflight_limit(self):
        c = await self.start(FakeECS(), max_inflight=0)
        r = await c.post("/v1/chat/completions", json={"messages": []})
        self.assertEqual(r.status, 429)
        self.assertEqual(self.ecs.updates, [])

    def test_config_from_env(self):
        c = Config.from_env({"MODEL_IDS": "a, b", "IDLE_TIMEOUT_S": "60", "PROXY_PATHS": "/v1/chat/completions,/v1/completions"})
        self.assertEqual(c.model_ids, ["a", "b"])
        self.assertEqual(c.idle_timeout_s, 60)
        self.assertEqual(c.proxy_paths, ["/v1/chat/completions", "/v1/completions"])
        self.assertFalse(c.allow_anonymous)
        self.assertTrue(Config.from_env({"ALLOW_ANONYMOUS": "true"}).allow_anonymous)


# ---------------------------------------------------------------------------------------------- multi-model
class MultiECS:
    """Several services in one cluster. A service's task is RUNNING `boot_polls` list_tasks calls after it goes to
    desiredCount 1, on capacity providers in `available` only (None: any); `kill(svc)` = spot interruption."""

    def __init__(self, services, boot_polls=2, available=None, strategy=None):
        self.svc = {s: {"desired": 0, "polls": 0, "cp": (strategy or {}).get(s)} for s in services}
        self.boot_polls, self.available = boot_polls, available
        self.updates: list[tuple] = []  # (service, desiredCount, capacity provider or None)
        self.calls: list[str] = []
        self.cis: list[dict] = []
        self.reactivated: list[str] = []
        self.lock = threading.Lock()

    def describe_services(self, cluster, services):
        self.calls.append("describe_services")
        out = []
        for name in services:
            st = self.svc[name]
            d = {"serviceName": name, "desiredCount": st["desired"]}
            if st["cp"]:
                d["capacityProviderStrategy"] = [{"capacityProvider": st["cp"], "weight": 1, "base": 0}]
            out.append(d)
        return {"services": out}

    def update_service(self, cluster, service, desiredCount, capacityProviderStrategy=None, forceNewDeployment=False):
        self.calls.append("update_service")
        with self.lock:
            st = self.svc[service]
            if capacityProviderStrategy:
                st["cp"] = capacityProviderStrategy[0]["capacityProvider"]
            self.updates.append((service, desiredCount, capacityProviderStrategy[0]["capacityProvider"]
                                 if capacityProviderStrategy else None))
            st["desired"], st["polls"] = desiredCount, 0

    def _placeable(self, st):
        return self.available is None or st["cp"] in self.available

    def list_tasks(self, cluster, serviceName, desiredStatus="RUNNING"):
        self.calls.append("list_tasks")
        with self.lock:
            st = self.svc[serviceName]
            if st["desired"] < 1 or not self._placeable(st):
                return {"taskArns": []}
            st["polls"] += 1
            return {"taskArns": [f"arn:task/{serviceName}"]}

    def describe_tasks(self, cluster, tasks):
        svc = tasks[0].split("/", 1)[1]
        with self.lock:
            status = "RUNNING" if self.svc[svc]["polls"] >= self.boot_polls else "PROVISIONING"
        return {"tasks": [{"taskArn": tasks[0], "lastStatus": status, "attachments": [
            {"type": "ElasticNetworkInterface", "details": [{"name": "privateIPv4Address", "value": "127.0.0.1"}]}]}]}

    def list_container_instances(self, cluster, status):
        return {"containerInstanceArns": [c["containerInstanceArn"] for c in self.cis if c["status"] == status]}

    def describe_container_instances(self, cluster, containerInstances):
        return {"containerInstances": [c for c in self.cis if c["containerInstanceArn"] in containerInstances]}

    def update_container_instances_state(self, cluster, containerInstances, status):
        self.reactivated += containerInstances
        for c in self.cis:
            if c["containerInstanceArn"] in containerInstances:
                c["status"] = status

    def kill(self, service):
        with self.lock:
            self.svc[service]["polls"] = 0

    def desired(self, service):
        return self.svc[service]["desired"]


class MultiASG:
    """Per-name ASGs; `fail` = names whose launches fail (a Failed scaling activity)."""

    def __init__(self, names, fail=(), instances=None):
        self.g = {n: {"desired": 0, "instances": list((instances or {}).get(n, []))} for n in names}
        self.fail, self.sets = set(fail), []

    def describe_auto_scaling_groups(self, AutoScalingGroupNames):
        n = AutoScalingGroupNames[0]
        return {"AutoScalingGroups": [{"AutoScalingGroupName": n, "DesiredCapacity": self.g[n]["desired"],
                                       "Instances": self.g[n]["instances"]}]}

    def set_desired_capacity(self, AutoScalingGroupName, DesiredCapacity, HonorCooldown):
        self.sets.append((AutoScalingGroupName, DesiredCapacity))
        self.g[AutoScalingGroupName]["desired"] = DesiredCapacity

    def set_instance_protection(self, AutoScalingGroupName, InstanceIds, ProtectedFromScaleIn):
        pass

    def describe_scaling_activities(self, AutoScalingGroupName, MaxRecords):
        if AutoScalingGroupName in self.fail and self.g[AutoScalingGroupName]["desired"] > 0:
            return {"Activities": [{"StatusCode": "Failed", "StartTime": time.time(),
                                    "StatusMessage": "We currently do not have sufficient trn1.2xlarge capacity"}]}
        return {"Activities": []}


def tagged_backend(state, tag):
    """Fake neuron_server answering with its tag; state: healthy, requests, drop (close before answering),
    cut (stream one chunk then close), delay (s before answering)."""

    async def health(request):
        return web.json_response({}, status=200 if state["healthy"] else 503)

    async def chat(request):
        body = await request.json()
        state["requests"] += 1
        if state.get("delay"):
            await asyncio.sleep(state["delay"])
        if state.get("drop"):
            state["drop"] -= 1
            request.transport.abort()
            return web.Response()
        if body.get("stream"):
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            await resp.write(f"data: {json.dumps({'choices': [{'delta': {'content': tag}}]})}\n\n".encode())
            if state.get("cut"):
                await asyncio.sleep(0.02)
                request.transport.abort()
                return resp
            await resp.write(b"data: [DONE]\n\n")
            return resp
        return web.json_response({"choices": [{"message": {"role": "assistant", "content": tag}}],
                                  "model": body.get("model")})

    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_post("/v1/chat/completions", chat)
    return app


class MultiModelTest(unittest.IsolatedAsyncioTestCase):
    """Two models, ornith (default) and cydonia, each its own service, ASG and fake backend (own port)."""

    async def asyncSetUp(self):
        self.st = {n: {"healthy": True, "requests": 0} for n in ("ornith", "cydonia")}
        self.srv = {n: TestServer(tagged_backend(self.st[n], n)) for n in self.st}
        for s in self.srv.values():
            await s.start_server()
        self.client = None

    async def asyncTearDown(self):
        if self.client:
            await self.client.close()
        for s in self.srv.values():
            await s.close()

    def spec(self, name, ids, **kw):
        return ModelSpec(name=name, ids=ids, service=f"svc-{name}", port=self.srv[name].port, **kw)

    async def start(self, ecs=None, asg=None, specs=None, **kw):
        kw.setdefault("cold_nonstream_mode", "wait")
        kw.setdefault("allow_anonymous", True)
        specs = specs or [self.spec("ornith", ["ornith-1.5"]), self.spec("cydonia", ["cydonia-24b", "cydonia"])]
        cfg = Config(poll_interval_s=0.02, keepalive_interval_s=0.02, max_wait_s=kw.pop("max_wait_s", 5),
                     idle_timeout_s=kw.pop("idle_timeout_s", 60), models=specs, default_model="ornith", **kw)
        cfg.check()
        self.ecs = ecs or MultiECS([m.service for m in specs])
        self.client = TestClient(TestServer(make_app(cfg, self.ecs, asg=asg)))
        await self.client.start_server()
        return self.client

    async def chat(self, model=None, stream=False):
        body = {"messages": [], "stream": stream}
        if model:
            body["model"] = model
        return await self.client.post("/v1/chat/completions", json=body)

    async def wait_for(self, pred, timeout=5.0):
        for _ in range(int(timeout / 0.05)):
            if pred():
                return True
            await asyncio.sleep(0.05)
        return pred()

    async def test_models_lists_all_ids_default_first(self):
        c = await self.start()
        data = await (await c.get("/v1/models")).json()
        self.assertEqual([m["id"] for m in data["data"]], ["ornith-1.5", "cydonia-24b", "cydonia"])
        h = await (await c.get("/health")).json()
        self.assertEqual(set(h["models"]), {"ornith", "cydonia"})
        self.assertEqual(self.ecs.calls, [])  # listing never wakes or even looks at a backend

    async def test_routes_to_own_service_only(self):
        c = await self.start()
        r = await self.chat("cydonia")
        self.assertEqual(r.status, 200)
        self.assertEqual((await r.json())["choices"][0]["message"]["content"], "cydonia")
        self.assertEqual(self.ecs.updates, [("svc-cydonia", 1, None)])  # ornith untouched
        self.assertEqual(self.st["ornith"]["requests"], 0)
        r = await self.chat()  # no model field: the default model
        self.assertEqual((await r.json())["choices"][0]["message"]["content"], "ornith")
        self.assertEqual(self.ecs.updates, [("svc-cydonia", 1, None), ("svc-ornith", 1, None)])
        h = await (await c.get("/health")).json()
        self.assertEqual(h["active"], 2)  # several models at once (cap 2)
        n = len(self.ecs.calls)
        r = await self.chat("cydonia-24b", stream=True)  # warm, second id of the same model
        self.assertIn('"cydonia"', (await r.read()).decode())
        self.assertEqual(self.ecs.calls[n:], [])

    async def test_unknown_model_404(self):
        c = await self.start()
        r = await self.chat("gpt-4")
        self.assertEqual(r.status, 404)
        err = (await r.json())["error"]
        self.assertEqual(err["type"], "model_not_found")
        self.assertIn("gpt-4", err["message"])
        self.assertEqual(self.ecs.updates, [])

    async def test_cap_reject(self):
        c = await self.start(max_active_models=1, cap_policy="reject", retry_after_s=90)
        self.assertEqual((await self.chat("ornith-1.5")).status, 200)
        r = await self.chat("cydonia")
        self.assertEqual(r.status, 503)
        self.assertEqual((await r.json())["error"]["type"], "capacity_exceeded")
        self.assertTrue(5 <= int(r.headers["Retry-After"]) <= 90)
        self.assertEqual(self.ecs.desired("svc-cydonia"), 0)
        self.assertEqual(self.ecs.desired("svc-ornith"), 1)
        self.assertEqual((await self.chat("ornith-1.5")).status, 200)  # the active model is unaffected

    async def test_cap_evict_longest_idle(self):
        c = await self.start(max_active_models=1, cap_policy="evict", cap_min_idle_s=0)
        self.assertEqual((await self.chat("ornith-1.5")).status, 200)
        r = await self.chat("cydonia")
        self.assertEqual(r.status, 200)
        self.assertEqual(self.ecs.updates, [("svc-ornith", 1, None), ("svc-ornith", 0, None), ("svc-cydonia", 1, None)])
        h = await (await c.get("/health")).json()
        self.assertEqual(h["models"]["ornith"]["state"], "stopped")
        self.assertEqual(h["active"], 1)

    async def test_cap_evict_never_evicts_busy_or_recent(self):
        c = await self.start(max_active_models=1, cap_policy="evict", cap_min_idle_s=0)
        self.assertEqual((await self.chat("ornith-1.5")).status, 200)
        self.st["ornith"]["delay"] = 0.5
        busy = asyncio.ensure_future(self.chat("ornith-1.5"))
        await asyncio.sleep(0.15)
        r = await self.chat("cydonia")  # ornith has a request in flight: not evictable
        self.assertEqual(r.status, 503)
        self.assertEqual((await busy).status, 200)
        self.assertEqual(self.ecs.desired("svc-cydonia"), 0)
        await self.client.close()
        self.st["ornith"]["delay"] = 0
        c = await self.start(max_active_models=1, cap_policy="evict", cap_min_idle_s=60)
        self.assertEqual((await self.chat("ornith-1.5")).status, 200)
        self.assertEqual((await self.chat("cydonia")).status, 503)  # idle < CAP_MIN_IDLE_S

    async def test_cap_counts_models_found_running_after_router_restart(self):
        ecs = MultiECS(["svc-ornith", "svc-cydonia"])
        ecs.svc["svc-cydonia"]["desired"] = 1  # left running by the previous router
        c = await self.start(ecs=ecs, max_active_models=1, cap_policy="reject")
        self.assertEqual((await self.chat("ornith-1.5")).status, 503)
        self.assertEqual(ecs.desired("svc-ornith"), 0)

    async def test_concurrent_cold_models_respect_cap(self):
        c = await self.start(max_active_models=1, cap_policy="reject")
        rs = await asyncio.gather(self.chat("ornith-1.5"), self.chat("cydonia"))
        self.assertEqual(sorted(r.status for r in rs), [200, 503])
        self.assertEqual(len([u for u in self.ecs.updates if u[1] == 1]), 1)

    async def test_idle_scale_down_per_model(self):
        specs = [self.spec("ornith", ["ornith-1.5"], idle_timeout_s=0.3), self.spec("cydonia", ["cydonia"])]
        c = await self.start(specs=specs, idle_timeout_s=60)
        self.assertEqual((await self.chat("ornith-1.5")).status, 200)
        self.assertEqual((await self.chat("cydonia")).status, 200)
        self.assertTrue(await self.wait_for(lambda: self.ecs.desired("svc-ornith") == 0))
        self.assertEqual(self.ecs.desired("svc-cydonia"), 1)  # its own (longer) idle timeout
        h = await (await c.get("/health")).json()
        self.assertEqual((h["models"]["ornith"]["state"], h["models"]["cydonia"]["state"]), ("stopped", "ready"))

    async def test_min_count_never_idle_stopped(self):
        specs = [self.spec("ornith", ["ornith-1.5"], idle_timeout_s=0.2, min_count=1), self.spec("cydonia", ["cydonia"])]
        c = await self.start(specs=specs)
        self.assertEqual((await self.chat("ornith-1.5")).status, 200)
        await asyncio.sleep(1.2)
        self.assertEqual(self.ecs.desired("svc-ornith"), 1)

    def fallback_specs(self, after=300.0):
        return [self.spec("ornith", ["ornith-1.5"], asg_name="asg-ornith", capacity_provider="cp-ornith",
                          fallback_asg_name="asg-ornith-od", fallback_capacity_provider="cp-ornith-od",
                          spot_fallback_after_s=after, idle_timeout_s=0.4),
                self.spec("cydonia", ["cydonia"], asg_name="asg-cydonia")]

    async def test_spot_fallback_on_failed_launch_and_back(self):
        asg = MultiASG(["asg-ornith", "asg-ornith-od", "asg-cydonia"], fail=["asg-ornith"])
        ecs = MultiECS(["svc-ornith", "svc-cydonia"], available={"cp-ornith-od"}, strategy={"svc-ornith": "cp-ornith"})
        c = await self.start(ecs=ecs, asg=asg, specs=self.fallback_specs())
        r = await self.chat("ornith-1.5")
        self.assertEqual(r.status, 200)
        self.assertIn(("svc-ornith", 1, "cp-ornith-od"), ecs.updates)
        self.assertEqual(asg.g["asg-ornith-od"]["desired"], 1)
        self.assertEqual(asg.g["asg-ornith"]["desired"], 0)
        self.assertTrue((await (await c.get("/health")).json())["models"]["ornith"]["on_demand_fallback"])
        # idle: back to 0 AND back to the spot capacity provider, both ASGs at 0
        self.assertTrue(await self.wait_for(lambda: ecs.desired("svc-ornith") == 0))
        self.assertEqual(ecs.updates[-1], ("svc-ornith", 0, "cp-ornith"))
        self.assertTrue(await self.wait_for(lambda: asg.g["asg-ornith-od"]["desired"] == 0))

    async def test_spot_fallback_after_timeout_without_instance(self):
        asg = MultiASG(["asg-ornith", "asg-ornith-od", "asg-cydonia"])
        ecs = MultiECS(["svc-ornith", "svc-cydonia"], available={"cp-ornith-od"}, strategy={"svc-ornith": "cp-ornith"})
        c = await self.start(ecs=ecs, asg=asg, specs=self.fallback_specs(after=0.2))
        self.assertEqual((await self.chat("ornith-1.5")).status, 200)
        self.assertIn(("svc-ornith", 1, "cp-ornith-od"), ecs.updates)

    async def test_spot_fallback_by_timeout_when_activities_denied(self):
        class Denied(MultiASG):
            def describe_scaling_activities(self, **kw):
                raise RuntimeError("AccessDenied: autoscaling:DescribeScalingActivities")
        asg = Denied(["asg-ornith", "asg-ornith-od", "asg-cydonia"], fail=["asg-ornith"])
        ecs = MultiECS(["svc-ornith", "svc-cydonia"], available={"cp-ornith-od"}, strategy={"svc-ornith": "cp-ornith"})
        c = await self.start(ecs=ecs, asg=asg, specs=self.fallback_specs(after=0.2))
        self.assertEqual((await self.chat("ornith-1.5")).status, 200)
        self.assertIn(("svc-ornith", 1, "cp-ornith-od"), ecs.updates)

    async def test_no_fallback_while_spot_instance_boots(self):
        asg = MultiASG(["asg-ornith", "asg-ornith-od", "asg-cydonia"],
                       instances={"asg-ornith": [{"InstanceId": "i-1", "LifecycleState": "Pending"}]})
        ecs = MultiECS(["svc-ornith", "svc-cydonia"], boot_polls=15, strategy={"svc-ornith": "cp-ornith"})
        c = await self.start(ecs=ecs, asg=asg, specs=self.fallback_specs(after=0.05))
        self.assertEqual((await self.chat("ornith-1.5")).status, 200)
        self.assertNotIn(("svc-ornith", 1, "cp-ornith-od"), ecs.updates)
        self.assertEqual(asg.g["asg-ornith-od"]["desired"], 0)

    async def test_spot_interruption_rewake_and_retry(self):
        c = await self.start(health_ttl_s=10, lost_retries=1)
        self.assertEqual((await self.chat("cydonia")).status, 200)
        # interruption: the instance goes away under a warm (health-cached) backend; the replacement task needs polls
        self.st["cydonia"].update(drop=1, healthy=False)
        self.ecs.kill("svc-cydonia")

        async def replacement_up():
            await asyncio.sleep(0.2)
            self.st["cydonia"]["healthy"] = True

        asyncio.ensure_future(replacement_up())
        with self.assertLogs("router", "INFO") as logs:
            r = await self.chat("cydonia")
        self.assertEqual(r.status, 200)
        self.assertEqual((await r.json())["choices"][0]["message"]["content"], "cydonia")
        self.assertTrue(any("retrying request after backend loss" in m for m in logs.output))
        self.assertEqual(self.st["cydonia"]["requests"], 3)  # first, the dropped one, the retry

    async def test_warm_nonstream_loss_retry_is_held_with_keepalives(self):
        c = await self.start(cold_nonstream_mode="keepalive")
        self.assertEqual((await self.chat("cydonia")).status, 200)
        self.st["cydonia"].update(drop=1, healthy=False)
        self.ecs.kill("svc-cydonia")

        async def replacement_up():
            await asyncio.sleep(0.2)
            self.st["cydonia"]["healthy"] = True

        asyncio.ensure_future(replacement_up())
        r = await self.chat("cydonia")
        raw = await r.read()
        self.assertEqual(r.status, 200)
        self.assertTrue(raw.startswith(b" "))  # keepalive bytes while the model is re-woken (Cloudflare 100 s limit)
        self.assertEqual(json.loads(raw)["choices"][0]["message"]["content"], "cydonia")

    async def test_backend_loss_reject_mode_is_503_and_rewakes(self):
        c = await self.start(cold_nonstream_mode="reject")
        self.st["cydonia"]["delay"] = 0
        r = await self.chat("cydonia")  # cold + reject: 503, warms up in the background
        self.assertEqual(r.status, 503)
        self.assertTrue(await self.wait_for(lambda: (self.client.app is not None) and self.ecs.desired("svc-cydonia") == 1))
        for _ in range(100):
            if (await (await self.client.get("/health")).json())["models"]["cydonia"]["state"] == "ready":
                break
            await asyncio.sleep(0.02)
        self.assertEqual((await self.chat("cydonia")).status, 200)
        self.st["cydonia"]["drop"] = 1
        r = await self.chat("cydonia")
        self.assertEqual(r.status, 503)
        self.assertIn("restarting", (await r.json())["error"]["message"])

    async def test_backend_loss_without_retry_is_503(self):
        c = await self.start(lost_retries=0)
        self.assertEqual((await self.chat("cydonia")).status, 200)
        self.st["cydonia"]["drop"] = 1
        r = await self.chat("cydonia")
        self.assertEqual(r.status, 503)
        self.assertIn("Retry-After", r.headers)

    async def test_stream_cut_mid_response_gets_error_event(self):
        c = await self.start()
        self.assertEqual((await self.chat("cydonia")).status, 200)
        self.st["cydonia"]["cut"] = True
        r = await self.chat("cydonia", stream=True)
        text = (await r.read()).decode()
        self.assertIn('"cydonia"', text)  # what was generated before the loss
        self.assertIn("backend lost mid-response", text)
        self.assertTrue(text.rstrip().endswith("data: [DONE]"))
        self.assertEqual(self.st["cydonia"]["requests"], 2)  # never re-sent once bytes went out
        h = await (await c.get("/health")).json()
        self.assertNotEqual(h["models"]["cydonia"]["state"], "ready")  # the next request re-wakes it

    async def test_draining_reactivation_scoped_to_own_asg(self):
        asg = MultiASG(["asg-ornith", "asg-cydonia"], instances={"asg-ornith": [{"InstanceId": "i-orn"}],
                                                                 "asg-cydonia": [{"InstanceId": "i-cyd"}]})
        ecs = MultiECS(["svc-ornith", "svc-cydonia"], boot_polls=3)
        idle = {"agentConnected": True, "runningTasksCount": 0, "pendingTasksCount": 0, "status": "DRAINING"}
        ecs.cis = [{"containerInstanceArn": "ci-orn", "ec2InstanceId": "i-orn", **idle},
                   {"containerInstanceArn": "ci-cyd", "ec2InstanceId": "i-cyd", **idle}]
        specs = [self.spec("ornith", ["ornith-1.5"], asg_name="asg-ornith"), self.spec("cydonia", ["cydonia"], asg_name="asg-cydonia")]
        c = await self.start(ecs=ecs, asg=asg, specs=specs)
        self.assertEqual((await self.chat("cydonia")).status, 200)
        self.assertEqual(ecs.reactivated, ["ci-cyd"])  # never another model's instance

    def test_models_json_from_env_and_checks(self):
        env = {"MODELS_JSON": json.dumps([
            {"name": "ornith", "ids": ["o"], "service": "behemoth-backend", "asg_name": "a", "idle_timeout_s": 120,
             "fallback_asg_name": None, "unknown_future_field": 1},
            {"name": "cydonia", "ids": ["c"], "service": "behemoth-backend-cydonia", "cold_nonstream_mode": "reject"}]),
            "DEFAULT_MODEL": "ornith", "MAX_ACTIVE_MODELS": "3", "CAP_POLICY": "reject", "IDLE_TIMEOUT_S": "300",
            "LOST_RETRIES": "2", "STREAM_LOSS_ERROR": "false", "ALLOW_ANONYMOUS": "true"}
        c = Config.from_env(env)
        c.check()
        ms = {m.name: m for m in c.resolved_models()}
        self.assertEqual((ms["ornith"].idle_timeout_s, ms["cydonia"].idle_timeout_s), (120, 300))
        self.assertEqual((ms["ornith"].cold_nonstream_mode, ms["cydonia"].cold_nonstream_mode), ("keepalive", "reject"))
        self.assertEqual((c.max_active_models, c.cap_policy, c.lost_retries, c.stream_loss_error), (3, "reject", 2, False))
        bad = [dict(models=[ModelSpec("a", ["x"], "s1"), ModelSpec("b", ["x"], "s2")]),  # duplicate id
               dict(models=[ModelSpec("a", ["x"], "s1")], default_model="zz"),
               dict(cap_policy="lru"),
               dict(models=[ModelSpec("a", ["x"], "s1", fallback_asg_name="f")])]  # fallback without providers
        for kw in bad:
            with self.assertRaises(SystemExit, msg=str(kw)):
                Config(allow_anonymous=True, **kw).check()


if __name__ == "__main__":
    unittest.main()
