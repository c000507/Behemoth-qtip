"""Router unit tests: fake ECS client + a fake backend HTTP server on localhost. Run: python3 -m pytest -q"""

import asyncio
import json
import threading
import unittest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from router import Config, make_app


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

    def update_service(self, cluster, service, desiredCount):
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
        kw.setdefault("cold_nonstream_mode", "wait")  # most tests check scaling state right after the response
        cfg = Config(backend_port=self.backend.port, poll_interval_s=0.02, keepalive_interval_s=0.02,
                     max_wait_s=kw.pop("max_wait_s", 5), idle_timeout_s=kw.pop("idle_timeout_s", 60),
                     health_ttl_s=kw.pop("health_ttl_s", 10), **kw)
        self.ecs = ecs
        self.client = TestClient(TestServer(make_app(cfg, ecs, asg=asg)))
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

    def test_config_from_env(self):
        c = Config.from_env({"MODEL_IDS": "a, b", "IDLE_TIMEOUT_S": "60", "PROXY_PATHS": "/v1/chat/completions,/v1/completions"})
        self.assertEqual(c.model_ids, ["a", "b"])
        self.assertEqual(c.idle_timeout_s, 60)
        self.assertEqual(c.proxy_paths, ["/v1/chat/completions", "/v1/completions"])


if __name__ == "__main__":
    unittest.main()
