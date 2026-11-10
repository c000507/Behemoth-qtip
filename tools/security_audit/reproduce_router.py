"""Local router probes with fake AWS clients; no AWS calls or provisioning.

Requires aiohttp. Demonstrates auth defaults, malformed-request scale-out, and
body-size mismatch. Actual repository router runs against localhost test servers.
"""
import asyncio
import json
from pathlib import Path
import sys

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'deploy/ecs-neuron/router'))
from router import Config, make_app
from test_router import FakeECS


async def cold(api_key):
    ecs=FakeECS(never_start=True)
    cfg=Config(api_key=api_key,cold_nonstream_mode='reject',max_wait_s=.05,poll_interval_s=.01,
               idle_timeout_s=3600)
    client=TestClient(TestServer(make_app(cfg,ecs)))
    await client.start_server()
    try:
        r=await client.post('/v1/chat/completions',data=b'not-json')
        await r.read()
        await asyncio.sleep(.15)
        return {'status':r.status,'fake_aws_desired_updates':ecs.updates}
    finally:
        await client.close()


async def oversized():
    async def health(request):return web.json_response({'healthy':True})
    async def chat(request):
        await request.read()  # Match native backend's 4-MiB body cap.
        return web.json_response({'ok':True})
    app=web.Application(client_max_size=4*1024**2)
    app.router.add_get('/health',health);app.router.add_post('/v1/chat/completions',chat)
    backend=TestServer(app);await backend.start_server()
    ecs=FakeECS(boot_polls=1)
    cfg=Config(backend_port=backend.port,cold_nonstream_mode='wait',poll_interval_s=.01,max_wait_s=1,
               idle_timeout_s=3600)
    client=TestClient(TestServer(make_app(cfg,ecs)));await client.start_server()
    try:
        body=json.dumps({'messages':[],'padding':'x'*(5*1024**2)}).encode()
        r=await client.post('/v1/chat/completions',data=body);await r.read()
        return {'request_bytes':len(body),'status':r.status,'fake_aws_desired_updates':ecs.updates,
                'meaning':'Oversized body admitted by router; backend rejected it after scale-out'}
    finally:
        await client.close();await backend.close()


async def main():
    results={'unauthenticated_malformed':await cold(None),'configured_key_control':await cold('test-only-key'),
             'backend_oversized':await oversized()}
    print(json.dumps(results,indent=2))


if __name__=='__main__':asyncio.run(main())
