"""Live CPU integration test against a tiny synthetic GLM dense checkpoint."""
import argparse
import json
import os
import pathlib
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=pathlib.Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="glm53-http-") as temp:
        root = pathlib.Path(temp)
        subprocess.run([str(args.build / "glm53_tests"), "--server-fixture", str(root / "model")], check=True)
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        env = dict(os.environ, OMP_NUM_THREADS="2")
        with (root / "server.log").open("w") as log:
            process = subprocess.Popen([str(args.build / "glm53_server"), "--model", str(root / "model"),
                                        "--port", str(port)], stdout=log, stderr=log, env=env)
            try:
                base = f"http://127.0.0.1:{port}"
                for _ in range(200):
                    if process.poll() is not None:
                        raise RuntimeError((root / "server.log").read_text())
                    try:
                        with urllib.request.urlopen(base + "/health", timeout=1) as r:
                            assert json.load(r)["healthy"]
                        break
                    except (OSError, urllib.error.URLError):
                        time.sleep(.05)
                else:
                    raise RuntimeError("server startup timeout")

                def post(payload):
                    request = urllib.request.Request(base + "/v1/chat/completions",
                        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(request, timeout=30) as r:
                        return r.read().decode()

                request = dict(model="glm53", messages=[dict(role="user", content="hi")],
                               max_tokens=3, temperature=0, seed=17)
                plain = json.loads(post(request))
                assert plain["usage"]["completion_tokens"] <= 3
                message = plain["choices"][0]["message"]
                assert isinstance(message["content"], str) and isinstance(message["reasoning_content"], str)
                again = json.loads(post(request))
                assert again["choices"] == plain["choices"], "recurrent state leaked between requests"
                request.update(stream=True, stream_options=dict(include_usage=True))
                events = post(request).split("\n\n")
                assert events[-2] == "data: [DONE]"
                accumulated = dict(content="", reasoning_content="")
                for event in events[:-2]:
                    record = json.loads(event.removeprefix("data: "))
                    if not record["choices"]:
                        assert record["usage"] == plain["usage"]
                        continue
                    delta = record["choices"][0]["delta"]
                    for key in accumulated:
                        accumulated[key] += delta.get(key, "")
                assert all(accumulated[k] == message[k] for k in accumulated)
                request["tools"] = []
                try:
                    post(request)
                    raise AssertionError("unsupported tools accepted")
                except urllib.error.HTTPError as error:
                    assert error.code == 400
            finally:
                process.terminate()
                process.wait(timeout=5)
    print("PASS GLM53 HTTP streaming/nonstreaming, usage, state reset and validation")


if __name__ == "__main__":
    main()
