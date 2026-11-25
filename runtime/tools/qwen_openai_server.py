#!/usr/bin/env python3
"""Minimal OpenAI-compatible HTTP front-end for the Qwen3.5 runtime (tools/qwen_worker).

The HF tokenizer renders the chat template (thinking models: the reasoning before </think> is
returned as `reasoning_content`); generation runs in one qwen_worker process (single sequence,
requests are serialised). Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions,
POST /v1/completions (both with optional "stream": true).

usage: qwen_openai_server.py --worker build-rocm/qwen_worker --model DIR [--gpus 0,1]
                             [--host 0.0.0.0] [--port 8027] [--max-context 32768] [--name NAME] [--tp]
"""
import argparse
import json
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from transformers import AutoTokenizer


class Worker:
    def __init__(self, args):
        cmd = [args.worker, "--model", args.model, "--gpus", args.gpus, "--max-context",
               str(args.max_context), "--chunk", str(args.chunk)] + (["--tp"] if args.tp else [])
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        ready = json.loads(self.p.stdout.readline())
        assert ready.get("ready"), ready
        self.max_context = ready["max_context"]
        self.lock = threading.Lock()

    def generate(self, req):
        """Yields ("t", id) per token, then ("done", info)."""
        with self.lock:
            self.p.stdin.write(json.dumps(req) + "\n")
            self.p.stdin.flush()
            while True:
                line = self.p.stdout.readline()
                if not line:
                    raise RuntimeError("worker exited")
                msg = json.loads(line)
                if "t" in msg:
                    yield "t", msg["t"]
                else:
                    yield "done", msg
                    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8027)
    ap.add_argument("--max-context", type=int, default=32768)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--tp", action="store_true", help="tensor parallel over the 2 GPUs (default: layer pipeline)")
    ap.add_argument("--name", default="qwen3.8-27b-fp8")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    stop_ids = sorted({i for i in [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>"),
                                   tok.eos_token_id] if isinstance(i, int) and i >= 0})
    think_end = tok.convert_tokens_to_ids("</think>")
    worker = Worker(args)
    print(f"serving {args.name} on {args.host}:{args.port} (stop ids {stop_ids})", flush=True)

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *a):
            print("%s %s" % (self.address_string(), fmt % a), flush=True)

        def _json(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == "/health":
                return self._json(200, {"status": "ok"})
            if self.path == "/v1/models":
                return self._json(200, {"object": "list", "data": [
                    {"id": args.name, "object": "model", "owned_by": "behemoth", "max_model_len": worker.max_context}]})
            self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except Exception as e:  # noqa: BLE001
                return self._json(400, {"error": {"message": f"bad json: {e}"}})
            chat = self.path == "/v1/chat/completions"
            if not chat and self.path != "/v1/completions":
                return self._json(404, {"error": {"message": "not found"}})
            try:
                if chat:
                    kw = {}
                    if body.get("chat_template_kwargs"):
                        kw.update(body["chat_template_kwargs"])
                    prompt = tok.apply_chat_template(body.get("messages", []), tools=body.get("tools"), tokenize=False,
                                                     add_generation_prompt=True, **kw)
                else:
                    prompt = body.get("prompt", "")
                    if isinstance(prompt, list):
                        prompt = prompt[0]
                ids = tok(prompt, add_special_tokens=False)["input_ids"]
            except Exception as e:  # noqa: BLE001
                return self._json(400, {"error": {"message": f"template/tokenize: {e}"}})
            max_new = int(body.get("max_completion_tokens") or body.get("max_tokens") or 4096)
            req = {"tokens": ids, "max_new": max_new, "temperature": float(body.get("temperature", 0.6 if chat else 0)),
                   "top_p": float(body.get("top_p", 0.95 if chat else 1.0)), "top_k": int(body.get("top_k", 20 if chat else 0)),
                   "stop": stop_ids}
            if "seed" in body and body["seed"] is not None:
                req["seed"] = int(body["seed"])
            # thinking: the template opens <think> in the prompt when the model thinks
            thinking = chat and prompt.rstrip().endswith("<think>")
            stream = bool(body.get("stream"))
            rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
            created = int(time.time())
            gen = worker.generate(req)
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True

            out_ids, info = [], {}
            in_reason = thinking
            sent_text = ""  # decoded prefix already streamed (reasoning or content part)
            part_start = 0

            def chunk(delta, finish=None):
                if chat:
                    obj = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": args.name,
                           "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                else:
                    obj = {"id": rid, "object": "text_completion", "created": created, "model": args.name,
                           "choices": [{"index": 0, "text": delta.get("content", ""), "finish_reason": finish}]}
                self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
                self.wfile.flush()

            try:
                if stream and chat:
                    chunk({"role": "assistant", "content": ""})
                for kind, v in gen:
                    if kind == "done":
                        info = v
                        break
                    if v in stop_ids:
                        continue
                    out_ids.append(v)
                    if not stream:
                        continue
                    if in_reason and v == think_end:
                        in_reason = False
                        part_start = len(out_ids)
                        sent_text = ""
                        continue
                    text = tok.decode(out_ids[part_start:], skip_special_tokens=False)
                    if text.endswith("�"):
                        continue  # incomplete UTF-8 sequence
                    d = text[len(sent_text):]
                    sent_text = text
                    if d:
                        chunk({"reasoning_content": d} if in_reason else {"content": d})
            except (BrokenPipeError, ConnectionResetError):
                for _ in gen:  # drain the worker
                    pass
                return
            finish = "stop" if info.get("done") == "stop" else "length"
            usage = {"prompt_tokens": len(ids), "completion_tokens": info.get("completion", len(out_ids)),
                     "total_tokens": len(ids) + info.get("completion", len(out_ids))}
            dec = info.get("decode_s", 0) or 1e-9
            print(f"{rid}: prompt {len(ids)} tok prefill {info.get('prefill_s', 0):.2f}s, {usage['completion_tokens']} tok "
                  f"in {dec:.2f}s ({usage['completion_tokens'] / dec:.1f} tok/s), {finish}", flush=True)
            if stream:
                chunk({}, finish)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            reasoning, content_ids = None, out_ids
            if thinking:
                if think_end in out_ids:
                    k = out_ids.index(think_end)
                    reasoning = tok.decode(out_ids[:k], skip_special_tokens=True).strip()
                    content_ids = out_ids[k + 1:]
                else:
                    reasoning, content_ids = tok.decode(out_ids, skip_special_tokens=True).strip(), []
            content = tok.decode(content_ids, skip_special_tokens=True)
            if chat:
                msg = {"role": "assistant", "content": content.lstrip("\n") if thinking else content}
                if reasoning is not None:
                    msg["reasoning_content"] = reasoning
                obj = {"id": rid, "object": "chat.completion", "created": created, "model": args.name,
                       "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage}
            else:
                obj = {"id": rid, "object": "text_completion", "created": created, "model": args.name,
                       "choices": [{"index": 0, "text": content, "finish_reason": finish}], "usage": usage}
            self._json(200, obj)

    ThreadingHTTPServer((args.host, args.port), H).serve_forever()


if __name__ == "__main__":
    main()
