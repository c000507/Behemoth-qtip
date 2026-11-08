#!/usr/bin/env python3
"""End-to-end structured-output checks against a running behemoth_server.

    python3 runtime/scripts/test_structured_json.py --host 127.0.0.1 --port 1337

Sends json_object / json_schema / grammar requests (non-streaming, streaming, with and
without reasoning, greedy and sampled, concurrent with unconstrained requests), validates
every answer with `jsonschema` when installed (a small built-in validator otherwise),
checks HTTP 400s for unsupported schemas, and prints the server's structured-output
metrics. Exit status 0 only when every check passes. Standard library only (+ optional
jsonschema).
"""
import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request

try:
    import jsonschema  # type: ignore
except ImportError:  # pragma: no cover
    jsonschema = None

MODEL = "cgg507/Behemoth-R1-123B-v2-aqlm"

PERSON = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "maxLength": 60},
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
        "favorite_color": {"type": "string", "enum": ["red", "green", "blue"]},
        "hobbies": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
        "address": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "zip": {"type": "string"}},
            "required": ["city", "zip"],
            "additionalProperties": False,
        },
        "employed": {"type": "boolean"},
    },
    "required": ["name", "age", "favorite_color", "hobbies", "address", "employed"],
    "additionalProperties": False,
}

ORDER = {
    "$defs": {
        "item": {
            "type": "object",
            "properties": {
                "sku": {"type": "string"},
                "quantity": {"type": "integer", "minimum": 1, "maximum": 99},
                "price": {"type": "number", "minimum": 0, "maximum": 10000},
                "note": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
            "required": ["sku", "quantity", "price", "note"],
            "additionalProperties": False,
        }
    },
    "type": "object",
    "properties": {
        "order_id": {"type": "string", "format": "uuid"},
        "date": {"type": "string", "format": "date"},
        "status": {"const": "open"},
        "items": {"type": "array", "items": {"$ref": "#/$defs/item"}, "minItems": 2, "maxItems": 4},
    },
    "required": ["order_id", "date", "status", "items"],
    "additionalProperties": False,
}


def mini_validate(schema, value, root=None):
    """Fallback validator for the schemas above."""
    root = root or schema
    if "$ref" in schema:
        node = root
        for part in schema["$ref"][2:].split("/"):
            node = node[part]
        return mini_validate(node, value, root)
    if "anyOf" in schema:
        return any(mini_validate(s, value, root) for s in schema["anyOf"])
    if "const" in schema:
        return value == schema["const"]
    if "enum" in schema and value not in schema["enum"]:
        return False
    t = schema.get("type")
    types = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}
    if t == "integer" and not (isinstance(value, int) and not isinstance(value, bool)):
        return False
    if t == "number" and not (isinstance(value, (int, float)) and not isinstance(value, bool)):
        return False
    if t in types and not isinstance(value, types[t]):
        return False
    if isinstance(value, dict):
        props = schema.get("properties", {})
        if any(k not in value for k in schema.get("required", [])):
            return False
        for k, v in value.items():
            if k in props:
                if not mini_validate(props[k], v, root):
                    return False
            elif schema.get("additionalProperties") is False:
                return False
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", 1 << 30):
            return False
        if "items" in schema and not all(mini_validate(schema["items"], v, root) for v in value):
            return False
    if isinstance(value, str):
        if len(value) > schema.get("maxLength", 1 << 30) or len(value) < schema.get("minLength", 0):
            return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return False
        if "maximum" in schema and value > schema["maximum"]:
            return False
    return True


def validate(schema, value):
    if jsonschema is not None:
        try:
            jsonschema.validate(value, schema, format_checker=jsonschema.FormatChecker())
            return True, ""
        except jsonschema.ValidationError as e:
            return False, e.message
    return mini_validate(schema, value), "mini validator"


class Client:
    def __init__(self, host, port, timeout):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout

    def post(self, body):
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def stream(self, body):
        body = dict(body, stream=True, stream_options={"include_usage": True})
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        content, reasoning, finish, chunks = "", "", None, 0
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if "error" in obj:
                    raise RuntimeError(obj["error"])
                for ch in obj.get("choices", []):
                    delta = ch.get("delta", {})
                    content += delta.get("content", "") or ""
                    reasoning += delta.get("reasoning_content", "") or ""
                    finish = ch.get("finish_reason") or finish
                    chunks += 1
        return content, reasoning, finish, chunks

    def metrics(self):
        with urllib.request.urlopen(self.base + "/metrics", timeout=30) as r:
            return json.loads(r.read())


def chat(prompt, **extra):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}]}
    body.update(extra)
    return body


def schema_format(name, schema, strict=True):
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": strict}}


class Runner:
    def __init__(self, client, max_tokens):
        self.c = client
        self.max_tokens = max_tokens
        self.failures = []
        self.passed = 0

    def check(self, name, ok, detail=""):
        print(("PASS " if ok else "FAIL ") + name + (f" -- {detail}" if detail else ""), flush=True)
        if ok:
            self.passed += 1
        else:
            self.failures.append(name)

    def answer(self, name, body, schema=None, expect_reasoning=None):
        t0 = time.time()
        status, r = self.c.post(body)
        dt = time.time() - t0
        if status != 200:
            self.check(name, False, f"HTTP {status}: {r}")
            return None
        msg = r["choices"][0]["message"]
        content, finish = msg.get("content", ""), r["choices"][0]["finish_reason"]
        usage = r.get("usage", {})
        try:
            value = json.loads(content)
        except ValueError as e:
            self.check(name, False, f"invalid JSON ({e}); finish={finish}; content={content[:300]!r}")
            return None
        ok, why = (validate(schema, value) if schema is not None else (isinstance(value, dict), "not an object"))
        detail = (f"{usage.get('completion_tokens')} tokens in {dt:.1f}s, finish={finish}, "
                  f"reasoning={len(msg.get('reasoning_content', ''))} chars: {content[:160]}")
        if ok and expect_reasoning is not None:
            has = bool(msg.get("reasoning_content"))
            if expect_reasoning != has:
                ok, why = False, f"reasoning_content present={has}, expected {expect_reasoning}"
        self.check(name, ok and finish == "stop", detail if ok else f"{why}; {detail}")
        return value

    def run(self, concurrency):
        mt = self.max_tokens
        # Unconstrained regression first.
        # Unconstrained regression: normal text, reasoning routed as before, no structured cost.
        before = self.c.metrics().get("structured_output", {}).get("tokens", 0)
        status, r = self.c.post(chat("Say hello in five words.", max_tokens=mt))
        after = self.c.metrics().get("structured_output", {}).get("tokens", 0)
        msg = r["choices"][0]["message"] if status == 200 else {}
        self.check("unconstrained regression",
                   status == 200 and (msg.get("content") or msg.get("reasoning_content")) and before == after,
                   f"finish={r['choices'][0]['finish_reason']} content={msg.get('content', '')[:80]!r} "
                   f"reasoning={len(msg.get('reasoning_content', ''))} chars, constrained tokens unchanged="
                   f"{before == after}" if status == 200 else str(r))
        self.answer("json_object (thinking allowed)",
                    chat("Describe a fictional cat as a JSON object with a few fields.", max_tokens=mt,
                         response_format={"type": "json_object"}))
        self.answer("json_object reasoning_effort=none",
                    chat("Describe a fictional cat as a JSON object.", max_tokens=mt,
                         response_format={"type": "json_object"}, reasoning_effort="none"),
                    expect_reasoning=False)
        self.answer("json_schema person (nested, enum, array)",
                    chat("Invent a person. Reply in JSON.", max_tokens=mt,
                         response_format=schema_format("person", PERSON)), PERSON)
        self.answer("json_schema order ($defs/$ref, anyOf, const, uuid/date, number bounds)",
                    chat("Create a sample shop order with two items as JSON.", max_tokens=mt,
                         response_format=schema_format("order", ORDER)), ORDER)
        self.answer("json_schema sampled (temperature 0.8, top_p 0.9, top_k 50)",
                    chat("Invent a person. Reply in JSON.", max_tokens=mt, temperature=0.8, top_p=0.9, top_k=50,
                         seed=7, response_format=schema_format("person", PERSON)), PERSON)
        self.answer("json_schema with a thinking prompt",
                    chat("Think step by step about what 17*23 is, then answer.", max_tokens=mt,
                         response_format=schema_format("calc", {
                             "type": "object", "properties": {"result": {"type": "integer"},
                                                             "explanation": {"type": "string", "maxLength": 200}},
                             "required": ["result", "explanation"], "additionalProperties": False})),
                    {"type": "object", "properties": {"result": {"type": "integer"},
                                                      "explanation": {"type": "string", "maxLength": 200}},
                     "required": ["result", "explanation"], "additionalProperties": False})
        # Streaming.
        try:
            content, reasoning, finish, chunks = self.c.stream(
                chat("Invent a person. Reply in JSON.", max_tokens=mt, response_format=schema_format("person", PERSON)))
            value = json.loads(content)
            ok, why = validate(PERSON, value)
            self.check("streaming json_schema", ok and finish == "stop",
                       f"{chunks} chunks, reasoning {len(reasoning)} chars, finish={finish}: {content[:120]}"
                       + ("" if ok else f" ({why})"))
        except Exception as e:  # noqa: BLE001
            self.check("streaming json_schema", False, repr(e))
        ticket = {"type": "object", "properties": {"id": {"type": "string", "pattern": "^[A-Z]{3}-\\d{4}$"},
                                                   "url": {"type": "string", "pattern": "^https://"}},
                  "required": ["id", "url"], "additionalProperties": False}
        self.answer("json_schema pattern (anchored regex, prefix)",
                    chat("Make up a support ticket with an id and a link. Reply in JSON.", max_tokens=mt,
                         response_format=schema_format("ticket", ticket)), ticket)
        # Raw grammar extension.
        status, r = self.c.post(chat("Is the sky blue? Answer yes or no.", max_tokens=mt, reasoning_effort="none",
                                     grammar='root ::= "yes" | "no"'))
        self.check("grammar extension (yes|no)",
                   status == 200 and r["choices"][0]["message"]["content"] in ("yes", "no"),
                   str(r["choices"][0]["message"]["content"] if status == 200 else r))
        # Rejections.
        for name, rf in [("pattern lookahead", schema_format("x", {"type": "string", "pattern": "(?=a)"})),
                         ("multipleOf", schema_format("x", {"type": "integer", "multipleOf": 3})),
                         ("unknown type", {"type": "xml"}),
                         ("missing schema", {"type": "json_schema", "json_schema": {"name": "x"}})]:
            status, r = self.c.post(chat("Hi", max_tokens=8, response_format=rf))
            err = r.get("error", {})
            self.check(f"400 for {name}", status == 400 and err.get("type") == "invalid_request_error"
                       and str(err.get("param", "")).startswith("response_format"), f"{status} {err.get('message')}")
        # Concurrent mixed batch: constrained and unconstrained requests decode together.
        results = [None] * concurrency

        def worker(i):
            if i % 2:
                results[i] = ("plain", self.c.post(chat(f"Write one sentence about the number {i}.", max_tokens=400)))
            else:
                results[i] = ("person", self.c.post(chat("Invent a person. Reply in JSON.", max_tokens=mt,
                                                         temperature=0.7, seed=i,
                                                         response_format=schema_format("person", PERSON))))
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        ok_all, details = True, []
        for kind, (status, r) in results:
            if status != 200:
                ok_all = False
                details.append(f"{kind}: HTTP {status}")
                continue
            content = r["choices"][0]["message"]["content"]
            if kind == "plain":
                content = content or r["choices"][0]["message"].get("reasoning_content", "")
            if kind == "person":
                try:
                    good, _ = validate(PERSON, json.loads(content))
                except ValueError:
                    good = False
                ok_all &= good and r["choices"][0]["finish_reason"] == "stop"
                details.append(f"person valid={good}")
            else:
                ok_all &= bool(content)
                details.append("plain ok" if content else "plain empty")
        self.check(f"concurrent mixed x{concurrency}", ok_all, f"{time.time() - t0:.1f}s: " + ", ".join(details))
        try:
            m = self.c.metrics().get("structured_output", {})
            print("metrics structured_output:", json.dumps(m))
            self.check("metrics report constraint overhead", "constraint_us_per_token" in m, json.dumps(m))
        except Exception as e:  # noqa: BLE001
            self.check("metrics", False, repr(e))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1337)
    ap.add_argument("--max-tokens", type=int, default=1500, help="output budget incl. the think block")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=900)
    args = ap.parse_args()
    if jsonschema is None:
        print("note: python3-jsonschema not installed; using the built-in subset validator")
    runner = Runner(Client(args.host, args.port, args.timeout), args.max_tokens)
    runner.run(args.concurrency)
    print(f"\n{runner.passed} passed, {len(runner.failures)} failed" +
          (": " + ", ".join(runner.failures) if runner.failures else ""))
    return 1 if runner.failures else 0


if __name__ == "__main__":
    sys.exit(main())
