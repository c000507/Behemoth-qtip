#!/usr/bin/env python3
"""End-to-end prefix-cache test against a running behemoth_server (PREFIX_CACHE_REPORT.md).

    python3 runtime/tests/prefix_cache_e2e.py [--host 127.0.0.1] [--port 1337]

Needs a server started with the prefix cache enabled (default --prefix-cache-slots 10). Uses
only the Python standard library. Checks, via usage.prompt_tokens_details.cached_tokens
(streaming with stream_options.include_usage) and /metrics:
  1. multi-turn: turn 2 (= turn-1 prompt + reply + new message) reuses turn 1's KV; TTFT drops;
  2. a turn cancelled by the client mid-decode is still cached (prompt + generated part);
  3. a request cancelled during prefill caches the prefilled chunks;
  4. a shared system prompt is reused across different conversations;
  5. concurrent multi-turn conversations: no errors, later turns hit;
  6. no leaked KV pages (free + prefix-cache GPU tier == total once idle), caps respected.
Exit status 0 when every check passes.
"""
import argparse
import http.client
import json
import socket
import sys
import threading
import time

HOST, PORT = "127.0.0.1", 1337
WORDS = ("amber basin cedar delta ember fjord granite harbor island jasper kelp lagoon meadow "
         "nickel orchard pebble quartz river summit timber upland valley willow yarrow zephyr").split()


def filler(words, seed):
    out = []
    for i in range(words):
        w = WORDS[(i * 7 + seed * 13 + (i // 5)) % len(WORDS)]
        out.append(w if i % 12 else f"\n{seed}.{i // 12}: {w}")
    return " ".join(out)


def metrics():
    c = http.client.HTTPConnection(HOST, PORT, timeout=60)
    c.request("GET", "/metrics")
    d = json.loads(c.getresponse().read())
    c.close()
    return d


def chat(messages, max_tokens=32, cancel_after_chunks=None, cancel_after_s=None):
    """Streams one completion. Returns dict(text, ttft_s, usage, finish, error, cancelled)."""
    body = {"messages": messages, "max_tokens": max_tokens, "stream": True, "temperature": 0,
            "stream_options": {"include_usage": True}}
    t0 = time.time()
    c = http.client.HTTPConnection(HOST, PORT, timeout=3600)
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    if cancel_after_s is not None:
        c.sock.settimeout(cancel_after_s)
    r = c.getresponse()
    res = {"text": "", "ttft_s": None, "usage": None, "finish": None, "error": None,
           "cancelled": False, "chunks": 0}
    try:
        while True:
            line = r.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            d = line[5:].strip()
            if d == b"[DONE]":
                break
            j = json.loads(d)
            if "error" in j:
                res["error"] = j["error"]
                break
            if j.get("usage"):
                res["usage"] = j["usage"]
            for ch in j.get("choices", []):
                txt = ch.get("delta", {}).get("content")
                if txt:
                    if res["ttft_s"] is None:
                        res["ttft_s"] = time.time() - t0
                    res["text"] += txt
                    res["chunks"] += 1
                if ch.get("finish_reason"):
                    res["finish"] = ch["finish_reason"]
            if cancel_after_chunks is not None and res["chunks"] >= cancel_after_chunks:
                res["cancelled"] = True
                break
    except socket.timeout:
        res["cancelled"] = True
    try:
        c.sock.shutdown(socket.SHUT_RDWR)
    except Exception:  # noqa
        pass
    c.close()
    res["wall_s"] = time.time() - t0
    return res


def cached(res):
    u = res.get("usage") or {}
    return (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)


def wait_idle(timeout=120):
    t = time.time()
    while time.time() - t < timeout:
        m = metrics()
        pc = m["prefix_cache"]
        if (m["active_requests"] == 0 and pc.get("transfers_in_flight", 0) == 0
                and pc.get("saves_pending", 0) == 0):
            return m
        time.sleep(0.2)
    raise RuntimeError("server did not become idle")


failures = 0


def check(ok, what):
    global failures
    print(("PASS " if ok else "FAIL ") + what, flush=True)
    failures += not ok


def main():
    global HOST, PORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--words", type=int, default=1500, help="turn-1 user message length")
    a = ap.parse_args()
    HOST, PORT = a.host, a.port
    m0 = wait_idle()
    check(m0["prefix_cache"].get("enabled"), "prefix cache enabled in /metrics")

    # 1. multi-turn
    u1 = {"role": "user", "content": filler(a.words, 1) + "\nList three of the words above."}
    t1 = chat([u1], max_tokens=48)
    p1 = t1["usage"]["prompt_tokens"]
    g1 = t1["usage"]["completion_tokens"]
    slots_after_t1 = wait_idle()["prefix_cache"]["slots_used"]
    msgs = [u1, {"role": "assistant", "content": t1["text"]},
            {"role": "user", "content": "Now name one more."}]
    t2 = chat(msgs, max_tokens=16)
    c2 = cached(t2)
    wait_idle()
    print(f"  turn 1: prompt {p1} tokens, {g1} generated, TTFT {t1['ttft_s']:.2f} s; "
          f"turn 2: prompt {t2['usage']['prompt_tokens']} tokens, cached {c2}, "
          f"TTFT {t2['ttft_s']:.2f} s")
    check(c2 >= p1, f"turn 2 reuses at least the turn-1 prompt ({c2} >= {p1})")
    # The reply is reused only where re-tokenizing the assistant text reproduces the generated
    # ids (the template renders " " + text; this model's replies start with <think> right after
    # [/INST], so the first reply token differs).
    print(f"  turn-1 reply tokens reused: {c2 - p1} of {g1}")
    check(metrics()["prefix_cache"]["slots_used"] == slots_after_t1,
          "turn 2 replaced the turn-1 slot (no second slot for the same conversation)")
    check(t2["ttft_s"] < t1["ttft_s"] / 2, "turn-2 TTFT below half of turn 1")
    wait_idle()
    # same turn-2 request again: hit on the extended slot (prompt - 1 tokens at most)
    t2b = chat(msgs, max_tokens=16)
    check(cached(t2b) == t2["usage"]["prompt_tokens"] - 1,
          f"repeated turn 2 hits the extended slot: cached {cached(t2b)} = prompt - 1")
    print(f"  repeated turn 2 greedy text identical: {t2b['text'] == t2['text']}")
    wait_idle()

    # 2. cancelled mid-decode
    mb = metrics()["prefix_cache"]
    ub = {"role": "user", "content": filler(a.words, 2) + "\nWrite a long story using these words."}
    tb = chat([ub], max_tokens=400, cancel_after_chunks=12)
    check(tb["cancelled"] and tb["finish"] is None, f"client cancelled after {tb['chunks']} chunks")
    m = wait_idle()
    pc = m["prefix_cache"]
    check(pc["saves_of_cancelled"] > mb["saves_of_cancelled"], "cancelled turn was saved")
    tb2 = chat([ub, {"role": "assistant", "content": tb["text"]},
                {"role": "user", "content": "Continue."}], max_tokens=8)
    pb = tb2["usage"]["prompt_tokens"]
    print(f"  cancelled turn: next turn prompt {pb}, cached {cached(tb2)}")
    check(cached(tb2) >= pb - 40, "next turn after a cancelled turn reuses prompt + partial reply")
    wait_idle()

    # 3. cancelled during prefill
    uc = {"role": "user", "content": filler(6 * a.words, 3) + "\nSummarize."}
    tc = chat([uc], max_tokens=8, cancel_after_s=8)
    check(tc["cancelled"] and tc["ttft_s"] is None, "client cancelled during prefill")
    wait_idle()
    tc2 = chat([uc], max_tokens=8)
    pc_tokens = tc2["usage"]["prompt_tokens"]
    print(f"  prefill cancelled after 8 s: rerun prompt {pc_tokens}, cached {cached(tc2)}")
    check(2048 <= cached(tc2) < pc_tokens, "prefilled chunks of the cancelled request reused")
    wait_idle()

    # 4. shared system prompt
    sysmsg = {"role": "system", "content": filler(a.words // 2, 4)}
    d1 = chat([sysmsg, {"role": "user", "content": "Pick a word."}], max_tokens=8)
    wait_idle()
    d2 = chat([sysmsg, {"role": "user", "content": "Pick two different words, please."}],
              max_tokens=8)
    print(f"  shared system prompt: conversation 2 prompt {d2['usage']['prompt_tokens']}, "
          f"cached {cached(d2)} (conversation 1 prompt {d1['usage']['prompt_tokens']})")
    check(cached(d2) >= d1["usage"]["prompt_tokens"] - 12,
          "second conversation reuses the shared system prompt")
    wait_idle()

    # 5. concurrent conversations
    results = {}

    def conversation(k):
        msgs = [{"role": "user", "content": filler(300, 10 + k) + "\nGive one word."}]
        out = []
        for turn in range(3):
            r = chat(msgs, max_tokens=12)
            out.append(r)
            msgs = msgs + [{"role": "assistant", "content": r["text"]},
                           {"role": "user", "content": f"Another one ({turn})."}]
        results[k] = out
    th = [threading.Thread(target=conversation, args=(k,)) for k in range(4)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    errors = [r["error"] for out in results.values() for r in out if r["error"]]
    later = [cached(r) for out in results.values() for r in out[1:]]
    print(f"  concurrent: cached tokens of turns 2-3: {later}")
    check(not errors and len(results) == 4, "4 concurrent 3-turn conversations without errors")
    check(all(c > 0 for c in later), "every later turn of the concurrent conversations hit")

    # 6. accounting
    m = wait_idle()
    pc = m["prefix_cache"]
    check(m["kv_pages_free"] + pc["gpu_tier_pages"] == m["kv_pages_total"]
          and pc["kv_pages_unaccounted"] == 0,
          f"no leaked KV pages (free {m['kv_pages_free']} + GPU tier {pc['gpu_tier_pages']} "
          f"== total {m['kv_pages_total']})")
    check(pc["slots_used"] <= pc["slots_max"] and pc["host_bytes_used"] <= pc["host_bytes_cap"],
          f"slots {pc['slots_used']}/{pc['slots_max']}, host bytes "
          f"{pc['host_bytes_used'] / 1e9:.2f}/{pc['host_bytes_cap'] / 1e9:.1f} GB")
    check(m["healthy"], "server healthy")
    print(json.dumps({k: v for k, v in pc.items()}, indent=1))
    print("PREFIX CACHE E2E " + ("FAILED" if failures else "PASSED"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
