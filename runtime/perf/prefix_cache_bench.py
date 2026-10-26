#!/usr/bin/env python3
"""Prefix-cache benchmarks for behemoth_server (PREFIX_CACHE_REPORT.md).

  prefix_cache_bench.py turn2 --sizes 2000,8000,16000,32000 --tag on
  prefix_cache_bench.py shared --sys 8000 --tag on
  prefix_cache_bench.py save --long 16000 --streams 4 --tag on
  prefix_cache_bench.py fill --size 2000 --n 12 --tag on
Prompts: /home/charles/qtip_server_bench/prompts.json ({size: {content, tokens}}). Every
conversation starts with a unique session line so different runs/sizes never share a prefix.
Raw results append to OUT/<tag>_<mode>.jsonl. Streaming client, per-token timestamps.
"""
import argparse, http.client, json, statistics, threading, time, uuid

HOST, PORT = "127.0.0.1", 1337
OUT = "/home/charles/prefix_cache_bench"
PROMPTS = "/home/charles/qtip_server_bench/prompts.json"


def metrics():
    c = http.client.HTTPConnection(HOST, PORT, timeout=60)
    c.request("GET", "/metrics")
    d = json.loads(c.getresponse().read())
    c.close()
    d.pop("last_step_profile", None)
    return d


def pc():
    return metrics().get("prefix_cache", {})


def wait_idle(timeout=600):
    t = time.time()
    while time.time() - t < timeout:
        m = metrics()
        p = m.get("prefix_cache", {})
        if m["active_requests"] == 0 and not p.get("transfers_in_flight") and not p.get("saves_pending"):
            return m
        time.sleep(0.2)
    raise RuntimeError("not idle")


def stream(messages, max_tokens, stop_flag=None, temperature=0.0, seed=None):
    body = {"messages": messages, "max_tokens": max_tokens, "stream": True,
            "temperature": temperature, "stream_options": {"include_usage": True}}
    if seed is not None:
        body["seed"] = seed
    t0 = time.time()
    times, finish, err, text, usage = [], None, None, "", None
    try:
        c = http.client.HTTPConnection(HOST, PORT, timeout=7200)
        c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        while True:
            if stop_flag is not None and stop_flag.is_set():
                finish = "client_stop"
                break
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
                err = j["error"]
                break
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices", []):
                txt = ch.get("delta", {}).get("content")
                if txt:
                    times.append(time.time())
                    text += txt
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
        c.close()
    except Exception as e:  # noqa
        err = repr(e)
    res = {"t0": t0, "t_end": time.time(), "times": times, "finish": finish, "error": err,
           "text": text, "usage": usage}
    if times:
        res["ttft_s"] = times[0] - t0
    return res


def cached(r):
    return ((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0)


def pct(a, q):
    a = sorted(a)
    return a[min(len(a) - 1, int(len(a) * q))] if a else None


def gaps_ms(times):
    return [1000 * (b - a) for a, b in zip(times, times[1:])]


def save(tag, mode, rec):
    with open(f"{OUT}/{tag}_{mode}.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def session():
    return f"Session {uuid.uuid4().hex[:12]}.\n"


PC_KEYS = ["hits", "misses", "gpu_tier_hits", "host_restores", "tokens_reused", "restore_ms_last",
           "restore_wall_ms_last", "restore_gbps_rank0_last", "restore_gbps_rank1_last",
           "save_ms_last", "save_wall_ms_last", "save_gbps_rank0_last", "save_gbps_rank1_last",
           "saves", "slots_used", "host_bytes_used", "host_bytes_pinned", "evictions",
           "gpu_tier_pages", "host_alloc_ms_total", "issue_ms_total"]


def pick(p):
    return {k: p.get(k) for k in PC_KEYS if k in p}


def cmd_turn2(a, P):
    for s in a.sizes.split(","):
        for run in range(a.runs):
            u1 = {"role": "user", "content": session() + P[s]["content"]}
            r1 = stream([u1], a.max_tokens)
            wait_idle()
            after1 = pc()
            msgs = [u1, {"role": "assistant", "content": r1["text"]},
                    {"role": "user", "content": "Summarize the story so far in one sentence."}]
            r2 = stream(msgs, a.max_tokens)
            wait_idle()
            after2 = pc()
            rec = {"size": s, "run": run, "turn1_prompt": (r1["usage"] or {}).get("prompt_tokens"),
                   "turn1_ttft_s": r1.get("ttft_s"), "turn1_tokens": len(r1["times"]),
                   "turn2_prompt": (r2["usage"] or {}).get("prompt_tokens"),
                   "turn2_cached": cached(r2), "turn2_ttft_s": r2.get("ttft_s"),
                   "turn2_itl_p50_ms": pct(gaps_ms(r2["times"]), .5),
                   "turn1_save": pick(after1), "turn2_restore": pick(after2),
                   "errors": [x for x in (r1["error"], r2["error"]) if x]}
            save(a.tag, "turn2", rec)


def cmd_shared(a, P):
    sysmsg = {"role": "system", "content": session() + P[a.sys]["content"]}
    r1 = stream([sysmsg, {"role": "user", "content": "Who is the main character?"}], a.max_tokens)
    wait_idle()
    r2 = stream([sysmsg, {"role": "user", "content": "Describe the setting in two sentences."}], a.max_tokens)
    wait_idle()
    rec = {"sys": a.sys, "conv1_prompt": (r1["usage"] or {}).get("prompt_tokens"),
           "conv1_ttft_s": r1.get("ttft_s"), "conv2_prompt": (r2["usage"] or {}).get("prompt_tokens"),
           "conv2_cached": cached(r2), "conv2_ttft_s": r2.get("ttft_s"), "pc": pick(pc()),
           "errors": [x for x in (r1["error"], r2["error"]) if x]}
    save(a.tag, "shared", rec)


def cmd_save(a, P):
    """Decode ITL of concurrent streams while a long request finishes and is saved."""
    stop = threading.Event()
    res = [None] * a.streams

    def w(i):
        res[i] = stream([{"role": "user", "content": P["100"]["content"]}], 4000, stop_flag=stop,
                        temperature=0.8, seed=3000 + i)
    th = [threading.Thread(target=w, args=(i,)) for i in range(a.streams)]
    for t in th:
        t.start()
        time.sleep(0.05)
    time.sleep(a.lead)
    before = pc()
    t_sub = time.time()
    L = stream([{"role": "user", "content": session() + P[a.long]["content"]}], 1)
    t_done = L["t_end"]
    time.sleep(a.tail)
    stop.set()
    for t in th:
        t.join()
    after = pc()
    save_wall = (after.get("save_wall_ms_last") or 0) / 1e3 if after.get("saves", 0) > before.get("saves", 0) else 0

    def window(lo, hi):
        return [x for r in res for (t1, x) in zip(r["times"][1:], gaps_ms(r["times"])) if lo <= t1 < hi]
    base = window(t_sub - a.lead + 2, t_sub)
    # the save is issued right after the long request's last step; its completion is seen at
    # the next step boundary, so [t_done, t_done + save_wall + 0.5 s] covers it
    during = window(t_done, t_done + max(save_wall, 0) + 0.5)
    after_w = window(t_done + max(save_wall, 0) + 0.5, t_done + a.tail)
    rec = {"tag": a.tag, "long": a.long, "streams": a.streams, "long_ttft_s": L.get("ttft_s"),
           "save_ms": after.get("save_ms_last"), "save_wall_ms": after.get("save_wall_ms_last"),
           "save_gbps": [after.get("save_gbps_rank0_last"), after.get("save_gbps_rank1_last")],
           "itl_base_p50_ms": pct(base, .5), "itl_base_p95_ms": pct(base, .95),
           "itl_base_max_ms": max(base) if base else None,
           "itl_save_window_p50_ms": pct(during, .5), "itl_save_window_p95_ms": pct(during, .95),
           "itl_save_window_max_ms": max(during) if during else None, "n_save_window": len(during),
           "itl_after_p50_ms": pct(after_w, .5), "itl_after_max_ms": max(after_w) if after_w else None,
           "errors": [r["error"] for r in res if r["error"]] + ([L["error"]] if L["error"] else [])}
    save(a.tag, "save", rec)


def cmd_fill(a, P):
    """n unique conversations of one size; then turn 2 of the first and last."""
    convs = []
    for i in range(a.n):
        u = {"role": "user", "content": session() + P[a.size]["content"]}
        r = stream([u], 8)
        m = wait_idle()
        p = m["prefix_cache"]
        convs.append((u, r))
        save(a.tag, "fill", {"i": i, "size": a.size, "prompt": (r["usage"] or {}).get("prompt_tokens"),
                             "ttft_s": r.get("ttft_s"), "slots_used": p["slots_used"],
                             "evictions": p["evictions"], "host_bytes_used": p["host_bytes_used"],
                             "host_bytes_pinned": p["host_bytes_pinned"], "kv_pages_free": m["kv_pages_free"],
                             "gpu_tier_pages": p["gpu_tier_pages"], "error": r["error"]})
    for i in (0, a.n - 1):
        u, r = convs[i]
        r2 = stream([u, {"role": "assistant", "content": r["text"]}, {"role": "user", "content": "Go on."}], 8)
        wait_idle()
        save(a.tag, "fill", {"turn2_of": i, "cached": cached(r2), "ttft_s": r2.get("ttft_s"),
                             "prompt": (r2["usage"] or {}).get("prompt_tokens"), "pc": pick(pc())})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--sizes", default="2000,8000,16000,32000")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--sys", default="8000")
    ap.add_argument("--long", default="16000")
    ap.add_argument("--streams", type=int, default=4)
    ap.add_argument("--lead", type=float, default=12)
    ap.add_argument("--tail", type=float, default=8)
    ap.add_argument("--size", default="2000")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--port", type=int, default=1337)
    a = ap.parse_args()
    PORT = a.port
    P = json.load(open(PROMPTS))
    {"turn2": cmd_turn2, "shared": cmd_shared, "save": cmd_save, "fill": cmd_fill}[a.cmd](a, P)
