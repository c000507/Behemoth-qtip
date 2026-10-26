"""Greedy chat continuations from a running behemoth_server (127.0.0.1:1337), for the QTIP vs AQLM
quality comparison. Usage: python3 perf/qtip_greedy.py TAG [MAX_TOKENS=96] > perf/results/qtip/TAG_greedy.jsonl"""
import json, sys, urllib.request

PROMPTS = [
    "What is the capital of Australia, and why was it chosen over Sydney and Melbourne?",
    "Explain in three sentences how a transformer language model generates text.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Translate into French: 'The weather is beautiful today, let's go for a walk by the river.'",
    "List the planets of the solar system in order from the Sun.",
    "Write a haiku about autumn rain.",
    "A train leaves at 3:15 pm and the trip takes 2 hours 50 minutes. When does it arrive?",
    "Write a short, vivid paragraph about a lighthouse keeper who discovers that the fog has started to whisper names.",
]
tag = sys.argv[1]
max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 96
for p in PROMPTS:
    body = json.dumps({"messages": [{"role": "user", "content": p}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request("http://127.0.0.1:1337/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    print(json.dumps({"tag": tag, "prompt": p, "text": d["choices"][0]["message"]["content"],
                      "usage": d.get("usage")}), flush=True)
