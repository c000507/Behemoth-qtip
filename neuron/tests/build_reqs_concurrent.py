"""Requests for the nki-fusion runs: build_reqs3.py MODEL_DIR REPO_DIR OUT_DIR (reuses SYSTEM/TOOLS/USER/build of fable/build_reqs.py).
Writes: req_ttft.json (sequential: ~1k/2k/4k/8k/12k-token prompts, max_new 8 -> TTFT / prefill tokens per s),
        req_par{4,8,16}_1k.json and req_par{4,8}_4k.json (concurrent sets of N distinct ~1.4k / ~4.4k-token tool prompts, max_new 128),
        req_t15.json (the 1364-token tool15 prompt), req_long.json (the 4398-token long prompt)."""
import json, sys
model, repo, out = sys.argv[1:4]
src = open('/home/ubuntu/work/fable/build_reqs.py').read().split('req = lambda')[0]
sys.argv = ['x', model, repo, out]
exec(src)
tok.no_truncation()
req = lambda t, n=128: {"tokens": t, "max_new": n, "temperature": 0, "top_k": 0, "top_p": 1, "seed": 0}
# tool15-style prompt (4 tools, trimmed system prompt, ~1.4k tokens), with a per-request variation so the N concurrent prompts differ
paras = SYSTEM.split("\n\n"); T4 = TOOLS[:4]
def build4(paras, user):
    s = "\n\n".join(paras)
    return [sp("<s>"), sp("[SYSTEM_PROMPT]")] + ids(s) + [sp("[/SYSTEM_PROMPT]"), sp("[AVAILABLE_TOOLS]")] + ids(json.dumps(T4)) + [sp("[/AVAILABLE_TOOLS]"), sp("[INST]")] + ids(user) + [sp("[/INST]")]
while True:
    t15 = build4(paras, USER)
    if len(t15) <= 1550 or len(paras) <= 3: break
    paras.pop(-2)
vary = lambda i: USER + f"\n\n(Request {i}: name the file you will read first, then proceed.)"
par1k = [build4(paras, vary(i)) for i in range(16)]
long_t = build(len(CONTEXT_LONG))
def build_long(user):
    s = SYSTEM + CONTEXT_LONG
    return [sp("<s>"), sp("[SYSTEM_PROMPT]")] + ids(s) + [sp("[/SYSTEM_PROMPT]"), sp("[AVAILABLE_TOOLS]")] + ids(json.dumps(TOOLS)) + [sp("[/AVAILABLE_TOOLS]"), sp("[INST]")] + ids(user) + [sp("[/INST]")]
par4k = [build_long(vary(i)) for i in range(8)]
# long-context prompts at target lengths (repo files pre-read into the system prompt)
def big(chars):
    ctx = ("\n# Repository context (pre-read for you)\n\n## neuron/cpp/neuron_model.cpp\n```cpp\n" + repo_text("neuron/cpp/neuron_model.cpp", chars) +
           "\n```\n\n## neuron/cpp/neuron_engine.cpp\n```cpp\n" + repo_text("neuron/cpp/neuron_engine.cpp", chars) +
           "\n```\n\n## neuron/SERVING.md\n" + repo_text("neuron/SERVING.md", chars // 2) + "\n\n## neuron/PERF_NOTES.md\n" + repo_text("neuron/PERF_NOTES.md", chars // 2))
    s = SYSTEM + ctx
    return [sp("<s>"), sp("[SYSTEM_PROMPT]")] + ids(s) + [sp("[/SYSTEM_PROMPT]"), sp("[AVAILABLE_TOOLS]")] + ids(json.dumps(TOOLS)) + [sp("[/AVAILABLE_TOOLS]"), sp("[INST]")] + ids(USER) + [sp("[/INST]")]
def target(n):
    lo, hi = 0, 200000
    for _ in range(18):
        mid = (lo + hi) // 2
        if len(big(mid)) < n: lo = mid
        else: hi = mid
    return big(hi)
ttft = [t15] + [target(n) for n in (4096, 8192, 12288)]   # 1.4k (tool15), 4k, 8k, 12k; the base prompt is already 2297 tokens so no 1k/2k 'big' variant
json.dump([req(t, 8) for t in ttft], open(f"{out}/req_ttft.json", "w"))
json.dump([req(t15)], open(f"{out}/req_t15.json", "w"))
json.dump([req(long_t)], open(f"{out}/req_long.json", "w"))
for n in (4, 8, 16):
    json.dump([req(t) for t in par1k[:n]], open(f"{out}/req_par{n}_1k.json", "w"))
for n in (4, 8):
    json.dump([req(t) for t in par4k[:n]], open(f"{out}/req_par{n}_4k.json", "w"))
print(json.dumps({"ttft": [len(t) for t in ttft], "t15": len(t15), "long": len(long_t), "par1k": [len(t) for t in par1k[:4]], "par4k": [len(t) for t in par4k[:2]]}))
