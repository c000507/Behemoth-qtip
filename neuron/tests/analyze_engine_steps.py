"""Aggregate step-log analysis for concurrent neuron_engine_cli runs: analyze_steps.py LABEL run.out run.err
Groups the steps by composition (decode bucket, prefill chunks in the step) and prints count / mean wall / mean layers_ms / tokens emitted,
plus the aggregate decode throughput of the PURE decode steps (no prefill chunk in the step), the total process wall, and the TTFT list."""
import json, re, sys
label, out_path, err_path = sys.argv[1:4]
txt = open(out_path).read()
js = txt[txt.index("\n{\n") + 1:] if "\n{\n" in txt else txt[txt.index("{"):]
d = json.loads(js)
steps = []
for line in open(err_path, errors="replace"):
    m = re.match(r"step rows=(\d+) drafts=(\d+) tokens=(\d+) wall_ms=([\d.]+) (\{.*\})", line)
    if m:
        p = json.loads(m.group(5))
        steps.append(dict(reqs=int(m.group(1)), drafts=int(m.group(2)), emitted=int(m.group(3)), wall=float(m.group(4)), launches=p["launches"],
                          bucket=p["decode_bucket"], tokens=p["tokens"], layers=p["layers_ms"], fwd=p["forward_ms"], head=p["head_ms"]))
reqs = d["requests"]
print(f"== {label}: process {d['seconds']:.1f} s, requests {len(reqs)}, prompt tokens {sum(r['prompt_tokens'] for r in reqs)}, output tokens {sum(len(r['tokens']) for r in reqs)}, "
      f"steps {len(steps)}, drafted {d['metrics']['drafted_tokens']} accepted {d['metrics']['accepted_drafts']}")
print("   ttft ms:", [round(r["first_token_ms"]) for r in reqs], " finish:", sorted(set(r["finish"] for r in reqs)))
groups = {}
for s in steps:
    # a decode launch holds <= 8 (or 16/32) rows; tokens beyond the rows are prefill chunk tokens
    prefill = s["bucket"] == "" or (s["tokens"] > 32 and s["launches"] > 1) or (s["bucket"] and s["tokens"] > int(s["bucket"].split("_")[0][3:].rstrip("s")))
    key = (s["bucket"] or "prefill-only", "mixed" if (prefill and s["bucket"]) else ("prefill" if prefill else "decode"))
    g = groups.setdefault(key, dict(n=0, wall=0.0, layers=0.0, emitted=0, tokens=0, drafts=0))
    g["n"] += 1; g["wall"] += s["wall"]; g["layers"] += s["layers"]; g["emitted"] += s["emitted"]; g["tokens"] += s["tokens"]; g["drafts"] += s["drafts"]
for (b, kind), g in sorted(groups.items()):
    print(f"   {b:12s} {kind:8s} steps={g['n']:3d} mean wall={g['wall']/g['n']:7.1f} ms mean layers={g['layers']/g['n']:7.1f} ms emitted/step={g['emitted']/g['n']:5.2f} tokens/step={g['tokens']/g['n']:6.1f} drafts/step={g['drafts']/g['n']:.2f}")
pure = [s for s in steps if s["bucket"] and not ((s["tokens"] > 32 and s["launches"] > 1) or s["tokens"] > int(s["bucket"].split("_")[0][3:].rstrip("s")))]
if pure:
    w = sum(s["wall"] for s in pure); e = sum(s["emitted"] for s in pure)
    print(f"   PURE decode steps: {len(pure)}, {e} tokens in {w:.0f} ms -> {1000*e/w:.1f} tok/s aggregate, {w/max(1,e):.2f} ms/token; per-step {w/len(pure):.1f} ms")
pf = [s for s in steps if s["bucket"] == ""]
if pf:
    w = sum(s["wall"] for s in pf); t = sum(s["tokens"] for s in pf); l = sum(s["layers"] for s in pf)
    print(f"   prefill-only steps: {len(pf)}, {t} prompt tokens in {w:.0f} ms -> {1000*t/w:.0f} tok/s; layers {l:.0f} ms ({100*l/w:.0f} %), non-layer {w-l:.0f} ms = {(w-l)/len(pf):.1f} ms/step")
