#!/usr/bin/env python3
"""Summarise behemoth_projection_bench JSON files: best mode per projection and layout,
per-layer sums, and the auto selection. Usage: summarize_round2.py FILE..."""
import json
import sys

ORDER = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
SHORT = {"self_attn.q_proj": "q", "self_attn.k_proj": "k", "self_attn.v_proj": "v",
         "self_attn.o_proj": "o", "mlp.gate_proj": "gate", "mlp.up_proj": "up",
         "mlp.down_proj": "down"}

for path in sys.argv[1:]:
    d = json.load(open(path))
    rows = None
    layouts = {}
    failed = []
    for r in d["results"]:
        if r.get("status") != "measured":
            continue
        rows = r["rows"]
        lay = layouts.setdefault(r["code_layout"], {})
        proj = lay.setdefault(r["projection"], {})
        proj[r["mode"]] = r["median_ms"]
        if not r.get("cross_path_pass", True):
            failed.append((r["code_layout"], r["projection"], r["mode"], r.get("max_abs_error")))
    print(f"== {path} rows={rows} cross_path_failures={len(failed)}")
    for f in failed:
        print("   FAIL", f)
    for lay, projs in layouts.items():
        best_sum = 0.0
        auto_sum = 0.0
        cells = []
        for name in ORDER:
            modes = projs.get(name, {})
            forced = {m: t for m, t in modes.items() if m != "auto"}
            if not forced:
                continue
            bm = min(forced, key=forced.get)
            best_sum += forced[bm]
            auto_sum += modes.get("auto", float("nan"))
            cells.append(f"{SHORT[name]} {forced[bm]:.3f} ({bm})")
        print(f"  {lay:12s} layer best {best_sum:.3f} ms  auto {auto_sum:.3f} ms | " + "; ".join(cells))
        # per-mode layer sums for the sorted-family modes
        mode_sums = {}
        for name in ORDER:
            for m, t in projs.get(name, {}).items():
                mode_sums.setdefault(m, []).append(t)
        interesting = {m: sum(v) for m, v in mode_sums.items() if len(v) == len(ORDER)}
        top = sorted(interesting.items(), key=lambda kv: kv[1])[:8]
        print("    layer sums: " + ", ".join(f"{m} {t:.3f}" for m, t in top))
