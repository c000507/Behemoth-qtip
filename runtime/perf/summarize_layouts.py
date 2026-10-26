#!/usr/bin/env python3
"""Summarise behemoth_projection_bench JSON (LAYOUT=all): per projection, best group-major
mode vs best sorted-layout mode, auto selection, and per-layer totals."""
import json, sys
order = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
for path in sys.argv[1:]:
    d = json.load(open(path))
    res = [r for r in d["results"] if r.get("status") == "measured"]
    rows = res[0]["rows"]
    print(f"## {path}  rows={rows}")
    print(f"{'projection':18s} {'stock best':22s} {'ms':>7s} | {'sorted best':22s} {'ms':>7s} | {'auto(sorted)':22s} {'ms':>7s} | {'sorted-lds':>10s} {'ratio':>6s} {'pass':>5s}")
    tot_stock = tot_sorted = tot_auto = tot_lds = 0.0
    for name in order:
        stock = [r for r in res if r["projection"] == name and r["code_layout"] == "group-major" and r["mode"] != "auto"]
        sorted_ = [r for r in res if r["projection"] == name and r["code_layout"] == "sorted" and r["mode"] != "auto"]
        auto = [r for r in res if r["projection"] == name and r["code_layout"] == "sorted" and r["mode"] == "auto"]
        if not stock or not sorted_:
            continue
        bs = min(stock, key=lambda r: r["median_ms"])
        # "sorted layout best" = modes that only need the sorted layout
        sl = [r for r in sorted_ if r["mode"].startswith("sorted") or r["mode"] == "blas"]
        bsl = min(sl, key=lambda r: r["median_ms"])
        lds = [r for r in sorted_ if r["mode"] == "sorted-lds"]
        lds_ms = lds[0]["median_ms"] if lds else float("nan")
        a = auto[0] if auto else None
        ok = all(r.get("cross_path_pass", False) for r in sorted_)
        print(f"{name:18s} {bs['mode']:22s} {bs['median_ms']:7.3f} | {bsl['mode']:22s} {bsl['median_ms']:7.3f} | "
              f"{(a['selected_mode'] if a else '-'):22s} {(a['median_ms'] if a else float('nan')):7.3f} | {lds_ms:10.3f} {bsl['median_ms']/bs['median_ms']:6.2f} {str(ok):>5s}")
        tot_stock += bs["median_ms"]; tot_sorted += bsl["median_ms"]; tot_lds += lds_ms
        tot_auto += a["median_ms"] if a else 0
    print(f"{'LAYER':18s} {'':22s} {tot_stock:7.3f} | {'':22s} {tot_sorted:7.3f} | {'':22s} {tot_auto:7.3f} | {tot_lds:10.3f} {tot_sorted/tot_stock:6.2f}")
    print()
