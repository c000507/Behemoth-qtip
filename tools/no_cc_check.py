#!/usr/bin/env python3
"""Enforcement: no emitter may reach neuronx-cc/jax at emit time.

Static AST call-graph over tools/lift_hip.py (+ tile_ir.py). Any emitter
that transitively reaches gen_oracle_custom / _jax_gen /
_generate_oracle_on_box is CLASS C (emit-time compiler) and reported FAIL.
Emitters referencing frozen oracle templates (shape_templates.json,
rms_oracles, ew_templates, ref dirs) are CLASS B (stopgap), reported WARN.
Pure synthesis = CLASS A, reported PASS.

Exit code 0 always; the report is the metric. --strict exits 1 if any C.
"""
import ast, sys, re
from pathlib import Path

TOOLS = Path(__file__).parent
CC_FUNCS = {"gen_oracle_custom", "_jax_gen", "_generate_oracle_on_box"}
B_MARKS = ("shape_templates", "rms_oracles", "ew_templates", "oracle-copy",
           "byte-copy", "oracle_copy")

def module_fns(path):
    tree = ast.parse(path.read_text())
    return tree

def call_graph(tree):
    """fn name -> set of names it calls (module-level funcs only)."""
    g = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            calls = set()
            for n in ast.walk(node):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                    calls.add(n.func.id)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                    calls.add(n.func.attr)
            g[node.name] = calls
    return g

def reaches(g, fn, seen=None):
    seen = seen or set()
    if fn in seen: return set()
    seen.add(fn)
    out = set()
    for callee in g.get(fn, ()):
        if callee in CC_FUNCS: out.add(callee)
        if callee in g: out |= reaches(g, callee, seen)
    return out

def classify():
    report = []
    for src in ("lift_hip.py", "tile_ir.py"):
        p = TOOLS / src
        if not p.exists(): continue
        tree = module_fns(p)
        g = call_graph(tree)
        raw = p.read_text()
        for fn, node in [(n.name, n) for n in tree.body if isinstance(n, ast.FunctionDef)]:
            if not fn.startswith("emit_") and fn not in ("build_graph", "synth"):
                continue
            cc = reaches(g, fn)
            body = ast.unparse(node)
            refs_b = any(m in body for m in B_MARKS)
            cls = "C" if cc else ("B" if refs_b else "A")
            report.append((src, fn, cls, sorted(cc)))
    return report

def main():
    strict = "--strict" in sys.argv
    rep = classify()
    print(f"{'module':<14} {'emitter':<28} class  reaches-cc")
    for src, fn, cls, cc in sorted(rep, key=lambda r: r[2]):
        tag = {"A": "PASS", "B": "WARN", "C": "FAIL"}[cls]
        print(f"{src:<14} {fn:<28} {cls}      {tag} {cc if cc else ''}")
    nC = sum(1 for r in rep if r[2] == 'C')
    nB = sum(1 for r in rep if r[2] == 'B')
    nA = sum(1 for r in rep if r[2] == 'A')
    print(f"\nA={nA} B={nB} C={nC}  (C=emit-time neuronx-cc: must go to 0)")
    if strict and nC: sys.exit(1)

if __name__ == "__main__":
    main()
