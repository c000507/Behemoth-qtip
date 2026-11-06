#!/usr/bin/env python3
"""Leave-one-in whole-model KL sensitivity sweep (BASE checkpoint + one layer file from UPGRADE).

Usage:
  sens_sweep.py eval NAME DIR                 evaluate an existing checkpoint dir
  sens_sweep.py sweep [--layers 0,1,..]       base + upgrade + one swap per layer (swap dir deleted after eval)
  sens_sweep.py mix NAME L1,L2,...            build mix (hardlinks), evaluate, KEEP dir (/mnt/evo/sens/mix-NAME)
Results (per-seq KL / nll) accumulate in /mnt/evo/sens/results.json (resume: finished names are skipped).
Needs a runtime where FP8 K/V is detected per tensor (the stock loader requires a global kv_format).
"""
import argparse, json, os, re, shutil, subprocess, sys, time

ROOT = "/mnt/evo/sens"
BASE = "/mnt/evo/Anubis-Mini-8B-v1-QTIP-ADAPTIVE"
UPG = "/mnt/evo/Anubis-Mini-8B-v1-QTIP-ADAPTIVE-fp8kv-t0.002-big"
TOOL = os.environ.get("KL_TOOL", f"{ROOT}/src/runtime/build/behemoth_cuda_model_kl")
TOKENS = "/mnt/evo/anubis_ref/heldout_1024.i32"
REF = "/mnt/evo/anubis_ref"
RES = f"{ROOT}/results.json"
N = int(os.environ.get("SENS_N", "8"))


def load_res():
    return json.load(open(RES)) if os.path.exists(RES) else {}


def lname(i):
    return f"model-layer-{i:05d}.safetensors"


def build_mix(path, layers):
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    layers = set(layers)
    for f in os.listdir(BASE):
        src = BASE if not (f.startswith("model-layer-") and int(f[12:17]) in layers) else UPG
        if f in ("model.safetensors.index.json",):
            continue
        if f == "adaptive_report.json":
            continue
        os.link(f"{src}/{f}", f"{path}/{f}")
    bi = json.load(open(f"{BASE}/model.safetensors.index.json"))["weight_map"]
    ui = json.load(open(f"{UPG}/model.safetensors.index.json"))["weight_map"]
    wm = {}
    for k, f in bi.items():
        if not f.startswith("model-layer-"):
            wm[k] = f
    for k, f in bi.items():
        if f.startswith("model-layer-") and int(f[12:17]) not in layers:
            wm[k] = f
    for k, f in ui.items():
        if f.startswith("model-layer-") and int(f[12:17]) in layers:
            wm[k] = f
    json.dump({"metadata": {}, "weight_map": wm}, open(f"{path}/model.safetensors.index.json", "w"))


def evaluate(name, d):
    res = load_res()
    if name in res:
        return res[name]
    os.makedirs(f"{ROOT}/logs", exist_ok=True)
    log = f"{ROOT}/logs/{name}.log"
    t0 = time.time()
    p = subprocess.run([TOOL, "--model", d, "--tokens", TOKENS, "--seq-len", "1024", "--ref-dir", REF,
                        "--nseq", str(N)], capture_output=True, text=True)
    wall = time.time() - t0
    open(log, "w").write(p.stderr + p.stdout)
    if p.returncode:
        raise RuntimeError(f"{name} failed: {p.stderr[-400:]}")
    seq = re.findall(r"seq (\d+): KL ([\d.]+)\s+nll ref ([\d.]+) quant ([\d.]+)", p.stderr)
    m = re.search(r"ppl ref ([\d.]+)\s+ppl quant ([\d.]+)", p.stdout)
    r = {"kl": [float(s[1]) for s in seq], "nll_ref": [float(s[2]) for s in seq],
         "nll_q": [float(s[3]) for s in seq], "ppl_ref": float(m[1]), "ppl_q": float(m[2]), "wall": wall,
         "dir": d}
    r["kl_mean"] = sum(r["kl"]) / len(r["kl"])
    res = load_res()
    res[name] = r
    json.dump(res, open(RES + ".tmp", "w"), indent=1)
    os.replace(RES + ".tmp", RES)
    print(f"{time.strftime('%H:%M:%S')} {name}: KL {r['kl_mean']:.5f} ppl {r['ppl_q']:.4f} ({wall:.0f}s)", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("a", nargs="*")
    ap.add_argument("--layers", default=",".join(map(str, range(32))))
    ar = ap.parse_args()
    if ar.cmd == "eval":
        evaluate(ar.a[0], ar.a[1])
    elif ar.cmd == "sweep":
        evaluate("base", BASE)
        evaluate("upgrade", UPG)
        for L in map(int, ar.layers.split(",")):
            if f"swap-{L:02d}" in load_res():
                continue
            d = f"{ROOT}/swap-{L:02d}"
            build_mix(d, [L])
            try:
                evaluate(f"swap-{L:02d}", d)
            finally:
                shutil.rmtree(d, ignore_errors=True)
    elif ar.cmd == "mix":
        d = f"{ROOT}/mix-{ar.a[0]}"
        build_mix(d, [int(x) for x in ar.a[1].split(",")])
        evaluate(f"mix-{ar.a[0]}", d)


if __name__ == "__main__":
    main()
