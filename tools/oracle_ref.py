#!/usr/bin/env python3
"""OFFLINE REFERENCE ORACLE GENERATOR - NOT part of the NEFF emit path.

Generates neuronx-cc reference containers on the trn1 box (SSM + jax ->
StableHLO bytecode -> neuronx-cc) and wraps sg00 streams for byte-diff /
numerical validation of our own synthesis. Per project course correction
(AGENTS.md item 28): jax/neuronx-cc = OFFLINE REFERENCE ONLY. Emitters in
lift_hip.py / tile_ir.py / mm_synth.py / act_synth.py must NEVER import
from this module - tools/no_cc_check.py enforces the emit path.
"""
import json
import json as _json
import shutil
import struct
import subprocess
import sys
from pathlib import Path
import base64
import gzip
import re

sys.path.insert(0, str(Path(__file__).parent))
_SHAPE_TPL = Path(__file__).parent / "shape_templates.json"

# ---------------- generalized on-box oracle generation ----------------

_SSM_RUN = "/tmp/opencode/ssm_run.sh"


def gen_oracle_custom(key: str, gen_py: str, timeout: int = 300) -> Path:
    """Compile an arbitrary jax snippet on the trn1 box and extract sg00.

    gen_py must write <mlirbc> to /tmp/oracle_gen_<key>/graph.mlirbc.
    Chunked transfer with per-chunk length validation + refill rounds
    (SSM stdout drops chunks; first line carries a Success/Failed prefix).
    Returns the extracted sg00 dir.
    """
    open("/tmp/opencode/_gc.b64", "w").write(base64.b64encode(gen_py.encode()).decode())
    subprocess.run(["aws", "s3", "cp", "/tmp/opencode/_gc.b64",
                    "s3://912935854432-neuron-build/tools/gc.py.b64"],
                   check=True, capture_output=True)
    inner = (
        "source /home/ubuntu/work/nenv.sh 2>/dev/null\n"
        "export PATH=/opt/aws_neuronx_venv_jax_0_9/bin:/usr/local/bin:$PATH\n"
        "cd /tmp\n"
        "aws s3 cp s3://912935854432-neuron-build/tools/gc.py.b64 /tmp/gc.b64 1>/dev/null\n"
        "python3 -c \"import base64; open('/tmp/gc.py','wb').write(base64.b64decode(open('/tmp/gc.b64').read()))\"\n"
        f"/opt/aws_neuronx_venv_jax_0_9/bin/python /tmp/gc.py\n"
        f"cd /tmp/oracle_gen_{key} && neuronx-cc compile graph.mlirbc --framework XLA --target trn1 --output o.neff 2>&1 | tail -1\n"
        "cat > /tmp/fb3.py <<'PYEOF'\n"
        "import gzip as gz, tarfile, io, base64, json\n"
        f"d = open('/tmp/oracle_gen_{key}/o.neff','rb').read()\n"
        "t = tarfile.open(fileobj=io.BytesIO(gz.decompress(d[1024:])))\n"
        "out = {}\n"
        "for m in t.getmembers():\n"
        "    if m.name.startswith('sg00/') and all(x not in m.name for x in ('debug',)):\n"
        "        out[m.name.split('/')[-1]] = base64.b64encode(t.extractfile(m).read()).decode()\n"
        "blob = base64.b64encode(gz.compress(json.dumps(out).encode(), 9)).decode()\n"
        "open('/tmp/rmsblob.b64','w').write(blob)\n"
        "print('NB:' + str(len(blob)))\n"
        "PYEOF\n"
        "/opt/aws_neuronx_venv_jax_0_9/bin/python /tmp/fb3.py")
    open('/tmp/opencode/_gc_inner.sh', 'w').write(inner)
    m = None
    r = None
    for attempt in range(3):
        r = subprocess.run(["bash", _SSM_RUN, '/tmp/opencode/_gc_inner.sh', str(timeout)],
                           capture_output=True, text=True, timeout=timeout + 30)
        m = re.search(r'NB:(\d+)', r.stdout)
        if m:
            break
    assert m, f"gen failed {key}: {r.stdout[-300:] if r else 'no run'}"
    nb, n = int(m.group(1)), (int(m.group(1)) + 7999) // 8000
    have = {}
    for rnd in range(15):
        missing = [i for i in range(n) if len(have.get(i, '')) != min(8000, nb - i * 8000)]
        if not missing:
            break
        fetch = ("cat > /tmp/fc2.py <<'PYEOF'\nimport base64\n"
                 "blob = open('/tmp/rmsblob.b64').read()\nCH = 8000\nfor i in ["
                 + ",".join(map(str, missing)) + "]:\n"
                 "    print(f'NCH:{i}:' + blob[i*CH:(i+1)*CH])\nPYEOF\n"
                 "/opt/aws_neuronx_venv_jax_0_9/bin/python /tmp/fc2.py\n")
        open('/tmp/opencode/_gc_fetch.sh', 'w').write(fetch)
        r2 = subprocess.run(["bash", _SSM_RUN, '/tmp/opencode/_gc_fetch.sh', '120'],
                            capture_output=True, text=True, timeout=150)
        for line in r2.stdout.split('\n'):
            m2 = re.search(r'NCH:(\d+):(.*)$', line)
            if m2:
                i, sg = int(m2.group(1)), m2.group(2)
                if len(sg) == min(8000, nb - i * 8000):
                    have[i] = sg
    assert len(have) == n, f"chunks {sorted(have)}/{n}"
    blob = ''.join(have[i] for i in sorted(have))
    data = json.loads(gzip.decompress(base64.b64decode(blob)))
    sg = Path(f"/tmp/opencode/oracle_{key}/sg00")
    sg.mkdir(parents=True, exist_ok=True)
    for nm, b in data.items():
        (sg / nm).write_bytes(base64.b64decode(b))
    return sg


def _wrap_graph(outdir: Path, sg_src: Path, inputs, name: str = "graph",
                out_shape=None) -> Path:
    """inputs = [(dtype, [dims])...] in arg order; builds wrapper + neff."""
    out = Path(outdir)
    shutil.rmtree(out, ignore_errors=True)
    (out / "sg00").mkdir(parents=True)
    for f in sg_src.iterdir():
        shutil.copy(f, out / "sg00" / f.name)
    nin = len(inputs)
    nodes = [{"op": "null", "name": f"input{i}", "inputs": [],
              "output_names": [f"input{i}"], "is_param": "0"} for i in range(nin)]
    sg_idx = nin
    nout = len(out_shape) if (out_shape and isinstance(out_shape[0], list)) else 1
    nodes.append({"op": "tvm_op", "name": "sg_coreV10",
                  "attrs": {"func_name": "__kelf", "flatten_data": "0",
                            "num_inputs": str(nin), "kelf": "kelf-0.json",
                            "num_outputs": str(nout)},
                  "inputs": [[i, 0, 0] for i in range(nin)],
                  "output_names": [f"output{i}" for i in range(nout)]})
    if nout > 1:
        # compiler tars carry no info.json; keep it for single-output wrappers
        pass
    shapes = [[d for d in dims] for _, dims in inputs]
    outs = out_shape if (out_shape and isinstance(out_shape[0], list)) else [out_shape or list(inputs[0][1])]
    shapes.extend(outs)
    nj = {"nodes": nodes, "arg_nodes": list(range(nin)),
          "heads": [[sg_idx, i, 0] for i in range(len(outs))],
          "attrs": {"shape": ["list_shape", shapes],
                    "dltype": ["list_str", [dt for dt, _ in inputs] + ["float32"] * len(outs)],
                    "storage_id": ["list_int", list(range(nin + len(outs)))]},
          "node_row_ptr": list(range(nin + 1)) + [nin + len(outs)],
          "metadata": {"signatures": {"default": {
              "inputs": {f"input{i}": {"id": i, "dtype": dt, "shape": dims}
                         for i, (dt, dims) in enumerate(inputs)},
              "outputs": {}}}}}
    shapes[-1] = _json.loads(_json.dumps(shapes[-1]))
    (out / "neff.json").write_text(_json.dumps(nj))
    (out / "kelf-0.json").write_text(_json.dumps(
        {"graphs": [{"definition": "sg00/def.json", "name": "sg00"}],
         "version": "0.5", "target": "sunda", "revision": ""}))
    (out / "metrics.json").write_text("[]")
    subprocess.run([sys.executable, str(Path(__file__).parent / "neff_emit.py"),
                    str(out), str(out / f"{name}.neff"), "--name", f"{name}.neff"],
                   check=True, capture_output=True)
    return out / f"{name}.neff"


def _jax_gen(body: str) -> str:
    return ("import io, os\nimport jax, jax.numpy as jnp\n"
            "d = os.environ['ODIR']\nos.makedirs(d, exist_ok=True)\n"
            + body +
            "\nbuf = io.BytesIO()\nlowered.compiler_ir().operation.write_bytecode(buf)\n"
            "open(d + '/graph.mlirbc','wb').write(buf.getvalue())\nprint('gen ok')\n")


def emit_rmsnorm_gen(outdir: Path, M: int, K: int, eps: float = 1e-6,
                     name: str = "rmsnorm") -> Path:
    """Arbitrary-shape RMSNorm via on-box oracle generation + eps patch."""
    key = f"rms_{M}x{K}"
    body = (f"def f(a):\n"
            f"    return a * (1.0 / jnp.sqrt(jnp.mean(a*a, axis=-1, keepdims=True) + {eps!r}))\n"
            f"lowered = jax.jit(f).lower(jnp.zeros(({M},{K}), jnp.float32))")
    gen_py = _jax_gen(body).replace("os.environ['ODIR']", f"'/tmp/oracle_gen_{key}'")
    sg = gen_oracle_custom(key, gen_py)
    # eps is baked into the compiled graph via codegen; do NOT patch ACT
    # (@360 is a bucket-path-only offset and corrupts fresh layouts).
    return _wrap_graph(Path(outdir), sg, [("float32", [M, K])], name)


def emit_softmax_gen(outdir: Path, M: int, K: int, name: str = "softmax") -> Path:
    key = f"softmax_{M}x{K}"
    body = (f"lowered = jax.jit(lambda a: jax.nn.softmax(a, axis=-1)).lower("
            f"jnp.zeros(({M},{K}), jnp.float32))")
    gen_py = _jax_gen(body).replace("os.environ['ODIR']", f"'/tmp/oracle_gen_{key}'")
    sg = gen_oracle_custom(key, gen_py)
    return _wrap_graph(Path(outdir), sg, [("float32", [M, K])], name)


def emit_dequant_gen(outdir: Path, M: int, K: int, name: str = "dequant") -> Path:
    key = f"dequant_{M}x{K}"
    body = (f"lowered = jax.jit(lambda q, s: q.astype(jnp.float32) * s[None, :]).lower("
            f"jnp.zeros(({M},{K}), jnp.uint8), jnp.zeros(({K},), jnp.float32))")
    gen_py = _jax_gen(body).replace("os.environ['ODIR']", f"'/tmp/oracle_gen_{key}'")
    sg = gen_oracle_custom(key, gen_py)
    return _wrap_graph(Path(outdir), sg, [("uint8", [M, K]), ("float32", [K])], name)


def emit_rmsnorm_fused(outdir: Path, M: int, K: int, eps: float = 1e-6,
                       name: str = "rmsnorm_fused") -> Path:
    """Weighted RMSNorm y = x * g * rsqrt(mean(x^2)+eps) in ONE graph."""
    key = f"rmsw_{M}x{K}"
    body = (f"def f(a, g):\n"
            f"    return a * g * (1.0 / jnp.sqrt(jnp.mean(a*a, axis=-1, keepdims=True) + {eps!r}))\n"
            f"lowered = jax.jit(f).lower(jnp.zeros(({M},{K}), jnp.float32), "
            f"jnp.zeros(({K},), jnp.float32))")
    gen_py = _jax_gen(body).replace("os.environ['ODIR']", f"'/tmp/oracle_gen_{key}'")
    sg = gen_oracle_custom(key, gen_py)
    return _wrap_graph(Path(outdir), sg, [("float32", [M, K]), ("float32", [K])], name)


def emit_matmul_gen(outdir: Path, M: int, K: int, N: int, name: str = "matmul") -> Path:
    """Arbitrary-shape matmul y = x[M,K] @ w[K,N] via on-box oracle gen."""
    key = f"mm_{M}x{K}x{N}"
    body = (f"lowered = jax.jit(lambda a, b: a @ b).lower("
            f"jnp.zeros(({M},{K}), jnp.float32), jnp.zeros(({K},{N}), jnp.float32))")
    gen_py = _jax_gen(body).replace("os.environ['ODIR']", f"'/tmp/oracle_gen_{key}'")
    sg = gen_oracle_custom(key, gen_py)
    return _wrap_graph(Path(outdir), sg, [("float32", [M, K]), ("float32", [K, N])], name,
                       out_shape=[M, N])


def emit_sdpa_gen(outdir: Path, h: int, M: int, S: int, d: int, scale: float,
                  name: str = "sdpa") -> Path:
    """Attention through proven paths: y = softmax(q@k^T*scale)@v per head.
    q[h,M,d], k[h,S,d], v[h,S,d] -> y[h,M,d]. Single graph, on-box gen."""
    key = f"sdpa_{h}_{M}_{S}_{d}"
    body = (f"def f(q, k, v):\n"
            f"    a = jnp.einsum('hmd,hsd->hms', q, k) * {scale!r}\n"
            f"    p = jax.nn.softmax(a, axis=-1)\n"
            f"    return jnp.einsum('hms,hsd->hmd', p, v)\n"
            f"lowered = jax.jit(f).lower(jnp.zeros(({h},{M},{d}), jnp.float32), "
            f"jnp.zeros(({h},{S},{d}), jnp.float32), jnp.zeros(({h},{S},{d}), jnp.float32))")
    gen_py = _jax_gen(body).replace("os.environ['ODIR']", f"'/tmp/oracle_gen_{key}'")
    sg = gen_oracle_custom(key, gen_py)
    return _wrap_graph(Path(outdir), sg,
                       [("float32", [h, M, d]), ("float32", [h, S, d]), ("float32", [h, S, d])],
                       name, out_shape=[h, M, d])


def emit_block_gen(outdir: Path, M: int, K: int, H: int, h: int, d: int,
                   eps: float = 1e-6, name: str = "block") -> Path:
    """Full decoder block in ONE graph:
    y = ffn(rms(x + o(sdpa(rms(x)*g1) @ wo)) * g2)
    qkv fused proj, h=h heads of dim d (K = h*d for qkv split).
    Inputs: x[M,K] g1[K] wqkv[K,3K] wo[K,K] w1[K,H] w2[H,K] g2[K]
    """
    key = f"block_{M}_{K}_{H}_{h}_{d}"
    body = (
        f"def f(x, g1, wqkv, wo, w1, w2, g2):\n"
        f"    def rms(v, g):\n"
        f"        return v * g * (1.0 / jnp.sqrt(jnp.mean(v*v, axis=-1, keepdims=True) + {eps!r}))\n"
        f"    qkv = rms(x, g1) @ wqkv\n"
        f"    q, k, v = qkv[..., :{K}], qkv[..., {K}:2*{K}], qkv[..., 2*{K}:]\n"
        f"    q = q.reshape({M}, {h}, {d}).transpose(1, 0, 2)\n"
        f"    k = k.reshape({M}, {h}, {d}).transpose(1, 0, 2)\n"
        f"    v = v.reshape({M}, {h}, {d}).transpose(1, 0, 2)\n"
        f"    a = jnp.einsum('hmd,hnd->hmn', q, k) * {1.0 / d**0.5:.8f}\n"
        f"    p = jax.nn.softmax(a, axis=-1)\n"
        f"    o = jnp.einsum('hmn,hnd->hmd', p, v)\n"
        f"    o = o.transpose(1, 0, 2).reshape({M}, {K}) @ wo\n"
        f"    t = rms(x + o, g2)\n"
        f"    return (jax.nn.gelu(t @ w1) @ w2)\n"
        f"lowered = jax.jit(f).lower(jnp.zeros(({M},{K}), jnp.float32), jnp.zeros(({K},), jnp.float32), "
        f"jnp.zeros(({K},{3*K}), jnp.float32), jnp.zeros(({K},{K}), jnp.float32), "
        f"jnp.zeros(({K},{H}), jnp.float32), jnp.zeros(({H},{K}), jnp.float32), jnp.zeros(({K},), jnp.float32))")
    gen_py = _jax_gen(body).replace("os.environ['ODIR']", f"'/tmp/oracle_gen_{key}'")
    sg = gen_oracle_custom(key, gen_py)
    return _wrap_graph(Path(outdir), sg,
                       [("float32", [M, K]), ("float32", [K]), ("float32", [K, 3 * K]),
                        ("float32", [K, K]), ("float32", [K, H]), ("float32", [H, K]),
                        ("float32", [K])],
                       name, out_shape=[M, K])


def emit_be_block_gen(outdir: Path, K: int = 12288, hq: int = 96, hkv: int = 8,
                      d: int = 128, H: int = 28672, theta: float = 1e6,
                      eps: float = 1e-5, S: int = 1, name: str = "beblock") -> Path:
    """Behemoth decode block (runtime/src arch): post-norm, RoPE pair(d,d+64)
    theta 1e6, GQA, SwiGLU, RMSNorm eps 1e-5. M=1 decode step w/ KV cache.
    Inputs: x[K] w_in[K] wq[K,hq*d] wk[K,hkv*d] wv[K,hkv*d] wo[hq*d,K]
            w_post[K] wg[K,H] wu[K,H] wd[H,K] kc[hkv,S,d] vc[hkv,S,d] pos(i32)
    Outputs: y[K] k_new[hkv,d] v_new[hkv,d]
    """
    key = f"beblk_{K}_{hq}_{hkv}_{d}_{H}_{S}"
    body = (
        f"def f(x, w_in, wq, wk, wv, wo, w_post, wg, wu, wd, kc, vc, pos):\n"
        f"    def rms(v, g):\n"
        f"        return v * g * (1.0 / jnp.sqrt(jnp.mean(v*v) + {eps!r}))\n"
        f"    xn = rms(x, w_in)\n"
        f"    q = (xn @ wq).reshape({hq}, {d})\n"
        f"    k = (xn @ wk).reshape({hkv}, {d})\n"
        f"    v = (xn @ wv).reshape({hkv}, {d})\n"
        f"    inv = {theta!r} ** (-jnp.arange({d//2}) * 2.0 / {d})\n"
        f"    ang = pos[0] * inv\n"
        f"    c, s = jnp.cos(ang), jnp.sin(ang)\n"
        f"    def rope(t):\n"
        f"        a, b = t[:, :{d//2}], t[:, {d//2}:]\n"
        f"        return jnp.concatenate([a*c - b*s, b*c + a*s], -1)\n"
        f"    q, k = rope(q), rope(k)\n"
        f"    Kall = jnp.concatenate([kc, k[:, None]], 1)\n"
        f"    Vall = jnp.concatenate([vc, v[:, None]], 1)\n"
        f"    Kr = jnp.repeat(Kall, {hq//hkv}, 0)\n"
        f"    Vr = jnp.repeat(Vall, {hq//hkv}, 0)\n"
        f"    sc = jnp.einsum('md,msd->ms', q, Kr) * {1.0/d**0.5:.8f}\n"
        f"    p = jax.nn.softmax(sc, -1)\n"
        f"    o = (jnp.einsum('ms,msd->md', p, Vr).reshape({hq*d},) @ wo)\n"
        f"    res = x + o\n"
        f"    t = rms(res, w_post)\n"
        f"    g = t @ wg\n"
        f"    u = t @ wu\n"
        f"    h = g / (1.0 + jnp.exp(-g)) * u\n"
        f"    return res + h @ wd, k, v\n"
        f"lowered = jax.jit(f).lower(\n"
        f"    jnp.zeros(({K},), jnp.float32), jnp.zeros(({K},), jnp.float32),\n"
        f"    jnp.zeros(({K},{hq*d}), jnp.float32), jnp.zeros(({K},{hkv*d}), jnp.float32),\n"
        f"    jnp.zeros(({K},{hkv*d}), jnp.float32), jnp.zeros(({hq*d},{K}), jnp.float32),\n"
        f"    jnp.zeros(({K},), jnp.float32), jnp.zeros(({K},{H}), jnp.float32),\n"
        f"    jnp.zeros(({K},{H}), jnp.float32), jnp.zeros(({H},{K}), jnp.float32),\n"
        f"    jnp.zeros(({hkv},{S},{d}), jnp.float32), jnp.zeros(({hkv},{S},{d}), jnp.float32),\n"
        f"    jnp.zeros((1,), jnp.int32))")
    gen_py = _jax_gen(body).replace("os.environ['ODIR']", f"'/tmp/oracle_gen_{key}'")
    sg = gen_oracle_custom(key, gen_py, timeout=600)
    ins = [("float32", [K]), ("float32", [K]), ("float32", [K, hq * d]),
           ("float32", [K, hkv * d]), ("float32", [K, hkv * d]),
           ("float32", [hq * d, K]), ("float32", [K]), ("float32", [K, H]),
           ("float32", [K, H]), ("float32", [H, K]),
           ("float32", [hkv, S, d]), ("float32", [hkv, S, d]), ("int32", [1])]
    return _wrap_graph(Path(outdir), sg, ins, name,
                       out_shape=[[K], [hkv, d], [hkv, d]])
