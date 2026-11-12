"""EXPERIMENT harness for nki_moe.py: the routed-expert part of a decode MoE block (one TP rank of Qwen3.5-397B: 512 experts,
egu [512, 64, 4096] fp8, ed [512, 32, 4096] fp8) as K chained blocks in one single-core NEFF, either through the NKI kernel
(form nki) or through qwen35_graphs.moe_routed (form torch, MOE_SLOTS=row semantics: slot r * 10 + j = expert j of row r).
Each block: n = bf16(rms(x)); routed = experts(n, idx_k, w); x = x + bf16(routed), with its own weight tensors and its own
expert indices ((top + offset_k) % 512, top a runtime input). layer_bench NEFF steps 1 times it; (K = 8 - K = 2) / 6 = ms per block.

  python bench_moe_nki.py build OUT T K {nki|torch} [key=value ...]   -> OUT/<form>[_opts]_T_K/graph.neff
        nki options: gu=slot|grid act=slot|grid down=tile4|half static=0|1 parts=gu,down
        torch options: env ROW_FORM (default launder), MOE_BMM
  python bench_moe_nki.py num OUT T {nki|torch} NEFF_IO [key=value ...]   (device job: K = 1, random fp8 weights, vs the CPU mirror)
"""
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('ROW_FORM', 'launder')
os.environ.setdefault('UNSAFE_FP8FNCAST', '1')
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glm_graphs as G  # noqa: E402
import qwen35_graphs as Q  # noqa: E402

F8 = G.F8
E, EI, H, TOPK = 512, 32, 4096, 10
g = SimpleNamespace(E=E, EIl=EI, H=H, topk=TOPK)


def offsets(T, K):
    r = np.random.default_rng(7)
    return [torch.from_numpy(np.stack([r.permutation(E)[:TOPK] for _ in range(T)])).long() if k else torch.zeros(T, TOPK, dtype=torch.long)
            for k in range(K)]


def make_fn(form, T, K, opts):
    offs = offsets(T, K)
    S = T * TOPK
    kernel = None
    if form == 'nki':
        import nki_moe
        kernel = nki_moe.make_moe(T, S, gu=opts.get('gu', 'slot'), act_form=opts.get('act', 'slot'), down=opts.get('down', 'tile4'),
                                  static=opts.get('static', '0') == '1', parts=tuple(opts.get('parts', 'gu,down').split(',')))

    def fn(x, top, w, *ws):
        for k in range(K):
            egu, egu_s, ed, ed_c = ws[4 * k:4 * k + 4]
            n = G.bf(x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6))
            tk = torch.remainder(top + offs[k].to(x.device), E)
            if form == 'nki':
                idx = tk.reshape(S)
                comb = (w.unsqueeze(1) * torch.eye(T, device=x.device).reshape(T, T, 1)).reshape(T, S)
                import nki_moe
                routed = kernel(*nki_moe.graph_inputs(n, idx, comb, egu_s, ed_c), egu, ed)
            else:
                routed = Q.moe_routed(n, (tk, w), {'egu': egu, 'egu_s': egu_s, 'ed': ed, 'ed_c': ed_c}, g, 'gather')[0]
            x = x + G.bf(routed)
        return x
    return fn


def weight_inputs(form, K, rand=None):
    out = []
    for k in range(K):
        if rand is None:
            egu, ed = G.zeros((E, 2 * EI, H), F8), G.zeros((E, EI, H), F8)
            egu_s, ed_c = torch.ones(E, 2 * EI), torch.ones(E, EI)
        else:
            # row-scaled E4M3 codes with max |code| 224 like the prepared checkpoint (glm_repack.to_fp8_rows)
            def q(wf):
                sc = torch.amax(torch.abs(wf), dim=-1).clamp(min=1e-30) / 224.0
                return (wf / sc.unsqueeze(-1)).to(F8), sc.float()
            egu, egu_s = q(torch.from_numpy(rand.standard_normal((E, 2 * EI, H), dtype=np.float32)) / 64)
            ed, ed_c = q(torch.from_numpy(rand.standard_normal((E, EI, H), dtype=np.float32)) / 16)
        if form == 'nki':
            import nki_moe
            egu = torch.from_numpy(nki_moe.relayout_egu(egu.view(torch.uint8).numpy())).view(F8)
            ed = ed.reshape(E * EI, H)
        out += [egu, egu_s, ed, ed_c]
    return out


def build(out, name, fn, inputs):
    import torch_neuronx
    d = Path(out) / name
    d.mkdir(parents=True, exist_ok=True)
    try:
        torch_neuronx.trace(fn, tuple(inputs), compiler_workdir=str(d))
    except RuntimeError as error:
        if 'neuronx-cc failed' not in str(error):
            raise
    here = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(here / 'tools_patch_fp8_hlo.py'), str(d / 'model' / 'graph.hlo'), str(d / 'model' / 'graph.hlo')], check=True)
    subprocess.run(['neuronx-cc', 'compile', 'model', '--framework', 'XLA', '--target', 'trn1', '--output', 'graph.neff'] +
                   os.environ.get('NCC_FLAGS', '').split(), cwd=d, check=True, stdout=subprocess.DEVNULL)
    return d


def main():
    mode, out, T = sys.argv[1], sys.argv[2], int(sys.argv[3])
    if mode == 'build':
        K, form, rest = int(sys.argv[4]), sys.argv[5], sys.argv[6:]
    else:
        K, form, neff_io, rest = 1, sys.argv[4], sys.argv[5], sys.argv[6:]
    opts = dict(a.split('=') for a in rest)
    tag = form + ''.join('_' + k + v.replace(',', '') for k, v in sorted(opts.items()))
    if form == 'torch':
        tag += '_' + (os.environ.get('ROW_FORM') or 'plain') + ('_bmm' if Q.MOE_BMM else '')
    fn = make_fn(form, T, K, opts)
    if mode == 'build':
        inputs = [torch.zeros(T, H), torch.zeros(T, TOPK, dtype=torch.long), torch.ones(T, TOPK) / TOPK] + weight_inputs(form, K)
        print('built', build(out, f'{tag}_{T}_{K}', fn, inputs), flush=True)
        return
    rand = np.random.default_rng(3)
    x = torch.from_numpy(rand.standard_normal((T, H)).astype(np.float32))
    top = torch.from_numpy(np.stack([rand.permutation(E)[:TOPK] for _ in range(T)])).long()
    if T > 1:
        top[1, :3] = top[0, :3]                                                         # duplicates across rows
    w = torch.softmax(torch.from_numpy(rand.standard_normal((T, TOPK)).astype(np.float32)), dim=-1)
    state = rand.bit_generator.state
    ref_w = weight_inputs('torch', 1, rand)
    ref = make_fn('torch', T, 1, {})(x, top, w, *ref_w)
    rand.bit_generator.state = state
    dev_w = weight_inputs(form, 1, rand)
    d = build(out, f'num_{tag}_{T}', fn, [x, top, w] + dev_w)
    io = d / 'io'
    io.mkdir(exist_ok=True)
    import glm_repack as RP
    for i, t in enumerate([x, top, w] + dev_w):
        (io / f'in{i}').write_bytes(RP.tensor_bytes(t))
    subprocess.run([neff_io, str(d / 'graph.neff'), str(io), '1'], check=True, stdout=subprocess.DEVNULL)
    y = torch.from_numpy(np.frombuffer((io / 'out0.r0').read_bytes(), np.float32).reshape(T, H).copy())
    rel = lambda a, b: float(torch.linalg.norm(a - b) / torch.linalg.norm(b))
    other = Path(out) / f'num_torch_launder_{T}' / 'io' / 'out0.r0'
    if form == 'nki' and other.exists():
        yt = torch.from_numpy(np.frombuffer(other.read_bytes(), np.float32).reshape(T, H).copy())
        print(f'NUM {tag} T={T}: kernel vs the torch form ON THE DEVICE: routed-part rel L2 {rel(y - x, yt - x):.3e}', flush=True)
    print(f'NUM {tag} T={T}: routed-part rel L2 {rel(y - x, ref - x):.3e}  (output {rel(y, ref):.3e}, |routed|/|x| {rel(ref, x):.3f}, finite {bool(torch.isfinite(y).all())})', flush=True)


if __name__ == '__main__':
    main()
