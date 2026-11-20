"""EXPERIMENT harness for nki_moe_dyn.py (distinct-expert decode MoE kernel), one NeuronCore.

K chained blocks in one single-core NEFF; each block: n = bf16(rms(x)); routed = experts(n, comb_k); x = x + bf16(routed), with its
own weights and its own routing: row t of block k selects experts (top[t] + off_k[t]) % 512, off_k drawn by `route`:
  uniform  every row 10 distinct experts uniformly at random (no sharing beyond chance: ~75 / 139 / 193 / 240 distinct at 8 / 16 / 24 / 32 rows)
  skew     experts drawn with Zipf-like popularity p(e) ~ 1 / (rank + 8) (fewer distinct experts, as real routers are)
  same     every row the same 10 experts
layer_bench NEFF steps 1 times it; (K = 8 - K = 2) / 6 = ms per block.

  python bench_moe_dyn.py build OUT T K FORM [key=value ...]   -> OUT/<form>[_opts]_T_K/graph.neff
      FORM dyn: the kernel (kmax=10T default); union: qwen35_graphs.moe_routed union form, rows in groups of `split` (default 8; the
      served MOE_SPLIT=8 form for T >= 16); rows: the row-slot form (MOE_SLOTS=row)
      options: route=uniform|skew|same  kmax=N  split=N
  python bench_moe_dyn.py num OUT T NEFF_IO [key=value ...]    (device job: K = 1, random fp8 weights; dyn and union vs the f32 mirror)
"""
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('UNSAFE_FP8FNCAST', '1')
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glm_graphs as G  # noqa: E402
import qwen35_graphs as Q  # noqa: E402

F8 = G.F8
E, EI, H, TOPK = 512, 32, 4096, 10
g = SimpleNamespace(E=E, EIl=EI, H=H, topk=TOPK)


def draw(T, route, r):
    if route == 'same':
        e = r.permutation(E)[:TOPK]
        return np.stack([e] * T)
    if route == 'skew':
        p = 1.0 / (np.arange(E) + 8.0)
        p = p / p.sum()
        perm = r.permutation(E)
        return np.stack([perm[r.choice(E, TOPK, replace=False, p=p)] for _ in range(T)])
    return np.stack([r.permutation(E)[:TOPK] for _ in range(T)])


def offsets(T, K, route):
    r = np.random.default_rng(7)
    return [torch.from_numpy(draw(T, route, r)).long() if k else torch.zeros(T, TOPK, dtype=torch.long) for k in range(K)]


def comb_of(tk, w):
    """[T, K] experts + weights -> [T, E] combine matrix (the route() output form)."""
    return (w.unsqueeze(-1) * (tk.unsqueeze(-1) == torch.arange(E, device=tk.device).reshape(1, 1, E)).float()).sum(dim=1)


def make_fn(form, T, K, opts):
    route = opts.get('route', 'uniform')
    offs = offsets(T, K, route)
    kmax = int(opts.get('kmax', min(TOPK * T, E)))
    split = int(opts.get('split', 8))
    import nki_moe_dyn as D

    def fn(x, top, w, *ws):
        w = torch.abs(w) + 1e-3          # every selected expert has a positive weight whatever the bench / profiler fills in
        for k in range(K):
            egu, egu_s, ed, ed_c = ws[4 * k:4 * k + 4]
            n = G.bf(x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6))
            tk = torch.remainder(top + offs[k].to(x.device), E)
            if form == 'dyn':
                comb = comb_of(tk, w)
                routed = D.moe_dyn(*D.graph_inputs(n, comb, egu_s, ed_c, kmax), egu, ed)
            elif form == 'dyn3':
                comb = comb_of(tk, w)
                U = int(opts.get('U', '4'))
                routed = D.moe_dyn3(*D.graph_inputs(n, comb, egu_s, ed_c, kmax, U=U), egu, ed, U=U)
            elif form == 'dyn4':
                comb = comb_of(tk, w)
                U, V = int(opts.get('U', '8')), int(opts.get('V', '2'))
                routed = D.moe_dyn4(*D.graph_inputs(n, comb, egu_s, ed_c, kmax, levels=(U, V)), egu, ed, U=U, V=V)
            elif form == 'dyn5':
                comb = comb_of(tk, w)
                U, V = int(opts.get('U', '8')), int(opts.get('V', '2'))
                routed = D.moe_dyn5(*D.graph_inputs5(n, comb, egu_s, ed_c, kmax, (U, V)), egu, ed, U=U, V=V)
            elif form == 'dyn6':
                comb = comb_of(tk, w)
                U, V = int(opts.get('U', '8')), int(opts.get('V', '2'))
                routed = D.moe_dyn6(*D.graph_inputs6(n, comb, egu_s, ed_c, kmax, (U, V)), egu, ed, U=U, V=V)
            elif form == 'dyn7':
                comb = comb_of(tk, w)
                U, V = int(opts.get('U', '8')), int(opts.get('V', '2'))
                routed = D.moe_dyn7(*D.graph_inputs7(n, comb, egu_s, ed_c, kmax, (U, V)), egu, ed, U=U, V=V)
            elif form == 'glue0':
                comb = comb_of(tk, w)
                routed = comb[:, :1] * n + egu_s[0, 0] + ed_c[0, 0] + egu[0, 0].float() + ed[0, 0].float()
            elif form == 'glue7':
                comb = comb_of(tk, w)
                gi = D.graph_inputs7(n, comb, egu_s, ed_c, kmax, (8, 2))
                routed = (gi[0].float() * gi[1] + gi[2].sum() + gi[3].sum() + gi[4].sum() + gi[5].float().sum() + gi[6].float().sum()
                          + egu[0, 0].float() + ed[0, 0].float())
            elif form == 'dyn8':
                comb = comb_of(tk, w)
                U, V = int(opts.get('U', '8')), int(opts.get('V', '2'))
                routed = D.moe_dyn8(*D.graph_inputs8(n, comb, egu_s, ed_c, kmax, (U, V)), egu, ed, U=U, V=V)
            elif form == 'dyn9':
                comb = comb_of(tk, w)
                U, V = int(opts.get('U', '8')), int(opts.get('V', '2'))
                routed = D.moe_dyn9(*D.graph_inputs9(n), comb, egu_s, ed_c, egu, ed, KM=kmax, U=U, V=V, order=int(opts.get('order', '1')))
            elif form == 'dyn2':
                comb = comb_of(tk, w)
                routed = D.moe_dyn2(*D.graph_inputs(n, comb, egu_s, ed_c, kmax, spare=2), egu, ed)
            elif form == 'rows':
                routed = Q.moe_routed(n, (tk, w), {'egu': egu, 'egu_s': egu_s, 'ed': ed, 'ed_c': ed_c}, g, 'gather')[0]
            else:
                comb = comb_of(tk, w)
                tt = {'egu': egu, 'egu_s': egu_s, 'ed': ed, 'ed_c': ed_c}
                sp = split if (T > split and T % split == 0) else T
                routed = torch.cat([Q.moe_routed(n[i:i + sp], comb[i:i + sp], tt, g, 'gather')[0] for i in range(0, T, sp)], dim=0)
            x = x + G.bf(routed)
        return x
    return fn


def weight_inputs(form, K, rand=None):
    import nki_moe
    out = []
    for k in range(K):
        if rand is None:
            egu, ed = G.zeros((E, 2 * EI, H), F8), G.zeros((E, EI, H), F8)
            egu_s, ed_c = torch.ones(E, 2 * EI), torch.ones(E, EI)
        else:
            def q(wf):
                sc = torch.amax(torch.abs(wf), dim=-1).clamp(min=1e-30) / 224.0
                return (wf / sc.unsqueeze(-1)).to(F8), sc.float()
            egu, egu_s = q(torch.from_numpy(rand.standard_normal((E, 2 * EI, H), dtype=np.float32)) / 64)
            ed, ed_c = q(torch.from_numpy(rand.standard_normal((E, EI, H), dtype=np.float32)) / 16)
        if form.startswith('dyn') or form.startswith('glue'):
            egu = torch.from_numpy(nki_moe.relayout_egu(egu.view(torch.uint8).numpy())).view(F8) if rand is not None else G.zeros((E * 128, 2048), F8)
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


def tag_of(form, opts):
    return form + ''.join('_' + k + v.replace(',', '') for k, v in sorted(opts.items()))


def num(out, T, neff_io, opts):
    """K = 1 block on the device (dyn and union forms, same random weights / routing) vs the exact f32 mirror (no residual)."""
    import glm_repack as RP
    import nki_moe_dyn as D
    route = opts.get('route', 'uniform')
    rand = np.random.default_rng(3)
    x = torch.from_numpy(rand.standard_normal((T, H)).astype(np.float32))
    top = torch.from_numpy(draw(T, route, rand)).long()
    w = torch.softmax(torch.from_numpy(rand.standard_normal((T, TOPK)).astype(np.float32)), dim=-1)
    state = rand.bit_generator.state
    wt = weight_inputs('union', 1, rand)
    n = G.bf(x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6))
    ref = D.mirror(n, comb_of(top, torch.abs(w) + 1e-3), *wt)
    ys = {}
    dform = opts.get('form', 'dyn')
    for form in (opts.pop('form', 'dyn'), 'union'):
        rand.bit_generator.state = state
        dev_w = weight_inputs(form, 1, rand)
        fn = make_fn(form, T, 1, opts)
        d = build(out, f'num_{tag_of(form, opts)}_{T}', fn, [x, top, w] + dev_w)
        io = d / 'io'
        io.mkdir(exist_ok=True)
        for i, t in enumerate([x, top, w] + dev_w):
            (io / f'in{i}').write_bytes(RP.tensor_bytes(t))
        subprocess.run([neff_io, str(d / 'graph.neff'), str(io), '1'], check=True, stdout=subprocess.DEVNULL)
        ys[form] = torch.from_numpy(np.frombuffer((io / 'out0.r0').read_bytes(), np.float32).reshape(T, H).copy()) - x
    refb = G.bf(ref)
    rel = lambda a, b: float(torch.linalg.norm(a - b) / torch.linalg.norm(b))
    nd = int((comb_of(top, torch.abs(w) + 1e-3) > 0).any(dim=0).sum())
    print(f'NUM T={T} route={route} distinct={nd}: dyn vs mirror {rel(ys[dform], refb):.3e} | union vs mirror {rel(ys["union"], refb):.3e} | '
          f'dyn vs union (device) {rel(ys[dform], ys["union"]):.3e} max|diff| {float((ys[dform] - ys["union"]).abs().max()):.3e} | '
          f'finite {bool(torch.isfinite(ys[dform]).all())}', flush=True)


def main():
    mode, out, T = sys.argv[1], sys.argv[2], int(sys.argv[3])
    if mode == 'num':
        return num(out, T, sys.argv[4], dict(a.split('=') for a in sys.argv[5:]))
    K, form = int(sys.argv[4]), sys.argv[5]
    opts = dict(a.split('=') for a in sys.argv[6:])
    fn = make_fn(form, T, K, opts)
    inputs = [torch.zeros(T, H), torch.zeros(T, TOPK, dtype=torch.long), torch.ones(T, TOPK) / TOPK] + weight_inputs(form, K)
    print('built', build(out, f'{tag_of(form, opts)}_{T}_{K}', fn, inputs), flush=True)


if __name__ == '__main__' and sys.argv[1] not in ('sim', 'dbg', 'hang'):
    main()


def sim(T, opts):
    """CPU: the kernel in the NKI simulator on host-built inputs vs the f32 mirror."""
    import nki
    import nki_moe_dyn as D
    route = opts.get('route', 'uniform')
    rand = np.random.default_rng(3)
    x = torch.from_numpy(rand.standard_normal((T, H)).astype(np.float32))
    top = torch.from_numpy(draw(T, route, rand)).long()
    w = torch.softmax(torch.from_numpy(rand.standard_normal((T, TOPK)).astype(np.float32)), dim=-1)
    state = rand.bit_generator.state
    wt = weight_inputs('union', 1, rand)
    rand.bit_generator.state = state
    egu_t, egu_s, ed, ed_c = weight_inputs('dyn', 1, rand)
    n = G.bf(x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6))
    comb = comb_of(top, w)
    ref = D.mirror(n, comb, *wt)
    kmax = int(opts.get('kmax', min(TOPK * T, E)))
    two = opts.get('form') == 'dyn2'
    U = int(opts.get('U', '4')) if opts.get('form') == 'dyn3' else 0
    lv = (int(opts.get('U', '8')), int(opts.get('V', '2'))) if opts.get('form') == 'dyn4' else None
    five = opts.get('form') in ('dyn5', 'dyn6', 'dyn7', 'dyn8')
    if opts.get('form') == 'dyn9':
        lv = (int(opts.get('U', '8')), int(opts.get('V', '2')))
        import ml_dtypes
        f8 = lambda t: t.contiguous().view(torch.uint8).numpy().view(ml_dtypes.float8_e4m3)
        n8, a = D.graph_inputs9(n)
        y = nki.simulate(D.moe_dyn9)(f8(n8), a.numpy(), comb.numpy(), egu_s.numpy(), ed_c.numpy(), f8(egu_t), f8(ed), KM=kmax, U=lv[0], V=lv[1], order=int(opts.get('order', '1')))
        y = torch.from_numpy(np.asarray(y, dtype=np.float32))
        print(f'SIM T={T}: kernel vs mirror {float(torch.linalg.norm(y - ref) / torch.linalg.norm(ref)):.3e}', flush=True)
        return
    if five:
        lv = (int(opts.get('U', '8')), int(opts.get('V', '2')))
        ins = {'dyn5': D.graph_inputs5, 'dyn6': D.graph_inputs6, 'dyn7': D.graph_inputs7, 'dyn8': D.graph_inputs8}[opts['form']](n, comb, egu_s, ed_c, kmax, lv)
    else:
        ins = D.graph_inputs(n, comb, egu_s, ed_c, kmax, spare=2 if two else 0, U=U, levels=lv)
    import ml_dtypes
    f8 = lambda t: t.contiguous().view(torch.uint8).numpy().view(ml_dtypes.float8_e4m3)
    kern = {'dyn5': D.moe_dyn5, 'dyn6': D.moe_dyn6, 'dyn7': D.moe_dyn7, 'dyn8': D.moe_dyn8}[opts['form']] if five else (D.moe_dyn2 if two else (D.moe_dyn3 if U else (D.moe_dyn4 if lv else D.moe_dyn)))
    kw = {'U': U} if U else ({'U': lv[0], 'V': lv[1]} if lv else {})
    y = nki.simulate(kern)(*[t.contiguous().numpy() if t.dtype != F8 else f8(t) for t in ins], f8(egu_t), f8(ed), **kw)
    y = torch.from_numpy(np.asarray(y, dtype=np.float32))
    rel = lambda a, b: float(torch.linalg.norm(a - b) / torch.linalg.norm(b))
    print(f'SIM T={T}: kernel vs mirror {rel(y, ref):.3e}', flush=True)


def dbg(out, T, neff_io, opts):
    """device: the kernel alone (graph glue included) returning its HBM scratch too; each phase vs the mirror."""
    import glm_repack as RP
    import nki_moe_dyn as D
    route = opts.get('route', 'uniform')
    rand = np.random.default_rng(3)
    x = torch.from_numpy(rand.standard_normal((T, H)).astype(np.float32))
    top = torch.from_numpy(draw(T, route, rand)).long()
    w = torch.softmax(torch.from_numpy(rand.standard_normal((T, TOPK)).astype(np.float32)), dim=-1)
    state = rand.bit_generator.state
    wt = weight_inputs('union', 1, rand)
    rand.bit_generator.state = state
    egu_t, egu_s, ed, ed_c = weight_inputs('dyn', 1, rand)
    n = G.bf(x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6))
    comb = comb_of(top, w)
    kmax = int(opts.get('kmax', min(TOPK * T, E)))
    ins = list(D.graph_inputs(n, comb, egu_s, ed_c, kmax))
    def fn(*a):
        y, acth, axh, wd_, hd_ = D.moe_dyn(*a, dbg=1, gu=int(opts.get('gu', '1')))
        return y * 1.0, acth * 1.0, axh.float(), wd_.float(), hd_ * 1.0
    args = ins + [egu_t, ed]
    d = build(out, f'dbg_{tag_of("dyn", opts)}_{T}', fn, args)
    io = d / 'io'
    io.mkdir(exist_ok=True)
    for i, t in enumerate(args):
        (io / f'in{i}').write_bytes(RP.tensor_bytes(t) if t.dtype != torch.int32 else t.contiguous().numpy().tobytes())
    subprocess.run([neff_io, str(d / 'graph.neff'), str(io), '1'], check=True, stdout=subprocess.DEVNULL)
    G_ = kmax // 8
    rd = lambda j, shp: torch.from_numpy(np.frombuffer((io / f'out{j}.r0').read_bytes(), np.float32).reshape(shp).copy())
    y, acth, axh = rd(0, (T, H)), rd(1, (G_, T, 256)), rd(2, (G_, 128, 2 * T))
    import ml_dtypes
    f8 = lambda t: t.contiguous().view(torch.uint8).numpy().view(ml_dtypes.float8_e4m3)
    import nki
    wdv, hdv = rd(3, (128, 8 * 2048)), rd(4, (T, 512))
    ys, ah, xh, wds, hds = nki.simulate(D.moe_dyn)(*[t.contiguous().numpy() if t.dtype != F8 else f8(t) for t in ins], f8(egu_t), f8(ed), dbg=1, gu=int(opts.get('gu', '1')))
    ng = int(ins[5].reshape(-1)[0])
    rel = lambda a, b: float(torch.linalg.norm(a - b) / max(float(torch.linalg.norm(b)), 1e-30))
    ah = torch.from_numpy(np.asarray(ah, np.float32)); xh = torch.from_numpy(np.asarray(xh, np.float32)); ys = torch.from_numpy(np.asarray(ys, np.float32))
    for gg in range(ng):
        print(f'DBG g={gg}: act {rel(acth[gg], ah[gg]):.3e}  ax {rel(axh[gg], xh[gg]):.3e}', flush=True)
    print(f'DBG ng={ng} y {rel(y, ys):.3e}', flush=True)
    wds = torch.from_numpy(np.asarray(wds).astype(np.float32)); hds = torch.from_numpy(np.asarray(hds, np.float32))
    print(f'DBG last group: w {rel(wdv, wds):.3e} acc {rel(hdv, hds):.3e}; w per slot ' + ' '.join(f'{rel(wdv[:, j*2048:(j+1)*2048], wds[:, j*2048:(j+1)*2048]):.2e}' for j in range(8)), flush=True)
    print('DBG acc per slot ' + ' '.join(f'{rel(hdv[:, j*64:(j+1)*64], hds[:, j*64:(j+1)*64]):.2e}' for j in range(8)), flush=True)
    print('DBG acc row0 dev', hdv[0, :6].tolist(), 'sim', hds[0, :6].tolist(), flush=True)


if __name__ == '__main__' and sys.argv[1] == 'dbg':
    dbg(sys.argv[2], int(sys.argv[3]), sys.argv[4], dict(a.split('=') for a in sys.argv[5:]))
if __name__ == '__main__' and sys.argv[1] == 'sim':
    sim(int(sys.argv[2]), dict(a.split('=') for a in sys.argv[3:]))


def hang(out, T, neff_io, opts):
    """device: one dyn9 block (K = 1) run with routings of every distinct-expert count 10 .. min(10 T, 512) in steps (loop-bound
    combinations: b1 / b2 / ng), each through neff_io under a timeout; prints the counts that do not finish or are wrong."""
    import glm_repack as RP
    import nki_moe_dyn as D
    rand = np.random.default_rng(3)
    x = torch.from_numpy(rand.standard_normal((T, H)).astype(np.float32))
    w = torch.softmax(torch.from_numpy(rand.standard_normal((T, TOPK)).astype(np.float32)), dim=-1)
    state = rand.bit_generator.state
    wt = weight_inputs('union', 1, rand)
    rand.bit_generator.state = state
    dev_w = weight_inputs('dyn', 1, rand)
    top0 = torch.zeros(T, TOPK, dtype=torch.long)
    fn = make_fn('dyn9', T, 1, opts)
    d = build(out, f'hang_{tag_of("dyn9", opts)}_{T}', fn, [x, top0, w] + dev_w)
    io = d / 'io'
    io.mkdir(exist_ok=True)
    for i, t in enumerate([x, top0, w] + dev_w):
        (io / f'in{i}').write_bytes(RP.tensor_bytes(t))
    n = G.bf(x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6))
    step = int(opts.get('step', '8'))
    for m in list(range(10, min(TOPK * T, E) + 1, step)) + [min(TOPK * T, E)]:
        perm = torch.from_numpy(rand.permutation(E)[:m])
        top = torch.stack([perm[(torch.arange(TOPK) + t * TOPK) % m] for t in range(T)])   # rows cycle over a pool of m experts
        (io / 'in1').write_bytes(top.numpy().tobytes())
        try:
            subprocess.run([neff_io, str(d / 'graph.neff'), str(io), '1'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        except Exception as e:  # noqa: BLE001
            print(f'HANG T={T} distinct={m}: {type(e).__name__}', flush=True)
            continue
        y = torch.from_numpy(np.frombuffer((io / 'out0.r0').read_bytes(), np.float32).reshape(T, H).copy()) - x
        ref = G.bf(D.mirror(n, comb_of(top, torch.abs(w) + 1e-3), *wt))
        ng = -(-m // 8)
        print(f'OK T={T} distinct={m} ng={ng} rel {float(torch.linalg.norm(y - ref) / torch.linalg.norm(ref)):.3e}', flush=True)


if __name__ == '__main__' and sys.argv[1] == 'hang':
    hang(sys.argv[2], int(sys.argv[3]), sys.argv[4], dict(a.split('=') for a in sys.argv[5:]))
