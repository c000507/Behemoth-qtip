"""gdn_block prefill forms vs the token loop, EXACT f32 (outputs and states; padding, carried state):
GDN_CHUNK with GDN_INV=diag (gdn_chunks2) and GDN_LOOP=fused.   python neuron/tests/test_qwen35_gdn_chunk2.py"""
import os, sys
from pathlib import Path
os.environ['EXACT'] = '1'
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qwen35_graphs as Q  # noqa: E402
import qwen35_moe_ref as R  # noqa: E402
import qwen35_prepare as P  # noqa: E402

torch.manual_seed(0)
rel = lambda a, b: float((a - b).norm() / b.norm().clamp(min=1e-30))
for cfg, tp in ((R.TINY397, 2), (R.TINY, 2)):
    g = Q.Geo(dict(cfg), tp)
    w = R.random_weights(dict(cfg), seed=4, scale=0.3)
    get = P.Getter(w)
    sh = {k[5:]: v for k, v in P.shard_layer(get, 0, 1, g, False).items() if k.startswith('attn.')}
    for Qn, nvalid, start, same in ((64, 64, 0, 0), (64, 37, 0, 0), (128, 128, 64, 0), (256, 200, 96, 0), (256, 256, 32, 1)):
        n = torch.randn(Qn, g.H)
        if same:   # repeated tokens: identical keys, the worst case for the triangular system
            n = n[:3].repeat(Qn // 3 + 1, 1)[:Qn].contiguous()
        pos = torch.tensor([start + i if i < nvalid else -1 for i in range(Qn)])
        st0 = {k: torch.randn(v[0]) * 0.2 for k, v in Q.state_shapes('gdn', g, 9, 4).items()}
        res = {}
        for name, C, inv, loop in (('loop', 0, 'rec', ''), ('fused', 0, 'rec', 'fused'), ('diag16', 16, 'diag', ''), ('diag32', 32, 'diag', ''), ('diag64', 64, 'diag', '')):
            Q.GDN_CHUNK, Q.GDN_INV, Q.GDN_LOOP = C, inv, loop
            y, st = Q.gdn_block(n, {**sh, 'pos': pos, 'slot': torch.tensor([2]), **{k: v.clone() for k, v in st0.items()}}, g, 1, Qn, lambda v: v)
            res[name] = (y[:nvalid], st['kstate'], st['cstate'])
        for name in ('fused', 'diag16', 'diag32', 'diag64'):
            ey, es = rel(res[name][0], res['loop'][0]), rel(res[name][1][2], res['loop'][1][2])
            print(f'E={g.E} Q={Qn} valid={nvalid} start={start} same={same} {name}: out rel {ey:.2e} state rel {es:.2e}')
            assert ey < 5e-4 and es < 5e-4 and torch.equal(res[name][2], res['loop'][2])
print('ok')
