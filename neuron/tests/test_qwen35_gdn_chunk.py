"""gdn_block with GDN_CHUNK (matmul-form chunked recurrence) vs the token loop, EXACT f32: outputs and states, with
padding and a carried-in state; several chunk sizes.   python neuron/tests/test_qwen35_gdn_chunk.py"""
import os, sys
from pathlib import Path
os.environ['EXACT'] = '1'
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qwen35_graphs as Q  # noqa: E402
import qwen35_moe_ref as R  # noqa: E402
import qwen35_prepare as P  # noqa: E402

torch.manual_seed(0)
m = torch.tril(torch.randn(3, 16, 16), diagonal=-1)
e = (Q.unit_lower_inverse(m) @ (torch.eye(16) + m) - torch.eye(16)).abs().max().item()
print(f'unit_lower_inverse: |inv (I + m) - I| max {e:.2e}'); assert e < 1e-4
for cfg, tp in ((R.TINY397, 2), (R.TINY, 2)):
    g = Q.Geo(dict(cfg), tp)
    w = R.random_weights(dict(cfg), seed=4, scale=0.3)       # larger weights: non-trivial gates / decays
    get = P.Getter(w)
    sh = {k[5:]: v for k, v in P.shard_layer(get, 0, 1, g, False).items() if k.startswith('attn.')}
    for Qn, nvalid, start in ((64, 64, 0), (64, 37, 0), (128, 128, 64), (32, 20, 96)):
        n = torch.randn(Qn, g.H)
        pos = torch.tensor([start + i if i < nvalid else -1 for i in range(Qn)])
        st0 = {k: torch.randn(v[0]) * 0.2 for k, v in Q.state_shapes('gdn', g, 9, 4).items()}
        res = {}
        for C in (0, 8, 16, 32):
            Q.GDN_CHUNK = C
            y, st = Q.gdn_block(n, {**sh, 'pos': pos, 'slot': torch.tensor([2]), **{k: v.clone() for k, v in st0.items()}}, g, 1, Qn, lambda v: v)
            res[C] = (y[:nvalid], st['kstate'], st['cstate'])
        for C in (8, 16, 32):
            rel = lambda a, b: float((a - b).norm() / b.norm().clamp(min=1e-30))
            ey, es = rel(res[C][0], res[0][0]), rel(res[C][1][2], res[0][1][2])
            print(f'E={g.E} Q={Qn} valid={nvalid} start={start} chunk={C}: out rel {ey:.2e} state rel {es:.2e}')
            assert ey < 2e-4 and es < 2e-4 and torch.equal(res[C][2], res[0][2])
print('ok')
