"""segment() (several whole layers in one graph) == the same layers through sublayer() one by one, with states, EXACT f32.
  python neuron/tests/test_qwen35_segment.py"""
import os, sys
from pathlib import Path
os.environ['EXACT'] = '1'
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qwen35_graphs as Q  # noqa: E402
import qwen35_moe_ref as R  # noqa: E402
import qwen35_prepare as P  # noqa: E402

cfg = dict(R.TINY397)
g = Q.Geo(cfg, 2)
w = R.random_weights(cfg, seed=2)
get = P.Getter(w)
ident = lambda v: v
for route, slots in (('topk', 'topk'), ('rank', 'onehot')):
    Q.ROUTE, Q.MOE_SLOTS = route, slots
    for Rr in (1, 3):
        torch.manual_seed(0)
        x = torch.randn(Rr, g.H)
        pos = torch.tensor([5 + i for i in range(Rr)])
        table = torch.stack([torch.arange(2) + 2 * i for i in range(Rr)])
        step = {'x': x, 'pos': pos, 'slot': torch.arange(Rr), 'table': table, 'wslot': torch.stack([table[i, 0] * 16 + pos[i] for i in range(Rr)]),
                'cos_sin': Q.cos_sin_rows(pos, g.rot, g.theta)}
        seg = [g.kind(l) for l in range(g.L)]
        sh = [P.shard_layer(get, l, 0, g, False) for l in range(g.L)]
        st0 = [{k: torch.randn(v[0]) * 0.1 for k, v in Q.state_shapes(seg[l], g, 9, 4).items()} for l in range(g.L)]
        # one by one
        xr, st_ref = x, {}
        for l in range(g.L):
            a = {k[5:]: v for k, v in sh[l].items() if k.startswith('attn.')}
            xr, st = Q.sublayer(seg[l], {**step, **a, **{k: v.clone() for k, v in st0[l].items()}, 'x': xr}, g, Rr, 1, 2, ident, 'gather')
            st_ref.update({f'l{l}.{k}': v for k, v in st.items()})
            xr, _ = Q.sublayer('moe', {**step, **{k[4:]: v for k, v in sh[l].items() if k.startswith('mlp.')}, 'x': xr}, g, Rr, 1, 2, ident, 'gather')
        t = dict(step)
        for l in range(g.L):
            t.update({f'l{l}.a.' + k[5:]: v for k, v in sh[l].items() if k.startswith('attn.')})
            t.update({f'l{l}.m.' + k[4:]: v for k, v in sh[l].items() if k.startswith('mlp.')})
            t.update({f'l{l}.{k}': v.clone() for k, v in st0[l].items()})
        xs, st_seg = Q.segment(seg, t, g, Rr, 1, 2, ident, 'gather')
        assert torch.equal(xs, xr), float((xs - xr).abs().max())
        assert st_seg.keys() == st_ref.keys() and all(torch.equal(st_seg[k], st_ref[k]) for k in st_ref)
        print(f'segment {seg} R={Rr} route={route}: identical to the sub-blocks ({len(st_ref)} states)')
