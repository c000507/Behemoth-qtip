"""Per-NeuronCore memory budget of a qwen3_5_moe config under TP (exact byte counts from qwen35_graphs.weight_shapes /
state_shapes, i.e. what the prepare step writes and the runtime allocates).
usage: CONFIG=config.json python qwen35_budget.py TP [PAGES] [SLOTS] [fp8|bf16]"""
import json, os, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen35_graphs as Q  # noqa: E402

SZ = {torch.float32: 4, torch.bfloat16: 2, Q.F8: 1, torch.int64: 8}


def nbytes(shapes):
    t = 0
    for shape, dt in shapes.values():
        n = 1
        for d in shape:
            n *= d
        t += n * SZ[dt]
    return t


def budget(cfg, tp, pages, slots, fp8=True):
    g = Q.Geo(cfg, tp)
    w = {k: nbytes(Q.weight_shapes(k, g, fp8)) for k in ('gdn', 'gqa', 'moe')}
    ex = {k: v for k, v in Q.weight_shapes('moe', g, fp8).items() if k.startswith('e')}
    n_gdn = sum(1 for l in range(g.L) if g.kind(l) == 'gdn')
    n_gqa = g.L - n_gdn
    head = g.VP // tp * g.H * (1 if fp8 else 2) + (g.VP // tp * 4 if fp8 else 0) + g.H * 4
    st = {k: nbytes(Q.state_shapes(k, g, pages, slots)) for k in ('gdn', 'gqa')}
    out = {'tp': tp, 'gdn_layers': n_gdn, 'gqa_layers': n_gqa,
           'attn_gdn': w['gdn'], 'attn_gqa': w['gqa'], 'moe': w['moe'], 'moe_experts': nbytes(ex), 'router_f32': g.E * g.H * 4,
           'weights': n_gdn * w['gdn'] + n_gqa * w['gqa'] + g.L * w['moe'], 'head': head,
           'kv': n_gqa * st['gqa'], 'kv_per_token': n_gqa * st['gqa'] / pages / Q.PAGE, 'gdn_state': n_gdn * st['gdn'], 'gdn_state_per_slot': n_gdn * st['gdn'] / slots}
    out['total'] = out['weights'] + out['head'] + out['kv'] + out['gdn_state']
    return out


if __name__ == '__main__':
    cfg = json.loads(Path(os.environ['CONFIG']).read_text())
    tp = int(sys.argv[1]); pages = int(sys.argv[2]) if len(sys.argv) > 2 else 4097; slots = int(sys.argv[3]) if len(sys.argv) > 3 else 9
    b = budget(cfg, tp, pages, slots, (sys.argv[4] if len(sys.argv) > 4 else 'fp8') == 'fp8')
    for k, v in b.items():
        print(f'{k:20s} {v:>16,.0f}' + (f'  {v / 2**30:8.3f} GiB' if v > 2**20 else ''))
