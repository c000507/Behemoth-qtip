"""Inputs for layer_bench LB_INPUTS=DIR (realistic data for the gathered-MoE graphs: random-normal router -> near-uniform routing,
the worst case for the distinct-expert kernel). Large tensors are written once into DIR/pool and symlinked per input index.

  python mk_lb_inputs.py GRAPH_DIR OUT_DIR      (reads GRAPH_DIR/meta.json)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glm_repack as RP  # noqa: E402
import qwen35_graphs as Q  # noqa: E402

F8 = Q.F8


def main():
    gdir, out = Path(sys.argv[1]), Path(sys.argv[2])
    meta = json.loads((gdir / 'meta.json').read_text())
    geo = meta['geometry']
    cfg = json.loads(Path(os.environ['CONFIG']).read_text())
    g = Q.Geo(cfg, geo['tp'])
    os.environ['EGU_T'] = '1' if meta.get('egu_t') else '0'
    Q.EGU_T = bool(meta.get('egu_t'))
    Q.ROUTER = meta.get('router', 'full')
    R, Qn, P = meta['R'], meta['Q'], meta['P']
    kinds = meta['kind'][len('qwen35_seq_'):].split('_') if meta['kind'].startswith('qwen35_seq_') else None
    shapes = {}
    shapes.update(Q.step_shapes('moe', R, Qn, P, g))
    if meta.get('seg'):     # 20-layer segments: attention + MoE weights per layer (states keep the bench's patterned fill)
        for i, a_kind in enumerate(meta['seg']):
            shapes.update(Q.step_shapes(a_kind, R, Qn, P, g))
            shapes.update({f'l{i}.a.{n}': v for n, v in Q.weight_shapes(a_kind, g, meta['fp8']).items()})
            shapes.update({f'l{i}.m.{n}': v for n, v in Q.weight_shapes('moe', g, meta['fp8']).items()})
    elif kinds:
        for i, k in enumerate(kinds):
            shapes.update({f'l{i}.m.{n}': v for n, v in Q.weight_shapes(k, g, meta['fp8']).items()})
    else:
        shapes.update(Q.weight_shapes('moe', g, meta['fp8']))
    out.mkdir(parents=True, exist_ok=True)
    pool = out / 'pool'
    pool.mkdir(exist_ok=True)
    rng = np.random.default_rng(11)

    def pooled(key, make):
        p = pool / key
        if not p.exists():
            p.write_bytes(make())
        return p

    for i, name in enumerate(meta['inputs']):
        shape, dt = shapes[name]
        base = name.split('.')[-1]
        dst = out / f'in{i}'
        for f in out.glob(f'in{i}.r*'):
            f.unlink()
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if dt == torch.int64 and base != 'rcols':     # positions / slots / page tables: the bench's zero fill (in bounds)
            continue
        if base in ('kstate', 'cstate', 'kv'):
            continue
        if base == 'rcols':
            cols = shape[0]
            for r in range(g.TP):
                (out / f'in{i}.r{r}').write_bytes((torch.arange(cols, dtype=torch.int64) + r * cols).numpy().tobytes())
            continue
        if base == 'rws':
            for r in range(g.TP):
                (out / f'in{i}.r{r}').write_bytes(torch.from_numpy(rng.standard_normal(shape, dtype=np.float32) * 0.02).numpy().tobytes())
            continue
        if dt == F8:
            key = f'f8_{"x".join(map(str, shape))}'
            src = pooled(key, lambda: RP.tensor_bytes((torch.from_numpy(rng.standard_normal(shape, dtype=np.float32)) * 32).clamp(-224, 224).to(F8)))
        elif base == 'x':
            src = pooled(f'x_{"x".join(map(str, shape))}', lambda: (torch.from_numpy(rng.standard_normal(shape, dtype=np.float32)) * 0.5).numpy().tobytes())
        elif base in ('norm', 'onorm', 'qn', 'kn'):
            src = pooled(f'ones_{"x".join(map(str, shape))}', lambda: np.ones(shape, np.float32).tobytes())
        elif base.endswith('_s') or base.endswith('_c'):
            src = pooled(f'sc_{"x".join(map(str, shape))}', lambda: np.full(shape, 2e-3, np.float32).tobytes())
        elif base == 'cos_sin':
            src = pooled(f'cs_{"x".join(map(str, shape))}', lambda: np.concatenate([np.ones((shape[0], shape[1] // 2), np.float32), np.zeros((shape[0], shape[1] // 2), np.float32)], 1).tobytes())
        elif base in ('alog', 'dt'):
            src = pooled(f'small_{"x".join(map(str, shape))}', lambda: np.full(shape, -1.0, np.float32).tobytes())
        elif dt == torch.float32:
            src = pooled(f'n_{"x".join(map(str, shape))}', lambda: (rng.standard_normal(shape, dtype=np.float32) * 0.02).tobytes())
        else:
            src = pooled(f'z_{"x".join(map(str, shape))}_{dt}', lambda: torch.zeros(shape, dtype=dt).numpy().tobytes())
        dst.symlink_to(src.resolve())
    print('inputs written', out, flush=True)


if __name__ == '__main__':
    main()
