"""Compare runner logits against model_oracle reference logits.

The oracle directory holds logits_<i>.bin (F32 [vocab]) from
neuron/tests/model_oracle.cpp; the runner directory holds logits_<i>.npy from
neuron/run_tokens.py --logits-out. Reports per-token relative L2, max absolute
error and greedy argmax agreement.
"""
import argparse
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('oracle'); p.add_argument('runner')
a = p.parse_args()
oracle = {int(f.stem.split('_')[1]): np.fromfile(f, dtype=np.float32)
          for f in sorted(Path(a.oracle).glob('logits_*.bin'))}
runner = {int(f.stem.split('_')[1]): np.load(f) for f in sorted(Path(a.runner).glob('logits_*.npy'))}
common = sorted(set(oracle) & set(runner))
if not common:
    raise SystemExit('No matching position indices found')
failed = False
for i in common:
    expected, actual = oracle[i].astype(np.float64), runner[i].astype(np.float64)
    if expected.shape != actual.shape:
        raise SystemExit(f'Shape mismatch at position {i}: {expected.shape} vs {actual.shape}')
    if not np.isfinite(actual).all():
        raise SystemExit(f'Nonfinite runner logits at position {i}')
    rel = np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30)
    top = np.argsort(expected)[-5:][::-1]
    got = int(np.argmax(actual)); want = int(np.argmax(expected))
    rank = int(np.where(np.argsort(actual)[::-1] == want)[0][0])
    margin = float(expected[top[0]] - expected[top[1]]) if len(top) > 1 else float('nan')
    print(f'position {i}: relative_l2={rel:.5g} max_abs={np.max(np.abs(actual-expected)):.5g} '
          f'argmax runner={got} oracle={want} oracle_rank_in_runner={rank} '
          f'oracle_top5={top.tolist()} logit_margin={margin:.4f}', flush=True)
    if got != want:
        failed = True
        print(f'  argmax mismatch at position {i}', flush=True)
print('argmax agreement:', 'FAILED' if failed else 'all match')
