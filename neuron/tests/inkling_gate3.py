"""GATE 3: the device's greedy tokens (neuron_engine_cli output) against the oracle, teacher-forced.

For every request: the oracle runs ONE forward over prompt + device generation and its argmax at each generating position is
compared with the device's token there. Variants: x8d8 (the fp8 weights the graphs use, as f32 values: experts per row,
dense projections per row) and ref (exact dequantised weights). Misses are listed with the oracle's top-2 margin.

usage: python inkling_gate3.py MODEL_DIR CACHE_DIR requests.json cli_out.json [variants=x8d8,ref]
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inkling_kl as K  # noqa: E402


def main():
    model_dir, cache, req_path, out_path = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]
    variants = (sys.argv[5] if len(sys.argv) > 5 else 'x8d8,ref').split(',')
    reqs = json.loads(Path(req_path).read_text())
    outs = json.loads(Path(out_path).read_text())['requests']
    for v in variants:
        m = K.make(model_dir, cache, v)
        agree = total = 0
        for i, (rq, o) in enumerate(zip(reqs, outs)):
            p, gen = rq['tokens'], o['tokens']
            if not gen:
                continue
            lg = m.forward(torch.tensor(p + gen[:-1]), {})[len(p) - 1:]
            top2 = torch.topk(lg, 2, dim=-1)
            pred = top2.indices[:, 0]
            dev = torch.tensor(gen)
            ok = pred == dev
            agree += int(ok.sum())
            total += len(gen)
            miss = [(j, int(dev[j]), int(pred[j]), round(float(top2.values[j, 0] - top2.values[j, 1]), 3),
                     round(float(top2.values[j, 0] - lg[j, dev[j]]), 3)) for j in range(len(gen)) if not ok[j]]
            print(f'{v} request {i}: {int(ok.sum())}/{len(gen)} top-1; misses (pos, device, oracle, oracle margin, device token deficit): {miss}', flush=True)
        print(f'{v}: GATE3 top-1 {agree}/{total} = {agree / max(total, 1):.3f}', flush=True)


if __name__ == '__main__':
    main()
