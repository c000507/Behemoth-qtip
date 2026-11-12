"""Teacher-forced oracle check of a Qwen3.5 MoE generation: the f32 CPU reference (qwen35_moe_ref.Ref, HF-validated) runs
the given token sequence (prompt + the tokens the device generated) in ONE pass with the checkpoint's weights streamed
layer by layer from model.safetensors (bf16 -> f32 on the fly: ~2 GB of RAM per layer, the 38 GB file is read once), and
reports at every generated position whether its argmax equals the device's token, plus the margin (top-1 - top-2 logit)
where they differ (near ties are the expected fp8 divergences).

usage: python neuron/tests/test_qwen35_oracle.py MODEL_DIR run.json [out.json]
  run.json: {"prompt": [ids], "generated": [ids]}  (as written by the end-to-end driver)
"""
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import glm_repack as RP  # noqa: E402
import qwen35_moe_ref as R  # noqa: E402


class LazyWeights:
    """HF name -> f32 tensor, read from the safetensors file on access; keeps only the tensors of one layer (plus the
    embedding / head / final norm) in memory."""

    def __init__(self, model_dir):
        self.ck = RP.Checkpoint(model_dir)
        self.cache = {}
        self.layer = None

    def __iter__(self):
        return iter(self.ck.map)

    def __contains__(self, k):
        return k in self.ck.map

    def __getitem__(self, k):
        if '.layers.' in k:
            l = int(k.split('.layers.')[1].split('.')[0])
            if l != self.layer:
                self.cache = {n: v for n, v in self.cache.items() if '.layers.' not in n}
                self.layer = l
        if k not in self.cache:
            self.cache[k] = self.ck.tensor(k)
        return self.cache[k]

    def items(self):
        raise RuntimeError('lazy')


def main():
    model, run = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
    cfg = json.loads((model / 'config.json').read_text())
    w = LazyWeights(model)
    ref = R.Ref.__new__(R.Ref)
    ref.g = R.Cfg(cfg)
    ref.w = w
    ref.p = w.ck.prefix
    prompt, gen = run['prompt'], run['generated']
    tokens = torch.tensor(prompt + gen[:-1], dtype=torch.int64)       # the last generated token has no teacher-forced target
    torch.set_num_threads(8)
    t0 = time.time()
    st = ref.init_state()
    logits = ref.forward(tokens, st)
    print(f'oracle forward over {len(tokens)} tokens in {time.time() - t0:.0f}s', flush=True)
    n0 = len(prompt) - 1                                              # logits[n0 + i] predicts gen[i]
    agree, rows = 0, []
    for i, tok in enumerate(gen):
        lg = logits[n0 + i]
        top = torch.topk(lg, 5)
        ok = int(top.indices[0]) == tok
        agree += ok
        rows.append({'i': i, 'device': tok, 'oracle_top1': int(top.indices[0]), 'agree': ok, 'margin': float(top.values[0] - top.values[1]),
                     'device_rank': int((lg > lg[tok]).sum()), 'device_gap': float(top.values[0] - lg[tok])})
        if not ok:
            print(f'  pos {i}: device {tok} oracle top-1 {int(top.indices[0])} (margin {rows[-1]["margin"]:.3f}, device token rank {rows[-1]["device_rank"]}, gap {rows[-1]["device_gap"]:.3f})')
    print(f'teacher-forced top-1 agreement: {agree}/{len(gen)}')
    out = {'agree': agree, 'n': len(gen), 'rows': rows}
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(json.dumps(out))


if __name__ == '__main__':
    main()
