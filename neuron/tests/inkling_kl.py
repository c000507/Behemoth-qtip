"""GATE 2: what the fp8 re-encode of the Inkling weights costs, measured with the f32 oracle on the real checkpoint.

Reference: inkling_ref.Ref over the exact dequantised experts (CACHE/L<l>.{gu,dn}.npy, as written by
test_inkling_oracle_hf.py) and the bf16 dense weights. Variants (the weights the Neuron graphs see, as f32 values):
  x8     experts E4M3 per output row (gate|up) / per input channel (down) - the prepared format
  x8d8   x8 + dense projections (q, gate, k, v, o, rel, head) E4M3 per output row
  xb128  experts E4M3 with one scale per 128 input columns (block form, report only)
Texts: a held-out text set (>= 2k tokens, one pass) and chat-template prompts. Reported per set: mean / p99 / max KL(ref || variant)
over positions (nats), top-1 agreement.

usage: python inkling_kl.py MODEL_DIR CACHE_DIR TEXT_FILE [variants=x8,x8d8,xb128]
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glm_repack as RP  # noqa: E402
import inkling_prepare as PR  # noqa: E402
import inkling_ref as R  # noqa: E402

torch.set_num_threads(os.cpu_count() or 8)
DENSE8 = ('wq_du', 'gate_attn', 'wk_dv', 'wv_dv', 'wo_ud', 'wr_du', 'unembed')


def fp8_rows(w):
    c, s = RP.to_fp8_rows(w)
    return c.float() * s.unsqueeze(-1)


def fp8_block(w, b):
    *lead, n = w.shape
    v = w.reshape(*lead, n // b, b)
    s = v.abs().amax(-1, keepdim=True).clamp(min=1e-30) / 224.0
    return ((v / s).to(torch.float8_e4m3fn).float() * s).reshape(w.shape)


def make(model_dir, cache, variant):
    cfg = json.loads((Path(model_dir) / 'config.json').read_text())
    st = PR.Safetensors(model_dir)
    wc = {}

    def W(n):
        if n not in wc:
            t = st[n].float()
            if variant == 'x8d8' and any(d in n for d in DENSE8):
                t = fp8_rows(t)
            wc[n] = t
        return wc[n]

    def X(l):
        gu = torch.from_numpy(np.load(cache / f'L{l:02d}.gu.npy', mmap_mode='r'))
        dn = torch.from_numpy(np.load(cache / f'L{l:02d}.dn.npy', mmap_mode='r'))
        if variant in ('x8', 'x8d8'):
            return fp8_rows(gu), fp8_rows(dn.transpose(1, 2)).transpose(1, 2)
        if variant == 'xb128':
            return fp8_block(gu, 128), fp8_block(dn.transpose(1, 2), 128).transpose(1, 2)
        return gu, dn
    return R.Ref(cfg, W, X)


def kl(ref_lp, lg):
    lq = torch.log_softmax(lg.float(), dim=-1)
    return (ref_lp.exp() * (ref_lp - lq)).sum(-1)


def main():
    model_dir, cache, text_file = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    variants = (sys.argv[4] if len(sys.argv) > 4 else 'x8,x8d8,xb128').split(',')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    text = Path(text_file).read_text()
    ids = tok.encode(text, add_special_tokens=False)[:2304]
    chats = ['Explain in two short paragraphs why the sky is blue.', 'Write a Python function that checks whether a number is prime.',
             'What were the main causes of the First World War?', 'Translate to French: The weather is lovely today.',
             'Give me three ideas for a short science-fiction story.']
    sets = {'text': [ids]}
    sets['chat'] = [tok.encode(tok.apply_chat_template([{'role': 'user', 'content': c}], tokenize=False, add_generation_prompt=True),
                               add_special_tokens=False) for c in chats]
    print('tokens: text', len(ids), 'chat', [len(c) for c in sets['chat']], flush=True)
    ref = make(model_dir, cache, 'ref')
    ref_lp, ref_top = {}, {}
    t0 = time.time()
    for name, seqs in sets.items():
        ref_lp[name], ref_top[name] = [], []
        for s in seqs:
            lg = ref.forward(torch.tensor(s), {})
            ref_lp[name].append(torch.log_softmax(lg, dim=-1).to(torch.bfloat16) if name == 'text' else torch.log_softmax(lg, dim=-1))
            ref_top[name].append(lg.argmax(-1))
            del lg
    print(f'reference done in {time.time() - t0:.0f}s', flush=True)
    for v in variants:
        m = make(model_dir, cache, v)
        t0 = time.time()
        for name, seqs in sets.items():
            ks, agree, n = [], 0, 0
            for s, lp, top in zip(seqs, ref_lp[name], ref_top[name]):
                lg = m.forward(torch.tensor(s), {})
                k = kl(lp.float(), lg)
                ks.append(k)
                agree += int((lg.argmax(-1) == top).sum())
                n += len(top)
                del lg
            k = torch.cat(ks)
            print(f'{v} {name}: {n} positions  KL mean {k.mean():.4f}  p99 {k.quantile(0.99):.4f}  max {k.max():.4f}  '
                  f'top-1 {agree}/{n} = {agree / n:.3f}   ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
