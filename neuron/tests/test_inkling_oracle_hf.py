"""GATE 1 on the real checkpoint: inkling_ref.Ref (f32) vs transformers' Inkling2Loop (PR 51050, f32) on the box CPU.

The 26B model dequantised to f32 is ~100 GB of experts: HF's own dequant op (Inkling2LoopDequantizeExperts + Interleave)
is wrapped so that every layer's result is written once to CACHE/L<l>.{gu,dn}.npy and the parameter becomes a read-only
memory map of that file (the page cache streams it; RAM stays ~10 GB). The oracle reads the same files, and its own
dequant (inkling_ref.dequant_experts) is checked against HF's for an NVFP4 and an MXFP8 layer.

Per prompt: the oracle greedy-decodes N tokens (prefill + step decode with its carried state), then HF runs ONE
teacher-forced f32 forward over prompt + generation; logits are compared at every position (rel L2, max |diff|, top-1).

usage: INKLING_TF=<transformers src> python test_inkling_oracle_hf.py MODEL_DIR CACHE_DIR [N]
  MODEL_DIR: the checkpoint (config.json, model.safetensors.index.json + shards, tokenizer files)
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if os.environ.get('INKLING_TF'):
    sys.path.insert(0, os.environ['INKLING_TF'])
import inkling_prepare as PR  # noqa: E402
import inkling_ref as R  # noqa: E402

torch.set_num_threads(os.cpu_count() or 8)


def memmap_tensor(path):
    a = np.load(path, mmap_mode='r')
    return torch.from_numpy(a)


def hf_model(model_dir, cache):
    """HF Inkling2LoopForConditionalGeneration in f32, experts as memmaps of HF's own dequantised tensors."""
    import transformers.core_model_loading as CML
    from transformers import AutoConfig
    from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopForCausalLM, Inkling2LoopForConditionalGeneration

    orig_dq, orig_il = CML.Inkling2LoopDequantizeExperts.convert, CML.Interleave.convert
    cache.mkdir(parents=True, exist_ok=True)

    def dq(self, input_dict, source_patterns, target_patterns, model=None, full_layer_name=None, **kw):
        name = full_layer_name or target_patterns[0]
        layer = int(name.split('layers.')[1].split('.')[0])
        which = 'gu' if 'gate_up' in name else 'dn'
        path = cache / f'L{layer:02d}.{which}.npy'
        if not path.exists():
            out = orig_dq(self, input_dict, source_patterns, target_patterns, model=model, full_layer_name=full_layer_name, **kw)
            t = next(iter(out.values())).float()
            if which == 'gu':   # HF applies Interleave(dim=1) next: do it here, write the final tensor
                t = next(iter(orig_il(CML.Interleave(dim=1), {'x': t}, ['x'], ['x']).values()))
            np.save(path, t.contiguous().numpy())
            del t, out
        m = memmap_tensor(path)
        m._ink_final = True
        return {target_patterns[0]: m}

    def il(self, input_dict, source_patterns, target_patterns, **kw):
        t = next(iter(input_dict.values()))
        t = t[0] if isinstance(t, list) else t
        if getattr(t, '_ink_final', False):
            return {target_patterns[0]: t}
        return orig_il(self, input_dict, source_patterns, target_patterns, **kw)

    CML.Inkling2LoopDequantizeExperts.convert = dq
    CML.Interleave.convert = il
    cfg = AutoConfig.from_pretrained(model_dir)
    if hasattr(cfg, 'quantization_config'):
        del cfg.quantization_config
    os.environ['HF_DEACTIVATE_ASYNC_LOAD'] = '1'
    t0 = time.time()
    cls = Inkling2LoopForCausalLM if cfg.model_type == 'inkling2_loop_text' else Inkling2LoopForConditionalGeneration
    m = cls.from_pretrained(model_dir, config=cfg, dtype=torch.float32).eval()
    print(f'HF model loaded in {time.time() - t0:.0f}s', flush=True)
    return m


def oracle(model_dir, cache):
    cfg = json.loads((Path(model_dir) / 'config.json').read_text())
    st = PR.Safetensors(model_dir)
    wc = {}

    def W(n):
        if n not in wc:
            wc[n] = st[n].float()
        return wc[n]

    def X(l):
        return memmap_tensor(cache / f'L{l:02d}.gu.npy'), memmap_tensor(cache / f'L{l:02d}.dn.npy')
    return R.Ref(cfg, W, X), st


def prompts(model_dir):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    chat = tok.apply_chat_template([{'role': 'user', 'content': 'Explain in two short paragraphs why the sky is blue.'}],
                                   tokenize=False, add_generation_prompt=True)
    return tok, [('chat-sky', tok.encode(chat, add_special_tokens=False)),
                 ('raw-paris', tok.encode('The capital of France is', add_special_tokens=False))]


def main():
    model_dir, cache = sys.argv[1], Path(sys.argv[2])
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    hf = hf_model(model_dir, cache)
    ref, st = oracle(model_dir, cache)
    for l in (0, 20):   # my dequant == HF's (NVFP4 layer 0, MXFP8 layer 20)
        gu, dn = R.dequant_experts(lambda n: st[n], l)
        a, b = ref.X(l)
        print(f'layer {l} dequant vs HF: gu max|d| {(gu - a).abs().max().item():.3e}  dn {(dn - b).abs().max().item():.3e}', flush=True)
    tok, ps = prompts(model_dir)
    worst = 0.0
    agree = total = 0
    for name, ids in ps:
        t0 = time.time()
        state = {}
        lg = ref.forward(torch.tensor(ids), state)
        gen, olog = [], [lg[-1]]
        for _ in range(N):
            nxt = int(olog[-1].argmax())
            gen.append(nxt)
            olog.append(ref.forward(torch.tensor([nxt]), state)[-1])
        t1 = time.time()
        seq = torch.tensor([ids + gen])
        with torch.no_grad():
            hl = hf(input_ids=seq).logits[0].float()
        t2 = time.time()
        # oracle logits at positions len(ids)-1 .. end (prefill last + every decode step) vs HF at the same positions
        ol = torch.stack(olog[:-1])
        hp = hl[len(ids) - 1:len(ids) - 1 + len(ol)]
        rel = ((ol - hp).norm(dim=-1) / hp.norm(dim=-1))
        top = (ol.argmax(-1) == hp.argmax(-1))
        full = ((lg - hl[:len(ids)]).norm() / hl[:len(ids)].norm()).item()
        worst = max(worst, rel.max().item(), full)
        agree += int(top.sum())
        total += len(top)
        print(f'{name}: {len(ids)} prompt + {N} generated; prefill logits rel {full:.2e}; per step rel max {rel.max():.2e} '
              f'mean {rel.mean():.2e}; max|d| {(ol - hp).abs().max():.2e}; top-1 {int(top.sum())}/{len(top)}; '
              f'oracle {t1 - t0:.0f}s HF {t2 - t1:.0f}s', flush=True)
        print('  text:', repr(tok.decode(gen)), flush=True)
    print(f'GATE1 worst rel {worst:.2e}, top-1 {agree}/{total}')


if __name__ == '__main__':
    main()
