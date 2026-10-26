#!/usr/bin/env python3
"""Calibration tokens for qtip_quant from the GM-CoT chat dataset (+ a general-text share).

Output: raw int32 [nseq][seqlen] (qtip_quant --tokens / --seq-len) + .json metadata + held-out conversation ids.
GM-CoT conversations are rendered with the model's chat template (keeping <think> reasoning) and cut into
contiguous seqlen windows; the general share is taken from an existing 2048-token calibration file
(consecutive sequences concatenated to seqlen).
"""
import argparse, json, random, re
import numpy as np
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/home/charles/train_data/train_cot_final.jsonl")
ap.add_argument("--tokenizer", default="/mnt/nvme/models/Behemoth-R1-123B-v2-aqlm-output")
ap.add_argument("--general", default="/mnt/nvme/qtip_quant/calib_2048x2048.i32")
ap.add_argument("--general-seqlen", type=int, default=2048)
ap.add_argument("--seqlen", type=int, default=8192)
ap.add_argument("--ntok", type=int, default=4194304)
ap.add_argument("--gm-frac", type=float, default=0.75)
ap.add_argument("--holdout-frac", type=float, default=0.1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

rng = random.Random(a.seed)
tok = AutoTokenizer.from_pretrained(a.tokenizer)
THINK = re.compile(r"<think>.*?</think>\s*", re.S)


def render(messages):
    # Behemoth-R1 emits its reasoning inline as <think>...</think> before the reply; the chat template ignores
    # reasoning_content, so inline it (dropping legacy inline <think> blocks inherited from the source data).
    out = []
    for m in messages:
        if m["role"] == "assistant":
            body = THINK.sub("", m["content"]).lstrip()
            r = (m.get("reasoning_content") or "").strip()
            m = {"role": "assistant", "content": f"<think>\n{r}\n</think>\n\n{body}" if r else body}
        else:
            m = {"role": m["role"], "content": m["content"]}
        out.append(m)
    return out


convs = [render(json.loads(l)["messages"]) for l in open(a.data)]
ids = list(range(len(convs)))
rng.shuffle(ids)
nhold = int(len(ids) * a.holdout_frac)
hold, train = sorted(ids[:nhold]), ids[nhold:]

nseq = a.ntok // a.seqlen
n_gm = int(round(nseq * a.gm_frac))
n_gen = nseq - n_gm

windows = []
for i in train:
    t = tok(tok.apply_chat_template(convs[i], tokenize=False), add_special_tokens=False).input_ids
    for s in range(0, len(t) - a.seqlen + 1, a.seqlen):
        windows.append(t[s:s + a.seqlen])
rng.shuffle(windows)
assert len(windows) >= n_gm, f"only {len(windows)} GM-CoT windows, need {n_gm}"
gm = np.array(windows[:n_gm], dtype=np.int32)

g = np.fromfile(a.general, dtype=np.int32).reshape(-1, a.general_seqlen)
per = a.seqlen // a.general_seqlen
assert g.shape[0] >= n_gen * per
gen = g[: n_gen * per].reshape(n_gen, a.seqlen)

allseq = np.concatenate([gm, gen])
order = list(range(nseq))
rng.shuffle(order)
allseq = allseq[order]
allseq.tofile(a.out)
meta = dict(nseq=nseq, seqlen=a.seqlen, ntok=int(allseq.size), gm_seqs=n_gm, general_seqs=n_gen,
            gm_windows_available=len(windows), data=a.data, general=a.general, seed=a.seed,
            chat_template=True, think_inlined=True, holdout_conversations=hold)
json.dump(meta, open(a.out + ".json", "w"))
print(json.dumps({k: v for k, v in meta.items() if k != "holdout_conversations"}), "holdout", len(hold))
