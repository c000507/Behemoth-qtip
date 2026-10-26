"""Calibration tokens for qtip_quant (same data choice as quantbench/make_calib.py):
packed sequences of --seqlen tokens, ~75% OpenWebText documents, ~25% SmolTalk chats rendered with the
model's chat template, BOS/EOS around every document.  Output: raw little-endian int32 [nseq][seqlen] + .json.
Default 2048 x 2048 = 4.19M tokens.
"""
import argparse, glob, json, os, random
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/mnt/evo/Behemoth-R1-123B-v2-FP8-Dynamic")
ap.add_argument("--nseq", type=int, default=2048)
ap.add_argument("--seqlen", type=int, default=2048)
ap.add_argument("--out", default="/mnt/nvme/qtip_quant/calib_2048x2048.i32")
ap.add_argument("--chat_frac", type=float, default=0.25)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

hub = os.path.expanduser("~/.cache/huggingface/hub")
owt = sorted(glob.glob(f"{hub}/datasets--Skylion007--openwebtext/snapshots/*/plain_text/train-000*-of-00080.parquet"))
smol = sorted(glob.glob(f"{hub}/datasets--HuggingFaceTB--smol-smoltalk/snapshots/*/data/train-0000*-of-00004.parquet"))
assert owt and smol, "datasets not in the HF cache"
tok = AutoTokenizer.from_pretrained(args.model)
rng = random.Random(args.seed)
n_chat = int(round(args.nseq * args.chat_frac))
need_web = (args.nseq - n_chat) * args.seqlen
need_chat = n_chat * args.seqlen


def texts(files, col, limit_tokens):
    out = []
    for f in files:
        t = pq.read_table(f, columns=[col]).column(col).to_pylist()
        rng.shuffle(t)
        out += t
        if len(out) * 600 > limit_tokens * 1.5:  # rough upper bound on what we need
            break
    return out


web = texts(owt, "text", need_web)
chats = texts(smol, "messages", need_chat)


def stream(items, render):
    for it in items:
        try:
            s = render(it)
        except Exception:
            continue
        ids = tok(s, add_special_tokens=False)["input_ids"]
        yield [tok.bos_token_id] + ids + [tok.eos_token_id]


seqs = []
for src, n in ((stream(web, lambda t: t), args.nseq - n_chat),
               (stream(chats, lambda m: tok.apply_chat_template(m, tokenize=False)), n_chat)):
    buf = []
    while n > 0:
        while len(buf) < args.seqlen:
            buf += next(src)
        seqs.append(buf[: args.seqlen])
        buf = buf[args.seqlen:]
        n -= 1
rng.shuffle(seqs)
a = np.asarray(seqs, dtype=np.int32)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
a.tofile(args.out)
json.dump({"nseq": a.shape[0], "seqlen": a.shape[1], "chat_frac": args.chat_frac, "seed": args.seed,
           "sources": [os.path.basename(f) for f in owt[:1] + smol[:1]]}, open(args.out + ".json", "w"))
print("saved", a.shape, args.out)
