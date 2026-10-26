#!/usr/bin/env python3
"""Prepare packed QAT data. No model weights or GPU required."""
import argparse
import json
from pathlib import Path
import random
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qtip_training.data import pack_documents, sha256_file, split_ids, tokenize_conversation, write_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="/home/charles/train_data/train_cot_final.jsonl")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--split-metadata", help="existing calibration .json; reuse its held-out ids")
    parser.add_argument("--split", choices=("train", "heldout"), default="train")
    parser.add_argument("--general", help="existing general calibration raw int32 file")
    parser.add_argument("--general-seqlen", type=int, default=2048)
    parser.add_argument("--seqlen", type=int, default=24576)
    parser.add_argument("--gm-frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, help="optional total packed token budget")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not 0 < args.gm_frac <= 1 or args.seqlen < 2 or args.general_seqlen < 2:
        parser.error("invalid mixture or sequence length")
    if args.max_tokens is not None and args.max_tokens < args.seqlen:
        parser.error("max-tokens must cover at least one sequence")
    if args.split == "heldout" and args.gm_frac != 1:
        parser.error("heldout preparation requires --gm-frac 1; evaluate general text separately")
    if args.gm_frac < 1 and not args.general:
        parser.error("--general is required for a general-text share")
    if Path(args.out).exists():
        parser.error("output already exists")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if not tokenizer.is_fast:
        parser.error("a fast tokenizer with offset mappings is required")
    with open(args.data) as handle:
        conversations = [json.loads(line)["messages"] for line in handle if line.strip()]
    recorded = None
    if args.split_metadata:
        recorded = json.loads(Path(args.split_metadata).read_text())["holdout_conversations"]
    train, held = split_ids(len(conversations), args.seed, recorded=recorded)
    selected = train if args.split == "train" else held
    documents = [(i, *tokenize_conversation(tokenizer, conversations[i])) for i in selected]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        parser.error("tokenizer needs a pad or eos token")
    rows = pack_documents(documents, args.seqlen, pad)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.max_tokens:
        # Reserve the requested share for general text without repeating GM docs.
        max_rows = args.max_tokens // args.seqlen
        rows = rows[:max(1, int(max_rows * args.gm_frac))]
    n_gm = len(rows)
    n_general = round(n_gm * (1 - args.gm_frac) / args.gm_frac)
    if args.max_tokens:
        n_general = min(n_general, args.max_tokens // args.seqlen - n_gm)
    if n_general:
        general = np.fromfile(args.general, dtype=np.int32).reshape(-1, args.general_seqlen)
        order = list(range(len(general)))
        rng.shuffle(order)
        required = n_general * args.seqlen
        if general.size < required:
            parser.error(f"general data has {general.size} tokens; need {required} without repetition")
        docs = [(-2 - i, general[i], np.ones(args.general_seqlen, dtype=np.uint8)) for i in order]
        # Nonnegative segments distinguish valid docs from padding (-1).
        docs = [(len(conversations) + j, ids, mask) for j, (_, ids, mask) in enumerate(docs)]
        rows.extend(pack_documents(docs, args.seqlen, pad)[:n_general])
    rng.shuffle(rows)
    meta = dict(split=args.split, seed=args.seed, data_sha256=sha256_file(args.data),
                tokenizer=args.tokenizer, holdout_conversations=held,
                training_conversations=train if args.split == "train" else [],
                evaluated_conversations=held if args.split == "heldout" else [],
                gm_sequences=n_gm, general_sequences=n_general,
                gm_frac=args.gm_frac, think_inlined=True, document_masking=True)
    if args.general:
        meta["general_sha256"] = sha256_file(args.general)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=out.name + ".", dir=out.parent))
    try:
        write_dataset(temporary / "data", rows, meta)
        (temporary / "data").rename(out)
    finally:
        shutil.rmtree(temporary)
    print(json.dumps(dict(nseq=len(rows), seqlen=args.seqlen, heldout=len(held), out=str(out))))


if __name__ == "__main__":
    main()
