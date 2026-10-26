"""Conversation-disjoint packing with assistant-target and document masks."""
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np
import torch

THINK = re.compile(r"<think>.*?</think>\s*", re.S)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def render(messages):
    result = []
    for message in messages:
        body = message.get("content") or ""
        if message["role"] == "assistant":
            body = THINK.sub("", body).lstrip()
            reasoning = (message.get("reasoning_content") or "").strip()
            if reasoning:
                body = f"<think>\n{reasoning}\n</think>\n\n{body}"
        result.append({"role": message["role"], "content": body})
    return result


def split_ids(count, seed=0, holdout_frac=0.1, recorded=None):
    ids = list(range(count))
    random.Random(seed).shuffle(ids)
    held = sorted(ids[:int(count * holdout_frac)]) if recorded is None else list(recorded)
    if len(set(held)) != len(held) or any(not 0 <= i < count for i in held):
        raise ValueError("invalid held-out conversation ids")
    return [i for i in ids if i not in set(held)], sorted(held)


def tokenize_conversation(tokenizer, messages):
    """Mask the exact full-template tokenization, never tokenize turns separately.

    Fast tokenizer offsets avoid BPE boundary errors from concatenating separately
    tokenized prompts/replies. Include assistant end-of-turn tokens. Fail closed
    for templates that rewrite earlier turns instead of guessing target spans.
    """
    messages = render(messages)
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    spans, previous = [], ""
    for i, message in enumerate(messages):
        turn = tokenizer.apply_chat_template(messages[:i + 1], tokenize=False, add_generation_prompt=False)
        if not full.startswith(turn) or not turn.startswith(previous):
            raise ValueError("chat template rewrites prefixes; provide a prefix-stable tokenizer template")
        if message["role"] == "assistant" and message["content"]:
            start = turn.find(message["content"], len(previous))
            if start < 0:
                raise ValueError("chat template changes assistant content; cannot identify loss span")
            spans.append((start, len(turn)))
        previous = turn
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    # Boundary-straddling tokens contain part of the answer and are targets too.
    mask = [int(end > start and any(end > lo and start < hi for lo, hi in spans)) for start, end in offsets]
    return np.asarray(encoded["input_ids"], dtype=np.int32), np.asarray(mask, dtype=np.uint8)


def pack_documents(documents, seqlen, pad_id):
    """Each input token is used at most once. Retain and mask the final short row."""
    if seqlen < 2:
        raise ValueError("sequence length must be at least two")
    rows, tokens, masks, segments, positions = [], [], [], [], []
    for doc_id, ids, loss in documents:
        if len(ids) != len(loss):
            raise ValueError("token and loss mask lengths differ")
        pos = 0
        while pos < len(ids):
            take = min(seqlen - len(tokens), len(ids) - pos)
            tokens.extend(ids[pos:pos + take])
            masks.extend(loss[pos:pos + take])
            segments.extend([doc_id] * take)
            positions.extend(range(pos, pos + take))
            pos += take
            if len(tokens) == seqlen:
                rows.append((tokens, masks, segments, positions))
                tokens, masks, segments, positions = [], [], [], []
    if tokens:
        missing = seqlen - len(tokens)
        rows.append((tokens + [pad_id] * missing, masks + [0] * missing,
                     segments + [-1] * missing, positions + [0] * missing))
    return rows


def write_dataset(out, rows, metadata):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    if not rows:
        raise ValueError("no packed sequences")
    files = {}
    for col, (name, dtype) in enumerate((("tokens.i32", np.int32), ("loss.u8", np.uint8),
                                        ("segments.i32", np.int32), ("positions.i32", np.int32))):
        np.asarray([row[col] for row in rows], dtype=dtype).tofile(out / name)
        files[name] = sha256_file(out / name)
    metadata = dict(metadata, format_version=1, nseq=len(rows), seqlen=len(rows[0][0]), files=files)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


class PackedDataset:
    def __init__(self, path, require_split="train"):
        self.path = Path(path)
        self.metadata = json.loads((self.path / "metadata.json").read_text())
        m = self.metadata
        if m.get("format_version") != 1 or (require_split and m.get("split") != require_split):
            raise ValueError(f"expected format v1 {require_split} dataset")
        self.arrays = {}
        for name, dtype in (("tokens.i32", np.int32), ("loss.u8", np.uint8),
                            ("segments.i32", np.int32), ("positions.i32", np.int32)):
            path = self.path / name
            if sha256_file(path) != m["files"][name]:
                raise ValueError(f"dataset checksum mismatch: {name}")
            if path.stat().st_size != m["nseq"] * m["seqlen"] * np.dtype(dtype).itemsize:
                raise ValueError(f"dataset size mismatch: {name}")
            self.arrays[name] = np.memmap(path, dtype=dtype, mode="r", shape=(m["nseq"], m["seqlen"]))
        if set(m.get("training_conversations", [])) & set(m.get("holdout_conversations", [])):
            raise ValueError("held-out conversations leaked into training")
        self.identity = sha256_file(self.path / "metadata.json")

    def __len__(self):
        return self.metadata["nseq"]

    def __getitem__(self, index):
        tokens, mask, segments, positions = (
            torch.tensor(self.arrays[k][index].astype(np.int64))
            for k in ("tokens.i32", "loss.u8", "segments.i32", "positions.i32"))
        labels = torch.full_like(tokens, -100)
        valid = (mask[1:] != 0) & (segments[1:] == segments[:-1]) & (segments[1:] >= 0)
        labels[:-1] = torch.where(valid, tokens[1:], -100)
        return tokens, labels, segments, positions
