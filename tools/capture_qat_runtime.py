#!/usr/bin/env python3
"""Capture real behemoth_eval token-id/logit traces for the strict QAT checker."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from safetensors.torch import load_file, save_file
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qtip_training.model import checkpoint_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True, help="safetensors with I64 tokens[T] and optional logit_positions[N]")
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if Path(args.out).exists() or args.chunk < 1:
        parser.error("output must be new and chunk positive")
    prompt = load_file(args.prompt)
    tokens = prompt["tokens"]
    selected = prompt.get("logit_positions", torch.tensor([len(tokens) - 1], dtype=torch.int64))
    if tokens.ndim != 1 or tokens.dtype != torch.int64 or not len(tokens):
        parser.error("expected nonempty I64 tokens[T]")
    if selected.ndim != 1 or selected.dtype != torch.int64 or not 1 <= len(selected) <= 128:
        parser.error("expected I64 logit_positions[N], 1 <= N <= 128")
    if selected.min() < 0 or selected.max() >= len(tokens):
        parser.error("logit position outside prompt")
    # The runtime trace path is one contiguous document. Silently flattening a
    # packed/document-masked training row would compare two different prompts.
    if "segments" in prompt and (prompt["segments"].shape != tokens.shape or not (prompt["segments"] == 0).all()):
        parser.error("runtime capture accepts one document (segments all zero)")
    if "positions" in prompt and not torch.equal(prompt["positions"], torch.arange(len(tokens))):
        parser.error("runtime capture requires positions 0..T-1")
    model = str(Path(args.model).resolve())
    with tempfile.TemporaryDirectory(prefix="qtip-runtime-trace-") as temporary:
        request, response = Path(temporary) / "request.json", Path(temporary) / "response.json"
        request.write_text(json.dumps(dict(tokens=tokens.tolist(), logit_positions=selected.tolist())))
        subprocess.run([args.eval_binary, model, "trace", str(request), str(response), str(args.chunk)], check=True)
        result = json.loads(response.read_text())
    if result["tokens"] != tokens.tolist() or result["logit_positions"] != selected.tolist() or result["model"] != model:
        raise ValueError("runtime trace does not match requested prompt/model")
    logits = torch.tensor(result["logits"], dtype=torch.float32)
    config = json.loads((Path(model) / "config.json").read_text())
    if logits.shape != (len(selected), config["vocab_size"]) or not torch.isfinite(logits).all():
        raise ValueError("invalid runtime logits")
    save_file(dict(tokens=tokens, logit_positions=selected, logits=logits,
                   segments=torch.zeros_like(tokens), positions=torch.arange(len(tokens))), args.out,
              metadata=dict(model_sha256=checkpoint_identity(model), producer="behemoth_eval trace",
                            chunk=str(result["chunk"]), attention_qp=result["attention_qp"]))


if __name__ == "__main__":
    main()
