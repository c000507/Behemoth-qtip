#!/usr/bin/env python3
"""Require bit-identical selected-position logits from a behemoth_server trace.

Trace safetensors: tokens[T] I64, logit_positions[N] I64, logits[N,vocab] F32;
optional segments[T] I64 and positions[T] I64. No trace is fabricated here.
"""
import argparse
import json
from pathlib import Path
import sys

from safetensors import safe_open
from safetensors.torch import load_file
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qtip_training.checkpoint import CPUStager
from qtip_training.kernels import load_backend
from qtip_training.model import HostModel
from qtip_training.streaming import Replica


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--engine", choices=("auto", "fast", "autograd"), default="auto")
    args = parser.parse_args()
    if args.backend == "auto":
        args.backend = "runtime" if torch.device(args.device).type == "cuda" else "runtime-reference"
    backend = load_backend(args.backend)
    if args.engine == "auto":
        args.engine = "fast" if args.backend == "runtime" else "autograd"
    trace = load_file(args.trace)
    tokens = trace["tokens"]
    indices = trace["logit_positions"]
    if tokens.ndim != 1 or tokens.dtype != torch.int64 or indices.ndim != 1 or indices.dtype != torch.int64:
        parser.error("expected I64 tokens[T] and logit_positions[N]")
    if len(indices) == 0 or indices.min() < 0 or indices.max() >= len(tokens):
        parser.error("logit_positions must select at least one token within the prompt")
    host = HostModel(args.model, pin=torch.device(args.device).type == "cuda")
    with safe_open(args.trace, "pt") as handle:
        identity = (handle.metadata() or {}).get("model_sha256")
    if identity is not None and identity != host.identity:
        parser.error("trace belongs to a different source model")
    if tokens.min() < 0 or tokens.max() >= host.config["vocab_size"]:
        parser.error("token outside vocabulary")
    expected = trace["logits"]
    if expected.shape != (len(indices), host.config["vocab_size"]) or expected.dtype != torch.float32:
        parser.error("expected F32 logits[N,vocab]")
    batch = (tokens, torch.full_like(tokens, -100), trace.get("segments", torch.zeros_like(tokens)),
             trace.get("positions", torch.arange(len(tokens))))
    replica = Replica(host, args.device, backend, len(tokens), engine=args.engine)
    stager, offset, exact, max_diff = CPUStager(), 0, True, 0.0
    results = []
    for chunk in replica.logits(batch, indices):
        result = stager.copy(chunk)
        reference = expected[offset:offset + len(result)]
        exact &= torch.equal(result, reference)
        max_diff = max(max_diff, float((result - reference).abs().max()))
        results.append(result)
        offset += len(result)
    # Approximate agreement, reported for diagnosis only: it never makes the check pass.
    actual = torch.cat(results).double()
    target = expected.double()
    p, q = target.log_softmax(-1), actual.log_softmax(-1)
    top5 = [len(set(a.tolist()) & set(b.tolist())) / 5 for a, b in zip(actual.topk(5).indices, target.topk(5).indices)]
    stats = dict(rel_rms=float((actual - target).norm() / target.norm()),
                 argmax_agreement=float((actual.argmax(-1) == target.argmax(-1)).double().mean()),
                 top5_overlap=sum(top5) / len(top5), kl_runtime_to_trainer=float((p.exp() * (p - q)).sum(-1).mean()))
    print(json.dumps(dict(bit_identical=exact, max_abs_diff=max_diff, positions=len(indices),
                          model_sha256=host.identity, backend=args.backend, engine=args.engine, **stats)))
    if not exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
