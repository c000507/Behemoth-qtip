#!/usr/bin/env python3
"""Time one real decoder layer forward / recompute+backward on one GPU.

Inputs are embeddings of real (or random) token ids; the layer's packed weights
are resident on the device, so this excludes host streaming (see the trainer's
--profile output for whole-step breakdowns).
"""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qtip_training.kernels import load_backend
from qtip_training.model import decoder_layer, HostModel, SU_GROUPS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--layer", type=int, default=40)
    parser.add_argument("--seqlen", default="2048")
    parser.add_argument("--engine", choices=("autograd", "fast"), default="fast")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--row-chunk", type=int, default=256)
    parser.add_argument("--attention-block", type=int, default=256)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    host = HostModel(args.model, layers=[args.layer], identity=False)
    backend = load_backend("runtime")
    weights = {k: v.to(device) for k, v in host.layers[0].items()}
    params = {k: torch.nn.Parameter(v.to(device).clone()) for k, v in host.initial[0].items()}
    for group in SU_GROUPS:
        for name in group[1:]:
            params[name] = params[group[0]]
    results = []
    for seqlen in [int(s) for s in args.seqlen.split(",")]:
        g = torch.Generator().manual_seed(0)
        tokens = torch.randint(0, host.config["vocab_size"], (seqlen,), generator=g)
        x = host.embedding[tokens].to(device=device, dtype=torch.bfloat16)
        segments = torch.zeros(seqlen, dtype=torch.long, device=device)
        positions = torch.arange(seqlen, device=device)
        dy = torch.randn(seqlen, x.shape[1], device=device) * 1e-3
        if args.engine == "fast":
            from qtip_training.fast import FastLayer, Timer, batch_geometry
            timer = Timer(args.profile)
            layer = FastLayer(host.config, device, backend, row_chunk=args.row_chunk, timer=timer)
            geometry = batch_geometry(segments.cpu(), positions, host.config, device)
        times = dict(forward=[], backward=[])
        for _ in range(args.repeat):
            if args.engine == "fast":
                timer.events.clear()
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            if args.engine == "fast":
                layer.decode(weights)
                out = layer.forward(x, params, geometry)
            else:
                with torch.no_grad():
                    out = decoder_layer(x, weights, params, host.config, segments, positions, backend,
                                        args.row_chunk, args.attention_block)
            torch.cuda.synchronize(device)
            times["forward"].append(time.perf_counter() - start)
            del out
            for p in params.values():
                p.grad = None
            start = time.perf_counter()
            if args.engine == "fast":
                layer.decode(weights)
                dx = layer.backward(x, params, geometry, dy.clone())
            else:
                value = x.detach().requires_grad_(True)
                out = decoder_layer(value, weights, params, host.config, segments, positions, backend,
                                    args.row_chunk, args.attention_block)
                out.backward(dy)
                dx = value.grad
                del out, value
            torch.cuda.synchronize(device)
            times["backward"].append(time.perf_counter() - start)
            del dx
        record = dict(engine=args.engine, seqlen=seqlen, layer=args.layer,
                      forward_s=min(times["forward"]), backward_s=min(times["backward"]),
                      peak_gb=torch.cuda.max_memory_allocated(device) / 1e9)
        if args.engine == "fast" and args.profile:
            record["sections_ms"] = timer.collect()
        print(json.dumps(record), flush=True)
        results.append(record)
        torch.cuda.reset_peak_memory_stats(device)


if __name__ == "__main__":
    main()
