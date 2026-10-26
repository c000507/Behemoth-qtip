#!/usr/bin/env python3
"""Stream QTIP layers through AdamW on continuous signs/scales and RMSNorms."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import resource
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qtip_training.checkpoint import CPUStager, export_model, load_checkpoint, save_checkpoint
from qtip_training.data import PackedDataset
from qtip_training.kernels import load_backend
from qtip_training.model import HostModel
from qtip_training.streaming import reduce_gradients, Replica


def lr_multiplier(step, total, warmup):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def parallel(pool, functions):
    futures = [pool.submit(function) for function in functions]
    return [future.result() for future in futures]


def no_core_dumps():
    """A crash of a process holding ~100-300 GB of pinned host memory would make apport
    write a multi-GB crash report into the root filesystem; disable dumps for this process."""
    try:
        import ctypes
        ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE = 0
    except (OSError, AttributeError):
        pass


def main():
    no_core_dumps()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, help="directory from make_train_gmcot.py")
    parser.add_argument("--work", required=True)
    parser.add_argument("--out", help="export complete HF QTIP directory after training")
    parser.add_argument("--devices", default="cuda:0,cuda:1", help="e.g. cuda:0,cuda:1 or cpu for tiny tests")
    parser.add_argument("--backend", default="auto", help="auto: native runtime on HIP, runtime-reference on CPU; or reference/module")
    parser.add_argument("--engine", choices=("auto", "fast", "autograd"), default="auto",
                        help="fast: dense-decode/bf16-GEMM/flash-attention engine (GPU default); autograd: reference path")
    parser.add_argument("--fast-rows", type=int, default=4096, help="fast engine token-row chunk")
    parser.add_argument("--profile", action="store_true", help="per-step GPU section, copy-engine and phase timings")
    parser.add_argument("--weight-rows", type=int, default=128)
    parser.add_argument("--row-chunk", type=int, default=256)
    parser.add_argument("--attention-block", type=int, default=256)
    parser.add_argument("--loss-chunk", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--norm-lr", type=float, default=1e-5)
    parser.add_argument("--norm-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=500000, help="pilot budget; never more than one epoch")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", help="path to an atomic training checkpoint")
    parser.add_argument("--stop-after-steps", type=int, help="interrupt cleanly without changing the resume schedule")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true", help="score a held-out/general packed dataset")
    args = parser.parse_args()
    if min(args.row_chunk, args.attention_block, args.loss_chunk, args.max_tokens, args.checkpoint_every) < 1:
        parser.error("chunk sizes, token budget and checkpoint frequency must be positive")
    if min(args.lr, args.norm_lr, args.clip) <= 0 or args.warmup_steps < 0 or args.norm_decay < 0:
        parser.error("invalid optimizer settings")
    devices = [torch.device(name.strip()) for name in args.devices.split(",")]
    if any(d.type not in ("cpu", "cuda") for d in devices) or len(set(map(str, devices))) != len(devices):
        parser.error("devices must be unique CPU/CUDA/HIP devices")
    if any(d.type != devices[0].type for d in devices):
        parser.error("do not mix CPU and GPU replicas")
    if devices[0].type == "cuda" and not torch.cuda.is_available():
        parser.error("no GPU available in this PyTorch build")
    if args.backend == "auto":
        args.backend = "runtime" if devices[0].type == "cuda" else "runtime-reference"
    if args.engine == "auto":
        args.engine = "fast" if args.backend == "runtime" else "autograd"
    if args.fast_rows < 1:
        parser.error("fast-rows must be positive")
    # Fail on a missing native build before allocating the full host model.
    backends = [load_backend(args.backend, args.weight_rows) for _ in devices]
    if args.stop_after_steps is not None and args.stop_after_steps < 1:
        parser.error("stop-after-steps must be positive")
    work = Path(args.work)
    if not args.eval_only and (work / "latest.pt").exists() and not args.resume:
        parser.error("work directory has a checkpoint; use --resume or a new --work directory")
    if args.out and Path(args.out).exists():
        parser.error("export output already exists")
    config = json.loads((Path(args.model) / "config.json").read_text())
    if devices[0].type == "cpu" and config["hidden_size"] > 1024:
        parser.error("CPU mode is for tiny verification models; select HIP devices for Behemoth")
    torch.manual_seed(args.seed)
    dataset = PackedDataset(args.data, require_split=None if args.eval_only else "train")
    seqlen = dataset.metadata["seqlen"]
    if args.max_tokens < seqlen:
        parser.error("token budget must include at least one sequence")
    count = min(len(dataset), args.max_tokens // seqlen)
    total_steps = math.ceil(count / len(devices))
    if args.backend in ("reference", "runtime-reference"):
        print(f"Using {args.backend}; full runtime parity and target GPU performance are unverified.", flush=True)
    print("Loading shared packed model and allocating pinned checkpoints...", flush=True)
    host = HostModel(args.model, pin=devices[0].type == "cuda")
    values = dataset.arrays["tokens.i32"]
    if values.min() < 0 or values.max() >= host.config["vocab_size"]:
        parser.error("dataset token ids exceed vocabulary")
    replicas = [Replica(host, device, backend, seqlen, args.row_chunk, args.attention_block, args.loss_chunk,
                        engine=args.engine, fast_rows=args.fast_rows, profile=args.profile)
                for device, backend in zip(devices, backends)]
    optimizers = [replica.optimizer(args.lr, args.norm_lr, args.norm_decay) for replica in replicas]
    signature = dict(model=host.identity, data=dataset.identity, backend=args.backend, weight_rows=args.weight_rows,
                     backend_identities=[getattr(backend, "identity", args.backend) for backend in backends],
                     row_chunk=args.row_chunk, attention_block=args.attention_block, loss_chunk=args.loss_chunk,
                     engine=args.engine, fast_rows=args.fast_rows if args.engine == "fast" else None,
                     replicas=len(devices), dtype=str(replicas[0].dtype), lr=args.lr, norm_lr=args.norm_lr,
                     norm_decay=args.norm_decay, warmup=args.warmup_steps, clip=args.clip, count=count,
                     max_tokens=args.max_tokens, seed=args.seed)
    progress = dict(step=0, position=0, tokens=0, supervised_tokens=0)
    if args.resume:
        progress = load_checkpoint(args.resume, replicas, optimizers, signature)
    work.mkdir(parents=True, exist_ok=True)
    stager = CPUStager()
    started = time.monotonic()
    starting_position = progress["position"]
    scored_loss, scored_count = 0.0, 0
    with ThreadPoolExecutor(max_workers=len(replicas)) as pool:
        while progress["position"] < count:
            offset = progress["position"]
            active = replicas[:min(len(replicas), count - offset)]
            batches = [dataset[offset + i] for i in range(len(active))]
            targets = sum(int((batch[1] != -100).sum()) for batch in batches)
            if targets == 0:
                progress["position"] += len(batches)
                progress["tokens"] += len(batches) * seqlen
                continue
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            step_started = time.monotonic()
            losses = parallel(pool, [lambda r=r, b=b: r.run(b, normalizer=targets, backward=not args.eval_only)
                                     for r, b in zip(active, batches)])
            if not all(math.isfinite(loss) for loss in losses):
                raise FloatingPointError("nonfinite loss; last atomic checkpoint remains resumable")
            norm = 0.0
            timings = dict(compute_s=time.monotonic() - step_started)
            if not args.eval_only:
                mark = time.monotonic()
                reduce_gradients(active)
                timings["reduce_s"] = time.monotonic() - mark
                mark = time.monotonic()
                # Tail steps may use fewer devices: broadcast the summed gradient
                # to idle replicas too, so every optimizer stays identical.
                for replica in replicas[len(active):]:
                    replica.gradients_from_host(active[0].grad_host)
                factor = lr_multiplier(progress["step"], total_steps, args.warmup_steps)
                for replica, optimizer in zip(replicas, optimizers):
                    with replica.context():
                        norm = float(torch.nn.utils.clip_grad_norm_(replica.parameters, args.clip, error_if_nonfinite=True))
                        for group, base_lr in zip(optimizer.param_groups, (args.lr, args.norm_lr)):
                            group["lr"] = base_lr * factor
                        optimizer.step()
                if replicas[0].cuda:
                    for replica in replicas:
                        torch.cuda.synchronize(replica.device)
                timings["optimizer_s"] = time.monotonic() - mark
            progress = dict(step=progress["step"] + 1, position=offset + len(batches),
                            tokens=progress["tokens"] + len(batches) * seqlen,
                            supervised_tokens=progress["supervised_tokens"] + targets)
            scored_loss += sum(losses)
            scored_count += targets
            record = dict(**progress, loss=sum(losses) / targets, grad_norm=norm,
                          elapsed_s=time.monotonic() - started,
                          tokens_per_s=(progress["position"] - starting_position) * seqlen / (time.monotonic() - started),
                          peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
            record["step_s"] = time.monotonic() - step_started
            record.update(timings)
            if replicas[0].cuda:
                record["peak_vram_bytes"] = [torch.cuda.max_memory_allocated(replica.device) for replica in replicas]
            if args.profile:
                record["replicas"] = []
                for replica in active:
                    detail = dict(device=str(replica.device), phases=getattr(replica, "phases", {}),
                                  copies=replica.slots.copy_stats())
                    if hasattr(replica, "timer"):
                        detail["gpu_sections_ms"] = replica.timer.collect()
                    record["replicas"].append(detail)
            print(json.dumps(record), flush=True)
            with (work / ("eval.jsonl" if args.eval_only else "train.jsonl")).open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            if not args.eval_only and progress["step"] % args.checkpoint_every == 0:
                mark = time.monotonic()
                save_checkpoint(work / "latest.pt", replicas[0], optimizers[0], progress, signature, stager)
                print(json.dumps(dict(snapshot=str(work / "latest.pt"), step=progress["step"],
                                      seconds=round(time.monotonic() - mark, 2))), flush=True)
            if args.stop_after_steps and progress["step"] >= args.stop_after_steps:
                break
    if not args.eval_only:
        save_checkpoint(work / "latest.pt", replicas[0], optimizers[0], progress, signature, stager)
        if args.out:
            export_model(host, stager.copy(replicas[0].named_parameters()), args.out)
            print(f"Exported QTIP model: {args.out}", flush=True)
    elif scored_count:
        mean = scored_loss / scored_count
        print(json.dumps(dict(assistant_nll=mean, perplexity=math.exp(min(mean, 700)), supervised_tokens=scored_count)))


if __name__ == "__main__":
    main()
