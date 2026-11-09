"""Compare two behemoth_cuda / glm53 teacher-forced logit traces position by position.

Reports top-1 agreement, KL(reference || candidate), the max absolute logit difference and
the smallest reference top-2 margin among disagreements. Exit status 1 when agreement or KL
miss the bounds (defaults are generous bounds for bf16 arithmetic differing by summation order).
"""
import argparse
import json
import math
import sys


def log_softmax(row):
    top = max(row)
    total = math.log(sum(math.exp(v - top) for v in row))
    return [v - top - total for v in row]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--min-agreement", type=float, default=0.95)
    parser.add_argument("--max-mean-kl", type=float, default=1e-3)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    a = json.load(open(args.reference))
    b = json.load(open(args.candidate))
    if a["tokens"] != b["tokens"] or a["logit_positions"] != b["logit_positions"]:
        print("traces cover different tokens/positions", file=sys.stderr)
        return 2
    agree = 0
    kls = []
    max_abs = 0.0
    smallest_margin = None
    for ra, rb in zip(a["logits"], b["logits"]):
        if len(ra) != len(rb):
            print("vocabulary mismatch", file=sys.stderr)
            return 2
        ia = max(range(len(ra)), key=ra.__getitem__)
        ib = max(range(len(rb)), key=rb.__getitem__)
        if ia == ib:
            agree += 1
        else:
            second = max(v for i, v in enumerate(ra) if i != ia)
            margin = ra[ia] - second
            smallest_margin = margin if smallest_margin is None else min(smallest_margin, margin)
        la, lb = log_softmax(ra), log_softmax(rb)
        kls.append(sum(math.exp(x) * (x - y) for x, y in zip(la, lb)))
        max_abs = max(max_abs, max(abs(x - y) for x, y in zip(ra, rb)))
    n = len(kls)
    mean_kl = sum(kls) / n
    label = args.label or f"{args.candidate} vs {args.reference}"
    print(f"{label}: positions {n}, top-1 agreement {agree}/{n} ({100 * agree / n:.1f}%), "
          f"KL mean {mean_kl:.2e} max {max(kls):.2e}, max |dlogit| {max_abs:.4f}"
          + (f", smallest reference margin among disagreements {smallest_margin:.4f}" if smallest_margin is not None else "")
          + ("" if max_abs > 0 else " (bitwise identical)"))
    ok = agree / n >= args.min_agreement and mean_kl <= args.max_mean_kl
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
