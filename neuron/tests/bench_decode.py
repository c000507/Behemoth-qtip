"""Measure steady-state device decode throughput of the Llama runner.

Prefill and the first decode tokens trigger compilation; those timings are
reported separately from the steady-state mean. Requires --device on an
attached Neuron device.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llama import LlamaRunner

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('checkpoint')
p.add_argument('--prompt', default='128000,791,6864,315,9822,374')
p.add_argument('--warmup', type=int, default=3, help='Decode tokens measured but excluded from the steady-state mean')
p.add_argument('--tokens', type=int, default=10, help='Total decode tokens')
a = p.parse_args()
ids = [int(x) for x in a.prompt.split(',')]
started = time.monotonic()
runner = LlamaRunner(a.checkpoint, simulate=False)
load_s = time.monotonic() - started
started = time.monotonic()
logits = None
for token in ids:
    logits = runner.forward_token(token)
prefill_s = time.monotonic() - started
timings = []
for i in range(a.tokens):
    token = int(np.argmax(logits))
    t = time.monotonic()
    logits = runner.forward_token(token)
    timings.append((token, time.monotonic() - t))
steady = [dt for _, dt in timings[a.warmup:]]
result = {'load_s': round(load_s, 3), 'prefill_tokens': len(ids), 'prefill_s': round(prefill_s, 3),
          'decode_ms': [(t, round(dt * 1000, 1)) for t, dt in timings],
          'steady_mean_ms': round(sum(steady) / len(steady) * 1000, 1) if steady else None,
          'steady_tokens_per_s': round(len(steady) / sum(steady), 3) if steady else None}
print(json.dumps(result))
