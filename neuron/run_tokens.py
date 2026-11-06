"""Run a single sequence by token ID through the experimental Llama path."""
import argparse
import json
import sys
import time
import numpy as np
from llama import LlamaRunner

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('checkpoint')
parser.add_argument('--tokens', required=True, help='Comma-separated input token IDs')
parser.add_argument('--generate', type=int, default=1, help='Number of greedy output IDs')
parser.add_argument('--device', action='store_true', help='Use NKI device execution instead of CPU simulation')
parser.add_argument('--progress', action='store_true', help='Report layer progress to stderr')
parser.add_argument('--logits-out', help='Write F32 logits for every forward pass to this directory as logits_<position>.npy')
args = parser.parse_args()
if args.generate < 0:
    parser.error('--generate must be nonnegative')
if args.logits_out:
    from pathlib import Path
    Path(args.logits_out).mkdir(parents=True, exist_ok=True)
ids = [int(x) for x in args.tokens.split(',') if x.strip()]
started=time.monotonic()
def progress(stage, current, total):
    print(f'elapsed_s={time.monotonic()-started:.1f} stage={stage} layer={current}/{total}', file=sys.stderr, flush=True)
runner = LlamaRunner(args.checkpoint, simulate=not args.device, progress=progress if args.progress else None)
if len(ids) + max(args.generate - 1, 0) > runner.max_context:
    parser.error('Prompt and generation exceed model context')
def emit(logits, position):
    if args.logits_out:
        np.save(f'{args.logits_out}/logits_{position}.npy', logits.astype(np.float32))
for i, token in enumerate(ids):
    logits = runner.forward_token(int(token))
    emit(logits, i)
result = []
for i in range(args.generate):
    token = int(np.argmax(logits))
    result.append(token)
    if i + 1 < args.generate:
        logits = runner.forward_token(token)
        emit(logits, len(ids) + i)
print(json.dumps({'prompt_tokens': len(ids), 'generated_tokens': result}))
