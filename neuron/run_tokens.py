"""Run a single sequence by token ID through the experimental Llama path."""
import argparse
import json
import numpy as np
from llama import LlamaRunner

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('checkpoint')
parser.add_argument('--tokens', required=True, help='Comma-separated input token IDs')
parser.add_argument('--generate', type=int, default=1, help='Number of greedy output IDs')
parser.add_argument('--device', action='store_true', help='Use NKI device execution instead of CPU simulation')
args = parser.parse_args()
if args.generate < 0:
    parser.error('--generate must be nonnegative')
ids = [int(x) for x in args.tokens.split(',') if x.strip()]
runner = LlamaRunner(args.checkpoint, simulate=not args.device)
if len(ids) + max(args.generate - 1, 0) > runner.max_context:
    parser.error('Prompt and generation exceed model context')
logits = runner.prefill(ids)
result = []
for i in range(args.generate):
    token = int(np.argmax(logits))
    result.append(token)
    if i + 1 < args.generate:
        logits = runner.forward_token(token)
print(json.dumps({'prompt_tokens': len(ids), 'generated_tokens': result}))
