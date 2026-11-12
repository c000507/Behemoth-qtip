"""End-to-end driver for Qwen3.5 MoE on the NeuronCores: renders chat prompts with the checkpoint's tokenizer (HF
`tokenizers` + the chat template through transformers when available, else neuron/cpp/tok_check), runs them through
neuron_engine_cli (the real Engine / scheduler / NeuronModel on the prepared fp8 set), decodes the greedy output and
writes run_<i>.json ({"prompt": ids, "generated": ids}) for test_qwen35_oracle.py.

usage: BEHEMOTH_NEURON_NEFF_DIR=GRAPHS python neuron/tests/qwen35_e2e.py PREPARED_DIR CLI_BINARY OUT_DIR [max_new] [prompt...]
  env: CONCURRENT=n (the same prompt n times in one submission; SEQUENTIAL=1: one after the other in the same engine
       process - the run-to-run reproducibility test without reloading the weights), THINK=0 (template with enable_thinking=false), FILL=n
       (prepend ~n tokens of filler text to the prompt: long-context timing)
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


class ToolTokenizer:
    """The same tokenizer / template through neuron/cpp/tok_check (TOK_CHECK=binary) when transformers is not installed."""

    def __init__(self, model):
        self.model, self.bin = str(model), os.environ.get('TOK_CHECK', 'tok_check')
        self.tmp = Path(os.environ.get('TMPDIR', '/tmp')) / f'tok_{os.getpid()}'
        self.tmp.mkdir(parents=True, exist_ok=True)

    def run(self, *args):
        return json.loads(subprocess.run([self.bin, self.model, *args], check=True, capture_output=True, text=True).stdout)

    def encode(self, text, add_special_tokens=False):
        (self.tmp / 't.txt').write_text(text)
        return self.run('--text', str(self.tmp / 't.txt'))['ids']

    def decode(self, ids, skip_special_tokens=False):
        (self.tmp / 'ids.json').write_text(json.dumps(list(ids)))
        return self.run('--decode', str(self.tmp / 'ids.json'))['text']

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        (self.tmp / 'm.json').write_text(json.dumps(messages))
        text = self.run(str(self.tmp / 'm.json'))['rendered']
        if not enable_thinking and text.endswith('<think>\n'):
            text += '\n</think>\n\n'   # the template's enable_thinking=false form
        return text

DEFAULT_PROMPTS = ['Explain in two paragraphs why the sky is blue.',
                   'Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.']


def main():
    model, cli, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    max_new = int(sys.argv[4]) if len(sys.argv) > 4 else 48
    prompts = sys.argv[5:] or DEFAULT_PROMPTS
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model) if AutoTokenizer else ToolTokenizer(model)
    think = os.environ.get('THINK', '0') != '0'
    fill = int(os.environ.get('FILL', '0'))
    conc = int(os.environ.get('CONCURRENT', '1'))
    reqs, metas = [], []
    for p in prompts:
        if fill:
            words = ('The history of computing spans mechanical calculators, electromechanical relays, vacuum tubes, transistors and integrated circuits. ' * (fill // 20 + 1))
            fill_ids = tok.encode(words, add_special_tokens=False)[:fill]
            p = tok.decode(fill_ids) + '\n\nAfter that long preamble, here is the question: ' + p
        text = tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True, enable_thinking=think)
        ids = tok.encode(text, add_special_tokens=False)
        for _ in range(conc):
            reqs.append({'tokens': ids, 'max_new': max_new, 'temperature': 0, 'top_k': 0, 'top_p': 1, 'seed': 0})
            metas.append({'prompt': p[-80:], 'ids': ids})
    (out / 'requests.json').write_text(json.dumps(reqs))
    t0 = time.time()
    r = subprocess.run([cli, str(model), str(out / 'requests.json')] + (['sequential'] if os.environ.get('SEQUENTIAL', '0') != '0' else []), capture_output=True, text=True)
    wall = time.time() - t0
    (out / 'cli_stderr.log').write_text(r.stderr)
    (out / 'cli_stdout.log').write_text(r.stdout)
    if r.returncode:
        print('neuron_engine_cli failed:', r.stderr[-2000:])
        sys.exit(1)
    res = None
    for m in re.finditer(r'^\{', r.stdout, re.M):   # libnrt logs land on stdout before the result JSON
        try:
            res = json.loads(r.stdout[m.start():])
            break
        except json.JSONDecodeError:
            continue
    if res is None:
        print('no result JSON in the CLI output:', r.stdout[-1000:])
        sys.exit(1)
    (out / 'cli_out.json').write_text(json.dumps(res, indent=1))
    total_new = 0
    for i, (req, m) in enumerate(zip(res['requests'], metas)):
        gen = req['tokens']
        total_new += len(gen)
        text = tok.decode(gen, skip_special_tokens=False)
        (out / f'run_{i}.json').write_text(json.dumps({'prompt': m['ids'], 'generated': gen, 'text': text, 'finish': req.get('finish')}))
        print(f'--- request {i}: prompt {len(m["ids"])} ids, {len(gen)} new tokens, finish={req.get("finish")}\n{text}\n')
    if conc > 1:   # identical requests: first differing token index of each vs request 0 (None = identical)
        g0 = res['requests'][0]['tokens']
        fd = [next((i for i, (x, y) in enumerate(zip(r_['tokens'], g0)) if x != y), None if len(r_['tokens']) == len(g0) else min(len(r_['tokens']), len(g0))) for r_ in res['requests']]
        print('REPRO first-diff vs request 0:', fd, 'lengths', [len(r_['tokens']) for r_ in res['requests']], 'errors', [r_.get('error') for r_ in res['requests'] if r_.get('error')])
    print(json.dumps({'wall_s': wall, 'engine_s': res.get('seconds'), 'new_tokens': total_new, 'metrics': res.get('metrics')}, indent=1))


if __name__ == '__main__':
    main()
