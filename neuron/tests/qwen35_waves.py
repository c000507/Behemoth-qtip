"""Several measurement waves in ONE engine process (the weights load once): each wave is n concurrent requests; a wave starts
when the previous one has finished (neuron_engine_cli "barrier" requests).

usage: BEHEMOTH_NEURON_NEFF_DIR=GRAPHS python neuron/tests/qwen35_waves.py PREPARED_DIR CLI_BINARY OUT_DIR SPEC...
  SPEC = n:max_new[:fill[:same]]   n streams, max_new tokens each, ~fill filler tokens before the prompt (0: none),
         same = 1: the n requests are the SAME prompt (default with fill), else distinct prompts of PROMPTS (file, one per line)
  -> OUT_DIR/w<k>/{run_<i>.json, cli_stderr.log (the wave's part of the step log)}, OUT_DIR/cli_stderr.log (all)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen35_e2e as E


def main():
    model, cli, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    specs = sys.argv[4:]
    out.mkdir(parents=True, exist_ok=True)
    tok = E.AutoTokenizer.from_pretrained(model) if E.AutoTokenizer else E.ToolTokenizer(model)
    pr = [l.rstrip('\n') for l in open(os.environ['PROMPTS'])] if os.environ.get('PROMPTS') else list(E.DEFAULT_PROMPTS)
    fixed = os.environ.get('FILL_PROMPT', 'Explain in two paragraphs why the sky is blue.')
    reqs, waves = [], []
    for sp in specs:
        f = sp.split(':')
        n, new, fill = int(f[0]), int(f[1]), int(f[2]) if len(f) > 2 else 0
        same = (f[3] == '1') if len(f) > 3 else fill > 0
        if same:
            ps = [fixed] * n
        elif n == 1:
            ps = [pr[1 % len(pr)]]     # as box/d.sh: the single-stream run uses the second prompt
        else:
            ps = [pr[i % len(pr)] for i in range(n)]
        first = len(reqs)
        for i, p in enumerate(ps):
            if fill:
                words = ('The history of computing spans mechanical calculators, electromechanical relays, vacuum tubes, transistors and integrated circuits. ' * (fill // 20 + 1))
                p = tok.decode(tok.encode(words, add_special_tokens=False)[:fill]) + '\n\nAfter that long preamble, here is the question: ' + p
            text = tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True, enable_thinking=os.environ.get('THINK', '0') != '0')
            ids = tok.encode(text, add_special_tokens=False)
            reqs.append({'tokens': ids, 'max_new': new, 'temperature': 0, 'top_k': 0, 'top_p': 1, 'seed': 0, 'barrier': i == 0 and first > 0})
        waves.append((sp, first, len(reqs)))
    (out / 'requests.json').write_text(json.dumps(reqs))
    r = subprocess.run([cli, str(model), str(out / 'requests.json')], capture_output=True, text=True)
    (out / 'cli_stderr.log').write_text(r.stderr)
    (out / 'cli_stdout.log').write_text(r.stdout)
    if r.returncode:
        print('neuron_engine_cli failed:', r.stderr[-2000:])
        sys.exit(1)
    res = None
    for m in E.re.finditer(r'^\{', r.stdout, E.re.M):
        try:
            res = json.loads(r.stdout[m.start():])
            break
        except json.JSONDecodeError:
            continue
    if res is None:
        print('no result JSON in the CLI output:', r.stdout[-1000:])
        sys.exit(1)
    (out / 'cli_out.json').write_text(json.dumps(res, indent=1))
    parts, cur = {}, None            # the step log by wave: lines after `wave-barrier <first request index>`
    head = []
    for l in r.stderr.splitlines(True):
        if l.startswith('wave-barrier '):
            cur = int(l.split()[1])
            parts[cur] = []
        elif cur is None:
            head.append(l)
        else:
            parts[cur].append(l)
    for k, (sp, a, b) in enumerate(waves):
        d = out / f'w{k}'
        d.mkdir(exist_ok=True)
        (d / 'cli_stderr.log').write_text(''.join(head if k == 0 else []) + ''.join(parts.get(a, [])))
        (d / 'spec.txt').write_text(sp + '\n')
        for i in range(a, b):
            q = res['requests'][i]
            (d / f'run_{i - a}.json').write_text(json.dumps({'prompt': reqs[i]['tokens'], 'generated': q['tokens'], 'text': tok.decode(q['tokens'], skip_special_tokens=False),
                                                           'finish': q.get('finish'), 'first_token_ms': q.get('first_token_ms'), 'last_token_ms': q.get('last_token_ms')}))
        print(f'WAVE w{k} {sp}: {b - a} requests, new tokens {[len(res["requests"][i]["tokens"]) for i in range(a, b)]}, errors {[res["requests"][i].get("error") for i in range(a, b) if res["requests"][i].get("error")]}')
    print(json.dumps({'engine_s': res.get('seconds'), 'metrics': res.get('metrics')})[:2000])


if __name__ == '__main__':
    main()
