#!/usr/bin/env python3
"""pf.py OUTDIR CASE...   CASE = name:tokens[:concurrency[:max_new]]
Streams chat completions against the local server, records TTFT / total / text / usage per request -> OUTDIR/<name>.json.
Prompts are deterministic slices of real source text (the engine sources), distinct per (case, stream index)."""
import json, sys, time, glob, threading, http.client, hashlib
out = sys.argv[1]
MODEL = 'ornith-1.5-35b-a3b-reap20-v3'
import os
if os.environ.get('PF_CORPUS') == 'prose':   # English technical prose (the repo's docs at 600b70f, frozen)
    corpus = ''.join(open(f, errors='replace').read() for f in sorted(glob.glob('/mnt/scratch/corpus2/docs/*.md')))
else:
    corpus = ''.join(open(f, errors='replace').read() for f in sorted(glob.glob('/mnt/scratch/corpus/neuron/cpp/*.cpp') + glob.glob('/mnt/scratch/corpus/runtime/src/*.cpp') + glob.glob('/mnt/scratch/corpus/neuron/*.py')))
CPT = float(open('/mnt/scratch/run/cpt').read()) if glob.glob('/mnt/scratch/run/cpt') else 3.0
def prompt(name, i, tokens):
    if name.endswith('+'):   # the prompt of <name> with more text appended before the question (shares its token prefix: prefix-cache test)
        base = prompt(name[:-1], i, tokens)
        return base.replace('\n\nIn two sentences', '\n\n' + corpus[1000:1000 + int(600 * CPT)] + '\n\nIn two sentences')
    n = int(tokens * CPT)
    off = int(hashlib.md5(f'{name}/{i}'.encode()).hexdigest(), 16) % max(1, len(corpus) - n - 1)
    off = corpus.rfind('\n', 0, off) + 1
    return f'[{name}/{i}] Here is a part of a code base.\n\n{corpus[off:off + n]}\n\nIn two sentences, what does this code do?'
def one(name, i, tokens, max_new, res):
    body = json.dumps({'model': MODEL, 'messages': [{'role': 'user', 'content': prompt(name, i, tokens)}], 'max_tokens': max_new, 'temperature': 0, 'stream': True, 'stream_options': {'include_usage': True}})
    c = http.client.HTTPConnection('127.0.0.1', 1337, timeout=3000)
    t0 = time.time(); c.request('POST', '/v1/chat/completions', body, {'Content-Type': 'application/json'})
    r = c.getresponse(); ttft = None; text = ''; usage = None; nchunks = 0; err = None
    if r.status != 200:
        res[i] = {'error': r.read().decode()[:500], 'status': r.status}; return
    for line in r:
        line = line.decode().strip()
        if not line.startswith('data:'): continue
        d = line[5:].strip()
        if d == '[DONE]': break
        j = json.loads(d)
        if 'error' in j: err = j['error']; break
        if j.get('usage'): usage = j['usage']
        for ch in j.get('choices', []):
            dl = ch.get('delta', {})
            s = (dl.get('reasoning_content') or '') + (dl.get('content') or '')
            if s:
                if ttft is None: ttft = time.time() - t0
                text += s; nchunks += 1
    res[i] = {'ttft': ttft, 'total': time.time() - t0, 'text': text, 'usage': usage, 'chunks': nchunks, 'error': err, 'timings': (usage or {}).get('timings')}
for case in sys.argv[2:]:
    p = case.split(':'); name, tokens = p[0], int(p[1]); conc = int(p[2]) if len(p) > 2 else 1; max_new = int(p[3]) if len(p) > 3 else 48
    res = [None] * conc; th = [threading.Thread(target=one, args=(name, i, tokens, max_new, res)) for i in range(conc)]
    t0 = time.time(); [t.start() for t in th]; [t.join() for t in th]; wall = time.time() - t0
    json.dump({'case': case, 'wall': wall, 'res': res}, open(f"{out}/{name.replace('+', 'x')}.json", 'w'), indent=1)
    for i, r in enumerate(res):
        if r.get('error'): print(f'{name}/{i} ERROR {r}'); continue
        u = r['usage'] or {}; pt = u.get('prompt_tokens', 0); ct = u.get('completion_tokens', 0)
        r['ttft'] = r['ttft'] if r['ttft'] is not None else r['total']
        dec = (r['total'] - r['ttft']) / max(1, ct - 1) * 1000
        print(f"{name}/{i} prompt={pt} ttft={r['ttft']:.2f}s -> {pt / r['ttft']:.0f} tok/s  new={ct} decode={dec:.1f} ms/tok  total={r['total']:.2f}s  md5={hashlib.md5(r['text'].encode()).hexdigest()[:8]}", flush=True)
    if conc > 1:
        pt = sum((r['usage'] or {}).get('prompt_tokens', 0) for r in res if not r.get('error')); mt = max(r['ttft'] for r in res if not r.get('error'))
        print(f'{name} aggregate: {pt} prompt tokens, last first-token at {mt:.2f}s -> {pt / mt:.0f} tok/s', flush=True)
