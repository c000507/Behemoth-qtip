#!/usr/bin/env python3
"""stall.py OUTDIR NAME TOKENS [DECODERS=1] [MAX_NEW=700] [DELAY=2.0]
How a prefill stalls running decode streams: DECODERS short-prompt streams start, DELAY seconds later one prompt of TOKENS tokens
arrives (prompt text = pf.py's slice NAME/0). Per decode stream: token-arrival gaps before and during the prefill
(window = the prefill request's start .. its first token) -> OUTDIR/stall_NAME.json and one summary line."""
import json, sys, time, glob, threading, http.client, hashlib
out, name, tokens = sys.argv[1], sys.argv[2], int(sys.argv[3])
ndec = int(sys.argv[4]) if len(sys.argv) > 4 else 1
max_new = int(sys.argv[5]) if len(sys.argv) > 5 else 700
delay = float(sys.argv[6]) if len(sys.argv) > 6 else 2.0
MODEL = 'ornith-1.5-35b-a3b-reap20-v3'
corpus = ''.join(open(f, errors='replace').read() for f in sorted(glob.glob('/mnt/scratch/corpus/neuron/cpp/*.cpp') + glob.glob('/mnt/scratch/corpus/runtime/src/*.cpp') + glob.glob('/mnt/scratch/corpus/neuron/*.py')))
CPT = float(open('/mnt/scratch/run/cpt').read())
def prompt(name, i, tokens):
    n = int(tokens * CPT)
    off = int(hashlib.md5(f'{name}/{i}'.encode()).hexdigest(), 16) % max(1, len(corpus) - n - 1)
    off = corpus.rfind('\n', 0, off) + 1
    return f'[{name}/{i}] Here is a part of a code base.\n\n{corpus[off:off + n]}\n\nIn two sentences, what does this code do?'
def stream(content, max_new, rec):
    body = json.dumps({'model': MODEL, 'messages': [{'role': 'user', 'content': content}], 'max_tokens': max_new, 'temperature': 0, 'stream': True, 'stream_options': {'include_usage': True}})
    c = http.client.HTTPConnection('127.0.0.1', 1337, timeout=3000)
    rec['t0'] = time.time(); c.request('POST', '/v1/chat/completions', body, {'Content-Type': 'application/json'})
    r = c.getresponse(); rec['at'] = []; rec['status'] = r.status
    for line in r:
        line = line.decode().strip()
        if not line.startswith('data:'): continue
        d = line[5:].strip()
        if d == '[DONE]': break
        j = json.loads(d)
        if j.get('usage'): rec['usage'] = j['usage']
        for ch in j.get('choices', []):
            dl = ch.get('delta', {})
            if (dl.get('reasoning_content') or '') + (dl.get('content') or ''): rec['at'].append(time.time())
    rec['t1'] = time.time()
decs = [{} for _ in range(ndec)]; pre = {}
th = [threading.Thread(target=stream, args=(f'[stall {i}] Write a long, detailed essay about the history of bridges, at least two thousand words.', max_new, decs[i])) for i in range(ndec)]
[t.start() for t in th]; time.sleep(delay)
tp = threading.Thread(target=stream, args=(prompt(name, 0, tokens), 8, pre)); tp.start(); tp.join(); [t.join() for t in th]
w0, w1 = pre['t0'], pre['at'][0] if pre.get('at') else pre['t1']
def gaps(at, lo, hi):
    v = [b - a for a, b in zip(at, at[1:]) if lo <= a and b <= hi + 1e9 and a < hi]
    return sorted(v)
res = {'name': name, 'prompt_tokens': (pre.get('usage') or {}).get('prompt_tokens'), 'prefill_ttft': w1 - w0, 'streams': []}
for d in decs:
    at = d['at']; before = gaps(at, 0, w0)[:]; before = sorted(b - a for a, b in zip(at, at[1:]) if b <= w0)
    during = sorted(b - a for a, b in zip(at, at[1:]) if b > w0 and a < w1)
    n_during = sum(1 for x in at if w0 <= x <= w1)
    res['streams'].append({'chunks': len(at), 'before_med_ms': 1000 * before[len(before) // 2] if before else None, 'during_n': n_during,
                           'during_max_ms': 1000 * during[-1] if during else None, 'during_med_ms': 1000 * during[len(during) // 2] if during else None,
                           'during_p90_ms': 1000 * during[int(len(during) * 0.9)] if during else None, 'during_chunks_per_s': n_during / (w1 - w0), 'ended_before_prefill_done': d['t1'] < w1,
                           'completion_tokens': (d.get('usage') or {}).get('completion_tokens')})
json.dump({**res, 'dec': [d['at'] for d in decs], 'window': [w0, w1]}, open(f'{out}/stall_{name}_{ndec}.json', 'w'))
s = res['streams']
print(f"stall {name} prompt={res['prompt_tokens']} decoders={ndec}: prefill ttft {res['prefill_ttft']:.2f}s ({(res['prompt_tokens'] or 0) / res['prefill_ttft']:.0f} tok/s); decode gap before med {s[0]['before_med_ms']:.1f} ms; during prefill: max gap {max(x['during_max_ms'] or 0 for x in s):.0f} ms, med {s[0]['during_med_ms']:.0f} ms, p90 {s[0]['during_p90_ms']:.0f} ms, {sum(x['during_chunks_per_s'] for x in s):.1f} stream chunks/s in all; ended early: {sum(x['ended_before_prefill_done'] for x in s)}", flush=True)
