"""Offline acceptance study of the MTP head (before any engine work): how many of its greedy drafts would the main model
accept, per draft position, on real generations.

Inputs: an e2e run of the fp8 engine (tests/qwen35_e2e.py with SEQUENTIAL=1 and NEURON_H_DUMP=<dir>/h.bin: the final
residual of every prompt and generated position, run_<i>.json with the token ids), the prepared set (embed.bin, the fp8
lm_head shards and the final norm) and the extracted `mtp.*` tensors (qwen35_fetch_mtp.py). The head runs on the CPU in
f32 (qwen35_moe_ref.Mtp) on the DEVICE's hidden states, teacher-forced along the device's greedy stream:

  entry p (true):  (h[p], x[p+1]) -> draft 1 = the token after x[p+1]          (exists once the main model has sampled x[p+1])
  chain j = 2..K:  (the head's output, its own draft j-1) at position p+j-1 -> draft j

Draft j is accepted iff drafts 1..j equal the stream's tokens x[p+2..p+1+j] (greedy verification). Reported per draft
position: the conditional acceptance a_j = P(draft j right | 1..j-1 right), the expected accepted drafts for K = 1..4 and
the tokens per verify step 1 + sum_j prod a. Context variants of the MTP attention: full (prompt + generated entries),
gen (only the entries of generated positions: no MTP pass over the prompt), none (no cache: every draft sees only its
own chain).

usage: qwen35_mtp_study.py PREPARED MTP_DIR E2E_DIR [K=4] [OUT.json]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glm_repack as RP  # noqa: E402
import qwen35_moe_ref as R  # noqa: E402
from qwen35_verify_prepared import read_entry  # noqa: E402


def load_prepared(root):
    man = json.loads((root / 'prepared.json').read_text())
    files, tp, V = man['files'], man['geometry']['tp'], man['geometry']['vocab']
    embed = read_entry(root / 'embed.bin', files['embed.bin'][0])                       # bf16 [V, H]
    parts, norm = [], None
    for r in range(tp):
        e = {x['name']: x for x in files[f'r{r}/head.bin']}
        w = read_entry(root / f'r{r}/head.bin', e['w']).float()
        if 'w_s' in e:
            w = w * read_entry(root / f'r{r}/head.bin', e['w_s']).reshape(-1, 1)
        parts.append(w)
        norm = read_entry(root / f'r{r}/head.bin', e['norm']).float()                   # (1 + w)
    return man, embed, torch.cat(parts)[:V].contiguous(), norm


def load_mtp(mtp_dir):
    ck = RP.Checkpoint(mtp_dir)
    return {k: ck.tensor(k) for k in ck.map if k.startswith('mtp.')}


def read_dump(path, H):
    rec = np.dtype([('slot', '<i8'), ('pos', '<i8'), ('x', '<f4', (H,))])
    a = np.fromfile(path, dtype=rec)
    seqs, cur = [], None
    for i in range(len(a)):                                                             # sequential requests: a new one starts at position 0
        if a['pos'][i] == 0:
            cur = []
            seqs.append(cur)
        assert a['pos'][i] == len(cur), 'the dump is not one sequence after the other (run the e2e with SEQUENTIAL=1)'
        cur.append(i)
    return [torch.from_numpy(a['x'][idx].copy()) for idx in seqs]


def main():
    prepared, mtp_dir, e2e = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    K = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    torch.set_grad_enabled(False)
    t0 = time.time()
    man, embed, head, fnorm = load_prepared(prepared)
    g = R.Cfg(man['config'])
    mtp = R.Mtp(man['config'], load_mtp(mtp_dir))
    print(f'loaded: main E={g.E}, MTP layer E={mtp.g.E} top-{mtp.g.topk}, {time.time() - t0:.0f}s', flush=True)
    xs, runs = [], []
    for d in str(e2e).split(','):                                                        # several e2e directories: comma-separated
        x1 = read_dump(Path(d) / 'h.bin', g.H)
        xs += x1
        runs += [json.loads((Path(d) / f'run_{i}.json').read_text()) for i in range(len(x1))]
    variants = ('full', 'gen', 'none')
    hit = {v: np.zeros(K) for v in variants}      # drafts 1..j all right
    tot = {v: np.zeros(K) for v in variants}      # chains that had a draft j to check
    main_agree = [0, 0]
    per_run = []
    for x, run in zip(xs, runs):
        toks = torch.tensor(run['prompt'] + run['generated'])
        n, N = len(run['prompt']), len(toks)
        assert x.shape[0] >= N - 1, (x.shape, N)                                         # the last generated token was never fed
        x = x[:N - 1]
        h = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + g.eps) * fnorm     # the main model's post-norm hidden
        # sanity: the dumped hidden reproduces the engine's greedy tokens through the fp8 head
        am = (h[n - 1:] @ head.T).argmax(-1)
        main_agree[0] += int((am == toks[n:]).sum()); main_agree[1] += N - n
        ps = torch.arange(n - 1, N - 2)                                                  # entries whose draft 1 (token p + 2) can be checked
        row = {'prompt': n, 'generated': N - n}
        for v in variants:
            lo = {'full': 0, 'gen': n - 1, 'none': None}[v]
            if lo is None:
                drafts = []
                for p in ps.tolist():                                                    # no cache: every chain starts from an empty state
                    st = mtp.init_state()
                    o = mtp.step(h[p:p + 1], embed[toks[p + 1:p + 2]].float(), [p], st)
                    drafts.append(mtp.chains(o, [0], st['k'], st['v'], embed, head, K - 1)[0])   # relative positions only matter
                drafts = torch.stack(drafts)
            else:
                st = mtp.init_state()
                idx = torch.arange(lo, N - 2)
                o = mtp.step(h[idx], embed[toks[idx + 1]].float(), idx, st)              # true entries lo .. N-3
                drafts = mtp.chains(o[ps - lo], ps - lo, st['k'], st['v'], embed, head, K - 1)
            ok = torch.ones(len(ps), dtype=torch.bool)
            acc = []
            for j in range(K):
                tgt = ps + 2 + j
                have = tgt < N
                ok = ok & have & (drafts[:, j] == toks[tgt.clamp(max=N - 1)])
                # denominator of a_j: the chains whose drafts 1..j-1 were right and whose target exists
                tot[v][j] += int(have.sum()); hit[v][j] += int(ok.sum())
                acc.append(int(ok.sum()))
            row[v] = acc
            print(f'  prompt {n:5d} gen {N - n:4d} {v:5s}: chains {len(ps)}, drafts 1..j all accepted: {acc}  ({time.time() - t0:.0f}s)', flush=True)
        per_run.append(row)
    out = {'main_head_agreement': main_agree, 'K': K, 'runs': per_run, 'variants': {}}
    print(f'\nfp8 head on the dumped hidden states reproduces the engine tokens: {main_agree[0]}/{main_agree[1]}')
    for v in variants:
        p_all = hit[v] / np.maximum(tot[v], 1)                                           # P(drafts 1..j all right)
        cond = [p_all[0]] + [p_all[j] / max(p_all[j - 1], 1e-9) for j in range(1, K)]
        exp = np.cumsum(p_all)
        out['variants'][v] = {'p_prefix': p_all.tolist(), 'conditional': cond, 'tokens_per_step': (1 + exp).tolist()}
        print(f'{v:5s} P(1..j accepted) ' + ' '.join(f'{a:.3f}' for a in p_all) + ' | conditional a_j ' + ' '.join(f'{a:.3f}' for a in cond) +
              ' | tokens per verify step, K=1..: ' + ' '.join(f'{1 + e:.2f}' for e in exp))
    if len(sys.argv) > 5:
        Path(sys.argv[5]).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
