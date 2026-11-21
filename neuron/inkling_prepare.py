"""Inkling 2 Mini checkpoint (original format: model.llm.*, NVFP4 / MXFP8 experts) -> per-rank prepared tensors for the
Neuron graphs (inkling_graphs.py) and the runtime, in the GLM / Qwen prepared file format:

  OUT/prepared.json             format inkling2-neuron-prepared, geometry, visits (layer, pass, kind, ctl), files
  OUT/embed.bin                 bf16 [vocab, hidden] = the NORMED embedding rms(embed) * embed_norm (host lookup)
  OUT/r{rank}/L{layer:03d}.bin  this rank's tensors of one layer, names as inkling_graphs.weight_shapes (loop layers also
                                carry the message-gate tensors mg / gd_* / gf_* / gg_*)
  OUT/r{rank}/head.bin          output depth-attention query oq, final norm, this rank's lm_head rows (+ scale)

Dense projections: E4M3 with a per-output-row f32 scale (max |code| 224) or bf16 (--weights bf16). Experts: dequantised
from NVFP4 (E2M1 x fp8 per-16 scale x per-expert amax / denominator) or MXFP8 (E4M3 x 2^(s-127) per 32) and re-encoded
as E4M3 per output row (gate|up) / per input channel (down, stored [E, I/TP, H]). Router f32, norms / convs / queries f32.
`shard_layer(get, l, rank, g, fp8)` is also the sharding the CPU tests use (get: a Source over a checkpoint dict).

usage: inkling_prepare.py MODEL OUT [--tp N] [--weights fp8|bf16] [--only l,l,..] [--jobs J]
"""
import argparse
import json
import multiprocessing
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glm_repack as RP  # noqa: E402  (to_fp8_rows, write_file)
import inkling_graphs as IG  # noqa: E402
import inkling_ref as R  # noqa: E402

P = R.P
NPDT = {'F32': (np.float32, torch.float32), 'BF16': (np.int16, torch.bfloat16), 'F8_E4M3': (np.uint8, torch.float8_e4m3fn),
        'U8': (np.uint8, torch.uint8), 'I64': (np.int64, torch.int64), 'F16': (np.float16, torch.float16)}


class Safetensors:
    """Raw tensors (native dtype) of a local checkpoint directory (one file or an index), memory-mapped."""

    def __init__(self, root):
        self.root = Path(root)
        idx = self.root / 'model.safetensors.index.json'
        if idx.exists():
            self.map = json.loads(idx.read_text())['weight_map']
        else:
            self.map = {k: 'model.safetensors' for k in self._header('model.safetensors')}
        self.headers = {}

    def _header(self, file):
        with open(self.root / file, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            h = json.loads(f.read(n))
        h.pop('__metadata__', None)
        for v in h.values():
            v['base'] = 8 + n
        return h

    def __contains__(self, name):
        return name in self.map

    def keys(self):
        return self.map.keys()

    def __getitem__(self, name):
        file = self.map[name]
        if file not in self.headers:
            self.headers[file] = self._header(file)
        e = self.headers[file][name]
        npdt, tdt = NPDT[e['dtype']]
        a, b = e['data_offsets']
        arr = np.fromfile(self.root / file, dtype=npdt, count=(b - a) // np.dtype(npdt).itemsize, offset=e['base'] + a)
        t = torch.from_numpy(arr)
        if tdt in (torch.bfloat16, torch.float8_e4m3fn):
            t = t.view(tdt)
        return t.reshape(e['shape'])


class Source:
    """name -> f32 (dense) and raw (experts) access over a dict or Safetensors."""

    def __init__(self, src):
        self.src = src

    def raw(self, name):
        return self.src[name]

    def __call__(self, name):
        return self.src[name].float()

    def has(self, name):
        return name in self.src


def proj(out, name, w, fp8):
    if fp8:
        out[name], out[name + '_s'] = RP.to_fp8_rows(w.float())
    else:
        out[name] = dense(w)


def dense(w):
    return w.float() if IG.EXACT else w.to(torch.bfloat16)


def expert_tensors(get, l, g, fp8=True):
    """dequantised experts of layer l -> per-rank dicts {egu, egu_s, ed, ed_c} (quantised once for all ranks)."""
    gu, dn = R.dequant_experts(get.raw, l)                                             # [E, 2I, H] (gate rows first), [E, H, I]
    return experts_from_dense(gu, dn, g, fp8)


def experts_from_dense(gu, dn, g, fp8=True):
    """f32 gate|up [E, 2I, H] (gate rows first) and down [E, H, I] -> per-rank expert dicts."""
    dnt = dn.transpose(1, 2).contiguous()                                              # [E, I, H]: rows = input channels
    I = g.I
    if fp8:
        c, s = RP.to_fp8_rows(gu)
        dc_, ds = RP.to_fp8_rows(dnt)
    out = []
    for r in range(g.TP):
        er = slice(r * g.EIl, (r + 1) * g.EIl)
        ur = slice(I + er.start, I + er.stop)
        d = {}
        if fp8:
            d['egu'] = torch.cat((c[:, er], c[:, ur]), dim=1).contiguous()
            d['egu_s'] = torch.cat((s[:, er], s[:, ur]), dim=1).contiguous()
            d['ed'] = dc_[:, er].contiguous()
            d['ed_c'] = ds[:, er].contiguous()
        else:
            d['egu'] = dense(torch.cat((gu[:, er], gu[:, ur]), dim=1))
            d['ed'] = dense(dnt[:, er])
        out.append(d)
    return out


def shard_layer(get, l, rank, g, fp8=True, experts=None):
    """-> {name: tensor} of layer l for one rank (inkling_graphs.weight_shapes names; the union over the layer's visit
    kinds). experts: precomputed expert_tensors(...)[rank] or None (computed here)."""
    c = g.c
    pre = f'{P}layers.{l}.'
    a = pre + 'attn.'
    swa = c.swa[l]
    kind = 's' if swa else 'g'
    hd, CK, K = g.hq * g.D, g.CK, g.K
    t = {}
    qr = slice(rank * hd, (rank + 1) * hd)
    proj(t, 'q', get(a + 'wq_du.weight')[qr], fp8)
    proj(t, 'gt', get(a + 'gate_attn.weight')[qr], fp8)
    kr = slice(rank * CK, (rank + 1) * CK)
    proj(t, 'k', get(a + 'wk_dv.weight')[kr], fp8)
    proj(t, 'v', get(a + 'wv_dv.weight')[kr], fp8)
    proj(t, 'o', get(a + 'wo_ud.weight')[:, qr], fp8)
    rr = slice(rank * g.hq * g.d_rel, (rank + 1) * g.hq * g.d_rel)
    proj(t, 'r', get(a + 'wr_du.weight')[rr], fp8)
    hr = slice(rank * g.hq, (rank + 1) * g.hq)
    t['sk'] = dense(get(a + 'attn_sink_proj.weight')[hr])
    t['qs'] = get(a + 'q_l2_clip.scale')[hr].reshape(-1).contiguous()
    t['rs'] = get(a + 'rel_l2_clip.scale')[hr].reshape(-1).contiguous()
    t['rp'] = get(a + 'rel_logits_proj.proj').contiguous()
    if swa:
        t['kc'] = get(a + 'k_sconv.weight').reshape(-1, K)[kr].contiguous()
        t['vc'] = get(a + 'v_sconv.weight').reshape(-1, K)[kr].contiguous()
        t['qn'] = get(pre + 'attn_norm.weight')
    else:
        # dynamic conv: static_w [K, C] (row 0 = current token) -> [C, K] with column K-1 = current (the static conv's
        # order); weight_proj rows (head, lag) -> this rank's heads, lag order reversed likewise
        for kv in 'kv':
            sw = get(a + f'{kv}_sconv.static_w')[:, kr]                                 # [K, CK]
            t[f'{kv}c'] = torch.flip(sw, dims=[0]).t().contiguous()
            wp = get(a + f'{kv}_sconv.weight_proj').reshape(g.kvh, K, g.H)[rank * g.kvl:(rank + 1) * g.kvl]
            t[f'{kv}w'] = dense(torch.flip(wp, dims=[1]).reshape(g.kvl * K, g.H))
        t['qn'] = get(pre + 'q_input_norm.weight')
        t['kn'] = get(pre + 'k_input_norm.weight')
        t['vn'] = get(pre + 'v_input_norm.weight')
    t['ac'] = get(pre + 'attn_sconv.weight').reshape(g.H, K).contiguous()
    t['mc'] = get(pre + 'mlp_sconv.weight').reshape(g.H, K).contiguous()
    t['mn'] = get(pre + 'mlp_norm.weight')
    nq = 1 if swa else 3
    t['aq'] = get(f'{P}attn_res.attn_queries.{l - 1}').reshape(nq, g.H) if l > 0 else torch.zeros(nq, g.H)
    t['mq'] = get(f'{P}attn_res.mlp_queries.{l}').reshape(1, g.H)
    if c.is_loop(l):
        ll = l - c.ls
        qa, qkv = get(f'{P}middle_cycle_cross_pass_queries'), get(f'{P}middle_cycle_cross_pass_kv_queries')
        t['mg'] = torch.stack([qa[2 * ll], qkv[2 * ll], qkv[2 * ll + 1], qa[2 * ll + 1]])
        for x, w in (('q', 'attn'), ('k', 'k'), ('v', 'v'), ('m', 'mlp')):
            b = pre + f'controllers.{w}.'
            t[f'gd_{x}'] = dense(get(b + 'down_weight'))
            t[f'gf_{x}'] = dense(get(b + 'forget_weight'))
            t[f'gg_{x}'] = dense(get(b + 'gain_weight'))
    t['rw'] = get(pre + 'mlp.gate.weight').contiguous()
    t['rb'] = get(pre + 'mlp.gate.bias').contiguous()
    t.update(experts if experts is not None else expert_tensors(get, l, g, fp8)[rank])
    for k_ in list(t):
        if t[k_].dtype == torch.float32:
            t[k_] = t[k_].contiguous()
    return t


def shard_head(get, rank, g, fp8=True):
    w = get(f'{P}unembed.weight')[:g.Vu]
    part = torch.zeros(g.VS, g.H)
    lo, hi = rank * g.VS, min((rank + 1) * g.VS, g.Vu)
    part[:hi - lo] = w[lo:hi]
    out = {'oq': get(f'{P}attn_res.output_query').reshape(1, g.H), 'norm': get(f'{P}norm.weight')}
    if fp8:
        out['w'], out['w_s'] = RP.to_fp8_rows(part)
    else:
        out['w'] = dense(part)
    return out


def normed_embedding(get, g, chunk=16384):
    e = get.raw(f'{P}embed.weight')
    w = get(f'{P}embed_norm.weight')
    out = torch.empty(e.shape, dtype=torch.float32 if IG.EXACT else torch.bfloat16)
    for i in range(0, e.shape[0], chunk):
        out[i:i + chunk] = R.rms(e[i:i + chunk].float(), w, g.eps).to(out.dtype)
    return out


def visit_table(g):
    return [{'visit': v, 'layer': l, 'pass': p, 'kind': g.kind(v), 'ctl': g.ctl(v)} for v, (l, p) in enumerate(g.visits)]


_W = {}


def _layer_job(l):
    a, g, out = _W['a'], _W['g'], _W['out']
    mark = out / f'.done/L{l:03d}.json'
    if mark.exists():
        return l, json.loads(mark.read_text()), 0.0
    t0 = time.time()
    if 'get' not in _W:
        _W['get'] = Source(Safetensors(a.model))
        torch.set_num_threads(max(1, a.threads))
    get = _W['get']
    fp8 = a.weights == 'fp8'
    ex = expert_tensors(get, l, g, fp8)
    files = {}
    for r in range(g.TP):
        tensors = shard_layer(get, l, r, g, fp8, experts=ex[r])
        tmp = out / f'r{r}/L{l:03d}.bin.tmp'
        files[f'r{r}/L{l:03d}.bin'] = RP.write_file(tmp, tensors)
        tmp.rename(out / f'r{r}/L{l:03d}.bin')
    mark.parent.mkdir(parents=True, exist_ok=True)
    mark.with_suffix('.tmp').write_text(json.dumps(files))
    mark.with_suffix('.tmp').rename(mark)
    return l, files, time.time() - t0


def _init_worker(a, cfg, tp, out):
    _W.update(a=a, g=IG.Geo(cfg, tp), out=Path(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('out')
    ap.add_argument('--tp', type=int, default=2)
    ap.add_argument('--weights', default='fp8', choices=['fp8', 'bf16'])
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    cfg = json.loads((Path(a.model) / 'config.json').read_text())
    g = IG.Geo(cfg, a.tp)
    get = Source(Safetensors(a.model))
    fp8 = a.weights == 'fp8'
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    if not (out / '.done/embed.json').exists():
        files['embed.bin'] = RP.write_file(out / 'embed.bin', {'embed': normed_embedding(get, g)})
        for r in range(g.TP):
            files[f'r{r}/head.bin'] = RP.write_file(out / f'r{r}/head.bin', shard_head(get, r, g, fp8))
        (out / '.done').mkdir(exist_ok=True)
        (out / '.done/embed.json').write_text(json.dumps(files))
    else:
        files.update(json.loads((out / '.done/embed.json').read_text()))
    print('embed + heads packed', flush=True)
    todo = [int(x) for x in a.only.split(',')] if a.only else list(range(g.L))
    _W.update(a=a, g=g, out=out)
    if a.jobs > 1:
        with multiprocessing.get_context('spawn').Pool(a.jobs, initializer=_init_worker, initargs=(a, cfg, a.tp, str(out))) as pool:
            for l, fl, dt in pool.imap_unordered(_layer_job, todo):
                files.update(fl)
                print(f'layer {l} packed in {dt:.0f}s', flush=True)
    else:
        _W['get'] = get
        for l in todo:
            l, fl, dt = _layer_job(l)
            files.update(fl)
            print(f'layer {l} packed in {dt:.0f}s', flush=True)
    if a.only:
        print('partial run (--only): no prepared.json written')
        return
    manifest = {'format': 'inkling2-neuron-prepared', 'version': 1, 'geometry': g.meta(), 'weights': a.weights,
                'visits': visit_table(g), 'eos': g.eos, 'max_context': g.max_context, 'files': files,
                'source': str(Path(a.model).resolve()), 'config': cfg}
    (out / 'prepared.json').write_text(json.dumps(manifest))
    (out / 'config.json').write_text(json.dumps(cfg))
    for name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'generation_config.json'):
        if (Path(a.model) / name).exists():
            shutil.copyfile(Path(a.model) / name, out / name)
    # the checkpoint's chat template rewritten for the server's minja (no trailing dict comma, no float / tojson kwargs /
    # dictsort kwargs; same output: tests/test_inkling_tokenizer.py), the original kept next to it
    if (Path(a.model) / 'chat_template.jinja').exists():
        shutil.copyfile(Path(a.model) / 'chat_template.jinja', out / 'chat_template.orig.jinja')
    shutil.copyfile(Path(__file__).resolve().parent / 'inkling_chat_template.jinja', out / 'chat_template.jinja')
    print('wrote', out)


if __name__ == '__main__':
    main()
