"""Qwen3.5 MoE checkpoint -> per-rank prepared tensors for the Neuron graphs (qwen35_graphs.py) and runtime.

Reads the HF layout (model.safetensors.index.json or a single model.safetensors; prefix model.language_model. or model.;
BF16 / F32; fused expert tensors experts.gate_up_proj [E, 2I, H] and experts.down_proj [E, H, I]) and writes, as
glm_repack.py does for GLM (same file format, so the GLM loader code applies):

  OUT/prepared.json            geometry, layer kinds, every tensor's file / offset / dtype / shape
  OUT/embed.bin                bf16 [vocab, hidden] (the host looks up embeddings)
  OUT/r{rank}/L{layer:03d}.bin this rank's tensors of one layer: attn.* (gdn | gqa) and mlp.* (moe), names as
                               qwen35_graphs.weight_shapes()
  OUT/r{rank}/head.bin         final norm and this rank's lm_head rows (vocab padded to a multiple of TP)

Projections become E4M3 with a per-output-row f32 scale (max |code| 224, the stack's FP8 convention); expert down_proj is
stored [E, I/TP, H] with a per-input-channel scale (the graph folds the gathered experts into one contraction). Norm
weights are stored as (1 + w) (Qwen's zero-centred RMSNorm) except linear_attn.norm (plain). --weights bf16 keeps bf16.
`shard_layer(get, l, rank, g, fp8)` is the sharding used by the CPU tests on random weights (get = HF name -> f32 tensor).

usage: qwen35_prepare.py MODEL OUT [--tp N] [--weights fp8|bf16] [--layers N] [--jobs J --threads T] [--only l,l,..]
pack_layer() quantises the experts once for all ranks (resumable per layer); see docs/neuron-qwen397.md for TP = 32.
MTP (`mtp.*`) and vision (`model.visual.*`) tensors are ignored. One layer at a time in RAM (~0.9 GB bf16 at E = 128).
"""
import argparse
import json
import multiprocessing
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glm_repack as RP  # noqa: E402  (Checkpoint reader, to_fp8_rows, bf16, write_file, tensor_bytes)
import qwen35_graphs as Q  # noqa: E402


def proj(out, name, w, fp8):
    if fp8:
        out[name], out[name + '_s'] = RP.to_fp8_rows(w)
    else:
        out[name] = store(w)


def store(w):
    """bf16 projection, or f32 when the CPU tests run the graphs in EXACT mode (no rounding anywhere)."""
    return w.float() if Q.EXACT else RP.bf16(w)


def shard_layer(get, l, rank, g, fp8=True, experts=True):
    """-> {'attn.<name>': tensor, 'mlp.<name>': tensor} of layer l for one rank (names / shapes = qwen35_graphs.weight_shapes)."""
    p = f'layers.{l}.'
    attn, mlp = {}, {}
    attn['norm'] = 1.0 + get(p + 'input_layernorm.weight').float()
    mlp['norm'] = 1.0 + get(p + 'post_attention_layernorm.weight').float()
    if g.kind(l) == 'gdn':
        a = p + 'linear_attn.'
        kd, vd = g.Hk * g.Dk, g.Hv * g.Dv
        qr = slice(g.key_of(rank) * g.Dk, (g.key_of(rank) + g.hk) * g.Dk)           # replicated key head when Hk < TP
        vr = slice(rank * g.Wv, (rank + 1) * g.Wv)
        rows = torch.cat((torch.arange(qr.start, qr.stop), kd + torch.arange(qr.start, qr.stop), 2 * kd + torch.arange(vr.start, vr.stop)))
        proj(attn, 'qkv', get(a + 'in_proj_qkv.weight')[rows], fp8)
        attn['conv'] = get(a + 'conv1d.weight').float().reshape(2 * kd + vd, g.K)[rows].contiguous()
        proj(attn, 'z', get(a + 'in_proj_z.weight')[vr], fp8)
        hr = slice(rank * g.hl, (rank + 1) * g.hl)
        proj(attn, 'b', get(a + 'in_proj_b.weight')[hr], fp8)
        proj(attn, 'a', get(a + 'in_proj_a.weight')[hr], fp8)
        proj(attn, 'o', get(a + 'out_proj.weight')[:, vr], fp8)
        attn['alog'] = get(a + 'A_log').float().reshape(-1)[hr].contiguous()
        attn['dt'] = get(a + 'dt_bias').float().reshape(-1)[hr].contiguous()
        attn['onorm'] = get(a + 'norm.weight').float()
    else:
        a = p + 'self_attn.'
        per = g.D * (2 if g.gate else 1)
        proj(attn, 'q', get(a + 'q_proj.weight')[rank * g.hq * per:(rank + 1) * g.hq * per], fp8)
        kr = slice(g.kv_of(rank) * g.D, (g.kv_of(rank) + g.kvl) * g.D)             # replicated KV head when kv_heads < TP
        proj(attn, 'k', get(a + 'k_proj.weight')[kr], fp8)
        proj(attn, 'v', get(a + 'v_proj.weight')[kr], fp8)
        proj(attn, 'o', get(a + 'o_proj.weight')[:, rank * g.hq * g.D:(rank + 1) * g.hq * g.D], fp8)
        attn['qn'] = 1.0 + get(a + 'q_norm.weight').float()
        attn['kn'] = 1.0 + get(a + 'k_norm.weight').float()
    m = p + 'mlp.'
    sr = slice(rank * g.SIl, (rank + 1) * g.SIl)
    proj(mlp, 'sg', get(m + 'shared_expert.gate_proj.weight')[sr], fp8)
    proj(mlp, 'su', get(m + 'shared_expert.up_proj.weight')[sr], fp8)
    proj(mlp, 'sd', get(m + 'shared_expert.down_proj.weight')[:, sr], fp8)
    mlp['sgate'] = get(m + 'shared_expert_gate.weight').float().reshape(-1)
    mlp['rw'] = get(m + 'gate.weight').float()
    if not experts:                                                                       # pack_layer adds them (quantised once for all ranks)
        return {**{'attn.' + k: v for k, v in attn.items()}, **{'mlp.' + k: v for k, v in mlp.items()}}
    er = slice(rank * g.EIl, (rank + 1) * g.EIl)
    gu_all = get(m + 'experts.gate_up_proj').float()                                    # [E, 2I, H]
    gu = torch.cat((gu_all[:, er], gu_all[:, g.I + er.start:g.I + er.stop]), dim=1)      # [E, 2 EIl, H]
    dn = get(m + 'experts.down_proj').float()[:, :, er].transpose(1, 2).contiguous()      # [E, EIl, H]
    if fp8:
        mlp['egu'], mlp['egu_s'] = RP.to_fp8_rows(gu)
        mlp['ed'], mlp['ed_c'] = RP.to_fp8_rows(dn)                                       # rows of the transposed down = its input channels
    else:
        mlp['egu'], mlp['ed'] = store(gu), store(dn)
    return {**{'attn.' + k: v for k, v in attn.items()}, **{'mlp.' + k: v for k, v in mlp.items()}}


def shard_head(get, rank, g, fp8=True):
    w = get('lm_head.weight').float()
    n = g.VP // g.TP
    part = torch.zeros(n, g.H)
    lo, hi = rank * n, min((rank + 1) * n, g.V)
    if hi > lo:
        part[:hi - lo] = w[lo:hi]
    out = {'norm': 1.0 + get('norm.weight').float()}
    if fp8:
        out['w'], out['w_s'] = RP.to_fp8_rows(part)
    else:
        out['w'] = store(part)
    return out


class Getter:
    """HF name (without the model prefix; 'lm_head.weight' as is) -> f32 tensor, from a glm_repack.Checkpoint or a dict."""

    def __init__(self, source):
        self.src = source
        if isinstance(source, dict):
            self.prefix = 'model.language_model.' if any(k.startswith('model.language_model.') for k in source) else 'model.'
        else:
            self.prefix = source.prefix

        self.cache = None                                                              # dict: memoise (pack_layer, many ranks)

    def __call__(self, name):
        if self.cache is not None and name in self.cache:
            return self.cache[name]
        full = name if name.startswith('lm_head') else self.prefix + name
        t = self.src[full].float() if isinstance(self.src, dict) else self.src.tensor(full)
        if self.cache is not None:
            self.cache[name] = t
        return t

    def chunk(self, name, e0, e1):
        """f32 [e1 - e0, ...] of a 3-D expert tensor without materialising the whole tensor (13 GB bf16 per layer at E = 512)."""
        full = self.prefix + name
        if isinstance(self.src, dict):
            return self.src[full][e0:e1].float()
        dtype, shape, arr = self.src.raw(full)
        per = 1
        for d in shape[1:]:
            per *= d
        a = np.array(arr[e0 * per:e1 * per])                                           # a private copy (the memmap is read-only)
        if dtype == 'BF16':
            t = torch.from_numpy(a.view(np.int16)).view(torch.bfloat16).float()
        else:
            t = torch.from_numpy(a.astype(np.float32, copy=False)).float()
        return t.reshape(e1 - e0, *shape[1:])


def pack_layer(get, l, g, fp8=True, chunk=32):
    """-> [rank] dicts equal to shard_layer(get, l, rank, g, fp8), computed once for all ranks: the expert tensors are
    converted chunk by chunk and (fp8) quantised once - the scales are per gate/up output row and per down input channel,
    so they do not depend on the rank's slice - then sliced along the intermediate axis. This is what makes TP = 32 over
    512 experts (6.4 B expert parameters per layer) a matter of seconds per layer instead of 32 full passes."""
    get.cache = {}
    ranks = [shard_layer(get, l, r, g, fp8, experts=False) for r in range(g.TP)]
    get.cache = None
    m = f'layers.{l}.mlp.experts.'
    E, I, H = g.E, g.I, g.H
    if fp8:
        gu, gs = torch.empty((E, 2 * I, H), dtype=Q.F8), torch.empty((E, 2 * I), dtype=torch.float32)
        dn, dc = torch.empty((E, I, H), dtype=Q.F8), torch.empty((E, I), dtype=torch.float32)
    else:
        dt = torch.float32 if Q.EXACT else torch.bfloat16
        gu, dn = torch.empty((E, 2 * I, H), dtype=dt), torch.empty((E, I, H), dtype=dt)
    for e0 in range(0, E, chunk):
        e1 = min(E, e0 + chunk)
        a = get.chunk(m + 'gate_up_proj', e0, e1)                                          # [e, 2I, H]
        b = get.chunk(m + 'down_proj', e0, e1).transpose(1, 2).contiguous()                # [e, I, H]: rows = input channels
        if fp8:
            gu[e0:e1], gs[e0:e1] = RP.to_fp8_rows(a)
            dn[e0:e1], dc[e0:e1] = RP.to_fp8_rows(b)
        else:
            gu[e0:e1], dn[e0:e1] = store(a), store(b)
    for r in range(g.TP):
        er = slice(r * g.EIl, (r + 1) * g.EIl)
        ur = slice(I + er.start, I + er.stop)
        ranks[r]['mlp.egu'] = torch.cat((gu[:, er], gu[:, ur]), dim=1)
        if fp8:
            ranks[r]['mlp.egu_s'] = torch.cat((gs[:, er], gs[:, ur]), dim=1)
        ranks[r]['mlp.ed'] = dn[:, er].contiguous()
        if fp8:
            ranks[r]['mlp.ed_c'] = dc[:, er].contiguous()
    return ranks


def pack_heads(get, g, fp8=True):
    get.cache = {}
    out = [shard_head(get, r, g, fp8) for r in range(g.TP)]
    get.cache = None
    return out


_W = {}


def _layer_job(l):
    """worker: one layer -> its per-rank files; returns the manifest entries (resumable: a .done marker per layer)."""
    a, g, out = _W['a'], _W['g'], _W['out']
    mark = out / f'.done/L{l:03d}.json'
    if mark.exists():
        return l, json.loads(mark.read_text()), 0.0
    t0 = time.time()
    if 'get' not in _W:
        _W['get'] = Getter(RP.Checkpoint(a.model))
        torch.set_num_threads(max(1, a.threads))
    ranks = pack_layer(_W['get'], l, g, a.weights == 'fp8')
    files = {}
    for r, tensors in enumerate(ranks):
        tmp = out / f'r{r}/L{l:03d}.bin.tmp'
        files[f'r{r}/L{l:03d}.bin'] = RP.write_file(tmp, tensors)
        tmp.rename(out / f'r{r}/L{l:03d}.bin')
    mark.parent.mkdir(parents=True, exist_ok=True)
    mark.with_suffix('.tmp').write_text(json.dumps(files))
    mark.with_suffix('.tmp').rename(mark)
    return l, files, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('out')
    ap.add_argument('--tp', type=int, default=2)
    ap.add_argument('--weights', default='fp8', choices=['fp8', 'bf16'])
    ap.add_argument('--jobs', type=int, default=1, help='layers packed in parallel (worker processes)')
    ap.add_argument('--threads', type=int, default=8, help='torch threads per worker')
    ap.add_argument('--only', default='', help='comma-separated layers (partial run for validation; no prepared.json)')
    ap.add_argument('--layers', type=int, default=None, help='keep only the first N layers (reduced-size validation)')
    a = ap.parse_args()
    cfg = json.loads((Path(a.model) / 'config.json').read_text())
    if a.layers:
        cfg = truncate_config(cfg, a.layers)
    g = Q.Geo(cfg, a.tp)
    get = Getter(RP.Checkpoint(a.model))
    fp8 = a.weights == 'fp8'
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {'embed.bin': RP.write_file(out / 'embed.bin', {'embed': RP.bf16(get('embed_tokens.weight'))})}
    for r, tensors in enumerate(pack_heads(get, g, fp8)):
        files[f'r{r}/head.bin'] = RP.write_file(out / f'r{r}/head.bin', tensors)
    print('embed + heads packed', flush=True)
    todo = [int(x) for x in a.only.split(',')] if a.only else list(range(g.L))
    _W.update(a=a, g=g, out=out)
    if a.jobs > 1:
        with multiprocessing.get_context('fork').Pool(a.jobs) as pool:
            for l, fl, dt in pool.imap_unordered(_layer_job, todo):
                files.update(fl)
                print(f'layer {l} ({g.kind(l)}, moe) packed in {dt:.0f}s', flush=True)
    else:
        _W['get'] = get
        for l in todo:
            l, fl, dt = _layer_job(l)
            files.update(fl)
            print(f'layer {l} ({g.kind(l)}, moe) packed in {dt:.0f}s', flush=True)
    if a.only:
        print('partial run (--only): no prepared.json written')
        return
    kinds = [{'attn': g.kind(l), 'mlp': 'moe'} for l in range(g.L)]
    manifest = {'format': 'qwen35-moe-neuron-prepared', 'version': 1, 'geometry': g.meta(), 'weights': a.weights, 'layers': kinds,
                'eos': g.eos, 'max_context': g.max_context, 'files': files, 'source': str(Path(a.model).resolve()), 'config': cfg}
    (out / 'prepared.json').write_text(json.dumps(manifest))
    (out / 'config.json').write_text(json.dumps(cfg))
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'special_tokens_map.json', 'generation_config.json'):
        if (Path(a.model) / name).exists():
            shutil.copyfile(Path(a.model) / name, out / name)
    print('wrote', out)


def truncate_config(cfg, n):
    cfg = json.loads(json.dumps(cfg))
    c = cfg.get('text_config', cfg)
    c['num_hidden_layers'] = n
    c['layer_types'] = c['layer_types'][:n]
    return cfg


if __name__ == '__main__':
    main()
