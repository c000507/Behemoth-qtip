"""GLM-5.3 checkpoint -> per-rank prepared tensors for the Neuron GLM graphs (glm_graphs.py) and runtime (cpp/glm_model.cpp).

Reads the original checkpoint layout (model.safetensors.index.json, prefix model.language_model. or model.; BF16/F32,
block-128 FP8 with weight_scale_inv, or per-channel FP8 with weight_scale) and writes, for TP ranks and EP groups:

  OUT/prepared.json            geometry, parallel layout, layer kinds, every tensor's file / offset / dtype / shape
  OUT/embed.bin                bf16 [vocab, hidden] (the host looks up embeddings)
  OUT/r{rank}/L{layer:03d}.bin this rank's tensors of one layer: attn.* (kda | mla) and mlp.* (dense | moe),
                               names as glm_graphs.weight_shapes()
  OUT/r{rank}/head.bin         final norm and this rank's lm_head rows (vocab padded to a multiple of TP)

Projections become E4M3 with a per-output-row f32 scale (max |code| 224, the existing Neuron FP8 path's convention);
routed-expert down_proj is stored [E/EP, EI/ETP, hidden] with a per-input-channel scale (glm_graphs.moe_block folds the
experts into one contraction). WEIGHTS=bf16 keeps bf16 projections (exactness checks).

usage: glm_repack.py MODEL OUT [--tp N] [--ep M] [--weights fp8|bf16] [--layers N]
"""
import argparse
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glm_graphs as G

DT = {'F32': (np.float32, 4), 'BF16': (np.uint16, 2), 'F16': (np.float16, 2), 'F8_E4M3': (np.uint8, 1), 'U8': (np.uint8, 1),
      'I64': (np.int64, 8)}
NAMES = {torch.float32: 'f32', torch.bfloat16: 'bf16', G.F8: 'f8e4m3', torch.int64: 'i64'}


class Checkpoint:
    def __init__(self, root):
        self.root = Path(root)
        idx = self.root / 'model.safetensors.index.json'
        if idx.exists():
            self.map = json.loads(idx.read_text())['weight_map']
        else:
            self.map = {k: 'model.safetensors' for k in self.header('model.safetensors')}
        self.prefix = 'model.language_model.' if 'model.language_model.embed_tokens.weight' in self.map else 'model.'
        self.headers = {}

    def header(self, file):
        with open(self.root / file, 'rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            h = json.loads(f.read(n))
        h.pop('__metadata__', None)
        for v in h.values():
            v['base'] = 8 + n
        return h

    def has(self, name):
        return name in self.map

    def raw(self, name):
        file = self.map[name]
        if file not in self.headers:
            self.headers[file] = self.header(file)
        e = self.headers[file][name]
        dt, size = DT[e['dtype']]
        a, b = e['data_offsets']
        arr = np.memmap(self.root / file, dtype=dt, mode='r', offset=e['base'] + a, shape=((b - a) // size,))
        return e['dtype'], e['shape'], arr

    def tensor(self, name):
        """-> torch f32 of the stored shape."""
        dtype, shape, arr = self.raw(name)
        if dtype == 'BF16':
            t = torch.from_numpy((arr.astype(np.uint32) << 16).view(np.float32))
        elif dtype == 'F8_E4M3':
            t = torch.from_numpy(np.array(arr)).view(G.F8).float()
        else:
            t = torch.from_numpy(np.array(arr, dtype=np.float32))
        return t.reshape(shape)

    def linear(self, base):
        """Dense f32 [out, in] of a linear (base without .weight; plain tensors such as the indexer gate also accepted)."""
        if self.has(base) and not self.has(base + '.weight'):
            return self.tensor(base)
        if self.has(base + '.weight_packed'):
            raise SystemExit(f'{base}: NVFP4-packed checkpoints are not supported by the Neuron repacker; use the source checkpoint')
        w = self.tensor(base + '.weight')
        if self.has(base + '.weight_scale_inv'):                          # block-128 FP8 (original GLM checkpoint)
            s = self.tensor(base + '.weight_scale_inv')
            bo, bi = -(-w.shape[0] // s.shape[0]), -(-w.shape[1] // s.shape[1])
            w = w * s.repeat_interleave(bo, 0)[:w.shape[0]].repeat_interleave(bi, 1)[:, :w.shape[1]]
        elif self.has(base + '.weight_scale'):                            # per-output-channel FP8
            w = w * self.tensor(base + '.weight_scale').reshape(-1, 1)
        return w


def to_fp8_rows(w):
    """f32 [..., out, in] -> (E4M3 codes, f32 scale [..., out]) with max |code| 224."""
    s = torch.amax(torch.abs(w), dim=-1).clamp(min=1e-30) / 224.0
    return (w / s.unsqueeze(-1)).to(G.F8), s.float()


def bf16(w):
    return w.to(torch.bfloat16)


class Packer:
    def __init__(self, ck, g, fp8=True):
        self.ck, self.g, self.fp8 = ck, g, fp8

    def proj(self, out, name, w):
        if self.fp8:
            out[name], out[name + '_s'] = to_fp8_rows(w)
        else:
            out[name] = bf16(w)

    def hc(self, out, p, which, norm):
        ck = self.ck
        out['hc_fn'] = ck.tensor(p + f'hc_{which}_fn').float()
        out['hc_base'] = ck.tensor(p + f'hc_{which}_base').float().reshape(-1)
        out['hc_scale'] = ck.tensor(p + f'hc_{which}_scale').float().reshape(-1)
        out['norm'] = ck.tensor(p + norm).float()

    def layer(self, l, rank):
        """-> {'attn.<name>': tensor, 'mlp.<name>': tensor} for one rank."""
        g, ck = self.g, self.ck
        p = ck.prefix + f'layers.{l}.'
        attn, mlp = {}, {}
        self.hc(attn, p, 'attn', 'input_layernorm.weight')
        self.hc(mlp, p, 'ffn', 'post_attention_layernorm.weight')
        a = p + 'self_attn.'
        if g.attn[l] == 'linear_attention':
            W, hl = g.wl, g.hl
            rows = slice(rank * W, (rank + 1) * W)
            for x in 'qkv':
                self.proj(attn, x, ck.linear(a + f'{x}_proj')[rows])
            attn['conv'] = torch.stack([ck.tensor(a + f'{x}_conv1d.weight').float().reshape(g.lheads * g.ldim, g.conv)[rows] for x in 'qkv'])
            self.proj(attn, 'fa', ck.linear(a + 'f_a_proj'))
            self.proj(attn, 'fb', ck.linear(a + 'f_b_proj')[rows])
            self.proj(attn, 'b', ck.linear(a + 'b_proj')[rank * hl:(rank + 1) * hl])
            self.proj(attn, 'ga', ck.linear(a + 'g_a_proj'))
            self.proj(attn, 'gb', ck.linear(a + 'g_b_proj')[rows])
            self.proj(attn, 'o', ck.linear(a + 'o_proj')[:, rows])
            attn['alog'] = ck.tensor(a + 'A_log').float().reshape(-1)[rank * hl:(rank + 1) * hl].contiguous()
            attn['dt'] = ck.tensor(a + 'dt_bias').float().reshape(-1)[rows].contiguous()
            attn['onorm'] = ck.tensor(a + 'o_norm.weight').float()
        else:
            ml = g.ml
            self.proj(attn, 'qa', ck.linear(a + 'q_a_proj'))
            attn['qa_norm'] = ck.tensor(a + 'q_a_layernorm.weight').float()
            self.proj(attn, 'qb', ck.linear(a + 'q_b_proj')[rank * ml * g.qdim:(rank + 1) * ml * g.qdim])
            self.proj(attn, 'kva', ck.linear(a + 'kv_a_proj_with_mqa'))
            attn['kva_norm'] = ck.tensor(a + 'kv_a_layernorm.weight').float()
            hd = g.qdim + g.vdim
            attn['kvb'] = bf16(ck.linear(a + 'kv_b_proj')[rank * ml * hd:(rank + 1) * ml * hd])
            self.proj(attn, 'o', ck.linear(a + 'o_proj')[:, rank * ml * g.vdim:(rank + 1) * ml * g.vdim])
            ix = a + 'indexer.'
            self.proj(attn, 'iq', ck.linear(ix + 'wq_b'))
            self.proj(attn, 'ik', ck.linear(ix + 'wk'))
            self.proj(attn, 'igate', ck.linear(ix + 'index_kpool_compress_gate'))
            self.proj(attn, 'iw', ck.linear(ix + 'weights_proj'))
            attn['ikn_w'] = ck.tensor(ix + 'k_norm.weight').float()
            attn['ikn_b'] = ck.tensor(ix + 'k_norm.bias').float()
            attn['ape'] = ck.tensor(ix + 'index_kpool_compress_ape').float().reshape(g.pool, g.idim)
        m = p + 'mlp.'
        if g.mlp[l] == 'dense':
            rows = slice(rank * g.ffl, (rank + 1) * g.ffl)
            self.proj(mlp, 'g', ck.linear(m + 'gate_proj')[rows])
            self.proj(mlp, 'u', ck.linear(m + 'up_proj')[rows])
            self.proj(mlp, 'd', ck.linear(m + 'down_proj')[:, rows])
        else:
            rows = slice(rank * g.SIl, (rank + 1) * g.SIl)
            self.proj(mlp, 'sg', ck.linear(m + 'shared_experts.gate_proj')[rows])
            self.proj(mlp, 'su', ck.linear(m + 'shared_experts.up_proj')[rows])
            self.proj(mlp, 'sd', ck.linear(m + 'shared_experts.down_proj')[:, rows])
            mlp['rw'] = ck.linear(m + 'gate').float()
            mlp['rb'] = ck.tensor(m + 'gate.e_score_correction_bias').float().reshape(-1)
            ids = g.expert_ids(rank)
            mlp['eids'] = torch.tensor(ids, dtype=torch.int64)
            j = rank % g.ETP
            er = slice(j * g.EIl, (j + 1) * g.EIl)
            gu = torch.stack([torch.cat((ck.linear(m + f'experts.{e}.gate_proj')[er], ck.linear(m + f'experts.{e}.up_proj')[er])) for e in ids])
            dn = torch.stack([ck.linear(m + f'experts.{e}.down_proj')[:, er].T.contiguous() for e in ids])   # [El, EIl, H]
            if self.fp8:
                mlp['egu'], mlp['egu_s'] = to_fp8_rows(gu)
                mlp['ed'], mlp['ed_c'] = to_fp8_rows(dn)      # rows of the transposed down = its input channels
            else:
                mlp['egu'], mlp['ed'] = bf16(gu), bf16(dn)
        return {**{'attn.' + k: v for k, v in attn.items()}, **{'mlp.' + k: v for k, v in mlp.items()}}

    def head(self, rank):
        g, ck = self.g, self.ck
        w = ck.linear('lm_head') if ck.has('lm_head.weight') else ck.linear(ck.prefix + 'lm_head')
        n = g.VP // g.TP
        part = torch.zeros(n, g.H)
        lo, hi = rank * n, min((rank + 1) * n, g.V)
        if hi > lo:
            part[:hi - lo] = w[lo:hi]
        out = {'norm': ck.tensor(ck.prefix + 'norm.weight').float()}
        if self.fp8:
            out['w'], out['w_s'] = to_fp8_rows(part)
        else:
            out['w'] = bf16(part)
        return out

    def embed(self):
        return self.ck.tensor(self.ck.prefix + 'embed_tokens.weight')


def tensor_bytes(t):
    if t.dtype == G.F8:
        return t.contiguous().view(torch.uint8).numpy().tobytes()
    if t.dtype == torch.bfloat16:
        return t.contiguous().view(torch.int16).numpy().tobytes()
    return t.contiguous().numpy().tobytes()


def write_file(path, tensors):
    entries, off = [], 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        for name, t in tensors.items():
            b = tensor_bytes(t)
            entries.append({'name': name, 'dtype': NAMES[t.dtype], 'shape': list(t.shape), 'offset': off, 'bytes': len(b)})
            f.write(b)
            pad = (-len(b)) % 64
            f.write(b'\0' * pad)
            off += len(b) + pad
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('out')
    ap.add_argument('--tp', type=int, default=2)
    ap.add_argument('--ep', type=int, default=None)
    ap.add_argument('--weights', default='fp8', choices=['fp8', 'bf16'])
    ap.add_argument('--layers', type=int, default=None, help='keep only the first N layers (reduced-size validation)')
    a = ap.parse_args()
    cfg = json.loads((Path(a.model) / 'config.json').read_text())
    if a.layers:
        cfg = truncate_config(cfg, a.layers)
    g = G.Geo(cfg, a.tp, a.ep)
    ck = Checkpoint(a.model)
    pk = Packer(ck, g, a.weights == 'fp8')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    emb = pk.embed()
    files['embed.bin'] = write_file(out / 'embed.bin', {'embed': bf16(emb)})
    for r in range(g.TP):
        files[f'r{r}/head.bin'] = write_file(out / f'r{r}/head.bin', pk.head(r))
    for l in range(g.L):
        for r in range(g.TP):
            files[f'r{r}/L{l:03d}.bin'] = write_file(out / f'r{r}/L{l:03d}.bin', pk.layer(l, r))
        print(f'layer {l} ({g.attn[l]}, {g.mlp[l]}) packed', flush=True)
    kinds = [{'attn': 'kda' if g.attn[l] == 'linear_attention' else 'mla', 'mlp': g.mlp[l]} for l in range(g.L)]
    manifest = {'format': 'glm53-neuron-prepared', 'version': 1, 'geometry': g.meta(), 'weights': a.weights,
                'layers': kinds, 'eos': g.eos, 'max_context': g.max_context, 'files': files, 'source': str(Path(a.model).resolve()),
                'config': cfg}
    (out / 'prepared.json').write_text(json.dumps(manifest))
    # the server loads the tokenizer and chat template from the same directory
    (out / 'config.json').write_text(json.dumps(cfg))
    for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'special_tokens_map.json', 'generation_config.json'):
        if (Path(a.model) / name).exists():
            shutil.copyfile(Path(a.model) / name, out / name)
    print('wrote', out)


def truncate_config(cfg, n):
    cfg = json.loads(json.dumps(cfg))
    c = cfg.get('text_config', cfg)
    c['num_hidden_layers'] = n
    for k in ('layer_types', 'mlp_layer_types', 'indexer_types'):
        if c.get(k):
            c[k] = c[k][:n]
    la = c.get('linear_attn_config', {})
    for k in ('kda_layers', 'full_attn_layers'):
        if k in la:
            la[k] = [i for i in la[k] if i < n]
    return cfg


if __name__ == '__main__':
    main()
