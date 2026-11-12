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

usage: qwen35_prepare.py MODEL OUT [--tp N] [--weights fp8|bf16] [--layers N]
MTP (`mtp.*`) and vision (`model.visual.*`) tensors are ignored. One layer at a time in RAM (~0.9 GB bf16 at E = 128).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

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


def shard_layer(get, l, rank, g, fp8=True):
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

    def __call__(self, name):
        full = name if name.startswith('lm_head') else self.prefix + name
        if isinstance(self.src, dict):
            return self.src[full].float()
        return self.src.tensor(full)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('out')
    ap.add_argument('--tp', type=int, default=2)
    ap.add_argument('--weights', default='fp8', choices=['fp8', 'bf16'])
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
    for r in range(g.TP):
        files[f'r{r}/head.bin'] = RP.write_file(out / f'r{r}/head.bin', shard_head(get, r, g, fp8))
    for l in range(g.L):
        for r in range(g.TP):
            files[f'r{r}/L{l:03d}.bin'] = RP.write_file(out / f'r{r}/L{l:03d}.bin', shard_layer(get, l, r, g, fp8))
        print(f'layer {l} ({g.kind(l)}, moe) packed', flush=True)
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
