"""Repack a compressed-tensors NVFP4 checkpoint (llm-compressor `nvfp4-pack-quantized`, e.g. a stock HF Llama / Mistral NVFP4 export)
into the per-layer layout the Neuron runtime loads: model-embed.safetensors (embed_tokens, lm_head, final norm) and
model-layer-NNNNN.safetensors (norms + projections), plus a config.json in the QTIP-hfized llama form the loader parses.

Tensor bytes are copied as stored, except the global scale: compressed-tensors dequantises as code * block_scale / weight_global_scale,
the runtime as code * block_scale * weight_global_scale, so the repacked file stores the reciprocal. input_global_scale is dropped
(weights-only on this backend). Only bias-free SiLU llama-shaped models without a sliding window are accepted (Mistral Small 3.x is one).

usage: python repack_compressed_tensors.py SRC_DIR DST_DIR
"""
import json
from pathlib import Path
import shutil
import struct
import sys

import numpy as np

PROJ = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')
QUIP = {'K': 2, 'L': 16, 'V': 2, 'codebook': 'bitshift', 'codebook_version': 0, 'decode_mode': 'quantlut_sym', 'skip_list': None,
        'split_for_tp': False, 'td_x': 16, 'td_y': 16, 'tlut_bits': 9}
COPY = ('generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja', 'README.md')


class Source:
    def __init__(self, root):
        self.root = root
        self.index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
        self.headers = {}

    def _header(self, file):
        if file not in self.headers:
            with open(self.root / file, 'rb') as f:
                n = struct.unpack('<Q', f.read(8))[0]
                self.headers[file] = (8 + n, json.loads(f.read(n)))
        return self.headers[file]

    def has(self, name):
        return name in self.index

    def get(self, name):
        """(dtype, shape, raw bytes)"""
        file = self.index[name]
        start, h = self._header(file)
        e = h[name]
        a, b = e['data_offsets']
        with open(self.root / file, 'rb') as f:
            f.seek(start + a)
            return e['dtype'], e['shape'], f.read(b - a)


def write(path, tensors):
    """tensors: list of (name, dtype, shape, bytes) -> safetensors file (sorted, contiguous)."""
    header, offset = {}, 0
    for name, dtype, shape, raw in sorted(tensors):
        header[name] = {'dtype': dtype, 'shape': list(shape), 'data_offsets': [offset, offset + len(raw)]}
        offset += len(raw)
    header['__metadata__'] = {'format': 'pt'}
    h = json.dumps(header, separators=(',', ':')).encode()
    h += b' ' * (-len(h) % 8)
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(h)))
        f.write(h)
        for _, _, _, raw in sorted(tensors):
            f.write(raw)


def main(src, dst):
    src, dst = Path(src), Path(dst)
    c = json.loads((src / 'config.json').read_text())
    q = c.get('quantization_config') or {}
    if q.get('quant_method') != 'compressed-tensors' or q.get('format') != 'nvfp4-pack-quantized':
        raise SystemExit('not a compressed-tensors nvfp4-pack-quantized checkpoint')
    if c['model_type'] not in ('llama', 'mistral') or c.get('hidden_act') != 'silu' or c.get('tie_word_embeddings') \
            or c.get('sliding_window') is not None or c.get('attention_bias') or c.get('mlp_bias'):
        raise SystemExit('needs a bias-free SiLU llama / mistral model, untied embeddings, no sliding window')
    s = Source(src)
    dst.mkdir(parents=True, exist_ok=True)
    L = int(c['num_hidden_layers'])
    write(dst / 'model-embed.safetensors', [(n, *s.get(n)) for n in ('model.embed_tokens.weight', 'lm_head.weight', 'model.norm.weight')])
    for layer in range(L):
        base = f'model.layers.{layer}.'
        out = [(base + n, *s.get(base + n)) for n in ('input_layernorm.weight', 'post_attention_layernorm.weight')]
        for p in PROJ:
            b = base + p + '.'
            if not s.has(b + 'weight_packed'):
                raise SystemExit(f'{b}weight_packed missing (only all-NVFP4 projections are supported)')
            for n in ('weight_packed', 'weight_scale'):
                out.append((b + n, *s.get(b + n)))
            dtype, shape, raw = s.get(b + 'weight_global_scale')
            g = np.frombuffer(raw, np.float32)
            if dtype != 'F32' or g.size != 1 or not np.isfinite(g).all() or g[0] <= 0:
                raise SystemExit(f'bad global scale {b}')
            out.append((b + 'weight_global_scale', 'F32', [1], (np.float32(1.0) / g).astype(np.float32).tobytes()))
        write(dst / f'model-layer-{layer:05d}.safetensors', out)
        print(f'layer {layer + 1}/{L}', flush=True)
    cfg = {k: v for k, v in c.items() if k != 'quantization_config'}
    cfg.update(model_type='llama', architectures=['LlamaForCausalLM'], attention_bias=False, mlp_bias=False, quip_params=QUIP,
               qtip_quant={'kv_format': 'qtip', 'layer_formats': ['nvfp4'] * L, 'source': {'repo_format': 'compressed-tensors nvfp4-pack-quantized',
                                                                                         'model_type': c['model_type']}})
    cfg.pop('sliding_window', None)
    (dst / 'config.json').write_text(json.dumps(cfg, indent=2))
    for n in COPY:
        if (src / n).exists():
            shutil.copy(src / n, dst / n)


if __name__ == '__main__':
    main(*sys.argv[1:3])
