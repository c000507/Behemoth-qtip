"""Repack a compressed-tensors NVFP4 checkpoint (llm-compressor `nvfp4-pack-quantized`, e.g. a stock HF Llama / Mistral NVFP4 export)
into the per-layer layout the Neuron runtime loads: model-embed.safetensors (embed_tokens, lm_head, final norm) and
model-layer-NNNNN.safetensors (norms + projections), plus a config.json in the QTIP-hfized llama form the loader parses.

Tensor bytes are copied as stored, except the global scale: compressed-tensors dequantises as code * block_scale / weight_global_scale,
the runtime as code * block_scale * weight_global_scale, so the repacked file stores the reciprocal. input_global_scale is dropped
(weights-only on this backend). Only bias-free SiLU llama-shaped models without a sliding window are accepted (Mistral Small 3.x is one).

--fp8: write the projections as row-scaled E4M3 instead (weight [out, in] F8_E4M3 with |code| <= 224, weight_scale [out] f32),
the form the runtime slices straight into its per-core shards at load (no dequantise / requantise pass, ~22 GB less f32 work for
a 24B model). From NVFP4 this is the runtime's load-time quantisation (amax / 224 per output row), except that row-parallel
projections (o / down) keep one scale per full row instead of one per rank's half, so outputs agree to FP8 rounding.
A compressed-tensors FP8 export (`float-quantized` / `naive-quantized`, per-tensor or per-channel weight_scale) is also accepted:
copied as stored (`--fp8` then re-scales rows whose codes use the e4m3fn range above 240, which the devices cannot represent).
So is Mistral's own FP8 export (HF `quant_method: fp8`, per-tensor `weight_scale_inv`, the dequantisation multiplier despite its name;
Devstral 2 / Mistral Large 3 class `ministral3` models): same copy, static activation_scale dropped. Their `rope_parameters`
(YaRN) are written as the llama-form `rope_scaling` + `rope_theta`.

usage: python repack_compressed_tensors.py [--fp8] SRC_DIR DST_DIR
"""
import json
from pathlib import Path
import shutil
import struct
import sys

import ml_dtypes
import numpy as np

E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
F8 = np.dtype(ml_dtypes.float8_e4m3fn)


def to_fp8_rows(w):
    """f32 [out, in] -> (E4M3 bytes [out, in] with |code| <= 224, f32 row scales [out]): the runtime's quantize_slice."""
    amax = np.abs(w).max(axis=1)
    scale = np.where(amax > 0, amax / 224.0, 1.0).astype(np.float32)
    return (w / scale[:, None]).astype(F8).view(np.uint8), scale

PROJ = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')
QUIP = {'K': 2, 'L': 16, 'V': 2, 'codebook': 'bitshift', 'codebook_version': 0, 'decode_mode': 'quantlut_sym', 'skip_list': None,
        'split_for_tp': False, 'td_x': 16, 'td_y': 16, 'tlut_bits': 9}
COPY = ('generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja', 'README.md',
        'tekken.json')


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


def main(src, dst, fp8=False):
    src, dst = Path(src), Path(dst)
    c = json.loads((src / 'config.json').read_text())
    q = c.get('quantization_config') or {}
    fmt = q.get('format')
    if q.get('quant_method') == 'fp8' and q.get('weight_block_size') is None:
        fmt = 'mistral-fp8'
    elif q.get('quant_method') != 'compressed-tensors' or fmt not in ('nvfp4-pack-quantized', 'float-quantized', 'naive-quantized'):
        raise SystemExit('not a compressed-tensors nvfp4-pack-quantized / float-quantized or per-tensor fp8 checkpoint')
    if c['model_type'] not in ('llama', 'mistral', 'ministral3') or c.get('hidden_act') != 'silu' or c.get('tie_word_embeddings') \
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
            if s.has(b + 'weight_packed'):
                dtype, shape, raw = s.get(b + 'weight_global_scale')
                g = np.frombuffer(raw, np.float32)
                if dtype != 'F32' or g.size != 1 or not np.isfinite(g).all() or g[0] <= 0:
                    raise SystemExit(f'bad global scale {b}')
                if not fp8:
                    for n in ('weight_packed', 'weight_scale'):
                        out.append((b + n, *s.get(b + n)))
                    out.append((b + 'weight_global_scale', 'F32', [1], (np.float32(1.0) / g).astype(np.float32).tobytes()))
                    continue
                _, (o, half), packed = s.get(b + 'weight_packed')
                packed = np.frombuffer(packed, np.uint8).reshape(o, half)
                _, _, bs = s.get(b + 'weight_scale')
                bs = np.frombuffer(bs, F8).reshape(o, half * 2 // 16).astype(np.float32)
                w = np.empty((o, half * 2), np.float32)
                w[:, 0::2] = E2M1[packed & 15]
                w[:, 1::2] = E2M1[packed >> 4]
                w = (w.reshape(o, -1, 16) * (bs / g[0])[:, :, None]).reshape(o, half * 2)
                q8, sc = to_fp8_rows(w)
                out += [(b + 'weight', 'F8_E4M3', [o, half * 2], q8.tobytes()), (b + 'weight_scale', 'F32', [o], sc.tobytes())]
            elif s.has(b + 'weight'):
                dtype, shape, raw = s.get(b + 'weight')
                if dtype != 'F8_E4M3' or len(shape) != 2:
                    raise SystemExit(f'{b}weight is {dtype}, expected F8_E4M3')
                sdt, _, sraw = s.get(b + ('weight_scale_inv' if fmt == 'mistral-fp8' else 'weight_scale'))
                sc = np.frombuffer(sraw, {'F32': np.float32, 'BF16': ml_dtypes.bfloat16, 'F16': np.float16}[sdt]).astype(np.float32).reshape(-1)
                sc = np.broadcast_to(sc, (shape[0],)).copy() if sc.size == 1 else sc
                codes = np.frombuffer(raw, np.uint8).reshape(shape)
                if fp8 and ((codes & 0x78) == 0x78).any():                            # e4m3fn range above 240: re-scale those rows
                    q8, rs = to_fp8_rows(codes.view(F8).astype(np.float32) * sc[:, None])
                    codes, sc = q8, rs
                out += [(b + 'weight', 'F8_E4M3', list(shape), codes.tobytes()), (b + 'weight_scale', 'F32', [shape[0]], sc.astype(np.float32).tobytes())]
            else:
                raise SystemExit(f'{b}: neither weight_packed (NVFP4) nor weight (FP8)')
        write(dst / f'model-layer-{layer:05d}.safetensors', out)
        print(f'layer {layer + 1}/{L}', flush=True)
    cfg = {k: v for k, v in c.items() if k not in ('quantization_config', 'rope_parameters')}
    rope = c.get('rope_parameters')
    if rope:
        cfg['rope_theta'] = rope['rope_theta']
        if rope.get('rope_type', rope.get('type')) not in (None, 'default'):
            cfg['rope_scaling'] = {k: v for k, v in rope.items() if k not in ('rope_theta', 'type')}
    kind = 'fp8' if fp8 or fmt != 'nvfp4-pack-quantized' else 'nvfp4'
    cfg.update(model_type='llama', architectures=['LlamaForCausalLM'], attention_bias=False, mlp_bias=False, quip_params=QUIP,
               qtip_quant={'kv_format': 'qtip', 'layer_formats': [kind] * L, 'source': {'repo_format': 'compressed-tensors ' + fmt,
                                                                                       'model_type': c['model_type']}})
    cfg.pop('sliding_window', None)
    (dst / 'config.json').write_text(json.dumps(cfg, indent=2))
    for n in COPY:
        if (src / n).exists():
            shutil.copy(src / n, dst / n)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a != '--fp8']
    main(*args[:2], fp8='--fp8' in sys.argv[1:])
