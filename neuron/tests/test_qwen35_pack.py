"""pack_layer (quantise once, slice per rank) == shard_layer per rank, bit for bit, on the 397B shape class at TP = 32 and
on the 35B tiny config at TP = 2; and the file path (safetensors bf16 checkpoint -> prepare main with --jobs) agrees too.
  python neuron/tests/test_qwen35_pack.py"""
import json, struct, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qwen35_graphs as Q  # noqa: E402
import qwen35_moe_ref as R  # noqa: E402
import qwen35_prepare as P  # noqa: E402
import glm_repack as RP  # noqa: E402


def same(a, b):
    assert a.keys() == b.keys(), (a.keys(), b.keys())
    for k in a:
        assert a[k].dtype == b[k].dtype and a[k].shape == b[k].shape, k
        assert torch.equal(a[k].view(torch.uint8) if a[k].dtype == Q.F8 else a[k], b[k].view(torch.uint8) if b[k].dtype == Q.F8 else b[k]), k


def write_safetensors(path, tensors):
    head, off, blobs = {}, 0, []
    for k, t in tensors.items():
        b = t.to(torch.bfloat16).contiguous().view(torch.int16).numpy().tobytes()
        head[k] = {'dtype': 'BF16', 'shape': list(t.shape), 'data_offsets': [off, off + len(b)]}
        off += len(b); blobs.append(b)
    h = json.dumps(head).encode()
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(h))); f.write(h); [f.write(b) for b in blobs]


for cfg, tp in ((R.TINY397, 32), (R.TINY397, 2), (R.TINY, 2)):
    w = {k: v.to(torch.bfloat16).float() for k, v in R.random_weights(dict(cfg), seed=5).items()}   # bf16-representable, as a checkpoint
    g = Q.Geo(dict(cfg), tp)
    for fp8 in (True, False):
        get = P.Getter(w)
        for l in range(g.L):
            fast = P.pack_layer(get, l, g, fp8, chunk=5)
            for r in range(tp):
                same(fast[r], P.shard_layer(get, l, r, g, fp8))
    print(f'pack_layer == shard_layer: E={g.E} TP={tp} ok')
    with tempfile.TemporaryDirectory(dir=str(Path.home())) as d:
        d = Path(d); (d / 'm').mkdir()
        pre = 'model.language_model.'
        write_safetensors(d / 'm/model.safetensors', {(k if k.startswith(('lm_head', 'model.')) else pre + k): v for k, v in w.items()})
        (d / 'm/config.json').write_text(json.dumps(cfg))
        subprocess.run([sys.executable, str(Path(P.__file__)), str(d / 'm'), str(d / 'o'), '--tp', str(tp), '--jobs', '2', '--threads', '1'], check=True, stdout=subprocess.DEVNULL)
        man = json.loads((d / 'o/prepared.json').read_text())
        get = P.Getter(w)
        for l in (0, g.L - 1):
            for r in (0, tp - 1):
                ref = P.shard_layer(get, l, r, g, True)
                raw = (d / f'o/r{r}/L{l:03d}.bin').read_bytes()
                ents = man['files'][f'r{r}/L{l:03d}.bin']
                assert {e['name'] for e in ents} == set(ref)
                for e in ents:
                    assert raw[e['offset']:e['offset'] + e['bytes']] == RP.tensor_bytes(ref[e['name']]), e['name']
        ref = P.shard_head(get, tp - 1, g, True)
        raw = (d / f'o/r{tp - 1}/head.bin').read_bytes()
        for e in man['files'][f'r{tp - 1}/head.bin']:
            assert raw[e['offset']:e['offset'] + e['bytes']] == RP.tensor_bytes(ref[e['name']]), e['name']
    print(f'prepare main (--jobs 2) files == shard_layer: TP={tp} ok')
