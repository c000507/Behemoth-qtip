"""Download a bounded row slice of an original BF16/F16/F32 HF tensor.

Public repositories only. Pin the resolved revision and verify every HTTP byte
range, avoiding accidental multi-GB shard downloads. Outputs .npy + provenance.
Requires requests, numpy, ml_dtypes. Does not download tokenizer or execute code.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import ml_dtypes
import numpy as np
import requests


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True)
    p.add_argument('--revision', default='main')
    p.add_argument('--tensor', required=True)
    p.add_argument('--row-start', type=int, default=0)
    p.add_argument('--rows', type=int, default=512)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    s = requests.Session()
    r = s.get(f'https://huggingface.co/api/models/{a.repo}/revision/{a.revision}', timeout=60)
    r.raise_for_status()
    rev = r.json()['sha']
    base = f'https://huggingface.co/{a.repo}/resolve/{rev}/'
    r = s.get(base+'model.safetensors.index.json', timeout=60)
    r.raise_for_status()
    shard = r.json()['weight_map'][a.tensor]

    def get_range(lo, hi):
        # Distinct query avoids intermediary caches confusing different ranges.
        with s.get(base+shard+f'?byte_range={lo}-{hi}', headers={'Range':f'bytes={lo}-{hi}'},
                   stream=True, timeout=90) as resp:
            resp.raise_for_status()
            if resp.status_code != 206 or not resp.headers.get('Content-Range','').startswith(f'bytes {lo}-{hi}/'):
                raise RuntimeError('Server did not honor byte range; refusing full shard download')
            raw = resp.raw.read(hi-lo+2)
            if len(raw) != hi-lo+1:
                raise RuntimeError('Byte range length mismatch')
            return raw

    length = struct.unpack('<Q', get_range(0,7))[0]
    if length > 100_000_000:
        raise ValueError('Unreasonable safetensors header')
    meta = json.loads(get_range(8,7+length))[a.tensor]
    dtype = {'BF16':ml_dtypes.bfloat16, 'F16':np.float16, 'F32':np.float32}[meta['dtype']]
    shape = meta['shape']
    if len(shape) != 2 or a.row_start < 0 or a.rows <= 0 or a.row_start+a.rows > shape[0]:
        raise ValueError('Invalid matrix row slice')
    stride = shape[1]*np.dtype(dtype).itemsize
    lo = 8+length+meta['data_offsets'][0]+a.row_start*stride
    count = a.rows*stride
    if count > 128*1024**2:
        raise ValueError('Slice exceeds 128 MiB limit')
    data = b''.join(get_range(i,min(i+4*1024**2,lo+count)-1) for i in range(lo,lo+count,4*1024**2))
    w = np.frombuffer(data, dtype=dtype).reshape(a.rows,shape[1]).astype(np.float32)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(a.out, w)
    a.out.with_suffix('.json').write_text(json.dumps(dict(repo=a.repo,revision=rev,tensor=a.tensor,
        original_shape=shape,source_dtype=meta['dtype'],row_start=a.row_start,rows=a.rows,
        source_bytes_sha256=hashlib.sha256(data).hexdigest()),indent=2)+'\n')
    print(f'{a.repo}@{rev}: {a.tensor} rows {a.row_start}:{a.row_start+a.rows}, {w.shape} -> {a.out}')


if __name__ == '__main__':
    main()
