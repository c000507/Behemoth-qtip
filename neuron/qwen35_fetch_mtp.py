"""Extract the multi-token-prediction head (`mtp.*` tensors) of a Qwen3.5 checkpoint into OUT/model.safetensors (+ config.json),
without the rest of the checkpoint: from a local HF directory, or from the Hub by HTTP range requests (index, the shard
headers, then only the byte ranges of the mtp tensors, coalesced). The result is a directory glm_repack.Checkpoint reads.

usage: qwen35_fetch_mtp.py SOURCE OUT      SOURCE = local checkpoint dir | Hub repo id (org/name[@revision])
"""
import json
import struct
import sys
import urllib.request
from pathlib import Path


def http(url, start=None, end=None):
    req = urllib.request.Request(url, headers={'User-Agent': 'curl/8'})
    if start is not None:
        req.add_header('Range', f'bytes={start}-{end - 1}')
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if attempt == 4:
                raise
            print('retry', url[-60:], e, flush=True)


class Source:
    def __init__(self, src):
        self.local = Path(src) if Path(src).is_dir() else None
        if not self.local:
            repo, _, rev = src.partition('@')
            self.base = f'https://huggingface.co/{repo}/resolve/{rev or "main"}/'

    def read(self, name, start=None, end=None):
        if self.local:
            with open(self.local / name, 'rb') as f:
                if start is None:
                    return f.read()
                f.seek(start)
                return f.read(end - start)
        return http(self.base + name, start, end)


def main():
    src, out = Source(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_bytes(src.read('config.json'))
    wmap = json.loads(src.read('model.safetensors.index.json'))['weight_map']
    names = [k for k in wmap if k.startswith('mtp.')]
    files = sorted({wmap[k] for k in names})
    print(len(names), 'mtp tensors in', files, flush=True)
    header, blobs, off = {}, [], 0
    for file in files:
        n = struct.unpack('<Q', src.read(file, 0, 8))[0]
        h = json.loads(src.read(file, 8, 8 + n))
        mine = sorted((h[k]['data_offsets'][0], k) for k in names if wmap[k] == file)
        # coalesce adjacent tensors into ranges of at most 256 MB
        runs = []
        for a, k in mine:
            b = h[k]['data_offsets'][1]
            if runs and runs[-1][1] == a and b - runs[-1][0] <= 256 << 20:
                runs[-1][1] = b
                runs[-1][2].append(k)
            else:
                runs.append([a, b, [k]])
        for a, b, ks in runs:
            data = src.read(file, 8 + n + a, 8 + n + b)
            assert len(data) == b - a, (file, a, b, len(data))
            for k in ks:
                ka, kb = h[k]['data_offsets']
                header[k] = {'dtype': h[k]['dtype'], 'shape': h[k]['shape'], 'data_offsets': [off + ka - a, off + kb - a]}
            blobs.append(data)
            off += b - a
        print(file, len(mine), 'tensors,', len(runs), 'ranges', flush=True)
    hj = json.dumps(header, separators=(',', ':')).encode()
    hj += b' ' * ((-len(hj)) % 8)
    with open(out / 'model.safetensors', 'wb') as f:
        f.write(struct.pack('<Q', len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)
    print('wrote', out / 'model.safetensors', off, 'bytes')


if __name__ == '__main__':
    main()
