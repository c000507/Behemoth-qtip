"""Where does the device's FP8 projection differ from the exact mirror (fp8 codes widened, f32 accumulation)?
Two one-core graphs, run through cpp/neff_io.cpp on random data:
  quant   y = matmul(fp8(h / a), w8^T) * a * s   (glm_graphs.lin: the activation is quantised in the graph)
  codes   y = matmul(h8, w8^T) * a * s           (h8 quantised on the host: only the fp8 matmul on the device)
(A graph returning fp8(h / a) as f32 is no probe of the conversion: neuronx-cc folds the round trip away.)
Measured on trn1 (neuronx-cc 2.26): both match the exact mirror to ~1e-7, i.e. E4M3 x E4M3 products accumulate in f32
and the in-graph quantisation rounds like torch (round to nearest even).
  python neuron/tests/glm_fp8_probe.py WORKDIR NEFF_IO_BINARY
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import glm_graphs as G
import glm_repack as RP

F8 = torch.float8_e4m3fn


def build(work, name, fn, args):
    import torch_neuronx
    d = Path(work) / name
    if (d / 'graph.neff').exists():
        return d
    try:
        torch_neuronx.trace(fn, args, compiler_workdir=str(d))
    except RuntimeError as e:
        if 'neuronx-cc failed' not in str(e):
            raise
    subprocess.run([sys.executable, str(HERE.parent / 'tools_patch_fp8_hlo.py'), str(d / 'model/graph.hlo'), str(d / 'model/graph.hlo')], check=True)
    subprocess.run(['neuronx-cc', 'compile', 'model', '--framework', 'XLA', '--target', 'trn1', '--output', 'graph.neff'], cwd=d, check=True,
                   stdout=subprocess.DEVNULL)
    return d


def run(neff_io, d, inputs, shape):
    io = d / 'io'
    io.mkdir(exist_ok=True)
    for i, t in enumerate(inputs):
        (io / f'in{i}').write_bytes(RP.tensor_bytes(t))
    subprocess.run([neff_io, str(d / 'graph.neff'), str(io), '1'], check=True, stdout=subprocess.DEVNULL)
    return torch.from_numpy(np.frombuffer((io / 'out0.r0').read_bytes(), np.float32).reshape(shape).copy())


def main():
    work, neff_io = sys.argv[1], sys.argv[2]
    torch.manual_seed(0)
    T, K, N = 8, 1024, 512
    h = torch.randn(T, K)
    w8, s = RP.to_fp8_rows(torch.randn(N, K) / 32)
    a = torch.amax(torch.abs(h), dim=-1, keepdim=True) / 224.0
    h8 = (h / a).to(F8)
    exact = torch.matmul(h8.float(), w8.float().T) * a * s.reshape(1, -1)
    rel = lambda x, y: float(torch.linalg.norm(x - y) / torch.linalg.norm(y))

    def quant(h, w8, s):
        a = torch.amax(torch.abs(h), dim=-1, keepdim=True).clamp(min=1e-20) / 224.0
        return torch.matmul((h / a).to(F8), w8.T).float() * a * s.reshape(1, -1)

    def codes(h8, a, w8, s):
        return torch.matmul(h8, w8.T).float() * a * s.reshape(1, -1)

    yq = run(neff_io, build(work, 'quant', quant, (h, w8, s)), (h, w8, s), (T, N))
    yc = run(neff_io, build(work, 'codes', codes, (h8, a, w8, s)), (h8, a, w8, s), (T, N))
    bf = torch.matmul(h, (w8.float() * s.reshape(-1, 1)).T)
    print(f'quant (activation quantised on the device) vs exact mirror: {rel(yq, exact):.3e}')
    print(f'codes (host-quantised activation)          vs exact mirror: {rel(yc, exact):.3e}')
    print(f'reference: exact FP8 mirror vs unquantised-activation matmul {rel(exact, bf):.3e}')


if __name__ == '__main__':
    main()
