"""PROBE: which new-nki (0.5.0) instructions pass neuronx-cc 2.26's verifier for trn1. CPU only.
  python probe_nki_ops.py OUT NAME      (NAME in OPS; prints PASS / FAIL NAME)"""
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault('UNSAFE_FP8FNCAST', '1')
import torch

import nki
import nki.isa as nisa
import nki.language as nl

T = 32


@nki.jit
def k(a, sel):
    y = nl.ndarray((T, 256), dtype=nl.float32, buffer=nl.shared_hbm)
    x = nl.ndarray((T, 256), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=x, src=a)
    o = nl.ndarray((T, 256), dtype=nl.float32, buffer=nl.sbuf)
    if sel == 0:
        m = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=m, op=nl.abs_max, data=x, axis=1)
        nisa.tensor_scalar(dst=o, data=x, op0=nl.multiply, operand0=m)
    elif sel == 1:
        m = nl.ndarray((T, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=m, op=nl.maximum, data=x, axis=1)
        nisa.tensor_scalar(dst=o, data=x, op0=nl.multiply, operand0=m)
    elif sel == 2:
        nisa.activation(dst=o, op=nl.abs, data=x)
    elif sel == 3:
        nisa.tensor_scalar(dst=o, data=x, op0=nl.abs_max, operand0=0.0)
    elif sel == 4:
        nisa.reciprocal(dst=o, data=x)
    elif sel == 5:
        nisa.activation(dst=o, op=nl.reciprocal, data=x)
    elif sel == 6:
        nisa.activation(dst=o, op=nl.silu, data=x)
    elif sel == 7:
        nisa.activation(dst=o, op=nl.sigmoid, data=x)
    elif sel == 8:
        nisa.tensor_scalar(dst=o, data=x, op0=nl.multiply, operand0=1.0 / 224.0, op1=nl.maximum, operand1=1e-20)
    elif sel == 9:
        h = nl.ndarray((T, 256), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=h, data1=x, data2=x, op=nl.multiply)
        nisa.tensor_copy(dst=o, src=h)
    elif sel == 10:
        h = nl.ndarray((T, 256), dtype=nl.float8_e4m3, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=h, data=x, op0=nl.multiply, operand0=2.0)
        nisa.tensor_copy(dst=o, src=h)
    elif sel == 11:
        nisa.memset(dst=o, value=0.0)
    nisa.dma_copy(dst=y, src=o)
    return y


OPS = ['absmax', 'max', 'abs_act', 'abs_ts', 'recip', 'recip_act', 'silu', 'sigmoid', 'ts2', 'tt_bf16', 'fp8', 'memset']


def main():
    out, name = Path(sys.argv[1]), sys.argv[2]
    import torch_neuronx
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    sel = OPS.index(name)
    try:
        try:
            torch_neuronx.trace(lambda a: k(a, sel) * 1.0, (torch.ones(T, 256),), compiler_workdir=str(d))
        except RuntimeError as error:
            if 'neuronx-cc failed' not in str(error):
                raise
        r = subprocess.run(['neuronx-cc', 'compile', 'model', '--framework', 'XLA', '--target', 'trn1', '--output', 'graph.neff'],
                           cwd=d, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(('PASS ' if r.returncode == 0 else 'FAIL ') + name, flush=True)
    except Exception as e:  # noqa: BLE001
        print('FAIL-FE ' + name + ' ' + str(e).replace('\n', ' ')[-200:], flush=True)


if __name__ == '__main__':
    main()
