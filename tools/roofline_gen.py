#!/usr/bin/env python3
"""Roofline benchmark generators (brief5 step 1).

Builds the measurement containers used for docs/neuron-roofline.md:
  - elementwise DMA sweep: emit_elementwise(n, scale) for n in N_EW
  - matmul w-streaming:    emit_matmul_synth at synth-supported shapes
  - compiler comparators:  oracle_ref (OFFLINE - runs on the trn1 box via
    jax+neuronx-cc; NOT part of any emit path)
  - launch-floor chains:   oracle_ref gen of n-op elementwise chains in
    one graph

Inputs/expected outputs are written next to each .neff (in0/in1/exp0).
Timing on the box: /home/ubuntu/work/nrtd/timing <graph.neff> <dir> 200.
"""
import shutil
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).parent
sys.path.insert(0, str(TOOLS))

OUT = Path("/tmp/opencode/roof")
N_EW = [16383, 65536, 262144, 1048576]  # elementwise sweep (P = n/128)
# NOTE: ew cap = P*16 <= 65535 -> P <= 4095 -> n <= ~524k for CORRECT numerics
# (at P=8192 the DVE sem counter overflows u16; numerics FAIL but the
# DMA-rate timing is still representative).

MM_SYNTH = [(64, 1024, 128), (32, 1024, 128), (48, 384, 128), (32, 128, 384)]
MM_ORACLE = [(32, 2048, 128), (32, 4096, 128), (512, 2048, 512)]  # comparator
CHAINS = [2, 4, 8, 16]


def emit_synth():
    import lift_hip
    import numpy as np
    rng = np.random.default_rng(21)
    for n in N_EW:
        d = OUT / f"ew{n}"
        d.mkdir(parents=True, exist_ok=True)
        neff = lift_hip.emit_elementwise(d, n=n, scale=2.5, name="ew")
        x = rng.standard_normal(n).astype(np.float32)
        w = rng.standard_normal(n).astype(np.float32)
        x.tofile(d / "in0"); w.tofile(d / "in1")
        (2.5 * x * w).tofile(d / "exp0")
        print("ew", n, neff.stat().st_size)
    for M, K, N in MM_SYNTH:
        d = OUT / f"mm{M}_{K}_{N}"
        d.mkdir(parents=True, exist_ok=True)
        neff = lift_hip.emit_matmul(M, K, N, d, name="mm")
        x = rng.standard_normal((M, K)).astype(np.float32)
        w = rng.standard_normal((K, N)).astype(np.float32)
        x.tofile(d / "in0"); w.tofile(d / "in1"); (x @ w).tofile(d / "exp0")
        print("mm", M, K, N, neff.stat().st_size)


def gen_oracle():
    """OFFLINE comparator: compile reference graphs on the trn1 box."""
    import oracle_ref
    import numpy as np
    rng = np.random.default_rng(33)
    for M, K, N in MM_ORACLE:
        d = OUT / f"oracle{M}_{K}_{N}"
        d.mkdir(parents=True, exist_ok=True)
        oracle_ref.emit_matmul_gen(OUT, M, K, N, name=f"mm{M}k{K}")
        x = rng.standard_normal((M, K)).astype(np.float32)
        w = rng.standard_normal((K, N)).astype(np.float32)
        x.tofile(d / "in0"); w.tofile(d / "in1"); (x @ w).tofile(d / "exp0")
        print("oracle mm", M, K, N)
    for nops in CHAINS:
        expr = "a"
        for _ in range(nops):
            expr = f"jax.nn.relu({expr}) * 1.0001"
        body = (f"lowered = jax.jit(lambda a: {expr}).lower("
                f"jnp.zeros((4096,), jnp.float32))")
        key = f"chain{nops}"
        sg = oracle_ref.gen_oracle_custom(
            key, oracle_ref._jax_gen(body).replace(
                "os.environ['ODIR']", f"'/tmp/oracle_gen_{key}'"))
        d = OUT / key
        d.mkdir(parents=True, exist_ok=True)
        a = rng.standard_normal(4096).astype(np.float32)
        a.tofile(d / "in0")
        r = a.copy()
        for _ in range(nops):
            r = np.maximum(r, 0) * 1.0001
        r.tofile(d / "exp0")
        print("chain", nops, sg)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if "--oracle" in sys.argv:
        gen_oracle()
    else:
        emit_synth()
