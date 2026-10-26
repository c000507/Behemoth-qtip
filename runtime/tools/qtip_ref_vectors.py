"""Reference vectors for the C++/HIP QTIP runtime tests (tests/qtip_tests.cpp).

Uses the unmodified QTIP reference code (Cornell-RelaxML/qtip, /mnt/nvme/quantbench/qtip) through
qtip_quant/tools/qtip_shim.py (pure-torch Hadamard, no CUDA kernels), on CPU.

For each requested linear of one layer of a qtip_quant output directory it writes into OUT/:
  L<layer>_<linear>.rows.f32     reference decode_compressed() of the kernel-layout trellis (rotated,
                                 Wscale-free hatW, fp16 LUT values as float) for the row ranges in
                                 L<layer>_<linear>.rows.json
  L<layer>_<linear>.x.f32        x = randn(8, n) (torch seed 1234 + linear index), float32
  L<layer>_<linear>.y.f32        reference QuantizedLinear(...).forward(x) (eval mode, float32 internals)
  L<layer>_<linear>.yd.f32       x @ hatW^T in float64 with hatW = diag(SV) T_m hatWr T_n^T diag(SU)
                                 (the exact linear the format defines; same construction as check_decode.py)
Usage:
  /home/charles/venvs/quantbench/bin/python tools/qtip_ref_vectors.py --model DIR --layer 40 --out OUT \
      [--linears self_attn.q_proj,mlp.down_proj]
"""
import argparse, json, os, sys
import torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--layer", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--linears", default="self_attn.q_proj,self_attn.k_proj,mlp.down_proj")
ap.add_argument("--shim", default="/home/charles/Development/Behemoth-R1-123B-v2-aqlm-codebase/qtip_quant/tools")
ap.add_argument("--threads", type=int, default=32)
args = ap.parse_args()
torch.set_num_threads(args.threads)
sys.path.insert(0, args.shim)
import qtip_shim  # noqa: E402

qtip_shim.install()
import torch._dynamo  # noqa: E402

torch._dynamo.config.disable = True
from lib.utils.kernel_decompress import decode_compressed  # noqa: E402
from lib.utils import matmul_hadU  # noqa: E402
from lib.codebook.bitshift import bitshift_codebook  # noqa: E402
from lib.linear import QuantizedLinear  # noqa: E402

os.makedirs(args.out, exist_ok=True)
fn = os.path.join(args.model, f"model-layer-{args.layer:05d}.safetensors")
f = safe_open(fn, "pt")
pre = f"model.layers.{args.layer}."
for li, lin in enumerate(args.linears.split(",")):
    t = {k: f.get_tensor(pre + lin + "." + k) for k in ("trellis", "SU", "SV", "tlut", "rcp", "tp_rank")}
    m, n = len(t["SV"]), len(t["SU"])
    cb = bitshift_codebook(L=16, K=2, V=2, tlut_bits=9, decode_mode="quantlut_sym", tlut=t["tlut"].float().half())
    hatWr = decode_compressed(16, 9, 2, 1, m, n, t["trellis"].view(-1), cb.lut.T).float()
    # row ranges: first 64 rows, first 64 rows of the second TP half, last 32 rows
    ranges = [[0, 64], [m // 2, m // 2 + 64], [m - 32, m]]
    rows = torch.cat([hatWr[a:b] for a, b in ranges], 0).contiguous()
    base = os.path.join(args.out, f"L{args.layer}_{lin}")
    rows.numpy().tofile(base + ".rows.f32")
    with open(base + ".rows.json", "w") as jf:
        json.dump({"m": m, "n": n, "ranges": ranges}, jf)
    torch.manual_seed(1234 + li)
    x = torch.randn(8, n, dtype=torch.float32)
    x.numpy().tofile(base + ".x.f32")
    ql = QuantizedLinear(n, m, 16, 16, 16, 2, 2, 9, "quantlut_sym", dtype=torch.float32)
    sd = {k: t[k] for k in ("trellis", "SU", "SV", "rcp", "tp_rank")}
    sd["tlut"] = t["tlut"].half()
    ql.load_state_dict(sd, strict=False)
    with torch.no_grad():
        y = ql(x)
    y.float().numpy().tofile(base + ".y.f32")
    Wd = matmul_hadU(hatWr.double())
    Wd = matmul_hadU(Wd.T.contiguous()).T
    hatW = Wd * t["SV"].double().unsqueeze(1) * t["SU"].double().unsqueeze(0)
    yd = x.double() @ hatW.T
    yd.float().numpy().tofile(base + ".yd.f32")
    print(json.dumps({"linear": lin, "m": m, "n": n,
                      "forward_vs_dense_rel": ((y.double() - yd).norm() / yd.norm()).item()}), flush=True)
