"""Decode a qtip_quant layer file with the QTIP reference code and compare with the binary's own reconstruction.

For every linear of layer L in <out>/model-layer-XXXXX.safetensors:
  1. hatW_ref = reference decode: lib.utils.kernel_decompress.decode_compressed (kernel-layout trellis -> fp16 LUT
     values), then the inverse RHT exactly as BitshiftLinear.cache_hatW / forward do it (matmul_hadU, SU, SV),
     giving the dense weight the QTIP model would use.
  2. compare with <dump>/L<L>_<linear>.hatW.f32 written by `qtip_quant --dump-dir` (max / relative error)
  3. optionally (--forward) run the reference QuantizedLinear module (eval mode, kernel layout, CPU) on random
     inputs and compare with x @ hatW_ref^T
  4. if --hess is given: proxy error of hatW_ref vs the original FP8 weight under that Hessian
"""
import argparse, json, os, sys
import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qtip_shim  # noqa: E402

LIN = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
       "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
HN = {"self_attn.q_proj": "qkv", "self_attn.k_proj": "qkv", "self_attn.v_proj": "qkv", "self_attn.o_proj": "o",
      "mlp.gate_proj": "gateup", "mlp.up_proj": "gateup", "mlp.down_proj": "down"}
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--layer", type=int, required=True)
ap.add_argument("--dump", default="")
ap.add_argument("--hess", default="")
ap.add_argument("--model", default="/mnt/evo/Behemoth-R1-123B-v2-FP8-Dynamic")
ap.add_argument("--linears", default=",".join(LIN))
ap.add_argument("--forward", action="store_true")
ap.add_argument("--results", default="")
args = ap.parse_args()

qtip_shim.install()
import torch._dynamo  # noqa: E402
torch._dynamo.config.disable = True
from lib.utils.kernel_decompress import decode_compressed  # noqa: E402
from lib.utils import matmul_hadU, get_hadK  # noqa: E402
from lib.codebook.bitshift import bitshift_codebook  # noqa: E402
from lib.linear import QuantizedLinear  # noqa: E402

fn = os.path.join(args.out, f"model-layer-{args.layer:05d}.safetensors")
f = safe_open(fn, "pt")
pre = f"model.layers.{args.layer}."
for lin in args.linears.split(","):
    t = {k: f.get_tensor(pre + lin + "." + k) for k in ("trellis", "SU", "SV", "tlut", "rcp", "tp_rank")}
    m, n = len(t["SV"]), len(t["SU"])
    assert t["trellis"].dtype == torch.int16 and t["trellis"].shape == (m // 16 * n // 16, 32)
    assert int(t["rcp"]) == 0
    cb = bitshift_codebook(L=16, K=2, V=2, tlut_bits=9, decode_mode="quantlut_sym", tlut=t["tlut"].float().half())
    # (1) reference decode (kernel layout) -> rotated, Wscale-free weight; fp16 LUT values
    hatWr = decode_compressed(16, 9, 2, 1, m, n, t["trellis"].view(-1), cb.lut.T).float()
    # BitshiftLinear forward:  y = SV * hadU_m( hadUt_n(x * SU) @ hatWr^T )   (scale 32 cancels)
    #   => W_eff = diag(SV) T_m hatWr T_n^T diag(SU), with matmul_hadU(rows) = X T^T
    Wd = matmul_hadU(hatWr.double())                  # hatWr T_n^T
    Wd = matmul_hadU(Wd.T.contiguous()).T             # T_m hatWr T_n^T
    hatW = Wd * t["SV"].double().unsqueeze(1) * t["SU"].double().unsqueeze(0)
    rec = dict(layer=args.layer, linear=lin, shape=[m, n])
    if args.dump:
        mine = torch.from_file(os.path.join(args.dump, f"L{args.layer}_{lin}.hatW.f32"), size=m * n,
                               dtype=torch.float32).view(m, n).double()
        rec["max_abs_diff_vs_binary"] = (mine - hatW).abs().max().item()
        rec["rel_fro_diff_vs_binary"] = ((mine - hatW).norm() / hatW.norm()).item()
    if args.forward:
        ql = QuantizedLinear(n, m, 16, 16, 16, 2, 2, 9, "quantlut_sym", dtype=torch.float32)
        sd = {k: t[k] for k in ("trellis", "SU", "SV", "rcp", "tp_rank")}
        sd["tlut"] = t["tlut"].half()
        ql.load_state_dict(sd, strict=False)
        x = torch.randn(8, n)
        with torch.no_grad():
            y = ql(x).double()
        yr = x.double() @ hatW.T
        rec["forward_rel_err"] = ((y - yr).norm() / yr.norm()).item()
    if args.hess:
        with open(os.path.join(args.model, "model.safetensors.index.json")) as fi:
            wm = json.load(fi)["weight_map"]
        k = pre + lin + ".weight"
        with safe_open(os.path.join(args.model, wm[k]), "pt") as fw:
            W = fw.get_tensor(k).double() * fw.get_tensor(k + "_scale").double().reshape(-1, 1)
        with safe_open(args.hess, "pt") as fh:
            H = fh.get_tensor(HN[lin]).double()
        d = W - hatW
        rec["proxy_ref_decode"] = (((d @ H) * d).sum() / ((W @ H) * W).sum()).item()
    print(json.dumps(rec), flush=True)
    if args.results:
        with open(args.results, "a") as fo:
            fo.write(json.dumps(rec) + "\n")
