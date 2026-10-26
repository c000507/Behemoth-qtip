"""Run the unmodified QTIP reference (RHT, block_LDL, LDLQ, bitshift_codebook.viterbi) on one layer of the FP8
checkpoint with the Hessians written by qtip_quant (<hess_dir>/layer<L>.safetensors), for parity with the binary.
Settings = paper 2-bit HYB: L=16 K=2 V=2 quantlut_sym tlut_bits=9, td 16x16, sigma_reg=1e-2, scale_override=0.9,
fp32 W/H, fp16 codebook / Viterbi (as qtip quantize_finetune_llama with an fp16 model).  Same proxy definitions as
qtip_quant: proxy (unregularized H, original domain) and proxy_rot_reg (regularized, rotated: what the reference prints).
Appends JSON lines to --results.
"""
import argparse, json, os, sys, time
import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qtip_shim  # noqa: E402

LIN = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
       "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
HN = {"self_attn.q_proj": "qkv", "self_attn.k_proj": "qkv", "self_attn.v_proj": "qkv", "self_attn.o_proj": "o",
      "mlp.gate_proj": "gateup", "mlp.up_proj": "gateup", "mlp.down_proj": "down"}

SHAPE = {"self_attn.q_proj": (12288, 12288), "self_attn.k_proj": (1024, 12288), "self_attn.v_proj": (1024, 12288),
         "self_attn.o_proj": (12288, 12288), "mlp.gate_proj": (28672, 12288), "mlp.up_proj": (28672, 12288),
         "mlp.down_proj": (12288, 28672)}  # Behemoth / Mistral-Large

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/mnt/evo/Behemoth-R1-123B-v2-FP8-Dynamic")
ap.add_argument("--hess", required=True, help="qtip_quant Hessian file layer<L>.safetensors")
ap.add_argument("--layer", type=int, default=40)
ap.add_argument("--linears", default=",".join(LIN))
ap.add_argument("--rows", type=int, default=0, help="random subset of output rows (0 = all)")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--results", default="ref_results.jsonl")
ap.add_argument("--tag", default="")
ap.add_argument("--wdump", default="", help="read W from qtip_quant --dump-dir files L<L>_<linear>.W.f32 (exact fp8*scale "
                "in fp32, identical to the checkpoint decode) instead of --model")
args = ap.parse_args()

qtip_shim.install()
from lib.utils import matmul_hadUt, matmul_hadU, block_LDL, regularize_H  # noqa: E402
from lib.codebook.bitshift import bitshift_codebook  # noqa: E402
from lib.algo import ldlq  # noqa: E402

dev = torch.device(args.device)


def load_w(lin):
    if args.wdump:
        m, n = SHAPE[lin]
        return torch.from_file(os.path.join(args.wdump, f"L{args.layer}_{lin}.W.f32"), size=m * n,
                               dtype=torch.float32).view(m, n).clone()
    with open(os.path.join(args.model, "model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]
    k = f"model.layers.{args.layer}.{lin}.weight"
    with safe_open(os.path.join(args.model, wm[k]), "pt") as f:
        w = f.get_tensor(k).float()
        s = f.get_tensor(k + "_scale").float()
    return w * s.reshape(-1, 1)


Hcache = {}


def load_h(name):
    if name not in Hcache:
        with safe_open(args.hess, "pt") as f:
            Hcache.clear()
            Hcache[name] = f.get_tensor(name)
    return Hcache[name]


def sync():
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


torch.manual_seed(args.layer)
qa = argparse.Namespace(td_x=16, td_y=16, L=16, K=2, V=2, tlut_bits=9, decode_mode="quantlut_sym")
cb = bitshift_codebook(L=16, K=2, V=2, tlut_bits=9, decode_mode="quantlut_sym").to(dev).to(torch.float16)
for lin in args.linears.split(","):
    W = load_w(lin)
    if args.rows and args.rows < W.shape[0]:
        g = torch.Generator().manual_seed(1234)
        idx = torch.randperm(W.shape[0], generator=g)[: args.rows].sort().values
        W = W[idx].contiguous()
    W = W.to(dev)
    H = load_h(HN[lin]).to(dev, torch.float32)
    m, n = W.shape
    sync(); t0 = time.time()
    SU = (torch.randn(n, device=dev).sign() + 1e-5).sign()
    SV = (torch.randn(m, device=dev).sign() + 1e-5).sign()
    HR = regularize_H(H.clone(), 1e-2)
    Wr = matmul_hadUt(matmul_hadUt(W.T * SV).T * SU)
    HRr = matmul_hadUt(matmul_hadUt(HR * SU).T * SU)
    Wscale = Wr.square().mean().sqrt() / (cb.lut.to(torch.float64).square().mean().sqrt().float() * 0.9)
    Wr = Wr / Wscale
    sync(); t_rht = time.time() - t0; t0 = time.time()
    LRr, _ = block_LDL(HRr, 16)
    LRr[torch.arange(n, device=dev), torch.arange(n, device=dev)] = 0
    sync(); t_ldl = time.time() - t0; t0 = time.time()
    hatWr, Qidxs = ldlq.LDLQ(Wr, LRr, cb, qa, for_kernel=True)
    sync(); t_ldlq = time.time() - t0
    Wr = Wr * Wscale; hatWr = hatWr * Wscale
    d = Wr - hatWr
    err_rot = (torch.trace(d @ HRr @ d.T) / torch.trace(Wr @ HRr @ Wr.T)).item()
    A = matmul_hadU(hatWr) * SU
    hatW = (matmul_hadU(A.T) * SV).T
    num = den = 0.0
    for s in range(0, m, 2048):
        w = W[s:s + 2048]; dd = w - hatW[s:s + 2048]
        num += ((dd @ H) * dd).sum().item(); den += ((w @ H) * w).sum().item()
    rec = dict(kind="qtip_ref", layer=args.layer, linear=lin, shape=[m, n], rows=args.rows, proxy=num / den,
               proxy_rot_reg=err_rot, t_rht=t_rht, t_ldl=t_ldl, t_ldlq_viterbi=t_ldlq, t_total=t_rht + t_ldl + t_ldlq,
               s_per_Mweight=(t_rht + t_ldl + t_ldlq) / (m * n) * 1e6, tag=args.tag, t=time.strftime("%F %T"))
    print(json.dumps(rec), flush=True)
    with open(args.results, "a") as f:
        f.write(json.dumps(rec) + "\n")
    del W, H, HR, Wr, HRr, LRr, hatWr, hatW, A, d
    if dev.type == "cuda":
        torch.cuda.empty_cache()
