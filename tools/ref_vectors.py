"""Produce reference test vectors with the unmodified QTIP python code (CPU) for tests/test_cpu (qtip_tests).
Writes raw little-endian .bin files into --out:
  had_{n}_x.bin (f64 [4,n]), had_{n}_u.bin (matmul_hadU), had_{n}_ut.bin (matmul_hadUt)      n in 96,1024,12288,28672
  vit_x.bin (f32 [B,256], fp16-representable), vit_states.bin (i32 [B,128]) from bitshift_codebook.quantize (fp32 cb)
  vit_hat.bin (f32 [B,256])
  pack_states.bin (i32 [T,128]) random valid tail-biting trellises for an m x n = 64 x 96 linear (T = 4*6 = 24),
  pack_plain.bin (u16 [T,32]) cb.pack_trellis, pack_kernel.bin (u16 [T,32]) after the has_kernel swizzle
  ldl_H.bin (f64 [n,n]) SPD, ldl_L.bin (f64 [n,n]) block_LDL(H,16)[0] with diag zeroed, n=96
  perm.bin (i64 [256]) ldlq._PERMUTE
  lut.bin (f32 [2,65536]) cb.lut (fp16-rounded)
"""
import argparse, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qtip_shim  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--nvit", type=int, default=24)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
qtip_shim.install()
import torch._dynamo
torch._dynamo.config.disable = True
from lib.utils import matmul_hadU, matmul_hadUt, block_LDL  # noqa: E402
from lib.codebook.bitshift import bitshift_codebook  # noqa: E402
from lib.algo import ldlq  # noqa: E402


def save(name, t):
    t.contiguous().numpy().tofile(os.path.join(args.out, name))


g = torch.Generator().manual_seed(7)
for n in (96, 1024, 12288, 28672):
    x = torch.randn(4, n, generator=g, dtype=torch.float64)
    save(f"had_{n}_x.bin", x)
    save(f"had_{n}_u.bin", matmul_hadU(x))
    save(f"had_{n}_ut.bin", matmul_hadUt(x))

cb = bitshift_codebook(L=16, K=2, V=2, tlut_bits=9, decode_mode="quantlut_sym")
# fp16-rounded lut but fp32 arithmetic (reference converts the codebook to the model dtype)
cb.lut = cb.lut.half().float()
cb.recons_state = cb.recons(cb.state)
save("lut.bin", cb.lut.float())
save("perm.bin", ldlq._PERMUTE.to(torch.int64))

B = args.nvit
X = (torch.randn(B, 256, generator=g) * 0.9).half().float()
hatX, st = cb.quantize(X)
save("vit_x.bin", X)
save("vit_states.bin", st.to(torch.int32))
save("vit_hat.bin", hatX.float())

# packing: random tail-biting trellises
m, n = 64, 96
T = (m // 16) * (n // 16)
s0 = torch.randint(0, 1 << 16, (T,), generator=g)
st = torch.zeros(T, 128, dtype=torch.int64)
st[:, 0] = s0
for i in range(1, 128):
    nb = torch.randint(0, 16, (T,), generator=g)
    if i >= 125:  # last 3 steps must reproduce the top 12 bits of state 0 (tail biting)
        nb = (s0 >> (12 - 4 * (i - 125))) & 15
    st[:, i] = ((st[:, i - 1] << 4) & 0xFFFF) + nb
packed = cb.pack_trellis(st.to(torch.int32))
save("pack_states.bin", st.to(torch.int32))
save("pack_plain.bin", packed.view(torch.int16))
kp = packed.view(torch.uint8).view(-1, 2).flip((-1,)).reshape(m // 16 // 2, 2, n // 16 // 2, 2, 16 * 16 // 8, 2) \
    .permute(0, 2, 4, 3, 1, 5).flip((-1,)).contiguous().flatten().view(torch.int16).reshape(packed.shape)
save("pack_kernel.bin", kp)
un = cb.unpack_trellis(packed, 256)
assert (un.to(torch.int64) == st).all(), "reference unpack mismatch"

nn_ = 96
A = torch.randn(nn_, nn_ * 2, generator=g, dtype=torch.float64)
H = A @ A.T / (2 * nn_) + 1e-2 * torch.eye(nn_, dtype=torch.float64)
L, D = block_LDL(H.clone(), 16)
L[torch.arange(nn_), torch.arange(nn_)] = 0
save("ldl_H.bin", H)
save("ldl_L.bin", L)
json.dump({"nvit": B, "pack_m": m, "pack_n": n, "ldl_n": nn_}, open(os.path.join(args.out, "manifest.json"), "w"))
print("ok", args.out)
