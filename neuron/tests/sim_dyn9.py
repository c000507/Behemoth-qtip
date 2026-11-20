"""CPU simulator debug of moe_dyn9's prologue (slot tables) against the expected values."""
import os, sys
from pathlib import Path
os.environ.setdefault('UNSAFE_FP8FNCAST', '1')
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, torch, ml_dtypes, nki
import bench_moe_dyn as B, nki_moe_dyn as D
T = int(sys.argv[1]); KM = 10 * T
rand = np.random.default_rng(3)
x = torch.from_numpy(rand.standard_normal((T, B.H)).astype(np.float32))
top = torch.from_numpy(B.draw(T, 'uniform', rand)).long()
w = torch.softmax(torch.from_numpy(rand.standard_normal((T, B.TOPK)).astype(np.float32)), dim=-1)
egu_t, egu_s, ed, ed_c = B.weight_inputs('dyn', 1, rand)
n = B.G.bf(x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6))
comb = B.comb_of(top, w)
f8 = lambda t: t.contiguous().view(torch.uint8).numpy().view(ml_dtypes.float8_e4m3)
n8, a = D.graph_inputs9(n)
y, idl, csl, gsl, dfl, cnth, sidh = nki.simulate(D.moe_dyn9)(f8(n8), a.numpy(), comb.numpy(), egu_s.numpy(), ed_c.numpy(), f8(egu_t), f8(ed), KM=KM, dbg=1)
used = np.nonzero((comb > 0).any(0).numpy())[0]
print('nd', len(used), 'cnt (b1 b2 ng nd)', np.asarray(cnth).ravel())
print('idl[:12]', np.asarray(idl).ravel()[:12], 'expected', used[:12])
print('idl ok', np.array_equal(np.asarray(idl).ravel()[:len(used)], used), 'pads', np.asarray(idl).ravel()[len(used):len(used)+4])
print('sidx[:8, 0]', np.asarray(sidh)[:8, 0])
print('csl ok', np.allclose(np.asarray(csl)[:len(used)], comb.numpy()[:, used].T), 'gsl ok', np.allclose(np.asarray(gsl)[:len(used)], egu_s.numpy()[used]))
print('dfl ok', np.array_equal(np.asarray(dfl)[:len(used)], used[:, None] * 32 + np.arange(32)))
