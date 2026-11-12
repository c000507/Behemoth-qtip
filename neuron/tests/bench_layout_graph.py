"""Micro NEFFs: does the activation layout matter? K MLP-like blocks (RMSNorm, fp8 quantisation, fp8 matmul to 640, SiLU gate,
fp8 matmul back to H, residual) on T decode rows, written row-major (x [T, H], as qwen35_graphs) or column-major (x carried
as [H, T]: the hidden axis first, weights applied from the left). No collectives; layer_bench NEFF steps 1 times them.
  python bench_layout_graph.py OUT T K {row|col}   ->  OUT/lay_<form>_T_K/graph.neff
(docs/neuron-qwen397-batch.md: the first extra decode row costs 3x the later ones.)
"""
import subprocess, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glm_graphs as G  # noqa: E402

out, T, K, form = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
H, O = 4096, 640
F8 = G.F8
import torch_neuronx  # noqa: E402


def row(x, w, w1, s1, w2):
    for _ in range(K):
        n = G.bf(x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6) * w)
        a = torch.amax(torch.abs(n), dim=-1, keepdim=True).clamp(min=1e-20) / 224.0
        h = G.bf(torch.matmul((n / a).to(F8), w1.T).float() * a * s1.reshape(1, -1))
        act = G.bf(h[:, :O // 2] * torch.sigmoid(h[:, :O // 2]) * h[:, O // 2:])
        a2 = torch.amax(torch.abs(act), dim=-1, keepdim=True).clamp(min=1e-20) / 224.0
        y = torch.matmul((act / a2).to(F8), w2).float() * a2
        x = x + G.bf(y)
    return x


def col(x, w, w1, s1, w2):
    xt = x.T                                                                           # [H, T]
    wc, sc = w.reshape(H, 1), s1.reshape(O, 1)
    for _ in range(K):
        n = G.bf(xt * torch.rsqrt(torch.mean(xt * xt, dim=0, keepdim=True) + 1e-6) * wc)
        a = torch.amax(torch.abs(n), dim=0, keepdim=True).clamp(min=1e-20) / 224.0
        h = G.bf(torch.matmul(w1, (n / a).to(F8)).float() * a * sc)                    # [O, T]
        act = G.bf(h[:O // 2] * torch.sigmoid(h[:O // 2]) * h[O // 2:])
        a2 = torch.amax(torch.abs(act), dim=0, keepdim=True).clamp(min=1e-20) / 224.0
        y = torch.matmul(w2.T, (act / a2).to(F8)).float() * a2                         # [H, T]
        xt = xt + G.bf(y)
    return xt.T


d = Path(out) / f'lay_{form}_{T}_{K}'
d.mkdir(parents=True, exist_ok=True)
inputs = (torch.zeros(T, H), torch.ones(H), G.zeros((O, H), F8), torch.ones(O), G.zeros((O // 2, H), F8))
try:
    torch_neuronx.trace(row if form == 'row' else col, inputs, compiler_workdir=str(d))
except RuntimeError as error:
    if 'neuronx-cc failed' not in str(error):
        raise
here = Path(__file__).resolve().parents[1]
subprocess.run([sys.executable, str(here / 'tools_patch_fp8_hlo.py'), str(d / 'model' / 'graph.hlo'), str(d / 'model' / 'graph.hlo')], check=True)
subprocess.run(['neuronx-cc', 'compile', 'model', '--framework', 'XLA', '--target', 'trn1', '--output', 'graph.neff'], cwd=d, check=True, stdout=subprocess.DEVNULL)
print('built', d)
