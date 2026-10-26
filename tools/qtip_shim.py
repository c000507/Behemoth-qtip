"""Make the QTIP reference code (github.com/Cornell-RelaxML/qtip) importable on ROCm without its CUDA kernels.
- fast_hadamard_transform: pure-torch Sylvester-Hadamard matmul (only used by inference path)
- qtip_kernels: empty stub (decode kernels not needed for encoding)
- glog: std logging
- HYB codebook ('quantlut_sym', tlut_bits=9, V=2) needs /tmp/kmeans_9_2.pt, which QTIP builds with scipy kmeans;
  we build the same thing (k-means of 2^20 2-D Gaussians, 512 centroids, std-normalized * 0.9682) in torch.
"""
import logging, os, sys, types
import torch

_here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qtip")
QTIP_DIR = os.environ.get("QTIP_DIR", _here if os.path.isdir(_here) else "/mnt/nvme/quantbench/qtip")

_had_cache = {}


def _hadamard_transform(x, scale=1.0):
    n = x.shape[-1]
    key = (n, x.device, x.dtype)
    if key not in _had_cache:
        H = torch.ones(1, 1)
        while H.shape[0] < n:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        _had_cache[key] = H.to(x.device, x.dtype)
    return (x @ _had_cache[key]) * scale


def install():
    fht = types.ModuleType("fast_hadamard_transform")
    fht.hadamard_transform = _hadamard_transform
    sys.modules.setdefault("fast_hadamard_transform", fht)
    sys.modules.setdefault("qtip_kernels", types.ModuleType("qtip_kernels"))
    glog = types.ModuleType("glog")
    lg = logging.getLogger("qtip")
    glog.info, glog.warning, glog.error = lg.info, lg.warning, lg.error
    sys.modules.setdefault("glog", glog)
    if QTIP_DIR not in sys.path:
        sys.path.insert(0, QTIP_DIR)
    _ensure_hyb_tlut(9, 2)


def _ensure_hyb_tlut(tlut_bits, V, iters=20):
    fname = f"/tmp/kmeans_{tlut_bits}_{V}.pt"
    if os.path.exists(fname):
        return
    g = torch.Generator().manual_seed(0)
    k = 2 ** tlut_bits
    c = torch.randn(k, V, generator=g, dtype=torch.float64)
    data = torch.randn(1 << 20, V, generator=g, dtype=torch.float64)
    for _ in range(iters):
        a = torch.cdist(data, c).argmin(1)
        s = torch.zeros_like(c).index_add_(0, a, data)
        cnt = torch.bincount(a, minlength=k).clamp_min(1).unsqueeze(1)
        c = torch.where(torch.bincount(a, minlength=k).unsqueeze(1) > 0, s / cnt, c)
    c = c.float()
    c = c / c.std(unbiased=False) * 0.9682458365518543
    torch.save(c, fname)
