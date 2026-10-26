"""Fast engine: dense per-layer weights, bf16 GEMMs, fused row transforms, flash attention.

Same forward math as decoder_layer with a device_rounding backend (runtime-reference
semantics), with a manual analytic backward instead of autograd:

- each layer's seven trellises are decoded once into one bf16 buffer (runtime decoder,
  BF16 LUT values: exactly the runtime-reference weights) and reused by every GEMM of
  that layer pass; q|k|v and gate|up are stacked, so a layer has 4 GEMMs per direction;
- GEMMs are bf16 x bf16 with FP32 accumulation and FP32 outputs (hipBLASLt via torch.mm);
  transposed GEMMs round the incoming gradient to bf16 (standard mixed precision);
- Hadamard/sign/scale stages are one fused row kernel each (native/train_kernels.hip);
- attention is AOTriton flash attention (varlen: packed documents are separate sequences,
  GQA without repeating K/V); backward reuses the saved logsumexp;
- token rows are processed in chunks of row_chunk, so intermediates are bounded by
  row_chunk x 57344 (gate|up), never T x T.

CPU (tiny verification models) uses the same code with reference decode/Hadamard and
dense attention, so it can be compared with the autograd path on any shape.
"""
from collections import defaultdict
import contextlib
import math
import os

import torch
import torch.nn.functional as F

from .kernels import decode_rows, hadamard
from .model import LINEARS, NORMS

Q, K, V, O, G, U, D = LINEARS
BF16, F32 = torch.bfloat16, torch.float32


def bf(t):
    return t.detach().to(BF16).float()


class Timer:
    """Per-section GPU time from CUDA events (sections must not nest)."""
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.events = []
        self.pool = []  # recycled events: no per-step event creation/destruction

    def event(self):
        return self.pool.pop() if self.pool else torch.cuda.Event(enable_timing=True)

    @contextlib.contextmanager
    def __call__(self, name):
        if not self.enabled:
            yield
            return
        start, end = self.event(), self.event()
        start.record()
        try:
            yield
        finally:
            end.record()
            self.events.append((name, start, end))

    def collect(self):
        totals = {}
        if self.events:
            self.events[-1][2].synchronize()
        for name, start, end in self.events:
            totals[name] = totals.get(name, 0.0) + start.elapsed_time(end)
            self.pool += [start, end]
        self.events = []
        return {k: round(v, 3) for k, v in sorted(totals.items())}


class Geometry:
    """Per-batch document runs (cu_seqlens) and RoPE tables, shared by all layers."""
    def __init__(self, segments, positions, config, device):
        segments = segments.cpu()
        t = len(segments)
        change = torch.nonzero(segments[1:] != segments[:-1]).flatten() + 1
        bounds = torch.cat([torch.tensor([0]), change, torch.tensor([t])])
        self.tokens = t
        self.runs = list(zip(bounds[:-1].tolist(), bounds[1:].tolist()))
        self.cu = bounds.to(device=device, dtype=torch.int32)
        self.max_len = int((bounds[1:] - bounds[:-1]).max())
        start, end = torch.empty(t, dtype=torch.int32), torch.empty(t, dtype=torch.int32)
        for a, b in self.runs:
            start[a:b], end[a:b] = a, b
        self.doc_start, self.doc_end = start.to(device), end.to(device)
        n = config["hidden_size"]
        d = config.get("head_dim", n // config["num_attention_heads"])
        theta = config.get("rope_theta", 10000.0)
        # Exactly kernels.rope's FP32 table arithmetic.
        inv = theta ** (-torch.arange(0, d, 2, device=device, dtype=F32) / d)
        angle = positions.to(device).to(F32)[:, None] * inv[None, :]
        self.cos2 = angle.cos().repeat(1, 2).contiguous()
        self.sin2 = angle.sin().repeat(1, 2).contiguous()
        self.cos, self.sin = self.cos2[:, None, :], self.sin2[:, None, :]


def batch_geometry(segments, positions, config, device):
    return Geometry(segments, positions, config, device)


def rope_forward(x, cos, sin):
    d = x.shape[-1]
    z = x.float()
    half = torch.cat((-z[..., d // 2:], z[..., :d // 2]), -1)
    return (z * cos + half * sin).to(BF16)


def rope_backward(dy, cos, sin):
    d = dy.shape[-1]
    z = dy.float()
    t = z * sin
    return z * cos + torch.cat((t[..., d // 2:], -t[..., :d // 2]), -1)


class FastLayer:
    def __init__(self, config, device, backend, row_chunk=4096, timer=None, attention=None):
        attention = attention or os.environ.get("QTIP_ATTENTION", "triton")
        if attention not in ("triton", "aotriton"):
            raise ValueError("attention must be triton or aotriton")
        self.attention = attention
        self.config = config
        self.device = torch.device(device)
        self.backend = backend
        self.n = n = config["hidden_size"]
        self.heads, self.kv = config["num_attention_heads"], config["num_key_value_heads"]
        self.hd = config.get("head_dim", n // self.heads)
        self.inter = config["intermediate_size"]
        self.eps = config.get("rms_norm_eps", 1e-5)
        self.qd, self.kd = self.heads * self.hd, self.kv * self.hd
        if row_chunk < 1:
            raise ValueError("row_chunk must be positive")
        self.row_chunk = row_chunk
        self.timer = timer or Timer(False)
        self.cuda = self.device.type == "cuda"
        self.native = self.cuda and hasattr(backend, "had")
        if self.cuda and not self.native:
            raise ValueError("the fast engine on GPUs needs the runtime backend (native row kernels)")
        # down is stored transposed ([inter, n]): its forward/recompute GEMMs become NN, which
        # hipBLASLt runs ~35% faster than NT for this shape on gfx1201 (the backward becomes NT).
        shapes = dict(qkv=(self.qd + 2 * self.kd, n), o=(n, self.qd), gu=(2 * self.inter, n), d=(self.inter, n))
        self.storage = torch.empty(sum(a * b for a, b in shapes.values()), dtype=BF16, device=self.device)
        self.w, offset = {}, 0
        for key, (a, b) in shapes.items():
            self.w[key] = self.storage[offset:offset + a * b].view(a, b)
            offset += a * b
        self.down_scratch = torch.empty((n, self.inter), dtype=BF16, device=self.device)
        self.parts = dict(qkv=((Q, 0), (K, self.qd), (V, self.qd + self.kd)), o=((O, 0),),
                          gu=((G, 0), (U, self.inter)), d=((D, 0),))

    # ------------------------------------------------------------------ primitives
    def empty(self, *shape, dtype=BF16):
        return torch.empty(shape, dtype=dtype, device=self.device)

    def decode(self, weights):
        """Decode one layer's packed weights (a device slot) into the dense buffer."""
        with self.timer("decode"):
            lut = self.backend.lut(weights[Q + ".tlut"]) if self.native else None
            for key, items in self.parts.items():
                w = self.down_scratch if key == "d" else self.w[key]
                width = w.shape[1]
                for name, start in items:
                    trellis = weights[name + ".trellis"]
                    m = trellis.numel() * 8 // width
                    dst = w[start:start + m]
                    if self.native:
                        self.backend.decode_into(trellis, lut, m, width, dst)
                    else:
                        lut_values = weights[name + ".tlut"].to(BF16).float()
                        dst.copy_(decode_rows(trellis, lut_values, m, width))
            self.w["d"].copy_(self.down_scratch.t())

    def had(self, x, n, transpose=False, pre=None, round_in=False, out32=None, out16=None, post=None):
        if self.native and n in self.backend.fast_dims:
            self.backend.had(x, n, transpose, pre, round_in, out32, out16, post)
            return
        v = x.float()
        if round_in:
            v = v.to(BF16).float()
        if pre is not None:
            v = v * pre
        y = hadamard(v, transpose)
        if out32 is not None:
            out32.copy_(y)
        if out16 is not None:
            out16.copy_(y * post if post is not None else y)

    def mm_nt(self, a, w):
        """a[R, k] . w[m, k]^T -> FP32 [R, m]."""
        if self.cuda:
            return torch.mm(a, w.t(), out_dtype=F32)
        return a.float() @ w.float().t()

    def mm_nn(self, a, w):
        """a[R, m] . w[m, k] -> FP32 [R, k]."""
        if self.cuda:
            return torch.mm(a, w, out_dtype=F32)
        return a.float() @ w.float()

    def rms(self, x, w):
        z = x.float()
        r = torch.rsqrt(z.square().mean(-1, keepdim=True) + self.eps)
        return (z * r * w).to(BF16), r

    @staticmethod
    def rms_back(x, r, w, dxn, acc, key):
        xhat = x.float() * r
        acc[key] += (dxn * xhat).sum(0)
        dxhat = dxn * w
        return r * (dxhat - xhat * (dxhat * xhat).mean(-1, keepdim=True))

    # Fused native kernels (Behemoth dims) with portable fallbacks (tiny CPU models).
    def fused(self, dim=None):
        return self.native and (dim is None or dim in self.backend.fast_dims)

    def rms_had(self, x, w, su, u, xn=None, r=None):
        if self.fused() and self.n == 12288:
            self.backend.rms_had(x, w, self.eps, su, u, xn, r)
            return
        y, factor = self.rms(x, w)
        if xn is not None:
            xn.copy_(y)
        if r is not None:
            r.copy_(factor.view_as(r))
        self.had(y, self.n, True, pre=su, out16=u)

    def gateup(self, z, e, gate, up, product, cg=None, cu=None):
        inter = self.inter
        if self.fused() and inter == 28672:
            self.backend.gateup(z, e[G], e[U], gate, up, product, cg, cu)
            return
        self.had(z[:, :inter], inter, round_in=True, out32=cg, out16=gate, post=e[G])
        self.had(z[:, inter:], inter, round_in=True, out32=cu, out16=up, post=e[U])
        product.copy_((F.silu(gate.float()) * up.float()).to(BF16))

    def colsum(self, a, b, acc, rs=None):
        if self.fused() and a.shape[1] % 1024 == 0:
            self.backend.colsum(a, b, acc, rs)
            return
        v = a * b.float()
        if rs is not None:
            v = v * rs.view(-1, 1)
        acc += v.sum(0)

    def swiglu_bwd(self, dxsu, su, gate, up):
        """Returns (dg, dup); dg reuses dxsu's storage."""
        if self.fused() and dxsu.shape[1] % 1024 == 0:
            du = torch.empty_like(dxsu)
            self.backend.swiglu_bwd(dxsu, su, gate, up, dxsu, du)
            return dxsu, du
        dp = dxsu.mul_(su)
        g = gate.float()
        s = torch.sigmoid(g)
        dup = dp * g * s
        return dp.mul_(up.float()).mul_(s * (1 + g * (1 - s))), dup

    def norm_backward(self, x, r, w, su, xn, dxsu, dh, acc, su_key, w_key):
        """dsu += colsum(dxsu * xn); dw += su * colsum(dxsu * x * r); dh += RMSNorm adjoint(dxsu * su)."""
        self.colsum(dxsu, xn, acc[su_key])
        if self.fused() and self.n == 12288:
            raw = torch.zeros_like(acc[w_key])
            self.backend.colsum(dxsu, x, raw, r)
            acc[w_key] += raw * su
            self.backend.rms_bwd(x, r, w, su, dxsu, dh)
            return
        dh += self.rms_back(x, r.view(-1, 1), w, dxsu * su, acc, w_key)

    def rope(self, y, heads, g, sl, backward=False):
        rows = y.shape[0]
        if self.fused() and self.hd == 128:
            out = self.empty(rows, heads, 128, dtype=F32 if backward else BF16)
            self.backend.rope(y.reshape(rows, heads * 128) if backward else y, heads, g.cos2[sl], g.sin2[sl], out,
                              backward)
            return out
        view = y.view(rows, heads, self.hd)
        return (rope_backward if backward else rope_forward)(view, g.cos[sl], g.sin[sl])

    def chunks(self, t):
        for start in range(0, t, self.row_chunk):
            yield slice(start, min(start + self.row_chunk, t))

    def effective(self, p):
        e = dict(su_qkv=bf(p[Q + ".SU"]), su_o=bf(p[O + ".SU"]), su_gu=bf(p[G + ".SU"]), su_d=bf(p[D + ".SU"]),
                 w1=bf(p[NORMS[0]]), w2=bf(p[NORMS[1]]))
        for name in LINEARS:
            e[name] = (p[name + ".SV"] * p[name + ".Wscale"]).detach().contiguous()
        return e

    # ------------------------------------------------------------------ attention
    def attn_forward(self, q, k, v, g):
        if self.cuda and self.attention == "triton":
            from . import flash
            o, lse = flash.forward(q, k, v, g.doc_start)
            return o, (lse,)
        if self.cuda:
            r = torch.ops.aten._flash_attention_forward(q, k, v, g.cu, g.cu, g.max_len, g.max_len, 0.0, True, False)
            return r[0], (r[1], r[2], r[3])
        out = torch.empty_like(q)
        lse = torch.empty((q.shape[1], q.shape[0]), dtype=F32)
        group = self.heads // self.kv
        for a, b in g.runs:
            qs = q[a:b].float().transpose(0, 1)
            ks = k[a:b].float().transpose(0, 1).repeat_interleave(group, 0)
            vs = v[a:b].float().transpose(0, 1).repeat_interleave(group, 0)
            s = qs @ ks.transpose(-1, -2) / math.sqrt(self.hd)
            s.masked_fill_(~torch.ones(b - a, b - a, dtype=torch.bool).tril(), -torch.inf)
            lse[:, a:b] = s.logsumexp(-1)
            out[a:b] = (s.softmax(-1) @ vs).transpose(0, 1).to(q.dtype)
        return out, (lse,)

    def attn_backward(self, do, q, k, v, o, saved, g):
        if self.cuda and self.attention == "triton":
            from . import flash
            return flash.backward(do, q, k, v, o, saved[0], g.doc_start, g.doc_end)
        if self.cuda:
            lse, rng, unused = saved
            r = torch.ops.aten._flash_attention_backward(do, q, k, v, o, lse, g.cu, g.cu, g.max_len, g.max_len,
                                                         0.0, True, rng, unused)
            return r[0], r[1], r[2]
        (lse,) = saved
        dq, dk, dv = (torch.zeros(t.shape, dtype=F32) for t in (q, k, v))
        group = self.heads // self.kv
        scale = 1 / math.sqrt(self.hd)
        for a, b in g.runs:
            qs = q[a:b].float().transpose(0, 1)
            ks = k[a:b].float().transpose(0, 1).repeat_interleave(group, 0)
            vs = v[a:b].float().transpose(0, 1).repeat_interleave(group, 0)
            dos = do[a:b].float().transpose(0, 1)
            s = qs @ ks.transpose(-1, -2) * scale
            s.masked_fill_(~torch.ones(b - a, b - a, dtype=torch.bool).tril(), -torch.inf)
            p = torch.exp(s - lse[:, a:b, None])
            delta = (dos * o[a:b].float().transpose(0, 1)).sum(-1, keepdim=True)
            ds = p * (dos @ vs.transpose(-1, -2) - delta) * scale
            dq[a:b] = (ds @ ks).transpose(0, 1)
            dk[a:b] = (ds.transpose(-1, -2) @ qs).reshape(self.kv, group, b - a, self.hd).sum(1).transpose(0, 1)
            dv[a:b] = (p.transpose(-1, -2) @ dos).reshape(self.kv, group, b - a, self.hd).sum(1).transpose(0, 1)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)

    # ------------------------------------------------------------------ forward
    def attention_half(self, x, e, g, keep=None):
        t, n, qd, kd = x.shape[0], self.n, self.qd, self.kd
        q = self.empty(t, self.heads, self.hd)
        k, v = self.empty(t, self.kv, self.hd), self.empty(t, self.kv, self.hd)
        if keep is not None:
            keep.update(xn1=self.empty(t, n), r1=self.empty(t, dtype=F32),
                        c_qkv=self.empty(t, qd + 2 * kd, dtype=F32))
        spans = ((Q, 0, qd), (K, qd, qd + kd), (V, qd + kd, qd + 2 * kd))
        for sl in self.chunks(t):
            rows = sl.stop - sl.start
            u = self.empty(rows, n)
            with self.timer("norm_transform"):
                self.rms_had(x[sl], e["w1"], e["su_qkv"], u, None if keep is None else keep["xn1"][sl],
                             None if keep is None else keep["r1"][sl])
            with self.timer("gemm"):
                z = self.mm_nt(u, self.w["qkv"])
            del u
            y = self.empty(rows, qd + 2 * kd)
            c = keep["c_qkv"][sl] if keep is not None else None
            with self.timer("hadamard"):
                for name, a, b in spans:
                    self.had(z[:, a:b], b - a, out32=None if c is None else c[:, a:b], out16=y[:, a:b], post=e[name])
            del z
            with self.timer("rope"):
                q[sl] = self.rope(y[:, :qd], self.heads, g, sl)
                k[sl] = self.rope(y[:, qd:qd + kd], self.kv, g, sl)
                v[sl] = y[:, qd + kd:].view(rows, self.kv, self.hd)
            del y
        with self.timer("attention"):
            o, saved = self.attn_forward(q, k, v, g)
        if keep is not None:
            keep.update(q=q, k=k, v=v, o=o, attn=saved, c_o=self.empty(t, n, dtype=F32))
        else:
            del q, k, v
        h = torch.empty_like(x)
        for sl in self.chunks(t):
            rows = sl.stop - sl.start
            u = self.empty(rows, qd)
            with self.timer("hadamard"):
                self.had(o[sl].view(rows, qd), qd, True, pre=e["su_o"], out16=u)
            with self.timer("gemm"):
                z = self.mm_nt(u, self.w["o"])
            del u
            y = self.empty(rows, n)
            with self.timer("hadamard"):
                self.had(z, n, out32=keep["c_o"][sl] if keep is not None else None, out16=y, post=e[O])
            del z
            with self.timer("residual"):
                torch.add(x[sl], y, out=h[sl])
        return h

    def mlp_rows(self, h, e, keep=None):
        rows, n, inter = h.shape[0], self.n, self.inter
        u = self.empty(rows, n)
        xn = self.empty(rows, n) if keep is not None else None
        r = self.empty(rows, dtype=F32) if keep is not None else None
        with self.timer("norm_transform"):
            self.rms_had(h, e["w2"], e["su_gu"], u, xn, r)
        with self.timer("gemm"):
            z = self.mm_nt(u, self.w["gu"])
        del u
        gate, up, product = self.empty(rows, inter), self.empty(rows, inter), self.empty(rows, inter)
        cg = self.empty(rows, inter, dtype=F32) if keep is not None else None
        cu = self.empty(rows, inter, dtype=F32) if keep is not None else None
        with self.timer("gateup_swiglu"):
            self.gateup(z, e, gate, up, product, cg, cu)
        del z
        u = self.empty(rows, inter)
        with self.timer("hadamard"):
            self.had(product, inter, True, pre=e["su_d"], out16=u)
        with self.timer("gemm"):
            z = self.mm_nn(u, self.w["d"])
        del u
        y = self.empty(rows, n)
        cd = self.empty(rows, n, dtype=F32) if keep is not None else None
        with self.timer("hadamard"):
            self.had(z, n, out32=cd, out16=y, post=e[D])
        del z
        with self.timer("residual"):
            out = h + y
        if keep is not None:
            keep.update(xn2=xn, r2=r, cg=cg, cu=cu, gate=gate, up=up, product=product, cd=cd)
        return out

    @torch.no_grad()
    def forward(self, x, params, g):
        """x bf16 [T, n] -> bf16 [T, n]; call decode() for this layer first."""
        e = self.effective(params)
        h = self.attention_half(x, e, g)
        out = torch.empty_like(h)
        for sl in self.chunks(x.shape[0]):
            out[sl] = self.mlp_rows(h[sl], e)
        return out

    # ------------------------------------------------------------------ backward
    def accumulators(self):
        n, qd, kd, inter = self.n, self.qd, self.kd, self.inter
        sizes = {Q: qd, K: kd, V: kd, O: n, G: inter, U: inter, D: n, "su_qkv": n, "su_o": qd, "su_gu": n,
                 "su_d": inter, "w1": n, "w2": n}
        return {key: torch.zeros(size, dtype=F32, device=self.device) for key, size in sizes.items()}

    @torch.no_grad()
    def backward(self, x, params, g, dout):
        """Recompute the layer from its input x and return dL/dx (FP32, in dout's storage).

        Parameter gradients are accumulated into params[...].grad (tied SU once per group).
        """
        e = self.effective(params)
        n, qd, kd, inter = self.n, self.qd, self.kd, self.inter
        t = x.shape[0]
        keep = {}
        h = self.attention_half(x, e, g, keep)
        acc = self.accumulators()
        dh = dout
        for sl in self.chunks(t):
            rows = sl.stop - sl.start
            m = {}
            self.mlp_rows(h[sl], e, m)
            dy = dh[sl]
            with self.timer("grad_reduce"):
                self.colsum(dy, m["cd"], acc[D])
            db = self.empty(rows, n)
            with self.timer("hadamard"):
                self.had(dy, n, True, pre=e[D], out16=db)
            with self.timer("gemm_t"):
                da = self.mm_nt(db, self.w["d"])
            del db
            dxsu = self.empty(rows, inter, dtype=F32)
            with self.timer("hadamard"):
                self.had(da, inter, out32=dxsu)
            del da
            with self.timer("grad_reduce"):
                self.colsum(dxsu, m["product"], acc["su_d"])
            with self.timer("swiglu_bwd"):
                dg, dup = self.swiglu_bwd(dxsu, e["su_d"], m["gate"], m["up"])
            del dxsu
            with self.timer("grad_reduce"):
                self.colsum(dg, m["cg"], acc[G])
                self.colsum(dup, m["cu"], acc[U])
            dbgu = self.empty(rows, 2 * inter)
            with self.timer("hadamard"):
                self.had(dg, inter, True, pre=e[G], out16=dbgu[:, :inter])
                self.had(dup, inter, True, pre=e[U], out16=dbgu[:, inter:])
            del dg, dup
            with self.timer("gemm_t"):
                da = self.mm_nn(dbgu, self.w["gu"])
            del dbgu
            dxsu = self.empty(rows, n, dtype=F32)
            with self.timer("hadamard"):
                self.had(da, n, out32=dxsu)
            del da
            with self.timer("norm_bwd"):
                self.norm_backward(h[sl], m["r2"], e["w2"], e["su_gu"], m["xn2"], dxsu, dh[sl], acc, "su_gu", "w2")
            del m, dxsu
        del h
        do = self.empty(t, self.heads, self.hd)
        for sl in self.chunks(t):
            rows = sl.stop - sl.start
            dy = dh[sl]
            with self.timer("grad_reduce"):
                self.colsum(dy, keep["c_o"][sl], acc[O])
            db = self.empty(rows, n)
            with self.timer("hadamard"):
                self.had(dy, n, True, pre=e[O], out16=db)
            with self.timer("gemm_t"):
                da = self.mm_nn(db, self.w["o"])
            del db
            dxsu = self.empty(rows, qd, dtype=F32)
            with self.timer("hadamard"):
                self.had(da, qd, out32=dxsu, out16=do[sl].view(rows, qd), post=e["su_o"])
            del da
            with self.timer("grad_reduce"):
                self.colsum(dxsu, keep["o"][sl].view(rows, qd), acc["su_o"])
            del dxsu
        del keep["c_o"]
        with self.timer("attention_bwd"):
            dq, dk, dv = self.attn_backward(do, keep["q"], keep["k"], keep["v"], keep["o"], keep["attn"], g)
        del do, keep["q"], keep["k"], keep["v"], keep["o"], keep["attn"]
        spans = ((Q, 0, qd), (K, qd, qd + kd), (V, qd + kd, qd + 2 * kd))
        for sl in self.chunks(t):
            rows = sl.stop - sl.start
            with self.timer("rope"):
                dys = (self.rope(dq[sl], self.heads, g, sl, backward=True).view(rows, qd),
                       self.rope(dk[sl], self.kv, g, sl, backward=True).view(rows, kd),
                       dv[sl].float().view(rows, kd))
            c = keep["c_qkv"][sl]
            db = self.empty(rows, qd + 2 * kd)
            for (name, a, b), dy in zip(spans, dys):
                with self.timer("grad_reduce"):
                    self.colsum(dy, c[:, a:b], acc[name])
                with self.timer("hadamard"):
                    self.had(dy, b - a, True, pre=e[name], out16=db[:, a:b])
            del dys
            with self.timer("gemm_t"):
                da = self.mm_nn(db, self.w["qkv"])
            del db
            dxsu = self.empty(rows, n, dtype=F32)
            with self.timer("hadamard"):
                self.had(da, n, out32=dxsu)
            del da
            with self.timer("norm_bwd"):
                self.norm_backward(x[sl], keep["r1"][sl], e["w1"], e["su_qkv"], keep["xn1"][sl], dxsu, dh[sl], acc,
                                   "su_qkv", "w1")
            del dxsu
        with self.timer("grad_reduce"):
            self.accumulate(params, acc)
        return dh

    @staticmethod
    def accumulate(p, acc):
        def add(param, value):
            if param.grad is None:
                param.grad = value.detach().clone().reshape(param.shape)
            else:
                param.grad.add_(value.reshape(param.shape))
        for name in LINEARS:
            sv, scale = p[name + ".SV"], p[name + ".Wscale"]
            add(sv, acc[name] * scale.detach())
            add(scale, (acc[name] * sv.detach()).sum())
        for key, name in (("su_qkv", Q), ("su_o", O), ("su_gu", G), ("su_d", D)):
            add(p[name + ".SU"], acc[key])
        add(p[NORMS[0]], acc["w1"])
        add(p[NORMS[1]], acc["w2"])


def final_norm(x, weight, eps):
    z = x.float()
    r = torch.rsqrt(z.square().mean(-1, keepdim=True) + eps)
    return (z * r * bf(weight)).to(BF16), r


@torch.no_grad()
def head_loss(layer, final, head, labels, scale=1.0, chunk=4096, backward=True):
    """Summed CE over target rows; dfinal = d(scale * loss)/dfinal (FP32) when backward."""
    valid = torch.nonzero(labels != -100).flatten()
    total = torch.zeros((), dtype=F32, device=final.device)
    dfinal = torch.zeros(final.shape, dtype=F32, device=final.device) if backward else None
    for idx in valid.split(chunk):
        logits = layer.mm_nt(final.index_select(0, idx), head)
        y = labels.index_select(0, idx)
        lse = torch.logsumexp(logits, -1)
        total += (lse - logits.gather(1, y[:, None]).squeeze(1)).sum()
        if backward:
            p = logits.sub_(lse[:, None]).exp_()
            p[torch.arange(len(idx), device=p.device), y] -= 1
            p.mul_(scale)
            dfinal.index_copy_(0, idx, layer.mm_nn(p.to(BF16), head))
        del logits
    return total, dfinal
