"""Fast engine (qtip_training/fast.py) against the autograd decoder_layer oracle.

CPU tests use the tiny model; GPU tests cover the native row kernels at the real
Behemoth dimensions and, when the real checkpoint is present, one real layer.
"""
import os
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from qtip_training.fast import FastLayer, Geometry, head_loss, rope_backward, rope_forward
from qtip_training.kernels import RuntimeReferenceBackend, hadamard, rope
from qtip_training.model import decoder_layer, HostModel, SU_GROUPS
from qtip_training.streaming import Replica
from test_training import batch, tiny_model

REAL_MODEL = Path(os.environ.get("QTIP_REAL_MODEL", "/mnt/nvme/qtip_quant/Behemoth-R1-123B-v2-QTIP-2bit"))


def rel(a, b):
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def params_of(host, layer, device="cpu"):
    params = {k: torch.nn.Parameter(v.to(device).clone()) for k, v in host.initial[layer].items()}
    for group in SU_GROUPS:
        for name in group[1:]:
            params[name] = params[group[0]]
    return params


def test_rope_backward_is_adjoint():
    torch.manual_seed(0)
    pos = torch.tensor([0, 1, 2, 0, 1, 5, 9])
    x = torch.randn(7, 3, 16, dtype=torch.float64)
    config = dict(hidden_size=48, num_attention_heads=3, head_dim=16, rope_theta=1e6)
    g = Geometry(torch.zeros(7, dtype=torch.long), pos, config, "cpu")
    reference = rope(x.transpose(0, 1).float(), pos, 1e6).transpose(0, 1)
    torch.testing.assert_close(rope_forward(x.bfloat16(), g.cos, g.sin).float(),
                               rope(x.transpose(0, 1).bfloat16(), pos, 1e6).transpose(0, 1).float())
    dy = torch.randn_like(x).float()
    lhs = (reference.double() * dy.double()).sum()
    rhs = (x.double() * rope_backward(dy, g.cos, g.sin).double()).sum()
    assert float(lhs) == pytest.approx(float(rhs), rel=1e-5)


def test_fast_layer_matches_autograd_oracle_cpu(tmp_path):
    host = tiny_model(tmp_path / "model", 1)
    backend = RuntimeReferenceBackend(32)
    _, _, segments, positions = batch()
    torch.manual_seed(5)
    x = (torch.randn(8, 64) * 0.5).bfloat16()
    dy = torch.randn(8, 64)
    ref_params = params_of(host, 0)
    value = x.clone().requires_grad_(True)
    expected = decoder_layer(value, host.layers[0], ref_params, host.config, segments, positions, backend, 3, 3)
    expected.backward(dy.bfloat16())
    for chunk in (3, 8):
        layer = FastLayer(host.config, "cpu", backend, row_chunk=chunk)
        geometry = Geometry(segments, positions, host.config, "cpu")
        params = params_of(host, 0)
        layer.decode(host.layers[0])
        out = layer.forward(x, params, geometry)
        assert rel(out.float(), expected.float()) < 1e-2
        dx = layer.backward(x, params, geometry, dy.clone())
        assert rel(dx, value.grad.float()) < 3e-2
        for key in params:
            assert rel(params[key].grad, ref_params[key].grad) < 3e-2, key


def test_fast_replica_loss_gradients_and_overfit_cpu(tmp_path):
    host = tiny_model(tmp_path / "model", 2)
    backend = RuntimeReferenceBackend(32)
    tokens, labels, segments, positions = batch()
    total = int((labels != -100).sum())
    auto = Replica(host, "cpu", backend, 8, 3, 3, 3)
    quick = Replica(host, "cpu", backend, 8, 3, 3, 3, engine="fast", fast_rows=3)
    a = auto.run(batch(), normalizer=total)
    b = quick.run(batch(), normalizer=total)
    assert b == pytest.approx(a, rel=1e-2)
    # The autograd path rounds inter-layer gradients to bf16; the fast engine keeps FP32.
    # Scalars (Wscale) are sums with cancellation, so compare the whole gradient vector
    # tightly and single tensors loosely.
    left = torch.cat([q.grad.flatten() for q in quick.named_parameters().values()])
    right = torch.cat([p.grad.flatten() for p in auto.named_parameters().values()])
    assert rel(left, right) < 2e-2
    for (key, p), q in zip(auto.named_parameters().items(), quick.named_parameters().values()):
        assert rel(q.grad, p.grad) < 0.15, key
    optimizer = quick.optimizer(0.015, 0.01)
    losses = []
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        losses.append(quick.run(batch(), normalizer=total) / total)
        optimizer.step()
    assert losses[-1] < losses[0] / 10


def test_head_loss_matches_cross_entropy_cpu(tmp_path):
    host = tiny_model(tmp_path / "model", 1)
    layer = FastLayer(host.config, "cpu", RuntimeReferenceBackend(32))
    torch.manual_seed(2)
    final = torch.randn(9, 64).bfloat16()
    labels = torch.tensor([1, -100, 4, 3, -100, 2, 12, 0, -100])
    head = host.head
    x = final.float().requires_grad_(True)
    reference = torch.nn.functional.cross_entropy(x @ head.float().T, labels, reduction="sum")
    reference.backward()
    loss, dfinal = head_loss(layer, final, head, labels, 0.5, chunk=2)
    assert float(loss) == pytest.approx(float(reference), rel=1e-5)
    assert rel(dfinal, 0.5 * x.grad) < 1e-2
    assert float(dfinal[labels == -100].abs().sum()) == 0


hip = pytest.mark.skipif(not torch.version.hip or not torch.cuda.is_available(), reason="ROCm hardware required")


@hip
@pytest.mark.parametrize("n", [1024, 12288, 28672])
@pytest.mark.parametrize("transpose", [False, True])
def test_native_row_transform_strided_scales_and_rounding(n, transpose):
    from qtip_training.runtime_backend import RuntimeBackend
    native = RuntimeBackend()
    device = "cuda:0"
    torch.manual_seed(n)
    with torch.cuda.device(device), torch.cuda.stream(torch.cuda.Stream()):
        x = torch.randn(19, n + 64, device=device)[:, 32:32 + n]
        pre, post = torch.randn(n, device=device), torch.randn(n, device=device)
        out32 = torch.empty(19, n, device=device)
        out16 = torch.empty(19, n + 8, device=device, dtype=torch.bfloat16)[:, 4:4 + n]
        native.had(x, n, transpose, pre=pre, out32=out32, out16=out16, post=post)
        expected = hadamard(x.double() * pre.double(), transpose)
        torch.testing.assert_close(out32.double(), expected, atol=1e-5, rtol=1e-5)
        assert ((out16.double() - (expected * post.double())).abs() <=
                (expected * post.double()).abs() * 2 ** -8 + 1e-6).all()
        xb = x.bfloat16()
        native.had(x, n, transpose, round_in=True, out32=out32)
        torch.testing.assert_close(out32.double(), hadamard(xb.double(), transpose), atol=1e-5, rtol=1e-5)
        native.had(xb, n, transpose, out32=out32)
        torch.testing.assert_close(out32.double(), hadamard(xb.double(), transpose), atol=1e-5, rtol=1e-5)


@hip
@pytest.mark.skipif(not (REAL_MODEL / "config.json").exists(), reason="real Behemoth QTIP checkpoint required")
def test_real_layer_fast_engine_matches_runtime_reference_oracle():
    """One real Behemoth layer (decode, transforms, GEMMs, flash attention, analytic backward)
    against the autograd decoder_layer with the runtime-reference backend (FP32 math)."""
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    host = HostModel(REAL_MODEL, layers=[3], identity=False)
    weights = {k: v.to(device) for k, v in host.layers[0].items()}
    t = 384
    g = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, host.config["vocab_size"], (t,), generator=g)
    x = host.embedding[tokens].to(device=device, dtype=torch.bfloat16) * 8
    segments = torch.tensor([0] * 200 + [1] * 184)
    positions = torch.cat([torch.arange(200), torch.arange(184)])
    dy = torch.randn(t, x.shape[1], generator=g).to(device) * 1e-2
    from qtip_training.runtime_backend import RuntimeBackend
    native = RuntimeBackend()
    layer = FastLayer(host.config, device, native, row_chunk=128)
    geometry = Geometry(segments, positions, host.config, device)
    params = params_of(host, 0, device)
    layer.decode(weights)
    out = layer.forward(x, params, geometry)
    dx = layer.backward(x, params, geometry, dy.clone())
    oracle = RuntimeReferenceBackend(512)
    ref_params = params_of(host, 0, device)
    value = x.clone().requires_grad_(True)
    expected = decoder_layer(value, weights, ref_params, host.config, segments.to(device), positions.to(device),
                             oracle, 128, 128)
    expected.backward(dy.bfloat16())
    assert rel(out.float() - x.float(), expected.float() - x.float()) < 1e-2
    assert rel(dx, value.grad.float()) < 2e-2
    # Vector gradients agree to ~0.5%. Scalar Wscale gradients are long sums with heavy
    # cancellation: both bf16 paths are 1-5% from an FP64 oracle there (docs/FINETUNE_PERF_REPORT.md).
    for key in params:
        assert rel(params[key].grad, ref_params[key].grad) < (0.1 if params[key].dim() == 0 else 2e-2), key


def test_snapshot_overwrite_is_atomic(tmp_path, monkeypatch):
    """latest.pt is replaced only by a complete file; a failed write keeps the previous snapshot."""
    from qtip_training import checkpoint as ckpt
    host = tiny_model(tmp_path / "model", 1)
    replica = Replica(host, "cpu", RuntimeReferenceBackend(32), 8, engine="fast")
    optimizer = replica.optimizer(1e-3, 1e-4)
    stager = ckpt.CPUStager()
    path = tmp_path / "work" / "latest.pt"
    ckpt.save_checkpoint(path, replica, optimizer, {"step": 10}, {"s": 1}, stager)
    first = path.read_bytes()

    def broken(state, handle):
        handle.write(b"partial")
        raise OSError("disk full")
    monkeypatch.setattr(ckpt.torch, "save", broken)
    with pytest.raises(OSError):
        ckpt.save_checkpoint(path, replica, optimizer, {"step": 20}, {"s": 1}, stager)
    monkeypatch.undo()
    assert path.read_bytes() == first
    assert sorted(p.name for p in path.parent.iterdir()) == ["latest.pt"]
    assert ckpt.load_checkpoint(path, [replica], [optimizer], {"s": 1}) == {"step": 10}
    ckpt.save_checkpoint(path, replica, optimizer, {"step": 20}, {"s": 1}, stager)
    assert ckpt.load_checkpoint(path, [replica], [optimizer], {"s": 1}) == {"step": 20}
