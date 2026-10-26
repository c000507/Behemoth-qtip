"""Cross-language QTIP checks against the actual runtime, with no downloaded model."""
import ctypes as C
import json
import subprocess
import sys

import pytest
from safetensors.torch import load_file, save_file
import torch

from test_training import ROOT, tiny_model, batch, reference_tlut
from qtip_training.checkpoint import CPUStager, export_model, load_checkpoint, save_checkpoint
from qtip_training.kernels import decode_rows, hadamard, load_backend, qtip_linear, RuntimeReferenceBackend
from qtip_training.model import HostModel, SU_GROUPS
from qtip_training.streaming import Replica, reduce_gradients


@pytest.fixture(scope="session")
def host_oracle(tmp_path_factory):
    build = tmp_path_factory.mktemp("native-host")
    subprocess.run([sys.executable, "-m", "cmake", "-S", str(ROOT / "qtip_training/native"),
                    "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"], check=True, capture_output=True)
    subprocess.run([sys.executable, "-m", "cmake", "--build", str(build), "-j", "2"],
                   check=True, capture_output=True)
    library = C.CDLL(str(build / "libqtip_training_host.so"))
    library.qtip_training_error.restype = C.c_char_p
    library.qtip_host_decode.argtypes = [C.c_void_p] * 2 + [C.c_int] * 5 + [C.c_void_p]
    library.qtip_host_forward.argtypes = [C.c_void_p] * 5 + [C.c_int] * 5 + [C.c_void_p]
    return library


@pytest.mark.parametrize("rounded", [False, True])
def test_runtime_decoder_exact(host_oracle, rounded):
    torch.manual_seed(17)
    m, n = 64, 96
    codes = torch.randint(-32768, 32767, (m * n // 256, 32), dtype=torch.int16)
    lut = reference_tlut()
    output = torch.empty(m, n)
    result = host_oracle.qtip_host_decode(codes.data_ptr(), lut.data_ptr(), m, n, 0, m, rounded, output.data_ptr())
    assert result == 0, host_oracle.qtip_training_error()
    effective = lut.bfloat16().float() if rounded else lut
    torch.testing.assert_close(output, decode_rows(codes, effective, m, n), atol=0, rtol=0)


@pytest.mark.parametrize("round_z", [False, True])
def test_runtime_projection_rounding_and_ste_adjoint(host_oracle, round_z):
    torch.manual_seed(3)
    m, n = 64, 96
    codes = torch.randint(-32768, 32767, (m * n // 256, 32), dtype=torch.int16)
    lut = reference_tlut()
    x, su, sv, scale = [a.requires_grad_() for a in (
        torch.randn(3, n), torch.randn(n), torch.randn(m), torch.tensor(0.1))]
    backend = RuntimeReferenceBackend(32)
    actual = qtip_linear(x, su, sv, scale, codes, lut, backend, round_z)
    su_file, sv_file = su.detach().bfloat16().float(), (sv * scale).detach()
    expected = torch.empty_like(actual)
    result = host_oracle.qtip_host_forward(codes.data_ptr(), lut.data_ptr(), su_file.data_ptr(),
        sv_file.data_ptr(), x.data_ptr(), 3, m, n, 1, round_z, expected.data_ptr())
    assert result == 0, host_oracle.qtip_training_error()
    torch.testing.assert_close(actual, expected, atol=8e-6, rtol=2e-5)

    def ste(a):
        return a + (a.bfloat16().to(a.dtype) - a).detach()

    w = decode_rows(codes, lut.bfloat16().float(), m, n)
    a = ste(hadamard(x * ste(su), True))
    b = a @ w.T
    if round_z:
        b = ste(b)
    oracle = hadamard(b) * (sv * scale)
    upstream = torch.randn_like(actual)
    left = torch.autograd.grad(actual, (x, su, sv, scale), upstream)
    right = torch.autograd.grad(oracle, (x, su, sv, scale), upstream)
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=3e-5)


def test_shared_signs_updated_once_resume_and_export(tmp_path):
    host = tiny_model(tmp_path / "model", 1)
    replica = Replica(host, "cpu", RuntimeReferenceBackend(), 8)
    optimizer = replica.optimizer(0.001, 0.0001)
    for group in SU_GROUPS:
        assert all(replica.params[0][key] is replica.params[0][group[0]] for key in group)
    assert len(replica.parameters) == len({id(p) for p in replica.parameters})
    assert sum(len(group["params"]) for group in optimizer.param_groups) == len(replica.parameters)
    replica.run(batch(), 6)
    reduce_gradients([replica])
    optimizer.step()
    stager = CPUStager()
    save_checkpoint(tmp_path / "state.pt", replica, optimizer, {}, {}, stager)
    load_checkpoint(tmp_path / "state.pt", [replica], [optimizer], {})
    export_model(host, stager.copy(replica.named_parameters()), tmp_path / "export")
    tensors = load_file(tmp_path / "export/model.safetensors")
    for group in SU_GROUPS:
        assert all(torch.equal(tensors["model.layers.0." + key], tensors["model.layers.0." + group[0]]) for key in group)
    HostModel(tmp_path / "export")  # Same compatibility checks as before training.
    incompatible = stager.copy(replica.named_parameters())
    incompatible["model.layers.0.self_attn.k_proj.SU"][0] += 1
    with pytest.raises(ValueError, match="shared SU"):
        export_model(host, incompatible, tmp_path / "invalid")
    state = torch.load(tmp_path / "state.pt", weights_only=True)
    state["format_version"] = 1
    torch.save(state, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="signature mismatch"):
        load_checkpoint(tmp_path / "old.pt", [replica], [optimizer], {})


@pytest.mark.parametrize("bad", ["su", "tlut"])
def test_rejects_checkpoints_runtime_would_reject(tmp_path, bad):
    host = tiny_model(tmp_path / "model", 1)
    path = host.path / "model.safetensors"
    tensors = load_file(path)
    name = "model.layers.0.self_attn.k_proj." + ("SU" if bad == "su" else "tlut")
    tensors[name].view(-1)[0] += 0.1
    save_file(tensors, path)
    with pytest.raises(ValueError, match="shared"):
        HostModel(host.path)


def test_native_backend_never_silently_falls_back():
    if not torch.version.hip:
        with pytest.raises(RuntimeError, match="ROCm"):
            load_backend("runtime")


def test_capture_rejects_packed_documents_before_invoking_runtime(tmp_path):
    prompt = tmp_path / "packed.safetensors"
    save_file(dict(tokens=torch.tensor([1, 2, 3]), segments=torch.tensor([0, 0, 1])), prompt)
    result = subprocess.run([sys.executable, str(ROOT / "tools/capture_qat_runtime.py"),
        "--model", "unused", "--eval-binary", "must-not-be-invoked", "--prompt", str(prompt),
        "--out", str(tmp_path / "trace.safetensors")], capture_output=True, text=True)
    assert result.returncode == 2
    assert "one document" in result.stderr
    assert not (tmp_path / "trace.safetensors").exists()


def test_logit_checker_rejects_other_source_model(tmp_path):
    host = tiny_model(tmp_path / "model", 1)
    prompt = tmp_path / "trace.safetensors"
    save_file(dict(tokens=torch.tensor([1, 2, 3]), logit_positions=torch.tensor([2]),
                   logits=torch.zeros(1, 96)), prompt, metadata={"model_sha256": "different-source"})
    result = subprocess.run([sys.executable, str(ROOT / "tools/check_qat_runtime.py"),
        "--model", str(host.path), "--trace", str(prompt), "--device", "cpu"], capture_output=True, text=True)
    assert result.returncode == 2
    assert "different source model" in result.stderr


@pytest.mark.skipif(not torch.version.hip or not torch.cuda.is_available(), reason="ROCm hardware required")
def test_native_projection_adjoint_slot_reuse_and_nondefault_stream():
    from qtip_training.runtime_backend import RuntimeBackend
    native = RuntimeBackend(weight_rows=32)  # Fail, don't skip, if a HIP build was requested but is missing.
    reference = RuntimeReferenceBackend(32)
    device = "cuda:0"
    with torch.cuda.device(device), torch.cuda.stream(torch.cuda.Stream()):
        m, n = 64, 1024
        codes = torch.randint(-32768, 32767, (m * n // 256, 32), dtype=torch.int16, device=device)
        lut = reference_tlut().to(device)
        x = torch.randn(17, n, device=device)
        dy = torch.randn(17, m, device=device)
        for factor in (1., 0.5):
            # Same data_ptr with new contents invalidates the LUT cache.
            lut.mul_(factor)
            torch.testing.assert_close(native.matmul(x, codes, lut, m, n),
                                       reference.matmul(x, codes, lut, m, n), atol=2e-4, rtol=2e-4)
            torch.testing.assert_close(native.matmul(dy, codes, lut, m, n, True),
                                       reference.matmul(dy, codes, lut, m, n, True), atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(native.hadamard(x), hadamard(x), atol=2e-5, rtol=2e-5)
