"""CPU verification for QAT math, full-layer parity, streaming, resume and export.

Run: OMP_NUM_THREADS=2 python -m pytest -q tests/test_training.py
These do not establish HIP or behemoth_server parity.
"""
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.torch import save_file
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qtip_training.checkpoint import CPUStager, export_model, load_checkpoint, save_checkpoint
from qtip_training.data import PackedDataset, pack_documents, render, split_ids, tokenize_conversation, write_dataset
from qtip_training.kernels import (attention, cross_entropy, decode_rows, hadamard, qtip_linear,
                                   ReferenceBackend, rmsnorm, rope)
from qtip_training.model import decoder_layer, HostModel, LINEARS, NORMS, SU_GROUPS
from qtip_training.streaming import reduce_gradients, Replica

torch.set_num_threads(2)


def vector(name, dtype, shape):
    return torch.from_numpy(np.fromfile(ROOT / "tests/refvec" / name, dtype=dtype)).reshape(shape)


def reference_tlut():
    source = (ROOT / "src/qtip_constants.hpp").read_text()
    bits = re.search(r"kTlutF32Bits\[1024\] = \{([^}]+)", source)[1]
    array = np.array([int(x, 16) for x in bits.split(",")], dtype=np.uint32).view(np.float32)
    return torch.from_numpy(array).reshape(512, 2).half()


@pytest.mark.parametrize("n", [96, 1024, 12288, 28672])
@pytest.mark.parametrize("transpose,suffix", [(False, "u"), (True, "ut")])
def test_hadamard_upstream_vectors(n, transpose, suffix):
    x = vector(f"had_{n}_x.bin", np.float64, (4, n))
    expected = vector(f"had_{n}_{suffix}.bin", np.float64, (4, n))
    torch.testing.assert_close(hadamard(x, transpose), expected, atol=2e-7, rtol=1e-7)
    torch.testing.assert_close(hadamard(hadamard(x, transpose), not transpose), x)


def test_decoder_upstream_vectors_and_row_tiles():
    codes = vector("pack_kernel.bin", np.int16, (-1, 32))
    states = vector("pack_states.bin", np.int32, (24, 128)).long()
    lut = vector("lut.bin", np.float32, (2, 65536))
    perm = vector("perm.bin", np.int64, (256,))
    sequence = torch.stack((lut[0, states], lut[1, states]), -1).reshape(24, 256)
    expected = sequence[:, torch.argsort(perm)].reshape(4, 6, 16, 16).permute(0, 2, 1, 3).reshape(64, 96)
    actual = decode_rows(codes, reference_tlut(), 64, 96)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(decode_rows(codes, reference_tlut(), 64, 96, 32, 64), expected[32:], atol=0, rtol=0)
    backend = ReferenceBackend(32)
    x, dy = torch.randn(3, 96), torch.randn(3, 64)
    torch.testing.assert_close(backend.matmul(x, codes, reference_tlut(), 64, 96), x @ expected.T)
    torch.testing.assert_close(backend.matmul(dy, codes, reference_tlut(), 64, 96, True), dy @ expected)


def test_linear_all_continuous_and_input_gradients():
    torch.manual_seed(4)
    m, n = 32, 96  # Includes a nonsymmetric hadK, catches transpose mistakes.
    codes = torch.randint(-32768, 32767, (m * n // 256, 32), dtype=torch.int16)
    lut = torch.randn(512, 2, dtype=torch.float64)
    x, su, sv, scale = [v.requires_grad_() for v in (torch.randn(2, n, dtype=torch.float64),
                          torch.randn(n, dtype=torch.float64), torch.randn(m, dtype=torch.float64),
                          torch.tensor(0.13, dtype=torch.float64))]
    backend = ReferenceBackend(32)
    assert torch.autograd.gradcheck(lambda *args: qtip_linear(*args, codes, lut, backend),
                                    (x, su, sv, scale), fast_mode=True)
    w = decode_rows(codes, lut, m, n, dtype=torch.float64)
    effective = hadamard(hadamard(w).T).T * sv[:, None] * su[None, :] * scale
    y = qtip_linear(x, su, sv, scale, codes, lut, backend)
    torch.testing.assert_close(y, x @ effective.T)


@pytest.mark.parametrize("block", [1, 3, 16])
def test_attention_dense_parity_and_finite_differences(block):
    torch.manual_seed(1)
    q = torch.randn(4, 7, 4, dtype=torch.float64, requires_grad=True)
    k, v = [torch.randn(2, 7, 4, dtype=torch.float64, requires_grad=True) for _ in range(2)]
    segments = torch.tensor([0, 0, 1, 1, 1, -1, -1])
    mask = torch.ones(7, 7, dtype=torch.bool).tril() & (segments[:, None] == segments[None, :])
    dense = F.scaled_dot_product_attention(q, k.repeat_interleave(2, 0), v.repeat_interleave(2, 0), attn_mask=mask)
    actual = attention(q, k, v, segments, block)
    torch.testing.assert_close(actual, dense)
    grad = torch.randn_like(actual)
    a = torch.autograd.grad(actual, (q, k, v), grad, retain_graph=True)
    b = torch.autograd.grad(dense, (q, k, v), grad)
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right)
    assert torch.autograd.gradcheck(lambda *args: attention(*args, segments, block), (q, k, v), fast_mode=True)


def test_norm_swiglu_rope_and_chunked_loss_gradients():
    torch.manual_seed(3)
    x = torch.randn(7, 8, dtype=torch.float64, requires_grad=True)
    norm = torch.randn(8, dtype=torch.float64, requires_grad=True)
    pos = torch.tensor([0, 1, 2, 0, 1, 0, 1])
    assert torch.autograd.gradcheck(lambda x, w: rmsnorm(x, w, 1e-5), (x, norm), fast_mode=True)
    assert torch.autograd.gradcheck(lambda x: rope(x[None], pos, 1e6), (x,), fast_mode=True)
    up = torch.randn_like(x, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x, u: F.silu(x) * u, (x, up), fast_mode=True)
    head = torch.randn(13, 8, dtype=torch.float64)
    labels = torch.tensor([1, -100, 4, 3, -100, 2, 12])
    loss = cross_entropy(x, head, labels, 3)
    reference = F.cross_entropy(x @ head.T, labels, reduction="sum")
    torch.testing.assert_close(loss, reference)
    torch.testing.assert_close(torch.autograd.grad(loss, x, retain_graph=True)[0], torch.autograd.grad(reference, x)[0])
    assert torch.autograd.gradcheck(lambda x: cross_entropy(x, head, labels, 2), (x,), fast_mode=True)
    torch.testing.assert_close(cross_entropy(x, head, torch.full_like(labels, -100)), torch.tensor(0., dtype=x.dtype))


def tiny_model(path, layers=3):
    path.mkdir()
    torch.manual_seed(13)
    config = dict(hidden_size=64, intermediate_size=96, num_attention_heads=4, num_key_value_heads=2,
                  head_dim=16, vocab_size=96, num_hidden_layers=layers, rope_theta=1e6, rms_norm_eps=1e-5,
                  quip_params=dict(L=16, K=2, V=2, tlut_bits=9, decode_mode="quantlut_sym", td_x=16, td_y=16))
    tensors = {"model.embed_tokens.weight": torch.randn(96, 64).bfloat16(),
               "lm_head.weight": (torch.randn(96, 64) * 0.2).bfloat16(),
               "model.norm.weight": torch.ones(64).bfloat16()}
    for layer in range(layers):
        prefix = f"model.layers.{layer}."
        for linear, (m, n) in zip(LINEARS, [(64, 64), (32, 64), (32, 64), (64, 64), (96, 64), (96, 64), (64, 96)]):
            p = prefix + linear + "."
            tensors[p + "SU"] = (torch.randint(0, 2, (n,)) * 2 - 1).bfloat16()
            tensors[p + "SV"] = (torch.randint(0, 2, (m,)) * 2 - 1).float() * 0.025
            tensors[p + "trellis"] = torch.randint(-32768, 32767, (m * n // 256, 32), dtype=torch.int16)
            tensors[p + "tlut"] = reference_tlut()
            tensors[p + "rcp"] = torch.tensor(0)
            tensors[p + "tp_rank"] = torch.tensor(8)
        for norm in NORMS:
            tensors[prefix + norm] = torch.ones(64).bfloat16()
        for group in SU_GROUPS:
            for name in group[1:]:
                tensors[prefix + name] = tensors[prefix + group[0]].clone()
    save_file(tensors, path / "model.safetensors", metadata={"format": "pt"})
    (path / "config.json").write_text(json.dumps(config))
    (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "model.safetensors" for k in tensors}}))
    return HostModel(path)


def batch(seed=1):
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, 96, (8,), generator=generator)
    labels = torch.cat((tokens[1:], torch.tensor([-100])))
    labels[3] = -100
    return tokens, labels, torch.tensor([0] * 4 + [1] * 4), torch.tensor([0, 1, 2, 3] * 2)


def dense_layer(x, frozen, params, config, segments, positions):
    """Independent dense autograd oracle: no custom QTIP/attention backwards."""
    def linear(x, name):
        su, sv = params[name + ".SU"], params[name + ".SV"]
        w = decode_rows(frozen[name + ".trellis"], frozen[name + ".tlut"], len(sv), len(su), dtype=x.dtype)
        w = hadamard(hadamard(w).T).T * sv[:, None] * su[None] * params[name + ".Wscale"]
        return x @ w.T

    def norm(x, key):
        return x * (x.square().mean(-1, keepdim=True) + config["rms_norm_eps"]).rsqrt() * params[key]

    def rot(x):
        inv = config["rope_theta"] ** (-torch.arange(0, 16, 2, dtype=x.dtype) / 16)
        a = positions[:, None] * inv[None, :]
        c, s = a.cos().repeat(1, 2), a.sin().repeat(1, 2)
        return x * c + torch.cat((-x[..., 8:], x[..., :8]), -1) * s

    h = norm(x, NORMS[0])
    q = rot(linear(h, "self_attn.q_proj").reshape(8, 4, 16).transpose(0, 1))
    k = rot(linear(h, "self_attn.k_proj").reshape(8, 2, 16).transpose(0, 1)).repeat_interleave(2, 0)
    v = linear(h, "self_attn.v_proj").reshape(8, 2, 16).transpose(0, 1).repeat_interleave(2, 0)
    mask = torch.ones(8, 8, dtype=torch.bool).tril() & (segments[:, None] == segments[None, :])
    a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(0, 1).reshape(8, 64)
    x = x + linear(a, "self_attn.o_proj")
    h = norm(x, NORMS[1])
    return x + linear(F.silu(linear(h, "mlp.gate_proj")) * linear(h, "mlp.up_proj"), "mlp.down_proj")


def test_full_layer_forward_and_every_parameter_gradient(tmp_path):
    host = tiny_model(tmp_path / "model", 1)
    params = {k: v.double().requires_grad_() for k, v in host.initial[0].items()}
    x = torch.randn(8, 64, dtype=torch.float64, requires_grad=True)
    _, _, segments, positions = batch()
    actual = decoder_layer(x, host.layers[0], params, host.config, segments, positions, ReferenceBackend(), 3, 3)
    expected = dense_layer(x, host.layers[0], params, host.config, segments, positions)
    torch.testing.assert_close(actual, expected)
    grad = torch.randn_like(x)
    values = [x, *params.values()]
    a = torch.autograd.grad(actual, values, grad)
    b = torch.autograd.grad(expected, values, grad)
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, atol=1e-9, rtol=1e-7)


def test_streamed_multilayer_gradients_match_resident_autograd(tmp_path):
    host = tiny_model(tmp_path / "model")
    replica = Replica(host, "cpu", ReferenceBackend(), 8, 3, 3, 3)
    tokens, labels, segments, positions = batch()
    total = int((labels != -100).sum())
    loss = replica.run(batch(), normalizer=total)
    grads = {k: v.grad.clone() for k, v in replica.named_parameters().items()}
    for p in replica.parameters:
        p.grad = None
    x = host.embedding[tokens].float()
    for i in range(3):
        x = dense_layer(x, host.layers[i], replica.params[i], host.config, segments, positions)
    x = x * (x.square().mean(-1, keepdim=True) + host.config["rms_norm_eps"]).rsqrt() * replica.norm
    expected = F.cross_entropy(x @ replica.head.T, labels, reduction="sum")
    (expected / total).backward()
    assert loss == pytest.approx(expected.item(), rel=1e-6)
    for key, value in replica.named_parameters().items():
        torch.testing.assert_close(grads[key], value.grad, atol=3e-6, rtol=2e-4)


def step(replicas, optimizers, batches):
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    count = sum(int((b[1] != -100).sum()) for b in batches)
    losses = [r.run(b, count) for r, b in zip(replicas, batches)]
    reduce_gradients(replicas)
    for optimizer in optimizers:
        optimizer.step()
    return sum(losses) / count


def test_checkpoint_exact_resume_identity_and_frozen_export(tmp_path):
    host = tiny_model(tmp_path / "model")
    replica = Replica(host, "cpu", ReferenceBackend(), 8, 4, 4, 4)
    optimizer = replica.optimizer(1e-3, 1e-4)
    step([replica], [optimizer], [batch()])
    state = {"step": 1, "position": 1, "tokens": 8, "supervised_tokens": 6}
    save_checkpoint(tmp_path / "latest.pt", replica, optimizer, state, {"model": host.identity}, CPUStager())
    expected_loss = step([replica], [optimizer], [batch(2)])
    other = Replica(host, "cpu", ReferenceBackend(), 8, 4, 4, 4)
    other_optimizer = other.optimizer(1e-3, 1e-4)
    assert load_checkpoint(tmp_path / "latest.pt", [other], [other_optimizer], {"model": host.identity}) == state
    loss = step([other], [other_optimizer], [batch(2)])
    assert expected_loss == loss
    for a, b in zip(replica.parameters, other.parameters):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    with pytest.raises(ValueError, match="signature mismatch"):
        load_checkpoint(tmp_path / "latest.pt", [other], [other_optimizer], {"model": "changed"})
    export_model(host, CPUStager().copy(replica.named_parameters()), tmp_path / "export")
    with safe_open(tmp_path / "model/model.safetensors", "pt") as before, safe_open(tmp_path / "export/model.safetensors", "pt") as after:
        assert set(before.keys()) == set(after.keys())
        changed = 0
        for key in before.keys():
            a, b = before.get_tensor(key), after.get_tensor(key)
            assert a.dtype == b.dtype and a.shape == b.shape
            if key.endswith((".SU", ".SV", "layernorm.weight")) or key == "model.norm.weight":
                changed += int(not torch.equal(a, b))
            else:
                assert torch.equal(a, b), key
        assert changed > 0
    reloaded = HostModel(tmp_path / "export")
    replay = Replica(reloaded, "cpu", ReferenceBackend(), 8)
    # Export casts SU/norms to original bf16; compare against exactly those casts.
    for key, param in replica.named_parameters().items():
        if key.endswith((".SU", "layernorm.weight")) or key == "model.norm.weight":
            param.data.copy_(param.data.bfloat16().float())
    assert replay.run(batch(), backward=False) == pytest.approx(replica.run(batch(), backward=False), rel=1e-6)


def test_host_data_parallel_token_weighting_and_replica_isolation(tmp_path):
    host = tiny_model(tmp_path / "model", 1)
    replicas = [Replica(host, "cpu", ReferenceBackend(), 8) for _ in range(2)]
    original = host.initial[0][LINEARS[0] + ".SU"].clone()
    optimizers = [r.optimizer(1e-3, 1e-4) for r in replicas]
    b1, b2 = batch(), batch(2)
    b2[1][2:] = -100  # Deliberately unequal supervised counts.
    total = sum(int((b[1] != -100).sum()) for b in (b1, b2))
    for r, b in zip(replicas, (b1, b2)):
        r.run(b, total)
    expected = [a.grad.clone() + b.grad.clone() for a, b in zip(replicas[0].parameters, replicas[1].parameters)]
    reduce_gradients(replicas)
    for r in replicas:
        for param, grad in zip(r.parameters, expected):
            torch.testing.assert_close(param.grad, grad)
    for optimizer in optimizers:
        optimizer.step()
    for a, b in zip(replicas[0].parameters, replicas[1].parameters):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(host.initial[0][LINEARS[0] + ".SU"], original, atol=0, rtol=0)
    # AdamW scalar steps and moments must not alias after a multi-replica resume.
    save_checkpoint(tmp_path / "parallel.pt", replicas[0], optimizers[0], {}, {}, CPUStager())
    load_checkpoint(tmp_path / "parallel.pt", replicas, optimizers, {})
    step(replicas, optimizers, [b1, b2])
    for a, b in zip(replicas[0].parameters, replicas[1].parameters):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_overfit_single_batch(tmp_path):
    host = tiny_model(tmp_path / "model", 1)
    replica = Replica(host, "cpu", ReferenceBackend(), 8, 8, 8, 8)
    optimizer = replica.optimizer(0.015, 0.01)
    losses = [step([replica], [optimizer], [batch()]) for _ in range(100)]
    assert losses[-1] < 0.05
    assert losses[-1] < losses[0] / 50


class CharTokenizer:
    """A deterministic test template with explicit role headers and turn EOS."""
    def apply_chat_template(self, messages, **kwargs):
        return "".join(f"[{m['role']}]" + m["content"] + "!" for m in messages)

    def __call__(self, text, **kwargs):
        return dict(input_ids=[ord(c) for c in text], offset_mapping=[(i, i + 1) for i in range(len(text))])


def test_conversation_split_reasoning_masks_packing_and_checksums(tmp_path):
    train, held = split_ids(731)
    assert len(held) == 73 and not set(train) & set(held)
    assert split_ids(731, recorded=held)[0] == train
    messages = [{"role": "user", "content": "hello"},
                {"role": "assistant", "content": "<think>old</think>answer", "reasoning_content": "new"},
                {"role": "user", "content": "next"}, {"role": "assistant", "content": "yes"}]
    ids, mask = tokenize_conversation(CharTokenizer(), messages)
    text = "".join(chr(i) for i in ids)
    targets = "".join(chr(i) for i, use in zip(ids, mask) if use)
    assert targets == "<think>\nnew\n</think>\n\nanswer!yes!"
    assert "old" not in text and "[assistant]" not in targets and "hello" not in targets
    assert render(messages)[1]["content"].startswith("<think>\nnew")
    rows = pack_documents([(0, [1, 2, 3], [0, 1, 1]), (1, [4, 5], [0, 1])], 8, 0)
    write_dataset(tmp_path / "data", rows, dict(split="train", holdout_conversations=[9], training_conversations=[0, 1]))
    dataset = PackedDataset(tmp_path / "data")
    tokens, labels, seg, pos = dataset[0]
    assert labels.tolist() == [2, 3, -100, 5, -100, -100, -100, -100]
    assert pos.tolist() == [0, 1, 2, 0, 1, 0, 0, 0]
    assert seg.tolist() == [0, 0, 0, 1, 1, -1, -1, -1]
    with (tmp_path / "data/tokens.i32").open("ab") as handle:
        handle.write(b"oops")
    with pytest.raises(ValueError, match="checksum"):
        PackedDataset(tmp_path / "data")


def test_cli_crash_resume_and_one_epoch_cap(tmp_path):
    host = tiny_model(tmp_path / "model", 1)
    documents = [(i, [1, 2, 3, 4, 5, 6, 7, 8], [1] * 8) for i in range(3)]
    write_dataset(tmp_path / "data", pack_documents(documents, 8, 0),
                  dict(split="train", training_conversations=[0, 1, 2], holdout_conversations=[9]))
    args = [sys.executable, str(ROOT / "tools/finetune_qtip.py"), "--model", str(host.path),
            "--data", str(tmp_path / "data"), "--work", str(tmp_path / "work"), "--devices", "cpu",
            "--max-tokens", "100000", "--row-chunk", "8", "--attention-block", "4", "--warmup-steps", "1"]
    subprocess.run(args + ["--stop-after-steps", "1"], check=True, capture_output=True, text=True)
    subprocess.run(args + ["--resume", str(tmp_path / "work/latest.pt"), "--out", str(tmp_path / "out")],
                   check=True, capture_output=True, text=True)
    state = torch.load(tmp_path / "work/latest.pt", weights_only=True)
    assert state["progress"]["position"] == 3 and state["progress"]["step"] == 3
    assert state["progress"]["tokens"] == 24
    assert (tmp_path / "out/model.safetensors.index.json").exists()


def test_data_preparer_with_local_fast_tokenizer(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    engine = Tokenizer(models.WordLevel({"[UNK]": 0, "[EOS]": 1, "hello": 2, "answer": 3, "reason": 4}, unk_token="[UNK]"))
    engine.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=engine, unk_token="[UNK]", eos_token="[EOS]")
    tokenizer.chat_template = "{% for m in messages %}{{ '[' + m['role'] + ']' + m['content'] + eos_token }}{% endfor %}"
    tokenizer.save_pretrained(tmp_path / "tokenizer")
    conversations = [dict(messages=[dict(role="user", content="hello"),
                                    dict(role="assistant", content="answer", reasoning_content="reason")]) for _ in range(10)]
    (tmp_path / "conversations.jsonl").write_text("\n".join(json.dumps(c) for c in conversations))
    (tmp_path / "calibration.json").write_text(json.dumps(dict(holdout_conversations=[7])))
    np.full((64, 8), 2, dtype=np.int32).tofile(tmp_path / "general.i32")
    args = [sys.executable, str(ROOT / "tools/make_train_gmcot.py"),
            "--data", str(tmp_path / "conversations.jsonl"), "--tokenizer", str(tmp_path / "tokenizer"),
            "--split-metadata", str(tmp_path / "calibration.json"), "--seqlen", "16",
            "--general", str(tmp_path / "general.i32"), "--general-seqlen", "8"]
    subprocess.run(args + ["--out", str(tmp_path / "data")], check=True, capture_output=True, text=True)
    prepared = PackedDataset(tmp_path / "data")
    assert prepared.metadata["holdout_conversations"] == [7]
    assert 7 not in prepared.metadata["training_conversations"]
    assert prepared.metadata["general_sequences"] > 0
    assert sum(int((prepared[i][1] != -100).sum()) for i in range(len(prepared))) > 0
    subprocess.run(args + ["--split", "heldout", "--gm-frac", "1", "--out", str(tmp_path / "held")],
                   check=True, capture_output=True, text=True)
    with pytest.raises(ValueError, match="train dataset"):
        PackedDataset(tmp_path / "held")
    assert PackedDataset(tmp_path / "held", require_split="heldout").metadata["evaluated_conversations"] == [7]


def test_logit_checker_accepts_exact_and_rejects_changed_fixture(tmp_path):
    # Synthetic fixture checks the checker itself, NOT behemoth_server parity.
    host = tiny_model(tmp_path / "model", 1)
    replica = Replica(host, "cpu", ReferenceBackend(), 8)
    selected = torch.tensor([0, 3, 7])
    b = batch()
    logits = torch.cat(list(replica.logits(b, selected)))
    fixture = dict(tokens=b[0], logit_positions=selected, logits=logits, segments=b[2], positions=b[3])
    save_file(fixture, tmp_path / "trace.safetensors")
    args = [sys.executable, str(ROOT / "tools/check_qat_runtime.py"), "--model", str(host.path),
            "--trace", str(tmp_path / "trace.safetensors"), "--device", "cpu", "--backend", "reference"]
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["bit_identical"]
    fixture["logits"][0, 0] += 0.01
    save_file(fixture, tmp_path / "trace.safetensors")
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 1
    assert not json.loads(result.stdout)["bit_identical"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HIP/CUDA hardware required")
def test_gpu_weight_and_checkpoint_slots_and_pinned_reduction(tmp_path):
    # Four layers force reuse of both slots in both directions, across steps.
    tiny_model(tmp_path / "model", 4)
    host = HostModel(tmp_path / "model", pin=True)
    device = Replica(host, "cuda:0", ReferenceBackend(), 8, 4, 3, 3, dtype=torch.float32)
    cpu = Replica(host, "cpu", ReferenceBackend(), 8, 4, 3, 3)
    opt_gpu, opt_cpu = device.optimizer(1e-3, 1e-4), cpu.optimizer(1e-3, 1e-4)
    stager = CPUStager()
    for seed in (1, 2, 3):
        left = step([device], [opt_gpu], [batch(seed)])
        right = step([cpu], [opt_cpu], [batch(seed)])
        assert left == pytest.approx(right, rel=1e-4)
        values = stager.copy(device.named_parameters())
        for key, value in cpu.named_parameters().items():
            torch.testing.assert_close(values[key], value, atol=5e-5, rtol=5e-4)
    save_checkpoint(tmp_path / "gpu.pt", device, opt_gpu, {"step": 3}, {}, stager)
    assert load_checkpoint(tmp_path / "gpu.pt", [device], [opt_gpu], {}) == {"step": 3}


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two HIP/CUDA devices required")
def test_two_gpu_host_gradient_reduction(tmp_path):
    tiny_model(tmp_path / "model", 1)
    host = HostModel(tmp_path / "model", pin=True)
    replicas = [Replica(host, f"cuda:{i}", ReferenceBackend(), 8) for i in range(2)]
    optimizers = [r.optimizer(1e-3, 1e-4) for r in replicas]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(r.run, batch(i + 1), 12) for i, r in enumerate(replicas)]
        assert all(np.isfinite(f.result()) for f in futures)
    reduce_gradients(replicas)
    for replica, optimizer in zip(replicas, optimizers):
        with replica.context():
            optimizer.step()
    stager = CPUStager()
    states = [stager.copy(r.named_parameters()) for r in replicas]
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], atol=0, rtol=0)
