"""Crash-safe training checkpoints and unchanged-format HF/QTIP export."""
import os
from pathlib import Path
import shutil
import tempfile

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from .model import LINEARS, validate_shared_su


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class CPUStager:
    """Reusable pinned buffers; no device -> pageable copies, including optimizer state."""
    def __init__(self):
        self.buffers = {}

    def copy(self, value, key="root"):
        if isinstance(value, torch.Tensor):
            value = value.detach()
            if value.device.type == "cpu":
                return value.clone()
            buffer_key = (key, tuple(value.shape), value.dtype)
            if buffer_key not in self.buffers:
                self.buffers[buffer_key] = torch.empty_like(value, device="cpu", pin_memory=True)
            buffer = self.buffers[buffer_key]
            with torch.cuda.device(value.device):
                buffer.copy_(value, non_blocking=True)
                torch.cuda.current_stream(value.device).synchronize()
            return buffer.clone()
        if isinstance(value, dict):
            return {k: self.copy(v, f"{key}/{k}") for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self.copy(v, f"{key}/{i}") for i, v in enumerate(value))
        return value


def save_checkpoint(path, replica, optimizer, progress, signature, stager):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(format_version=2, signature=signature, progress=progress,
                 trainable=stager.copy(replica.named_parameters(), "parameters"),
                 optimizer=stager.copy(optimizer.state_dict(), "optimizer"),
                 torch_rng=torch.get_rng_state())
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path, replicas, optimizers, signature):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format_version") != 2 or state["signature"] != signature:
        raise ValueError("resume signature mismatch (source model, data, devices, backend or training schedule)")
    for i in range(len(replicas[0].params)):
        validate_shared_su(state["trainable"], f"model.layers.{i}.")
    for replica, optimizer in zip(replicas, optimizers):
        with replica.context(), torch.no_grad():
            parameters = replica.named_parameters()
            if parameters.keys() != state["trainable"].keys():
                raise ValueError("resume parameter names differ")
            for key, param in parameters.items():
                if param.shape != state["trainable"][key].shape:
                    raise ValueError(f"resume parameter shape mismatch: {key}")
                # Small parameter copies; use pinned memory on HIP too.
                source = state["trainable"][key]
                param.copy_(source.pin_memory() if replica.cuda else source, non_blocking=replica.cuda)
            # Optimizer load_state_dict internally moves tensors. Pin sources so
            # H2D state transfer never depends on pageable large staging buffers.
            def pin(value):
                if isinstance(value, torch.Tensor):
                    # load_state_dict can retain CPU tensor storage, including
                    # AdamW's scalar step. Each replica needs independent state.
                    return value.pin_memory() if replica.cuda else value.clone()
                if isinstance(value, dict):
                    return {k: pin(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [pin(v) for v in value]
                return value
            optimizer.load_state_dict(pin(state["optimizer"]))
    torch.set_rng_state(state["torch_rng"])
    return state["progress"]


def export_model(host, trainable, output):
    """Write complete shards atomically. Never mutate/link writable source shards."""
    output = Path(output).resolve()
    if output == host.path or host.path in output.parents or output in host.path.parents:
        raise ValueError("export must be outside the source model directory")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".", dir=output.parent))
    replacements = {}
    for i in range(len(host.layers)):
        prefix = f"model.layers.{i}."
        validate_shared_su(trainable, prefix)
        for linear in LINEARS:
            replacements[prefix + linear + ".SU"] = trainable[prefix + linear + ".SU"]
            replacements[prefix + linear + ".SV"] = (trainable[prefix + linear + ".SV"] *
                                                       trainable[prefix + linear + ".Wscale"])
    for key, value in trainable.items():
        if key.endswith("layernorm.weight") or key == "model.norm.weight":
            replacements[key] = value
    try:
        for filename in sorted(set(host.weight_map.values())):
            with safe_open(host.path / filename, framework="pt", device="cpu") as source:
                tensors = {key: (replacements[key].to(dtype=source.get_tensor(key).dtype).contiguous()
                                 if key in replacements else source.get_tensor(key)) for key in source.keys()}
                if any(not torch.isfinite(value).all() for key, value in tensors.items() if key in replacements):
                    raise ValueError("refusing to export nonfinite trainable parameters")
                save_file(tensors, temporary / filename, metadata=source.metadata())
            with open(temporary / filename, "rb") as handle:
                os.fsync(handle.fileno())
        for source in host.path.iterdir():
            if source.is_file() and source.suffix != ".safetensors":
                shutil.copy2(source, temporary / source.name)
                with open(temporary / source.name, "rb") as handle:
                    os.fsync(handle.fileno())
        # SV folds the learned scale back into the original format; Wscale is
        # present only in training checkpoints, never in the model's weight_map.
        fsync_directory(temporary)
        temporary.rename(output)
        fsync_directory(output.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
