"""Double-buffered weights/checkpoints and host-reduced data parallel gradients."""
from contextlib import nullcontext

import torch

from .kernels import cross_entropy, rmsnorm, norm_weight
from .model import decoder_layer, SU_GROUPS


class Slots:
    def __init__(self, host, device, shape, dtype):
        self.host, self.device = host, torch.device(device)
        self.cuda = self.device.type == "cuda"
        if self.cuda and not host.pin:
            raise ValueError("GPU replicas require a pinned shared HostModel")
        self.copy = torch.cuda.Stream(device=device) if self.cuda else None
        self.offload = torch.cuda.Stream(device=device) if self.cuda else None
        self.ready = [torch.cuda.Event() for _ in range(2)] if self.cuda else [None, None]
        self.used = [torch.cuda.Event() for _ in range(2)] if self.cuda else [None, None]
        self.used_once = [False, False]
        self.weights = [{key: torch.empty_like(value, device=device) for key, value in host.layers[0].items()}
                        for _ in range(2)]
        self.inputs = [torch.empty(shape, device=device, dtype=dtype) for _ in range(2)]
        self.checkpoints = [torch.empty(shape, dtype=dtype, pin_memory=self.cuda) for _ in host.layers]
        self.saved = [torch.cuda.Event() for _ in host.layers] if self.cuda else [None] * len(host.layers)

    def save(self, layer, x):
        if self.cuda:
            produced = torch.cuda.Event()
            produced.record(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self.offload):
                self.offload.wait_event(produced)
                self.checkpoints[layer].copy_(x, non_blocking=True)
                self.saved[layer].record(self.offload)
                x.record_stream(self.offload)
        else:
            self.checkpoints[layer].copy_(x)

    def prefetch(self, layer, slot, backward=False):
        with torch.cuda.stream(self.copy) if self.cuda else nullcontext():
            if self.cuda and self.used_once[slot]:
                self.copy.wait_event(self.used[slot])
            for key, value in self.host.layers[layer].items():
                self.weights[slot][key].copy_(value, non_blocking=self.cuda)
            if backward:
                if self.cuda:
                    self.copy.wait_event(self.saved[layer])
                self.inputs[slot].copy_(self.checkpoints[layer], non_blocking=self.cuda)
            if self.cuda:
                self.ready[slot].record(self.copy)

    def acquire(self, slot):
        if self.cuda:
            torch.cuda.current_stream(self.device).wait_event(self.ready[slot])
        return self.weights[slot], self.inputs[slot]

    def release(self, slot):
        if self.cuda:
            self.used[slot].record(torch.cuda.current_stream(self.device))
            self.used_once[slot] = True

    def layers(self, reverse=False):
        order = list(range(len(self.host.layers)))
        if reverse:
            order.reverse()
        for j, layer in enumerate(order[:2]):
            self.prefetch(layer, j, backward=reverse)
        for j, layer in enumerate(order):
            slot = j % 2
            weights, value = self.acquire(slot)
            yield layer, weights, value
            self.release(slot)
            if j + 2 < len(order):
                self.prefetch(order[j + 2], slot, backward=reverse)


class Replica:
    def __init__(self, host, device, backend, seqlen, row_chunk=256, attn_block=256, loss_chunk=128,
                 dtype=None):
        self.host, self.device, self.backend = host, torch.device(device), backend
        self.dtype = dtype or (torch.bfloat16 if self.device.type == "cuda" or backend.device_rounding else torch.float32)
        self.cuda = self.device.type == "cuda"
        self.row_chunk, self.attn_block, self.loss_chunk = row_chunk, attn_block, loss_chunk
        with self.context():
            self.params = [{key: torch.nn.Parameter(value.to(device=self.device, dtype=torch.float32).clone())
                            for key, value in layer.items()} for layer in host.initial]
            for layer in self.params:
                for group in SU_GROUPS:
                    for name in group[1:]:
                        layer[name] = layer[group[0]]
            self.norm = torch.nn.Parameter(host.final_norm.to(device=self.device, dtype=torch.float32).clone())
            self.head = host.head.to(device=self.device, dtype=self.dtype)
            self.embedding_stage = torch.empty((seqlen, host.config["hidden_size"]),
                                               dtype=host.embedding.dtype, pin_memory=self.cuda)
            self.slots = Slots(host, self.device, self.embedding_stage.shape, self.dtype)
            # Tied signs accumulate gradients from every use, but are reduced,
            # clipped and updated once. Preserve checkpoint names for export.
            self.parameters = list({id(p): p for layer in self.params for p in layer.values()}.values()) + [self.norm]
            count = sum(p.numel() for p in self.parameters)
            # Reused pinned buffers for every gradient reduction/checkpoint transfer.
            self.grad_host = torch.empty(count, dtype=torch.float32, pin_memory=self.cuda)
            self.grad_device = torch.empty(count, dtype=torch.float32, device=self.device)

    def context(self):
        return torch.cuda.device(self.device) if self.cuda else nullcontext()

    def named_parameters(self):
        return {**{f"model.layers.{i}.{key}": value for i, layer in enumerate(self.params)
                   for key, value in layer.items()}, "model.norm.weight": self.norm}

    def optimizer(self, lr, norm_lr, norm_decay=0.01):
        signs, norms = [], [self.norm]
        seen = set()
        for layer in self.params:
            for key, value in layer.items():
                if id(value) in seen:
                    continue
                seen.add(id(value))
                (norms if key.endswith("layernorm.weight") else signs).append(value)
        return torch.optim.AdamW([dict(params=signs, lr=lr, weight_decay=0.0),
                                  dict(params=norms, lr=norm_lr, weight_decay=norm_decay)], foreach=False)

    def run(self, batch, normalizer=1, backward=True):
        if normalizer <= 0:
            raise ValueError("loss normalizer must be positive")
        with self.context():
            tokens, labels, segments, positions = batch
            if tokens.shape != self.embedding_stage.shape[:1]:
                raise ValueError("batch length differs from allocated sequence length")
            torch.index_select(self.host.embedding, 0, tokens, out=self.embedding_stage)
            x = self.embedding_stage.to(device=self.device, dtype=self.dtype, non_blocking=self.cuda)
            labels, segments, positions = (t.to(self.device) for t in (labels, segments, positions))
            with torch.no_grad():
                for layer, weights, _ in self.slots.layers():
                    if backward:
                        self.slots.save(layer, x)
                    x = decoder_layer(x, weights, self.params[layer], self.host.config, segments, positions,
                                      self.backend, self.row_chunk, self.attn_block)
            with torch.set_grad_enabled(backward):
                x = x.detach().requires_grad_(backward)
                final = rmsnorm(x, norm_weight(self.norm, self.backend), self.host.config.get("rms_norm_eps", 1e-5))
                loss = cross_entropy(final, self.head, labels, self.loss_chunk)
                if backward:
                    (loss / normalizer).backward()
            loss_value = loss.detach().item()
            if backward:
                dx = x.grad
                del x, final, loss
                for layer, weights, saved in self.slots.layers(reverse=True):
                    value = saved.detach().requires_grad_(True)
                    out = decoder_layer(value, weights, self.params[layer], self.host.config, segments, positions,
                                        self.backend, self.row_chunk, self.attn_block)
                    out.backward(dx)
                    dx = value.grad
                    del out, value
            return loss_value

    @torch.no_grad()
    def logits(self, batch, indices):
        """Small selected-position logit chunks for offline runtime parity checks."""
        with self.context():
            tokens, _, segments, positions = batch
            torch.index_select(self.host.embedding, 0, tokens, out=self.embedding_stage)
            x = self.embedding_stage.to(device=self.device, dtype=self.dtype, non_blocking=self.cuda)
            segments, positions = (value.to(self.device) for value in (segments, positions))
            for layer, weights, _ in self.slots.layers():
                x = decoder_layer(x, weights, self.params[layer], self.host.config, segments, positions,
                                  self.backend, self.row_chunk, self.attn_block)
            x = rmsnorm(x, norm_weight(self.norm, self.backend), self.host.config.get("rms_norm_eps", 1e-5))
            head = self.head.float()
            for chunk in indices.split(self.loss_chunk):
                yield x[chunk.to(self.device)].float() @ head.T

    def gradients_to_host(self):
        with self.context(), torch.no_grad():
            offset = 0
            for param in self.parameters:
                view = self.grad_device[offset:offset + param.numel()]
                if param.grad is None:
                    view.zero_()
                else:
                    view.copy_(param.grad.reshape(-1))
                offset += param.numel()
            self.grad_host.copy_(self.grad_device, non_blocking=self.cuda)
            if self.cuda:
                torch.cuda.current_stream(self.device).synchronize()

    def gradients_from_host(self, host_gradient):
        with self.context(), torch.no_grad():
            # Each replica has a pinned buffer, also for host -> device.
            if host_gradient is not self.grad_host:
                self.grad_host.copy_(host_gradient)
            self.grad_device.copy_(self.grad_host, non_blocking=self.cuda)
            offset = 0
            for param in self.parameters:
                if param.grad is None:
                    param.grad = torch.empty_like(param)
                param.grad.copy_(self.grad_device[offset:offset + param.numel()].view_as(param))
                offset += param.numel()


def reduce_gradients(replicas):
    # Local losses are divided by GLOBAL supervised token count; sum, don't
    # average per-device means (assistant counts can be very different).
    for replica in replicas:
        replica.gradients_to_host()
    result = replicas[0].grad_host
    for replica in replicas[1:]:
        result.add_(replica.grad_host)
    for replica in replicas:
        replica.gradients_from_host(result)
