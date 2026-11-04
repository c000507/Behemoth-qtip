"""Optional independent oracle: pip install torch transformers safetensors.

Run: python hf_parity.py --build BUILD_DIR
Uses only tiny synthetic checkpoints. Python is not a runtime dependency.
"""
import argparse
import json
import pathlib
import subprocess
import tempfile

import torch
from safetensors.torch import load_file
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel


def oracle(path):
    raw = json.loads((path / "config.json").read_text())["text_config"]
    raw["pad_token_id"] = 0
    assert raw["hidden_size"] <= 64, "This test is for tiny fixtures only"
    config = Glm5NextTextConfig(**raw)
    config._attn_implementation = "eager"
    model = Glm5NextTextModel(config).float().eval()
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    files = {f: load_file(str(path / f)) for f in set(index.values())}

    def tensor(name):
        return files[index[name]][name].float()

    def weight(base):
        if base + ".weight" in index:
            return tensor(base + ".weight")
        packed = files[index[base + ".weight_packed"]][base + ".weight_packed"]
        codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(packed.shape[0], -1)
        levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6])
        scales = tensor(base + ".weight_scale").repeat_interleave(16, dim=-1)
        global_scale = tensor(base + ".weight_global_scale")
        return (levels[(codes & 7).long()] * (1 - 2 * (codes >> 3).float()) *
                scales * global_scale).to(torch.bfloat16).float()

    prefix = "model.language_model."
    state = {}
    for key in model.state_dict():
        parts = key.split(".")
        if ".attn_hc." in key or ".ffn_hc." in key:
            mapped = key.replace(".attn_hc.", ".hc_attn_").replace(".ffn_hc.", ".hc_ffn_")
            state[key] = tensor(prefix + mapped)
        elif ".self_attn.conv1d.weight" in key:
            base = prefix + key.removesuffix("conv1d.weight")
            state[key] = torch.cat([tensor(base + p + "_conv1d.weight") for p in ("q", "k", "v")])
        elif ".mlp.experts.gate_up_proj" in key:
            base = prefix + key.removesuffix("gate_up_proj")
            state[key] = torch.stack([torch.cat([weight(base + str(e) + ".gate_proj"),
                                                weight(base + str(e) + ".up_proj")])
                                      for e in range(raw["n_routed_experts"])])
        elif ".mlp.experts.down_proj" in key:
            base = prefix + key.removesuffix("down_proj")
            state[key] = torch.stack([weight(base + str(e) + ".down_proj")
                                      for e in range(raw["n_routed_experts"])])
        else:
            mapped = prefix + key.replace(".forget_gate.", ".")
            state[key] = weight(mapped.removesuffix(".weight")) if key.endswith("_proj.weight") else tensor(mapped)
    model.load_state_dict(state, strict=True)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    with torch.no_grad():
        output = model(ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
        return torch.nn.functional.linear(output, weight("lm_head"))[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=pathlib.Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory(prefix="glm53-parity-") as folder:
        root = pathlib.Path(folder)
        for kind, flag in (("moe", "--fixture"), ("dense", "--dense-fixture")):
            source = root / (kind + "-source")
            subprocess.run([str(args.build / "glm53_tests"), flag, str(source)], check=True)
            quant = root / (kind + "-nvfp4")
            subprocess.run([str(args.build / "glm53"), "quantize", "--model", str(source),
                            "--out", str(quant)], check=True)
            request = root / "request.json"
            request.write_text(json.dumps(dict(tokens=list(range(1, 8)), logit_positions=list(range(7)))))
            for path in (source, quant):
                trace = root / (path.name + "-trace.json")
                subprocess.run([str(args.build / "glm53"), "trace", "--model", str(path),
                                "--tokens", str(request), "--out", str(trace), "--arithmetic", "fp32"], check=True)
                native = torch.tensor(json.loads(trace.read_text())["logits"])
                hf = oracle(path)
                error = (native - hf).norm() / hf.norm().clamp_min(1e-30)
                maximum = (native - hf).abs().max()
                print(path.name, "relative L2", float(error), "max absolute", float(maximum))
                assert error < 1e-4 and maximum < 3e-5, "GLM53 logits differ from Transformers"
    print("PASS Transformers GLM5Next oracle: source + NVFP4, seven-token incremental/native vs full HF")


if __name__ == "__main__":
    main()
