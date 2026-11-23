#!/usr/bin/env python3
"""Dump HF transformers reference logits for a Qwen3.5 checkpoint (CPU, fp32).

usage: python3 qwen35_hf_ref.py <model_dir> [prompt] [out.npy]
Writes float32 [T, vocab] logits for the tokenised prompt (no BOS/chat template).
"""
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = sys.argv[1] if len(sys.argv) > 1 else "/mnt/nvme/models/Qwen3.5-0.8B"
prompt = sys.argv[2] if len(sys.argv) > 2 else "The quick brown fox jumps over the lazy dog."
out = sys.argv[3] if len(sys.argv) > 3 else "/tmp/qwen35_hf_logits.npy"

tok = AutoTokenizer.from_pretrained(model_dir)
ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
print("tokens:", ids)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, device_map="cpu")
with torch.no_grad():
    logits = model(torch.tensor([ids])).logits[0].float().numpy()
np.save(out, logits)
print("logits", logits.shape, "->", out)
print("argmax[-1]", int(logits[-1].argmax()), tok.decode([int(logits[-1].argmax())]))
for t in range(min(3, len(ids))):
    print(f" argmax[{t}]", int(logits[t].argmax()))
