# Whole-model quality reference for large teachers

`qtip_reference_logits` streams an original model through the quantizer's BF16 forward path, one layer at a time. It produces the reference log-probability files consumed by `runtime/behemoth_cuda_model_kl`, allowing a 70B BF16 teacher to be evaluated without fitting its full weights in VRAM. Embeddings, final norm and head must be BF16. Each sequence is an independent causal document.

```sh
qtip_reference_logits --model ORIGINAL_BF16 --tokens heldout.i32 \
  --seq-len 512 --nseq 2 --out teacher-reference

behemoth_cuda_model_kl --model QUANTIZED --tokens heldout.i32 \
  --seq-len 512 --nseq 2 --ref-dir teacher-reference --json whole-model-kl.json
```

Use exactly the same held-out token file, sequence length and vocabulary for both commands. The output is one `seqNNN.f32` file per sequence, containing row-major `[sequence_length, vocabulary]` float32 log probabilities, plus a manifest. Existing reference sequence files are not overwritten.

This measures end-to-end KL and perplexity, unlike the adaptive quantizer's isolated-layer KL. The reference uses the native quantizer's batched BF16 prefill; the comparison tool uses the serving engine's incremental decode. Small differences from their floating-point reduction orders can therefore contribute to the score. The reference is not a Hugging Face numerical-equivalence certificate.
