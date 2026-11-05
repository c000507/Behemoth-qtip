# Layer-zero Q/K/V audit

`qtip_qkv_audit` checks whether a large Hessian proxy corresponds to real projection error. It supports BF16 sources and runs with either quantizer GPU backend.

```sh
qtip_qkv_audit --model /models/Anubis-70B-v1 \
  --tokens calibration/tokens.i32 --ntok 4194304 --batch-rows 4096 \
  --sigma-reg 0.01 --ldl-threads 16 --out qkv-audit.json
```

Use the quantization run's token order and batch size to reproduce its layer-zero input covariance. The tool streams embeddings through the original input RMSNorm, accumulates the covariance with the production block schedule, and quantizes Q/K/V together using seed 0. It then measures `sum((X W_original - X W_quantized)^2) / sum((X W_original)^2)` directly over the calibration tokens. The candidate weights are rounded to BF16 as in adaptive layer evaluation.

The JSON reports direct relative SSE and RMSE beside `proxy` and `proxy_rot_reg`. These are projection measures; `kl_qtip` in `adaptive_report.json` measures the final next-token distribution after the candidate layer and all remaining original layers. Their numerical values are not interchangeable.

No full-model activation buffer is allocated. This allows a targeted audit without another 68.7 GB activation chain for a 70B model. The tool audits layer zero only; later layers require their actual incoming activations.
