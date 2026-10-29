# Diagnose a repeating generation

`qtip_diagnose` replays the same token sequence and compares model predictions
before repetition starts. It supports original FP8, QTIP, and mixed checkpoints.
It does not train or modify model weights. Build it with the native trainer:

```sh
cmake --build build-training --target qtip_diagnose -j8
cmake --build runtime/build-rocm --target behemoth_server behemoth_eval -j8
```

## Capture an actual failure

Restart your usual server with `BEHEMOTH_DIAGNOSTIC_DIR=/path/to/captures` in its
environment. Completed requests write a JSON file there, including the rendered
prompt, original request, resolved sampling settings/seed, exact prompt token IDs, and generated token IDs
delivered by the engine, before stop-string/display filtering. All requests in
that server process are captured while this variable is set. Unset it to stop
capturing. Cancelled/disconnected/error requests do not currently produce files.

Run your repeating conversation. The server logs the capture filename. This
preserves token boundaries and includes tokens whose decoded text may be hidden.

```sh
build-training/qtip_diagnose prepare \
  --capture /path/to/captures/chatcmpl-1-STAMP.json --out replay.json
```

This finds the first repeated exact 32-token span within the generated output
(with at least 32 tokens between starts). It selects up to 128 predictions around
that repeat and one block-output position just before it. This detector is an
onset hint: repeated quotations, formatting, and other legitimate repetition can
match. Inspect `repeat_start` and `previous_match`. To choose the actual onset:

```sh
build-training/qtip_diagnose prepare --capture CAPTURE.json \
  --position 1400 --out replay-manual.json
```

Positions are zero-based indices into **prompt plus generated tokens**. Prediction
at position `t` predicts `tokens[t+1]`. Preparing the request retains the exact
prefix, trims the unused tail, and inserts no BOS, template, or reasoning tags.
You can also supply `tokens` and `logit_positions` directly in a JSON request.
Optional `activation_positions` holds at most eight positions, and optional
`focus_tokens` lists token IDs to score in addition to the actual next token.
Tags containing several tokens must be assessed as sequences, not a single ID.

## Compare the source, quantized, and fitted model

Run traces sequentially so each pass can use the GPU arena. Weights stream one
block at a time; no full-model activation tape is allocated.

```sh
build-training/qtip_diagnose trace --model ORIGINAL_FP8 --request replay.json \
  --device 0 --arena-gb 24 --rows 1024 --out teacher.json
build-training/qtip_diagnose trace --model MIXED_BEFORE_QAT --request replay.json \
  --device 0 --arena-gb 24 --rows 1024 --out before.json
build-training/qtip_diagnose trace --model FINAL_MIXED --request replay.json \
  --device 0 --arena-gb 24 --rows 1024 --out final.json
build-training/qtip_diagnose compare --reference teacher.json \
  --candidate final.json --out teacher-vs-final.json
build-training/qtip_diagnose compare --reference before.json \
  --candidate final.json --out before-vs-final.json
```

All output paths must be new files. CPU trace mode (`--device cpu`) is limited to
tiny verification models. Preparation and comparison run on CPU without GPUs.
Host memory must hold the checkpoint's compressed/source weights. Eight selected
activation rows can also make traces large; one row is usually enough initially.
`--rows` chunks MLP work, not the attention context.

The trace backend uses the trainer's BF16-activation, FP32-accumulate forward
with full causal attention. It does **not** reproduce the inference FP8 KV cache,
query/probability quantization, server scheduling, or sampler. Comparisons across
checkpoints on this common backend help identify weight-induced differences.
Use the same tokenizer and source model family; matching IDs alone cannot prove
that two different vocabularies have the same meanings. Tokenizer hashes are
checked when available in both traces.

## Compare actual inference paths

The comparison mode also reads `behemoth_eval trace` outputs. For example:

```sh
BEHEMOTH_QTIP_BF16_PREFILL_ROWS=512 \
  runtime/build-rocm/behemoth_eval FINAL_MIXED trace replay.json runtime.json 256
BEHEMOTH_QTIP_BF16_PREFILL_ROWS=512 BEHEMOTH_ATTN_LEGACY=1 \
  runtime/build-rocm/behemoth_eval FINAL_MIXED trace replay.json legacy.json 256
BEHEMOTH_QTIP_BF16_PREFILL_ROWS=512 BEHEMOTH_ATTN_QP=bf16 \
  runtime/build-rocm/behemoth_eval FINAL_MIXED trace replay.json bf16-qp.json 256
build-training/qtip_diagnose compare --reference runtime.json \
  --candidate legacy.json --out native-vs-legacy.json
build-training/qtip_diagnose compare --reference final.json \
  --candidate runtime.json --out native-forward-vs-inference.json
```

Use a clean environment for each run; inherited `BEHEMOTH_ATTN_*` variables still
apply. Repeat with trace chunk `1` to process all tokens incrementally, and with
your server's prefill chunk. Match your server's projection/prefill options where
possible. This is a standalone inference replay and does not reproduce a mixed
server batch or a previously restored prefix cache.

## Read the report

Reports contain per-position KL(reference || candidate), both entropies, top
token IDs/log probabilities, actual-next-token ranks/log probabilities, and
candidate-minus-reference actual-next-token log probability. Values use natural
logs and raw, untempered logits; sampling penalties are not applied. Tied tokens
receive the same best rank. Summary fields identify peak divergence and the
first selected position with different argmax predictions.

At the repeat's first token, look at position `repeat_start-1`. A positive
`actual_next_logprob_delta` means the candidate favors that repeated token more
than the reference does. Compare the preceding positions too: after a repeated
passage has already been fed back as history, both models may predict more of it.
High probability on the forced continuation by itself does not diagnose a loop.

Where both native traces include block outputs, the report includes relative RMS
error and cosine similarity per layer at each selected position. A sharp increase
suggests blocks to inspect. This includes accumulated prefix error and does not
prove that the block caused repetition; validate a suspect using a subsequent
precision/restoration experiment. Runtime traces do not include block outputs.

Tests exercise full tiny-model FP8-teacher/mixed-student replay, exact next-token
alignment, known KL/entropy values, logit-shift invariance, matching-trace
identity, mismatch rejection, and repeated-span detection. GPU/server capture
execution and real-model repetition diagnosis require the deployment hardware.
