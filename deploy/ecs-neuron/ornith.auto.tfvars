# Serving switch: Devstral -> Ornith 1.5 35B-A3B REAP20 v3 FP8 (Qwen3.5 MoE, 205 experts, TP=2 fp8, KV8 int8, 64k context).
# Terraform loads *.auto.tfvars after terraform.tfvars, so these override the copy in s3://behemoth-tfstate-*/ecs-neuron/.
# Everything else (network, images, router, tunnel) still comes from that file. Contains no secrets.
# Revert: delete this file (or git revert the commit) and run the pipeline: the Devstral prefixes are untouched in S3.
model_s3_prefix    = "models/ornith-reap20-v3-fp8"
graphs_s3_prefix   = "graphs/ornith-reap20-v3-kv8"
prepared_s3_prefix = "" # the Qwen3.5 prepared dir (r0/ r1/ L<layer>.bin) is the model dir itself
backend_model_name = "ornith-1.5-35b-a3b-reap20-v3"
model_ids          = ["ornith-1.5-35b-a3b-reap20-v3"]

backend_extra_server_args = "--prefill-chunk 256 --max-batch-requests 8"

# Replaces (does not merge with) backend_extra_env of terraform.tfvars: the Devstral staging vars (STAGE_DIR, STREAM_UNLINK,
# STREAM_AHEAD, EMBED_RESIDENT) do not apply. STREAM_LOAD=0: the prepared dir has no model-layer-*.bin files to stream.
backend_extra_env = {
  STREAM_LOAD                         = "0"
  BEHEMOTH_NEURON_QWEN35_MIN_DECODE_R = "4" # workaround for the 1-row gdn NEFF race ("Nonfinite model logits"); 15.1 tok/s at 1 stream
}
