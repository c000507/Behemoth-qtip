# Served models (MULTI_MODEL.md). Terraform loads *.auto.tfvars after terraform.tfvars, so this overrides the copy in
# s3://behemoth-tfstate-*/ecs-neuron/; with `models` set, the single-model variables there (model_s3_prefix, model_ids,
# backend_extra_env, ...) are ignored. Everything else (network, images, router, tunnel) still comes from that file.
# Contains no secrets. Previously ornith.auto.tfvars (single model): its Ornith settings are the "ornith" entry, unchanged.
# Rollback: MULTI_MODEL.md "Rollback" (spot off: use_spot = false per entry; Cydonia off: delete its entry). Do NOT
# git revert the Terraform code after an apply without the reverse moved blocks listed there.

default_model     = "ornith" # requests without a `model` field
max_active_models = 2        # x 8 vCPU (trn1.2xlarge) <= 32 vCPU 'All Trn Spot Instance Requests' quota
cap_policy        = "evict"  # cold model beyond the cap: stop the longest-idle idle model, else 503 + Retry-After

models = {
  # Ornith 1.5 35B-A3B REAP20 v3 FP8 (Qwen3.5 MoE, 205 experts, TP=2 fp8, KV8 int8, 64k context). legacy_names: the
  # resources of the single-model stack (service behemoth-backend, capacity provider behemoth-trn-spot) are this entry.
  ornith = {
    legacy_names       = true
    model_s3_prefix    = "models/ornith-reap20-v3-fp8"
    graphs_s3_prefix   = "graphs/ornith-reap20-v3-kv8-seg20-mtp-pf1024" # = -pf (docs/neuron-qwen35-moe.md section 13: chunked GDN, blocked GQA, grouped MoE with dense fallback) + the 9 Q=1024 prefill NEFFs of section 14 (pfbuild_wide.sh 1024); needs the ornith-chunk4096 engine. Rollback: the -pf prefix with --prefill-chunk 256 and no --token-budget
    prepared_s3_prefix = "" # the Qwen3.5 prepared dir (r0/ r1/ L<layer>.bin) is the model dir itself
    model_ids          = ["ornith-1.5-35b-a3b-reap20-v3"]
    model_name         = "ornith-1.5-35b-a3b-reap20-v3"
    shm_size_mib       = 6144 # as served so far (backend_shm_size_mib of terraform.tfvars)

    # --prefix-cache-gb 4: host K/V tier of the prefix cache (int8 KV, 162.5 KiB per 16-token page, ~385k tokens).
    # --prefill-chunk 1024 --token-budget 1032: 1024-token prefill chunks (section 14: ~+37 % prefill on long cold prompts;
    # decode streams advance one token per chunk while a prompt prefills, median gap 711 ms vs 277 ms at 256).
    extra_server_args = "--prefill-chunk 1024 --token-budget 1032 --max-batch-requests 8 --prefix-cache-gb 4"
    # STREAM_LOAD=0: the prepared dir has no model-layer-*.bin files to stream (no Devstral-style staging vars).
    # gdn_{1,2,4}_1_32 are GDN_SLOT=bound NEFFs (docs/neuron-reap-nonfinite.md). Prefix cache with recurrent-state
    # snapshots (docs/neuron-reap-prefix-cache.md): 62.8 MiB per snapshot every 2048 tokens on the 1024-token chunk grid,
    # <= 4 GB of snapshots + <= 4 GB of K/V host pages.
    extra_env = {
      STREAM_LOAD                            = "0"
      BEHEMOTH_NEURON_QWEN35_PREFIX_CACHE    = "1"
      BEHEMOTH_NEURON_QWEN35_PREFIX_ALIGN    = "1"
      BEHEMOTH_NEURON_QWEN35_PREFIX_INTERVAL = "2048"
      BEHEMOTH_NEURON_QWEN35_BUCKET_FIT      = "1" # short prompts / tails use the Q=64 / Q=256 buckets instead of a padded Q=1024 chunk
      BEHEMOTH_NEURON_QWEN35_PREFIX_STATE_GB = "4"
      BEHEMOTH_NEURON_PREFIX_HOST_GB         = "4"
      # idle OpenMP threads (sampling) must not spin: spinning ones delayed the rank workers by ~2.7 ms per step (docs/neuron-reap-launches.md)
      OMP_WAIT_POLICY                        = "PASSIVE"
      # MTP speculative decoding, 1 draft per step while ONE stream decodes (docs/neuron-reap-mtp.md): the head (mtp.json + r0/r1 mtp.bin)
      # is pulled with the model dir (models/ornith-reap20-v3-fp8/mtp/); 2+ concurrent streams decode without drafts
      BEHEMOTH_NEURON_MTP_DIR                = "/data/models/ornith-reap20-v3-fp8/mtp"
      BEHEMOTH_NEURON_SPEC_MTP               = "1"
      BEHEMOTH_NEURON_SPEC_MTP_ADAPT         = "1" # k in {0, 1} by measured acceptance and step times: no loss on low-acceptance text
    }

    use_spot      = false # spot disabled (spot placement score 1/10: launches fail); on-demand with a warm pool
    spot_fallback = false
  }

  # Cydonia 24B FP8 (Mistral-Small architecture: the Llama/Mistral engine path, as Devstral), TP=2, fp8 KV, 32k context.
  # Same graphs as Devstral (identical geometry); prepared shards of models/cydonia-24b-fp8.
  cydonia = {
    model_s3_prefix    = "models/cydonia-24b-fp8"
    graphs_s3_prefix   = "graphs/cydonia-24b-fp8-kv"
    prepared_s3_prefix = "prepared/cydonia-24b-fp8-tp2-wt-mlptiled"
    model_ids          = ["cydonia-24b"]
    shm_size_mib       = 6144

    extra_server_args = "--mixed-prefill-tokens 256 --prefill-interval 2" # as Devstral was served
    # streamed load, layer files staged in RAM and deleted once on the devices, at most 8 (~5 GB) ahead of the upload
    extra_env = {
      STREAM_LOAD                    = "1"
      STAGE_DIR                      = "/dev/shm/layers"
      STREAM_UNLINK                  = "1"
      STREAM_AHEAD                   = "8"
      BEHEMOTH_NEURON_EMBED_RESIDENT = "1"
    }

    use_spot      = false # spot disabled, see ornith
    spot_fallback = false
  }
}
