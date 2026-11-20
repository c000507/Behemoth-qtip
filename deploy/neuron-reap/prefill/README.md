Box-side drivers of the prefill study (docs/neuron-qwen35-moe.md section 13). They assume the dev trn1.2xlarge layout:
/mnt/scratch/{model (prepared set), gprod (production graphs), src (this tree), BUILD* (engines), corpus, corpus2 (frozen
prompt sources: `git archive 600b70f neuron runtime` / `docs`), sets/<name> (symlinked graph sets), run/<tag>} and live in
/root/op (setup.sh mounts the instance store and pulls everything; upd.sh refreshes the sources and rebuilds an engine).

  matrix.sh TAG GRAPHS BUILD     fresh server (production env), the prompt matrix (pf.py), step summary (steps.py)
  par.sh TAG GRAPHS BUILD [ENV]  logits of 7 prompts x 8 greedy steps (MTP off) -> run/TAG/logits.bin; cmp.py BASE NEW
  cache.sh TAG GRAPHS BUILD      prefix-cache exactness (full hit, partial hit vs a cold server)
  ref.sh DIAGDIR N...            f32 prepared-weights reference of the N-th request; cmpref.py REFDIR TAG...
  loads.py server.log            expert-load table from BEHEMOTH_NEURON_QWEN35_MOE_LOAD_LOG=1 (MOE_GSTAT graph)
  lb.sh DIR... / devtest.sh DIR...   layer_bench / device-vs-mirror test of NEFF directories
  gb.sh OUT name:kind:R:Q:P:ENV=V,..   ad-hoc NEFF builds; mkset.sh NAME BASE bucket=DIR...   symlinked graph set
