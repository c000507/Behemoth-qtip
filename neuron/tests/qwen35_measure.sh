#!/bin/bash
# End-to-end timing of the Qwen3.5 MoE backend on the box (neuron_engine_cli + tests/qwen35_e2e.py): short and 8k / 32k / 64k
# fills, 1 and 4 concurrent streams; prints decode ms/step and prefill tok/s from the BEHEMOTH_NEURON_STEP_LOG lines.
# Paths as used on the trn1 box (/mnt/scratch: prepared set, graphs_sub bucket set, BUILD/neuron_engine_cli, BUILD/tok_check).
export PATH=/usr/local/bin:/opt/aws/neuron/bin:$PATH LD_LIBRARY_PATH=/opt/aws/neuron/lib:$LD_LIBRARY_PATH
S=/mnt/scratch; cd $S; unset NEURON_LOGICAL_NC_CONFIG
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
export NEURON_RT_VISIBLE_CORES=0-1 NEURON_RT_ROOT_COMM_ID=127.0.0.1:62128 BEHEMOTH_NEURON_NEFF_DIR=$S/graphs_sub BEHEMOTH_PREFILL_CHUNK=256 TOK_CHECK=$S/BUILD/tok_check TMPDIR=$S/e2e BEHEMOTH_NEURON_STEP_LOG=1
unset NEURON_X_DUMP
M=$S/e2e/measure; mkdir -p $M
stats() { # name log: decode steps (rows>=1, tokens==rows) mean wall and prefill summary
  python3 - "$1" "$2" <<'P'
import re, sys, json
name, log = sys.argv[1], sys.argv[2]
dec, pre = [], []
for l in open(log):
    if not l.startswith('step'): continue
    rows = int(re.search(r'rows=(\d+)', l).group(1)); toks = int(re.search(r'tokens=(\d+)', l).group(1)); wall = float(re.search(r'wall_ms=([0-9.]+)', l).group(1))
    (dec if toks == rows else pre).append((rows, toks, wall))
if dec:
    d = dec[1:] if len(dec) > 1 else dec
    w = sum(x[2] for x in d) / len(d); r = sum(x[0] for x in d) / len(d)
    print(f'{name}: decode steps {len(d)}, mean rows {r:.1f}, mean step {w:.1f} ms -> {1000 * r / w:.1f} tok/s aggregate, {w / 1:.1f} ms/token per stream')
if pre:
    t = sum(x[1] for x in pre); w = sum(x[2] for x in pre)
    print(f'{name}: prefill {t} tokens in {len(pre)} steps, {w:.0f} ms -> {1000 * t / w:.0f} tok/s')
P
}
run() { # name max_new env...
  n=$1; mn=$2; shift 2
  env "$@" timeout 3000 python repo/neuron/tests/qwen35_e2e.py $S/prepared $S/BUILD/neuron_engine_cli $M/$n $mn "Explain in two paragraphs why the sky is blue." > $M/$n.log 2>&1; echo "$n rc=$?"
  grep -E "engine step failed|NONFINITE" $M/$n/cli_stderr.log | head -2
  stats $n $M/$n/cli_stderr.log
}
run short1 200 CONCURRENT=1
run short4 200 CONCURRENT=4
run fill8k 32 FILL=8000
run fill32k 32 FILL=32000
run fill64k 32 FILL=64000
run fill8k_c4 32 FILL=8000 CONCURRENT=4
head -c 1200 $M/short1.log
echo MEASURE_DONE
