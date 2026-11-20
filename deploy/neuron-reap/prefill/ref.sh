#!/bin/bash
# ref.sh DIAGDIR N...: f32 prepared-weights reference logits (prefill end) for the N-th diag records -> /mnt/scratch/ref/ref<N>.npy (cores 4-7)
. /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
S=/mnt/scratch; D=$1; shift; mkdir -p $S/ref
for n in "$@"; do
  f=$(ls $D/chatcmpl-$n-*.json | head -1)
  [ -f $S/ref/ref$n.npy ] && continue
  OMP_NUM_THREADS=2 taskset -c 4-7 python $S/src/neuron/tests/qwen35_prepared_ref.py $S/model $f $S/ref/ref$n.npy > $S/ref/ref$n.log 2>&1
  tail -1 $S/ref/ref$n.log
done
echo REF_DONE
