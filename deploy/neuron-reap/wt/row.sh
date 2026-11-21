#!/bin/bash
# row.sh: task 7, MOE_SLOTS=row for the 1-row decode (seg20_1_P + moe_1_1_32) on top of the WT set; set sets/wtrow; E2E: par (identity
# vs sets/wt) + decode timing (MTP off and on) -> log/row.txt; then quiet R=8 segment bench + devtest of the WT dense fallbacks
S=/mnt/scratch; L=$S/log/row.txt; O=$S/grow
while ! grep -q MEAS_DONE $S/log/meas.txt 2>/dev/null; do sleep 20; done
sp=$(python3 /root/op/wtspecs.py $S/gprod WT=1 MOE_SLOTS=row | grep -E '^(seg20_1_[0-9]+|moe_1_1_32):')
echo "$sp" >> $L
JOBS=1 bash /root/op/gb.sh $O $sp >> $L 2>&1
a=""; for d in $O/*/; do a="$a $(basename $d)=$d"; done
bash /root/op/mkset.sh wtrow $S/sets/wt $a >> $L
bash /root/op/lb.sh $S/sets/wt/seg20_1_32 $O/seg20_1_32 $S/sets/wt/seg20_1_4096 $O/seg20_1_4096 >> $L 2>&1
export CHUNK=1024 BUDGET=1032
bash /root/op/par.sh r_par $S/sets/wtrow $S/BUILD2 BEHEMOTH_NEURON_QWEN35_BUCKET_FIT=1 >> $L 2>&1
/opt/aws_neuronx_venv_pytorch_2_9/bin/python /root/op/cmp.py $S/run/w_par/logits.bin $S/run/r_par/logits.bin >> $L 2>&1
cmp -s $S/run/w_par/logits.bin $S/run/r_par/logits.bin && echo "w_par vs r_par BIT-IDENTICAL" >> $L || echo "w_par vs r_par differ" >> $L
bash /root/op/dec.sh r_dn $S/sets/wtrow $S/BUILD2 BEHEMOTH_NEURON_SPEC_MTP=0 >> $L 2>&1; grep "rows=1 " $S/run/r_dn/steps.txt >> $L
bash /root/op/dec.sh r_dm $S/sets/wtrow $S/BUILD2 >> $L 2>&1; grep "rows=1 " $S/run/r_dm/steps.txt >> $L
echo "== r8 segs" >> $L; bash /root/op/lb.sh $S/sets/wt/seg20_8_* >> $L 2>&1
bash /root/op/devtest.sh $S/sets/wt/moe_1_256_32 $S/sets/wt/moe_1_1024_32 >> $L 2>&1
echo ROW_DONE >> $L
