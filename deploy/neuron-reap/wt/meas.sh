#!/bin/bash
# meas.sh: same-session comparison prod (gprod + BUILD) vs WT (sets/wt + BUILD2), production args/env -> log/meas.txt
# steps (MEAS env, default all): dec decn mx st par cache par2 soak
S=/mnt/scratch; L=$S/log/meas.txt; P=$S/gprod; W=${WSET:-$S/sets/wt}
export CHUNK=1024 BUDGET=1032
F=BEHEMOTH_NEURON_QWEN35_BUCKET_FIT=1
st() { echo "== $(date -u +%T) $*" >> $L; }
for step in ${MEAS:-dec decn mx st par cache par2 soak}; do
  case $step in
    dec) for t in p w; do G=$P; B=$S/BUILD; [ $t = w ] && { G=$W; B=$S/BUILD2; }
           st dec $t; bash /root/op/dec.sh ${t}_dm $G $B >> $L 2>&1; cat $S/run/${t}_dm/steps.txt $S/run/${t}_dm/mem.txt >> $L; grep -E 'md5|nonfinite' $S/run/${t}_dm/dec.txt >> $L; done ;;
    decn) for t in p w; do G=$P; B=$S/BUILD; [ $t = w ] && { G=$W; B=$S/BUILD2; }
           st decn $t; bash /root/op/dec.sh ${t}_dn $G $B BEHEMOTH_NEURON_SPEC_MTP=0 >> $L 2>&1; cat $S/run/${t}_dn/steps.txt >> $L; grep -E 'md5|nonfinite' $S/run/${t}_dn/dec.txt >> $L; done ;;
    mx) for t in p w; do G=$P; B=$S/BUILD; [ $t = w ] && { G=$W; B=$S/BUILD2; }
           st matrix $t; bash /root/op/matrix.sh ${t}_mx $G $B $F >> $L 2>&1; grep -v '^metrics' $S/run/${t}_mx/matrix.txt >> $L; cat $S/run/${t}_mx/steps.txt >> $L; done ;;
    st) for t in p w; do G=$P; B=$S/BUILD; [ $t = w ] && { G=$W; B=$S/BUILD2; }
           st stall $t; bash /root/op/stall.sh ${t}_st $G $B $F >> $L 2>&1; cat $S/run/${t}_st/stall.txt >> $L; done ;;
    par) for t in p w; do G=$P; B=$S/BUILD; [ $t = w ] && { G=$W; B=$S/BUILD2; }
           st par $t; bash /root/op/par.sh ${t}_par $G $B $F >> $L 2>&1; cat $S/run/${t}_par/par.txt >> $L; done
         st cmp p_par w_par; /opt/aws_neuronx_venv_pytorch_2_9/bin/python /root/op/cmp.py $S/run/p_par/logits.bin $S/run/w_par/logits.bin >> $L 2>&1
         st cmpref; /opt/aws_neuronx_venv_pytorch_2_9/bin/python /root/op/cmpref.py $S/ref p_par w_par >> $L 2>&1 ;;
    cache) st cache w; bash /root/op/cache.sh w_cache $W $S/BUILD2 $F >> $L 2>&1; cat $S/run/w_cache/cache.txt >> $L ;;
    par2) st par2 w; bash /root/op/par.sh w_par2 $W $S/BUILD2 $F >> $L 2>&1; cat $S/run/w_par2/par.txt >> $L
          st cmp w_par w_par2; /opt/aws_neuronx_venv_pytorch_2_9/bin/python /root/op/cmp.py $S/run/w_par/logits.bin $S/run/w_par2/logits.bin >> $L 2>&1
          cmp -s $S/run/w_par/logits.bin $S/run/w_par2/logits.bin && echo "w_par vs w_par2: BIT-IDENTICAL" >> $L || echo "w_par vs w_par2: differ" >> $L ;;
    soak) st soak w; bash /root/op/soak.sh w_soak $W $S/BUILD2 3 >> $L 2>&1; cat $S/run/w_soak/soak.txt >> $L ;;
  esac
  aws s3 sync --only-show-errors --exclude '*' --include '*.txt' --include '*.json' $S/run s3://behemoth-neuron-models-912935854432/tmp/ornith-wt/run
  aws s3 cp --only-show-errors $L s3://behemoth-neuron-models-912935854432/tmp/ornith-wt/log/meas.txt
done
echo MEAS_DONE >> $L
