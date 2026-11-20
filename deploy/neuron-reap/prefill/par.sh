#!/bin/bash
# par.sh TAG GRAPHDIR BUILD [ENV...]: logits of the first 8 greedy steps of 7 prompts (MTP off: one sampled row per launch) -> run/TAG/logits.bin
S=/mnt/scratch; TAG=$1; G=$2; B=$3; shift 3
echo 2.9 > $S/run/cpt
mkdir -p $S/run/$TAG; : > $S/run/$TAG/logits.bin
bash /root/op/srv.sh $TAG $G $B NEURON_LOGITS_DUMP=$S/run/$TAG/logits.bin BEHEMOTH_NEURON_SPEC_MTP=0 BEHEMOTH_DIAGNOSTIC_DIR=$S/run/$TAG/diag "$@" > $S/run/$TAG.srv 2>&1 || { cat $S/run/$TAG.srv; exit 1; }
cd $S; for e in "$@"; do export "$e"; done
python3 /root/op/pf.py run/$TAG short:40:1:8 p1k:1000:1:8 q2k:2000:1:8 p4k:4000:1:8 q6k:6000:1:8 p8k:8000:1:8 p32k:32000:1:8 > run/$TAG/par.txt 2>&1
pkill -f neuron_server; sleep 2
grep -c -i nonfinite run/$TAG/server.log >> run/$TAG/par.txt
python3 /root/op/steps.py run/$TAG/server.log | grep "tokens=256 " >> run/$TAG/par.txt
grep -o '"moe_fallbacks":[0-9]*' run/$TAG/server.log | sort | uniq -c >> run/$TAG/par.txt
ls -la run/$TAG/logits.bin >> run/$TAG/par.txt; echo PAR_DONE >> run/$TAG/par.txt
