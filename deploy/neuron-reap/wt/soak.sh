#!/bin/bash
# soak.sh TAG GRAPHDIR BUILD N: N fresh server starts (prod env, chunk 1024), each: 8 concurrent x 1024 new tokens, 4 x 1024, 1 x 2048
# (MTP on) and a 6k-token prompt + 512 new; nonfinite count per start -> run/TAG/soak.txt
S=/mnt/scratch; TAG=$1; G=$2; B=$3; N=${4:-3}
echo 2.9 > $S/run/cpt; mkdir -p $S/run/$TAG; : > $S/run/$TAG/soak.txt
for i in $(seq 1 $N); do
  CHUNK=1024 BUDGET=1032 bash /root/op/srv.sh ${TAG}_$i $G $B BEHEMOTH_NEURON_QWEN35_BUCKET_FIT=1 > $S/run/${TAG}_$i.srv 2>&1 || { cat $S/run/${TAG}_$i.srv >> $S/run/$TAG/soak.txt; continue; }
  cd $S; python3 /root/op/pf.py run/${TAG}_$i k8:300:8:1024 k4:300:4:1024 k1:300:1:2048 k6k:6000:1:512 > run/${TAG}_$i/pf.txt 2>&1
  echo "start $i: nonfinite $(grep -c -i nonfinite run/${TAG}_$i/server.log), errors $(grep -c ERROR run/${TAG}_$i/pf.txt), $(grep -c md5 run/${TAG}_$i/pf.txt) requests; md5 $(grep -o 'md5=[0-9a-f]*' run/${TAG}_$i/pf.txt | tr '\n' ' ')" >> $S/run/$TAG/soak.txt
  pkill -f neuron_server; sleep 3
done
echo SOAK_DONE >> $S/run/$TAG/soak.txt
