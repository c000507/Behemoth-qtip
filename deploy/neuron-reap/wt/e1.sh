#!/bin/bash
# e1: engine WT load path. sets/wt0 (decode-prof minimal WT set) on BUILD2 + the prod prepared dir (load-time transpose) vs the same
# buckets from gprod on BUILD (control); then BUILD2 must refuse a mixed set. CHUNK 64, MTP off, 1 stream x 3 x 256.
S=/mnt/scratch; L=$S/log/e1.txt; N=/root/decode-prof-neffs
while ! grep -q W2A_DONE $S/log/w2.txt; do sleep 20; done
bash /root/op/mkset.sh wt0 $N/wset seg20_1_32=$N/x4/seg20wt_1_32 >> $L
mkdir -p $S/sets/nwt0; find $S/sets/nwt0 -maxdepth 1 -type l -delete
for n in $(ls $S/sets/wt0); do ln -s $S/gprod/$n $S/sets/nwt0/$n; done
CHUNK=64 bash /root/op/dec1.sh e1_wt $S/sets/wt0 $S/BUILD2 >> $L 2>&1
grep -E "weights resident|WT" $S/run/e1_wt/server.log | head -3 >> $L
CHUNK=64 bash /root/op/dec1.sh e1_nwt $S/sets/nwt0 $S/BUILD >> $L 2>&1
for t in e1_wt e1_nwt; do echo "== $t"; grep -E "md5|nonfinite|metrics" $S/run/$t/dec.txt >> $L; grep -iE "median|step" $S/run/$t/steps.txt | head -4 >> $L; done
# mixed set must be refused
bash /root/op/mkset.sh mix0 $S/sets/wt0 head_1_1=$S/gprod/head_1_1 >> $L
CHUNK=64 bash /root/op/srv.sh e1_mix $S/sets/mix0 $S/BUILD2 BEHEMOTH_NEURON_SPEC_MTP=0 >> $L 2>&1; grep -m2 -i "disagree\|error" $S/run/e1_mix/server.log >> $L
pkill -f neuron_server
echo E1_DONE >> $L
