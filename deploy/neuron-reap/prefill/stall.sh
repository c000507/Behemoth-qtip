#!/bin/bash
# stall.sh TAG GRAPHDIR BUILD [ENV...]: fresh server, decode-stall cases -> run/TAG/stall.txt
S=/mnt/scratch; TAG=$1
echo 2.9 > $S/run/cpt
bash /root/op/srv.sh "$@" > $S/run/$TAG.srv 2>&1 || { cat $S/run/$TAG.srv; exit 1; }
cd $S
( python3 /root/op/stall.py run/$TAG s24k 24000 1 1500 2.0; python3 /root/op/stall.py run/$TAG s24kb 24000 4 1500 3.0; python3 /root/op/stall.py run/$TAG s6k 6000 1 600 2.0 ) > run/$TAG/stall.txt 2>&1
python3 /root/op/steps.py run/$TAG/server.log >> run/$TAG/stall.txt 2>&1
echo STALL_DONE >> run/$TAG/stall.txt
