#!/bin/bash
# dec1.sh TAG GRAPHS [BUILD] [ENV...]: fresh server (MODEL_DIR, CHUNK env honoured; MTP off), 1-stream decode x3 (short prompt, 256 new) + LTRACE
# -> run/TAG/{dec.txt,steps.txt}
S=/mnt/scratch; TAG=$1; G=$2; BLD=${3:-$S/BUILD}; shift 3 2>/dev/null || shift $#
echo 2.9 > $S/run/cpt
bash /root/op/srv.sh $TAG $G $BLD "$@" BEHEMOTH_NEURON_SPEC_MTP=0 BEHEMOTH_NEURON_QWEN35_LTRACE=${LTRACE:-100} BEHEMOTH_NEURON_QWEN35_BUCKET_FIT=1 > $S/run/$TAG.srv 2>&1 || { cat $S/run/$TAG.srv; exit 1; }
cd $S
python3 /root/op/pf.py run/$TAG warm:40:1:16 a1:40:1:256 b1:40:1:256 c1:40:1:256 > run/$TAG/dec.txt 2>&1
python3 /root/op/steps.py run/$TAG/server.log > run/$TAG/steps.txt 2>&1
grep request_metrics run/$TAG/server.log | python3 -c 'import sys,json; [print("metrics", {k: d.get(k) for k in ("prompt_tokens","completion_tokens","decode_tokens_per_s")}) for d in (json.loads(l.split(" ",1)[1]) for l in sys.stdin)]' >> run/$TAG/dec.txt
echo "nonfinite: $(grep -c -i nonfinite run/$TAG/server.log)" >> run/$TAG/dec.txt
pkill -f neuron_server; sleep 3
echo DEC_DONE >> run/$TAG/dec.txt
