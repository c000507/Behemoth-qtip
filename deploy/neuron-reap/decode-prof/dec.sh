#!/bin/bash
# dec.sh TAG GRAPHS [BUILD] [ENV...]: fresh prod-env server (CHUNK 1024 / BUDGET 1032 / BUCKET_FIT), decode at 1/2/4/8 streams (short prompts, 256 new)
# -> run/TAG/{dec.txt,steps.txt}. MTP on unless BEHEMOTH_NEURON_SPEC_MTP=0 is passed.
S=/mnt/scratch; TAG=$1; G=$2; BLD=${3:-$S/BUILD}; shift 3 2>/dev/null || shift $#
echo 2.9 > $S/run/cpt
CHUNK=${CHUNK:-1024} BUDGET=${BUDGET-1032} bash /root/op/srv.sh $TAG $G $BLD "$@" BEHEMOTH_NEURON_QWEN35_BUCKET_FIT=1 > $S/run/$TAG.srv 2>&1 || { cat $S/run/$TAG.srv; exit 1; }
cd $S
python3 /root/op/pf.py run/$TAG warm:40:1:16 d1:40:1:256 d2:40:2:256 d4:40:4:256 d8:40:8:256 > run/$TAG/dec.txt 2>&1
python3 /root/op/steps.py run/$TAG/server.log > run/$TAG/steps.txt 2>&1
grep request_metrics run/$TAG/server.log | python3 -c 'import sys,json; [print("metrics", {k: d.get(k) for k in ("prompt_tokens","completion_tokens","decode_tokens_per_s","active_start")}) for d in (json.loads(l.split(" ",1)[1]) for l in sys.stdin)]' >> run/$TAG/dec.txt
echo "nonfinite: $(grep -c -i nonfinite run/$TAG/server.log)" >> run/$TAG/dec.txt
pkill -f neuron_server; sleep 3
echo DEC_DONE >> run/$TAG/dec.txt
