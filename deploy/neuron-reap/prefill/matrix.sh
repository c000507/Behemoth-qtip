#!/bin/bash
# matrix.sh TAG GRAPHDIR [BUILD] [ENV...]: fresh server, the fixed prompt matrix, step summary -> /mnt/scratch/run/TAG/{matrix.txt,steps.txt}
S=/mnt/scratch; TAG=$1
echo 2.9 > $S/run/cpt
bash /root/op/srv.sh "$@" > $S/run/$TAG.srv 2>&1 || { cat $S/run/$TAG.srv; exit 1; }
cd $S
python3 /root/op/pf.py run/$TAG short:40:1:200 p1k:1000 p4k:4000 p8k:8000 p32k:32000 p60k:60000 c4x8k:8000:4 c8x4k:4000:8 short4:40:4:120 > run/$TAG/matrix.txt 2>&1
python3 /root/op/steps.py run/$TAG/server.log > run/$TAG/steps.txt 2>&1
echo "nonfinite: $(grep -c -i nonfinite run/$TAG/server.log)" >> run/$TAG/matrix.txt; grep -E 'Qwen35: 40 layers|MTP: ready' run/$TAG/server.log >> run/$TAG/matrix.txt; grep request_metrics run/$TAG/server.log | python3 -c 'import sys,json; [print("metrics", {k: d.get(k) for k in ("prompt_tokens","cached_tokens","prefill_ms","prefill_tokens_per_s","decode_tokens_per_s","ttft_ms","active_start")}) for d in (json.loads(l.split(" ",1)[1]) for l in sys.stdin)]' >> run/$TAG/matrix.txt
echo MATRIX_DONE >> run/$TAG/matrix.txt
aws s3 sync --only-show-errors --exclude '*' --include '*.txt' --include '*.json' run/$TAG s3://behemoth-neuron-models-912935854432/tmp/ornith-chunk4096/run/$TAG
