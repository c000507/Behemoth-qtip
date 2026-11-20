#!/bin/bash
# cache.sh TAG GRAPHDIR BUILD: prefix-cache exactness. Server 1: p8k cold, p8k again (full hit), p8k+ (partial hit: restore + prefill).
# Server 2: p8k+ cold. Logits (8 greedy steps each, MTP off) -> run/TAG/{warm,cold}.bin; compare with cmpcache.py.
S=/mnt/scratch; TAG=$1; G=$2; B=$3
echo 2.9 > $S/run/cpt; mkdir -p $S/run/$TAG; cd $S
: > run/$TAG/warm.bin; : > run/$TAG/cold.bin
bash /root/op/srv.sh $TAG $G $B NEURON_LOGITS_DUMP=$S/run/$TAG/warm.bin BEHEMOTH_NEURON_SPEC_MTP=0 > run/$TAG.srv 2>&1 || { cat run/$TAG.srv; exit 1; }
python3 /root/op/pf.py run/$TAG p8k:8000:1:8 > run/$TAG/cache.txt 2>&1
python3 /root/op/pf.py run/$TAG p8k:8000:1:8 p8k+:8000:1:8 q6k:6000:1:8 q6k:6000:1:8 q6k+:6000:1:8 >> run/$TAG/cache.txt 2>&1
grep request_metrics run/$TAG/server.log | python3 -c 'import sys,json; [print("warm metrics", {k: d.get(k) for k in ("prompt_tokens","cached_tokens","prefill_ms","ttft_ms")}) for d in (json.loads(l.split(" ",1)[1]) for l in sys.stdin)]' >> run/$TAG/cache.txt
cp run/$TAG/server.log run/$TAG/server_warm.log
bash /root/op/srv.sh $TAG $G $B NEURON_LOGITS_DUMP=$S/run/$TAG/cold.bin BEHEMOTH_NEURON_SPEC_MTP=0 > run/$TAG.srv 2>&1 || { cat run/$TAG.srv; exit 1; }
python3 /root/op/pf.py run/$TAG p8k+:8000:1:8 q6k+:6000:1:8 q6k:6000:1:8 >> run/$TAG/cache.txt 2>&1
pkill -f neuron_server; sleep 2
/opt/aws_neuronx_venv_pytorch_2_9/bin/python - $S/run/$TAG >> run/$TAG/cache.txt <<'P'
import sys, numpy as np
V = 248320; d = sys.argv[1]
w = np.memmap(d + '/warm.bin', np.float32).reshape(-1, V); c = np.memmap(d + '/cold.bin', np.float32).reshape(-1, V)
print('records warm', w.shape[0], 'cold', c.shape[0])
def cmp(name, a, b):
    rel = [float(np.linalg.norm(a[i] - b[i]) / np.linalg.norm(b[i])) for i in range(8)]
    print(f'{name}: max rel L2 over 8 steps {max(rel):.3e}; bit-identical {bool((a == b).all())}; top-1 {sum(int(a[i].argmax() == b[i].argmax()) for i in range(8))}/8')
cmp('p8k full hit vs its cold run', w[8:16], w[0:8])
cmp('p8k+ partial hit vs cold server', w[16:24], c[0:8])
cmp('q6k (padded 256-token tail chunk) full hit vs its cold run', w[32:40], w[24:32])
cmp('q6k+ partial hit vs cold server', w[40:48], c[8:16])
cmp('q6k cold, second server vs first server', c[16:24], w[24:32])
P
echo "nonfinite: $(grep -c -i nonfinite run/$TAG/server.log run/$TAG/server_warm.log | tr "\n" " ")" >> run/$TAG/cache.txt
echo CACHE_DONE >> run/$TAG/cache.txt
