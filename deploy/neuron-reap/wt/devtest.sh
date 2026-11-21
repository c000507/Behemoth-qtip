#!/bin/bash
# devtest.sh DIR...: tests/test_qwen35_device.py (device NEFF vs the CPU mirror, random weights of the real shapes) for single NEFF
# dirs, with the graph module's switches taken from each NEFF's meta.json
. /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
export PATH=/usr/local/bin:/opt/aws/neuron/bin:$PATH NEURON_RT_VISIBLE_CORES=0-1 NEURON_RT_ROOT_COMM_ID=127.0.0.1:62128
S=/mnt/scratch; mkdir -p $S/iotmp
for d in "$@"; do
  d=$(realpath $d)
  envs=$(python3 - $d/meta.json <<'P'
import json, sys
m = json.load(open(sys.argv[1]))
e = {'MOE': m.get('moe', 'all'), 'MOE_CAP': m.get('moe_cap') or 32, 'MOE_HOT': m.get('moe_hot', 16), 'MOE_GG': m.get('moe_gg', 'idx'), 'MOE_GD': m.get('moe_gd', 'slot2'),
     'MOE_GE': m.get('moe_ge', 0), 'GDN_CHUNK': m.get('gdn_chunk', 0), 'GDN_INV': m.get('gdn_inv', 'rec'),
     'GDN_SLOT': m.get('gdn_slot', ''), 'KV8': 1 if m.get('kv8') else 0, 'KV_CODE': m.get('kv_code', 'int8'), 'ATT_BLOCK': m.get('att_block', 0), 'ROUTE': m.get('route', 'topk'),
     'MOE_GF': m.get('moe_gf', 'onehot'), 'MOE_GB': m.get('moe_gb', 'sc1'), 'MOE_TIERS': m.get('moe_tiers', '').replace(',', '+'), 'MOE_TAIL': m.get('moe_tail', 'nat'), 'MOE_GSLOTS': m.get('moe_gslots', 3072), 'MOE_GBLK': m.get('moe_gblk', 0), 'MOE_TSUB': m.get('moe_tsub', 0), 'ATT_QBLOCK': m.get('att_qblock', 0),
     'WT': 1 if m.get('wt') else 0, 'WT_MOE': m.get('wt_moe', ''), 'WT_GRP': m.get('wt_grp') or 8, 'GDN_RSPLIT': m.get('gdn_rsplit', 0), 'MOE_SLOTS': m.get('moe_slots', 'topk'),
     'VERIFY': 1 if m.get('verify') else 0, 'ATT_SPLIT': m.get('att_split', 0), 'ATT_FOLD': m.get('att_fold', 0), 'TP': m['geometry']['tp'], 'PAGES': m['pages'], 'SLOTS': m['slots']}
print(' '.join(f'{k}={v}' for k, v in e.items()))
P
)
  ( cd $(dirname $d) && env $envs IO_TMP=$S/iotmp python $S/src/neuron/tests/test_qwen35_device.py $S/model/config.json $(dirname $d) $S/BUILD/neff_io $(basename $d) 2>&1 | grep -v "^{" | tail -3 )
done
