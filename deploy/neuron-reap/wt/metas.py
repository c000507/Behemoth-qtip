#!/usr/bin/env python3
"""metas.py DIR: one line per NEFF dir: name kind R Q P + non-default build flags from meta.json"""
import json, sys, os
D = sys.argv[1]
DEF = {'moe_slots': 'topk', 'moe_bmm': False, 'router': 'full', 'lin_cse': False, 'gdn_dec': False, 'gdn_form': '', 'row_form': '', 'moe_split': 0,
       'moe_kernel': '', 'moe_kstat': 0, 'moe_kskip': False, 'moe_kyd': 0, 'moe_kdense': 0, 'moe_kfast': 0, 'moe_kpre': False, 'egu_t': False,
       'route': 'topk', 'ar_bf16': '0', 'gdn_chunk': 0, 'gdn_inv': 'rec', 'gdn_loop': '', 'rlogit': 'f32', 'att_bf16': False, 'kv8': False,
       'kv_code': 'int8', 'att_block': 0, 'att_gather': 'index', 'gdn_gather': 'index', 'gdn_einsum': 'mm', 'gdn_slot': '', 'extra_outputs': [],
       'moe_cap': 0, 'moe_hot': 16, 'moe_gg': 'idx', 'moe_ge': 0, 'moe_stat': 0, 'moe_gd': 'slot2', 'moe_gf': 'onehot', 'moe_gb': 'sc1', 'moe_tiers': '',
       'moe_tail': 'nat', 'moe_gslots': 3072, 'moe_gblk': 0, 'moe_tsub': 0, 'att_qblock': 0, 'verify': False, 'wt': False, 'wt_moe': '', 'gdn_rsplit': 0,
       'prof_noar': False, 'prof_fixed': False, 'fp8': True, 'pages': 4097, 'slots': 9, 'mtp_experts': None, 'D': 0}
for n in sorted(os.listdir(D)):
    p = os.path.join(D, n, 'meta.json')
    if not os.path.exists(p):
        print(n, 'NO META'); continue
    m = json.load(open(p))
    extra = {k: v for k, v in m.items() if k not in ('kind', 'R', 'Q', 'P', 'T', 'S', 'inputs', 'states', 'geometry', 'neff', 'seg', 'outputs') and DEF.get(k, '__') != v}
    print(n, m['kind'], m.get('R', m.get('T')), m.get('Q', m.get('S')), m.get('P', ''), ' '.join(f'{k}={v}' for k, v in sorted(extra.items())))
