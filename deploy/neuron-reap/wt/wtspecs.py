#!/usr/bin/env python3
"""wtspecs.py SRCSET [ENV=V ...]: gb.sh specs (name:kind:R:Q:P:ENV=V,...) that rebuild every NEFF of a graph set with the build flags
recorded in its meta.json, plus the extra env given (e.g. WT=1 WT_MOE=grp WT_GRP=8). One spec per line; segments / Q>=1024 MoE last."""
import json, os, sys
import fnmatch
D, extra = sys.argv[1], [a for a in sys.argv[2:] if '@' not in a]
OVR = [a.split('@', 1) for a in sys.argv[2:] if '@' in a]   # NAMEGLOB@ENV=V[,ENV=V]: per-NEFF extra env (after the common one; a later key wins)
ENV = {'moe': 'MOE', 'gdn_slot': 'GDN_SLOT', 'gdn_chunk': 'GDN_CHUNK', 'gdn_inv': 'GDN_INV', 'kv8': 'KV8', 'kv_code': 'KV_CODE', 'att_block': 'ATT_BLOCK',
       'att_qblock': 'ATT_QBLOCK', 'moe_tsub': 'MOE_TSUB', 'moe_cap': 'MOE_CAP', 'moe_hot': 'MOE_HOT', 'moe_ge': 'MOE_GE', 'moe_gf': 'MOE_GF',
       'moe_gb': 'MOE_GB', 'moe_gblk': 'MOE_GBLK', 'verify': 'VERIFY', 'mtp_experts': 'MTP_EXPERTS', 'moe_slots': 'MOE_SLOTS', 'route': 'ROUTE',
       'router': 'ROUTER', 'gdn_rsplit': 'GDN_RSPLIT', 'gdn_form': 'GDN_FORM', 'gdn_dec': 'GDN_DEC', 'rlogit': 'RLOGIT', 'ar_bf16': 'AR_BF16',
       'moe_gd': 'MOE_GD', 'moe_gg': 'MOE_GG', 'moe_tiers': 'MOE_TIERS', 'moe_tail': 'MOE_TAIL', 'gdn_loop': 'GDN_LOOP', 'att_gather': 'ATT_GATHER',
       'lin_cse': 'LIN_CSE', 'att_bf16': 'ATT_BF16', 'egu_t': 'EGU_T', 'moe_kernel': 'MOE_KERNEL'}
SAME = {'moe_slots': 'topk', 'route': 'topk', 'router': 'full', 'gdn_rsplit': 0, 'gdn_form': '', 'gdn_dec': False, 'rlogit': 'f32', 'ar_bf16': '0',
        'moe_gd': 'slot2', 'moe_gg': 'idx', 'moe_tiers': '', 'moe_tail': 'nat', 'gdn_loop': '', 'att_gather': 'index', 'lin_cse': False,
        'att_bf16': False, 'egu_t': False, 'moe_kernel': '', 'kv_code': 'int8', 'gdn_chunk': 0, 'att_block': 0, 'att_qblock': 0, 'moe_tsub': 0,
        'moe_cap': 0, 'moe_ge': 0, 'moe_gblk': 0, 'verify': False, 'kv8': False, 'gdn_slot': ''}
def val(v):
    return '1' if v is True else '0' if v is False else str(v)
small, big = [], []
for n in sorted(os.listdir(D)):
    p = os.path.join(D, n, 'meta.json')
    if not os.path.exists(p):
        continue
    m = json.load(open(p))
    k = m['kind'][len('qwen35_'):]
    env = []
    for key, e in ENV.items():
        if key in m and SAME.get(key, '__none__') != m[key]:
            if key == 'moe_hot' and not m.get('moe_cap'):
                continue
            if key in ('moe_gb', 'moe_gf') and m.get('moe') != 'group':
                continue
            env.append(f'{e}={val(m[key])}')
    if m.get('pages', 4097) != 4097 and k.startswith(('gqa', 'seg')):
        env.append(f'PAGES={m["pages"]}')
    env += extra
    for pat, e in OVR:
        if fnmatch.fnmatch(n, pat):
            env += e.split(',')
    keep = {}
    for e in env:                      # one value per key, the last one wins
        keep[e.split('=', 1)[0]] = e
    env = list(keep.values())
    if k == 'head':
        spec = f'{n}:head:{m["T"]}:{m["S"]}'
        if env:   # gb.sh: name:kind:R:Q:P:envs; a head has no P, the env goes after an empty field
            spec = f'{n}:head:{m["T"]}:{m["S"]}::' + ','.join(env)
    else:
        spec = f'{n}:{k}:{m["R"]}:{m["Q"]}:{m["P"]}' + (':' + ','.join(env) if env else '')
    (big if k.startswith('seg') or (k == 'moe' and m['Q'] >= 1024) else small).append(spec)
print('\n'.join(small + big))
