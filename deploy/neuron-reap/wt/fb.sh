#!/bin/bash
# fb.sh: after wtbuild: the dense MOE=all fallbacks of the WT set (Q=256 / 1024: 64-token sub-blocks, 5-expert groups) into gwt,
# then a reproduction check of wtspecs.py: a few prod NEFFs rebuilt WITHOUT WT must match gprod byte for byte -> log/fb.txt
S=/mnt/scratch; L=$S/log/fb.txt
while ! grep -q WTB_DONE $S/gwt/build.txt 2>/dev/null; do sleep 30; done
specs=$(python3 /root/op/wtspecs.py $S/gprod WT=1 "moe_1_256_32@WT_MOE=grp,WT_GRP=5,MOE_TSUB=64" "moe_1_1024_32@WT_MOE=grp,WT_GRP=5,MOE_TSUB=64" | grep -E '^(moe_1_256_32|moe_1_1024_32):')
echo "$specs" >> $L
JOBS=1 bash /root/op/gb.sh $S/gwt $specs >> $L 2>&1
rs=$(python3 /root/op/wtspecs.py $S/gprod | grep -E '^(gdn_1_1_32|moe_1_64_32|gqa_1_1024_512|moeg_1_256_32|head_8_8|mtp_1_1_32|gdn_1_256_32):')
JOBS=3 bash /root/op/gb.sh $S/repro $rs >> $L 2>&1
for d in $S/repro/*/; do n=$(basename $d); python3 - $d/meta.json $S/gprod/$n/meta.json $n >> $L <<'P'
import json, sys
a, b = (json.load(open(f)) for f in sys.argv[1:3])
diff = sorted(k for k in set(a) | set(b) if k != 'neff' and a.get(k, '<missing>') != b.get(k, '<missing>'))
print('repro', sys.argv[3], 'meta identical to prod' if not diff else 'meta differs: ' + ', '.join(f'{k}={a.get(k, "-")}/{b.get(k, "-")}' for k in diff))
P
done
echo FB_DONE >> $L
