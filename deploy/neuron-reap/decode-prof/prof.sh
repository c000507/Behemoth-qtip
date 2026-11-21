#!/bin/bash
# prof.sh OUTDIR NEFFDIR...: 2-rank neuron-profile capture (rank 0 and 1) + summary-json + json export per NEFF dir -> OUTDIR/<name>.{ntff,sum.json,json}
export PATH=/usr/local/bin:/opt/aws/neuron/bin:$PATH HOME=/root NEURON_RT_VISIBLE_CORES=0-1 NEURON_RT_ROOT_COMM_ID=127.0.0.1:62128
O=$1; shift; mkdir -p $O
for d in "$@"; do
  n=$(basename $d); cd $O
  rm -rf $n.cap; mkdir -p $n.cap; cd $n.cap
  neuron-profile capture -n $d/graph.neff -s $O/$n.ntff --collectives-workers-per-node 2 --collectives-worker-count 2 --num-exec ${NEXEC:-5} --profile-nth-exec ${NTH:-5} ${PEXTRA} > $O/$n.cap.log 2>&1 || echo "$n capture rc $?"
  ls $O | grep "^$n" | head -20 > $O/$n.files
  for f in $(ls $O/$n*.ntff 2>/dev/null); do
    b=${f%.ntff}
    neuron-profile view -n $d/graph.neff -s $f --output-format summary-json > $b.sum.json 2> $b.view.err
    [ -n "$NOJSON" ] || neuron-profile view -n $d/graph.neff -s $f --output-format json --output-file $b.json >> $b.view.err 2>&1
  done
  echo "$n profiled: $(ls $O/$n*.ntff 2>/dev/null | wc -l) ntff"
done
echo PROF_DONE
