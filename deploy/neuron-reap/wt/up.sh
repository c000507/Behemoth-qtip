#!/bin/bash
# up.sh SET NAME: stage SET (graph.neff + meta.json per NEFF dir, symlinks resolved) + MANIFEST.md5 and upload to s3 tmp/ornith-wt/NAME/
S=/mnt/scratch; SET=$1; N=$2; U=$S/up/$N; mkdir -p $U
for d in $SET/*/; do n=$(basename $d); [ -f $d/graph.neff ] || continue; mkdir -p $U/$n; cp -L $d/graph.neff $d/meta.json $U/$n/; done
(cd $U && find . -name graph.neff -o -name meta.json | sort | sed 's#^\./##' | xargs md5sum > MANIFEST.md5)
aws s3 sync --only-show-errors $U s3://behemoth-neuron-models-912935854432/tmp/ornith-wt/$N/ && echo "UP_DONE $N $(ls $U | grep -vc MANIFEST) dirs $(du -sh $U | cut -f1)"
