#!/bin/bash
# xb.sh OUT SPEC...: decode NEFF builds with the production decode settings (lbuild.sh: MOE=gather KV8=1 GDN_SLOT=bound PAGES 4097 SLOTS 9)
#   SPEC = name:kind:R:Q:P[:ENV=V,...]; JOBS (default 2) parallel; log OUT/<name>.log, OUT/build.txt
OUT=$1; shift; mkdir -p $OUT
CONFIG=/mnt/scratch/model/config.json SRC=/mnt/scratch/src JOBS=${JOBS:-2} bash /mnt/scratch/src/deploy/neuron-reap/lbuild.sh $OUT "$@" 2>&1 | tee -a $OUT/build.txt
echo XB_DONE >> $OUT/build.txt
