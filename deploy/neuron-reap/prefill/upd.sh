#!/bin/bash
# upd.sh [BUILDDIR]: refresh /mnt/scratch/src from S3 (and /root/op scripts); with BUILDDIR also rebuild the engine there
export PATH=/usr/local/bin:/opt/aws/neuron/bin:$PATH
S=/mnt/scratch; T=s3://behemoth-neuron-models-912935854432/tmp/ornith-prefill
aws s3 cp --only-show-errors --recursive $T/box /root/op
aws s3 cp --only-show-errors $T/src.tgz $S/src.tgz && tar -xzf $S/src.tgz -C $S/src && echo "src updated"
if [ -n "$1" ]; then (cd $S/src && PATH=$S/shim:$PATH bash neuron/cpp/build.sh $1 srv) > $S/log/build_$(basename $1).log 2>&1; tail -1 $S/log/build_$(basename $1).log; grep -E "error" $S/log/build_$(basename $1).log | head; fi
