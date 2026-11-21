#!/bin/bash
# upd.sh: refresh /mnt/scratch/src and /root/op from the S3 scratch prefix (no engine rebuild)
B=s3://behemoth-neuron-models-912935854432/tmp/ornith-wt
aws s3 sync --only-show-errors $B/box /root/op && chmod +x /root/op/*.sh
aws s3 cp --only-show-errors $B/src.tgz /mnt/scratch/src.tgz && tar -xzf /mnt/scratch/src.tgz -C /mnt/scratch/src && echo upd ok
