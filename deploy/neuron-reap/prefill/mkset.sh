#!/bin/bash
# mkset.sh NAME BASE [bucket=DIR ...]: graph set /mnt/scratch/sets/NAME = symlinks to BASE's buckets, with the listed buckets replaced
# (bucket=- removes one). 
S=/mnt/scratch/sets/$1; B=$(realpath $2); shift 2
mkdir -p $S; find $S -maxdepth 1 -type l -delete
for d in $B/*/; do ln -s $(realpath $d) $S/$(basename $d); done
for kv in "$@"; do n=${kv%%=*}; d=${kv#*=}; [ -L $S/$n ] && unlink $S/$n; [ "$d" = - ] || ln -s $(realpath $d) $S/$n; done
ls $S | wc -l
