#!/bin/bash
d=$1
for m in "burst 600" "pulse 600 75 50" "steady 600"; do echo "$(date +%T) dev $d start $m"; ./gpustress $d $m; echo "$(date +%T) dev $d end $m rc=$?"; done
