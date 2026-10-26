#!/bin/bash
d=$1
for m in "pcie 480" "mixed 480"; do echo "$(date +%T) dev $d start $m"; ./gpustress $d $m; echo "$(date +%T) dev $d end $m rc=$?"; done
