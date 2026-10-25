#!/bin/bash
for m in "spike 480 0,1 20 1" "spike 300 0,1 3 0.3" "syncpulse 300 0,1 37 37" "spike 240 0,1 75 20"; do echo "$(date +%T) start $m"; ./gpustress2 $m; echo "$(date +%T) end rc=$?"; done
