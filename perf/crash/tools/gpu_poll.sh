#!/bin/bash
# 1 Hz: timestamp + vendor id of both GPUs; logs only changes + a heartbeat every 60 s
prev=""; n=0
while true; do
  a=$(sudo setpci -s 33:00.0 VENDOR_ID 2>&1); b=$(sudo setpci -s cc:00.0 VENDOR_ID 2>&1)
  cur="$a $b"
  if [ "$cur" != "$prev" ] || [ $((n % 60)) -eq 0 ]; then echo "$(date +%F_%T.%N) gpu0=$a gpu1=$b"; fi
  prev=$cur; n=$((n+1)); sleep 1
done
