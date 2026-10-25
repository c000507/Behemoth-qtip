#!/bin/bash
# 5 Hz poller. Per tick: vendor id of both GPUs, upstream-port PRIMARY_BUS (0x00 => bridge reset),
# root-port AER UESta/CESta, hwmon power(uW)/sclk(Hz)/temp(mC)/vddgfx(mV) per card.
# Full line every tick to CSV; state changes + 60 s heartbeat to stdout.
OUT=${1:-/tmp/gpu_poll2.csv}
H1=/sys/class/drm/card1/device/hwmon/hwmon1; H2=/sys/class/drm/card2/device/hwmon/hwmon2
rd() { timeout 1 cat "$1" 2>/dev/null || echo NA; }
prev=""; n=0
echo "ts,v0,v1,up0_bus,up1_bus,rp0_ue,rp0_ce,rp1_ue,rp1_ce,p0,p1,f0,f1,t0,t1,vg0,vg1" >> "$OUT"
while true; do
  v0=$(setpci -s 33:00.0 VENDOR_ID 2>/dev/null); v1=$(setpci -s cc:00.0 VENDOR_ID 2>/dev/null)
  b0=$(setpci -s 31:00.0 PRIMARY_BUS 2>/dev/null); b1=$(setpci -s ca:00.0 PRIMARY_BUS 2>/dev/null)
  ue0=$(setpci -s 30:02.0 0x104.L 2>/dev/null); ce0=$(setpci -s 30:02.0 0x110.L 2>/dev/null)
  ue1=$(setpci -s c9:02.0 0x104.L 2>/dev/null); ce1=$(setpci -s c9:02.0 0x110.L 2>/dev/null)
  p0=$(rd $H1/power1_average); p1=$(rd $H2/power1_average)
  f0=$(rd $H1/freq1_input); f1=$(rd $H2/freq1_input)
  t0=$(rd $H1/temp1_input); t1=$(rd $H2/temp1_input)
  g0=$(rd $H1/in0_input); g1=$(rd $H2/in0_input)
  ts=$(date +%F_%T.%N)
  echo "$ts,$v0,$v1,$b0,$b1,$ue0,$ce0,$ue1,$ce1,$p0,$p1,$f0,$f1,$t0,$t1,$g0,$g1" >> "$OUT"
  cur="$v0 $v1 $b0 $b1 $ue0 $ce0 $ue1 $ce1"
  if [ "$cur" != "$prev" ] || [ $((n % 300)) -eq 0 ]; then
    echo "$ts gpu0=$v0 gpu1=$v1 upbus=$b0/$b1 rp0_ue=$ue0 ce=$ce0 rp1_ue=$ue1 ce=$ce1 P=$((${p0//NA/0}/1000000))/$((${p1//NA/0}/1000000))W sclk=$((${f0//NA/0}/1000000))/$((${f1//NA/0}/1000000)) T=$((${t0//NA/0}/1000))/$((${t1//NA/0}/1000))"
  fi
  prev=$cur; n=$((n+1)); sleep 0.2
done
