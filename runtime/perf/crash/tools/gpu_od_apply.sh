#!/bin/bash
# Apply OverDrive operating point to both R9700s (card1, card2). Needs amdgpu.ppfeaturemask with bit 14 (0x4000) set.
# Chosen 2026-10-25: -500 MHz sclk offset, 0 mV, 210 W cap (both cards). 250 W crashed in 5.7 min, 300 W in 1.6 min, -100 mV gave a compute fault. Fallback: ~/gpu_vf_cap.sh (DPM level 1).
# Usage: gpu_od_apply.sh [SCLK_OFFSET_MHz] [VDDGFX_OFFSET_mV] [POWER_CAP_W]   (defaults = chosen stable point)
# Revert to stock: gpu_od_apply.sh 0 0 300   (or: echo r > pp_od_clk_voltage; echo c > pp_od_clk_voltage)
SOFF=${1:--500}; VOFF=${2:-0}; PCAP=${3:-210}
rc=0
for c in card1 card2; do
  d=/sys/class/drm/$c/device; h=$(ls -d $d/hwmon/hwmon* | head -1)
  if [ ! -e $d/pp_od_clk_voltage ]; then echo "$c: no pp_od_clk_voltage (OverDrive masked?)"; rc=1; continue; fi
  echo auto | sudo tee $d/power_dpm_force_performance_level >/dev/null
  echo 0 | sudo tee $d/pp_power_profile_mode >/dev/null   # BOOTUP_DEFAULT profile
  echo "s $SOFF"  | sudo tee $d/pp_od_clk_voltage >/dev/null || rc=1
  echo "vo $VOFF" | sudo tee $d/pp_od_clk_voltage >/dev/null || rc=1
  echo c          | sudo tee $d/pp_od_clk_voltage >/dev/null || rc=1
  echo $((PCAP*1000000)) | sudo tee $h/power1_cap >/dev/null || rc=1
  echo "$c: $(grep -A1 -E "OD_SCLK_OFFSET|OD_VDDGFX_OFFSET" $d/pp_od_clk_voltage | grep -v -- -- | tr "\n" " ") cap=$(( $(cat $h/power1_cap)/1000000 ))W level=$(cat $d/power_dpm_force_performance_level)"
done
exit $rc
