#!/bin/bash
# Cap both R9700s GFX clock to DPM level 1 (~2300 MHz, ~0.78 V) — workaround for dual-card PCIe drop-off.
# Resets at every boot; run after boot (e.g. from kernel.sh). Revert: echo auto > .../power_dpm_force_performance_level
for c in card1 card2; do
  d=/sys/class/drm/$c/device
  echo manual | sudo tee $d/power_dpm_force_performance_level >/dev/null
  echo 1 | sudo tee $d/pp_dpm_sclk >/dev/null
  echo "$c: level=$(cat $d/power_dpm_force_performance_level) sclk=$(tr "\n" " " < $d/pp_dpm_sclk)"
done
