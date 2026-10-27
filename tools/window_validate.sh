#!/bin/bash
# One behemoth_server stop window (<= ~1 h) for real-layer validation and timing of qtip_quant on .229.
# Stops the server by exact PID, runs the steps below, and ALWAYS restarts the server with the identical command line.
# Usage: bash tools/window_validate.sh <steps, e.g. "1 2 3 4 5 6"> ; log: $Q/window.log
set -u
Q=/mnt/nvme/qtip_quant
REPO=/home/charles/Development/Behemoth-R1-123B-v2-aqlm-codebase/qtip_quant
RT=/home/charles/Development/Behemoth-R1-123B-v2-aqlm-codebase/cpp_runtime
BIN=$REPO/build/qtip_quant
TOK=$Q/calib_2048x2048.i32
STEPS=${1:-"1 2 3 4 5 6"}
LOG=$Q/window.log
exec >> $LOG 2>&1
say() { echo "[$(date +%T)] $*"; }
gpus_ok() { [ "$(sudo setpci -s 33:00.0 VENDOR_ID)" = 1002 ] && [ "$(sudo setpci -s cc:00.0 VENDOR_ID)" = 1002 ]; }
od_ok() {  # policy since 13:00: DPM manual, sclk level 1
  for c in 1 2; do
    [ "$(cat /sys/class/drm/card$c/device/power_dpm_force_performance_level)" = manual ] || return 1
    grep -q "^2: .*\*" /sys/class/drm/card$c/device/pp_dpm_sclk && return 1  # never on the top level
  done
  return 0
}
mes_errors() { sudo dmesg | grep -ciE "mes.*(fail|hang|timeout)|ring .* timeout|gpu reset" ; }
PID=$(cat /tmp/beh_server.pid)
CMD=$(tr '\0' ' ' < /proc/$PID/cmdline)
CWD=$(readlink /proc/$PID/cwd)
say "window start; server pid $PID cwd $CWD cmd: $CMD"
echo "$CMD" > $Q/beh_server_cmdline.txt
[ "$CWD" = "$RT" ] || { say "unexpected server cwd, abort"; exit 1; }
cp /tmp/beh_server.log $Q/beh_server.log.before_window 2>/dev/null
restart_server() {
  say "restarting server: $CMD"
  cd $RT && setsid nohup $CMD > /tmp/beh_server.log 2>&1 < /dev/null &
  sleep 5
  pgrep -x behemoth_server > /tmp/beh_server.pid
  say "server pid $(cat /tmp/beh_server.pid)"
  for i in $(seq 1 90); do
    grep -q Listening /tmp/beh_server.log && break
    sleep 10
  done
  say "log: $(grep Listening /tmp/beh_server.log)"
  say "health: $(curl -s -m 10 127.0.0.1:1337/health)"
}
trap restart_server EXIT
kill $PID
for i in $(seq 1 30); do kill -0 $PID 2>/dev/null || break; sleep 2; done
if kill -0 $PID 2>/dev/null; then say "server ignored SIGTERM, SIGKILL $PID"; kill -9 $PID; fi
for i in $(seq 1 30); do kill -0 $PID 2>/dev/null || break; sleep 2; done
if kill -0 $PID 2>/dev/null; then say "server pid $PID still alive: abort without restart"; trap - EXIT; exit 1; fi
sleep 5
for c in 1 2; do say "card$c vram used $(( $(cat /sys/class/drm/card$c/device/mem_info_vram_used) / 1048576 )) MiB"; done
od_ok || { say "DPM not manual/level1: running gpu_vf_cap.sh"; bash ~/gpu_vf_cap.sh; }
MES0=$(mes_errors)
for s in $STEPS; do
  gpus_ok || { say "GPU(s) missing before step $s: stop"; exit 1; }
  [ "$(mes_errors)" = "$MES0" ] || { say "new MES/ring errors in dmesg before step $s: stop"; exit 1; }
  od_ok || { say "DPM state changed before step $s: stop"; exit 1; }
  say "=== step $s"
  case $s in
  1) # forward / Hessian check on layers 0,1 with 2 sequences (GPU 0)
     rm -rf $Q/fwd && $BIN --mode collect --layers 0:2 --ntok 4096 --gpus 0 --keep-hessians --tokens $TOK \
        --work $Q/fwd/work --out $Q/fwd/out --dump-acts $Q/fwd/acts ;;
  2) # layer 40/41 Hessians, 512k tokens (forward 0..39, collect 40, 41), single GPU (steady load)
     $BIN --mode collect --layers 40:42 --ntok 524288 --gpus 0 --keep-hessians --tokens $TOK \
        --work $Q/par/work --out $Q/par/out --hess-dir $Q/par/hess ;;
  3) # quantize layer 40 on one GPU, with dumps for the decode check
     rm -rf $Q/q1 && $BIN --mode quantize --layers 40:41 --gpus 0 --hess-dir $Q/par/hess --keep-hessians \
        --work $Q/q1/work --out $Q/q1/out --dump-dir $Q/q1/dump ;;
  4) # quantize layers 40 and 41 concurrently on both GPUs
     rm -rf $Q/q2 && $BIN --mode quantize --layers 40:42 --gpus 0,1 --hess-dir $Q/par/hess --keep-hessians \
        --work $Q/q2/work --out $Q/q2/out ;;
  5) # native CPU reference checks against the committed numerical fixtures
     "$REPO/build/qtip_tests_cpu" "$REPO/tests/refvec" ;;
  6) # full pipeline with 4.19M tokens on both GPUs, layers 0..1 (collect + quantize)
     rm -rf $Q/run && $BIN --mode run --layers 0:2 --ntok 4194304 --gpus 0,1 --tokens $TOK \
        --work $Q/run/work --out $Q/run/out ;;
  esac
  say "step $s exit $?"
done
gpus_ok || say "GPU(s) missing at the end"
say "window steps done"
