# Root cause: both Radeon AI PRO R9700s falling off the PCIe bus (192.168.0.229)

Date: 2026-10-25, 18:30-21:30. Nine crashes total (7 before this investigation, 2 reproduced on purpose
here as crash 8 and crash 9). Raw artifacts: `crash_logs/gpu_crash7/`, `crash_logs/gpu_crash8/`
(fast-poll transition log with microsecond timestamps, 1 Hz power trace, dmesg) and
`~/gpu_crash9/` on .229. Helper tools left on .229: `~/fastpoll` (200 Hz config-space poller),
`~/gpu_poll2.sh` (1-2 Hz CSV: vendor IDs, bridge bus numbers, root-port AER, power/clock/temp/VDDGFX),
`~/gpu_recover_probe.sh`, `~/gpu_vf_cap.sh`, `~/stress/gpustress{,2}` (+ `.hip` sources).

## Conclusion (confidence: high for the mechanism class, medium-high for the exact component)

**The cards are not hanging; they are being hard-reset by a power event on the shared 12 V feed
(the one GPU PSU + the single breakout cable feeding both cards).** Every crash is a
*simultaneous fundamental reset of both cards* (within one 5 ms poll iteration, 46-47 us apart in the
log), seen by both root ports as a **PCIe Surprise Down** on two independent CPU sockets. The trigger
is both cards running at their stock voltage/frequency point at the same time under the behemoth
TP=2 decode load (~300 W each, 2.9-3.2 GHz, 0.98-1.07 V VDDGFX, all-reduce lockstep so the current
steps of both cards coincide). Capping the GFX clock of **both** cards (DPM level 1: ~2.3 GHz,
~0.77 V, ~165 W under the same load) made the identical server + load run **45 min without a
fault**; un-capping only GPU 0 ran another **24 min without a fault**; un-capping GPU 1 as well (both
stock) killed both cards **4 min later**. Software, firmware/driver stack, GFXOFF, SCOPE_SYS, transport
type and the specific PSU unit are all excluded as the cause (details below). Synthetic loads
(600 W sustained, synchronized 0->600 W steps at 13 Hz and ~300 Hz, heavy PCIe/host-memory traffic)
never reproduce it, so this is not a simple "PSU too small" story; it is a transient at the
specific operating point the behemoth kernels drive the cards to (high clock AND high power at the
same time, in lockstep on both cards). The one element common to both cards that fits a
simultaneous reset on two sockets is the shared 12 V path.

Why not a per-card / firmware fault: GPU 0 alone at full stock V/f for 24 min under the same server
did not fail; it only fails when GPU 1 is also at stock V/f. A per-card fault would not care about
the other card's clock. Why not software: the same server process, same code, same load, same
transport ran 69 minutes across the capped phases and died 4 minutes after the only change
(DPM level of the second card). Why not the PSU unit: replaced after crash 3, crashes 4-9 happened
with the new one; the breakout cable (and the connectors/wiring between PSU and cards) stayed.
Why not a host-initiated reset: a directed hot reset / SBR does not set Surprise Down, and nothing
in dmesg precedes the loss (amdgpu, pcieport, ACPI all silent at the instant of death).

## Signature of every crash (verified in crashes 7, 8, 9 with 200 Hz polling)

From `crash_logs/gpu_crash8/fastpoll.log` (crash 8, 19:31:18; crash 9 identical at 21:18:34):

```
19:31:18.071973 gpu0_vid   00001002 -> 0000ffff      both GPUs gone in the same poll iteration
19:31:18.072019 gpu1_vid   00001002 -> 0000ffff
19:31:18.072030 up0_bus    00333231 -> ffffffff      card upstream switch ports gone too
19:31:18.072040 up1_bus    00cccbca -> ffffffff
19:31:18.072069 rp0_uesta  00000000 -> 00000020      root port 30:02.0: Surprise Down Error
19:31:18.072078 rp0_cesta  00000000 -> 00000001      ... + Receiver Error
19:31:18.072089 rp0_lnksta 00003103 -> 00001101      link down (reports 2.5 GT/s x1)
19:31:18.072098 rp1_uesta  00000000 -> 00000020      root port c9:02.0: Surprise Down Error
19:31:18.072107 rp1_cesta  00000000 -> 00000001
19:31:18.072127 rp1_lnksta 00003104 -> 00001101
19:31:18.082366 rp0_lnksta 00001101 -> 00001903      retrain 8 GT/s
19:31:18.123386 rp0_lnksta 00001903 -> 00001904      ... 16 GT/s
19:31:18.164399 up0_bus    ffffffff -> 00000000      card upstream port back, config RESET (bus numbers 0)
19:31:18.164460 rp0_lnksta 00001904 -> 00003104      link fully up again (92 ms after loss)
```

- Fresh-boot baseline of both root ports: `SDES- RxErr- CorrErr- NonFatalErr-`. After every crash:
  `SDES+ RxErr+ CorrErr+ NonFatalErr+`. The link to each card dropped unexpectedly and came back.
- After the crash the card's upstream switch port (31:00.0 / ca:00.0) reads
  `Bus: primary=00 secondary=00 subordinate=00`, `Command: I/O- Mem- BusMaster-`, `CommClk-`:
  **power-on default config space**. The downstream port and GPU read 0xffff only because the reset
  bridge forwards nothing. `~/gpu_recover_probe.sh` wrote the bus numbers back after crash 8: the
  downstream port and the GPU then answered (`vid=1002 devid=7551`, BAR0 cleared) - the ASIC is
  alive but in reset state (needs VBIOS POST + driver init). In-OS recovery
  (`echo 1 > .../31:00.0/remove` then rescan) hangs in D state inside amdgpu's reset path, so the
  only recovery is the BMC power cycle (graceful reboot hangs for the same reason).
- The `snd_hda_intel ... D3hot to D0, device inaccessible` / `CORB reset timeout` lines are the first
  *logged* symptom; they are a consequence (the HDA function's config space is gone), not a cause.
  `ERREVENT_ATHUB_INTERRUPT` and the "1048560 umc errors" in crashes 4-6 are the driver reading
  all-ones from a card that is already gone. `/health` stays `healthy:true` because the server never
  touches the GPUs in that handler.
- Devcoredumps are only produced when the driver later attempts a reset; crash 7 had none.
- BMC SEL has no power events at any crash time (it only monitors the system PSU rails; the GPU PSU
  is invisible to the BMC).

## Experiment table

All on kernel 7.0.0-38, amdgpu DKMS 7.1.9, SMU 104.80.0, ROCm 10.1 (crashes 1-6 happened on the
older stack: same signature). Power/clock/VDDGFX from hwmon (`power1_average`, `freq1_input`,
`in0_input`).

| # | Config | GPUs | Operating point | Duration | Result |
|---|--------|------|-----------------|----------|--------|
| Crash 7 (18:29) | behemoth_server zerocopy+sorted, 1 request | both, TP=2 | stock | ~6 min after start | both lost, same second |
| A | `gpustress` burst (2 kHz kernel+sync) / pulse (13 Hz 0<->300 W) / steady, two independent processes, no collectives, no host memory | both | 300 W each, 2.9-3.0 GHz, ~0.98 V (compute-bound, power-limited) | 30 min | survived, no AER bits |
| B | `gpustress` pcie (64 MB pinned H2D+D2H + zero-copy reads of host memory, 1.5-1.8 kHz syncs) then mixed compute+PCIe | both | up to 300 W | 16 min | survived |
| C (crash 8) | behemoth_server, identical command line, `load_bench` c=4 continuous | both, TP=2 | 300-340 W each for 46 s, last sample 3068/3157 MHz, 1031/1056 mV | died 49 s after the first request | both lost within 46 us |
| D | `gpustress2` barrier-synchronized spikes: memory-streaming phase (3.43-3.48 GHz, 1160 mV, 200 W) then both cards start a heavy burst at the same instant; shapes 20 ms/1 ms, 3 ms/0.3 ms, idle 37 ms/heavy 37 ms, 75 ms/20 ms | both | 200-300 W, up to 1160 mV | 22 min | survived |
| E | behemoth_server identical, `load_bench` c=4 continuous, **both cards DPM level 1** | both, TP=2 | 160-187 W each, ~2290 MHz, 0.77 V | **45 min, 51 batches (408 requests), 16.3 tok/s aggregate** | survived |
| F1 | same server process, GPU 0 set back to `auto`, GPU 1 still capped | both, TP=2 | GPU 0: 300 W, 3.0-3.15 GHz, 1.02-1.07 V; GPU 1: 165 W | 24 min | survived |
| F2 (crash 9) | same server process, GPU 1 also `auto` (= stock) | both, TP=2 | 300 W each, 2.9-3.1 GHz, 0.98-1.0 V | **died after 4 min** | both lost within 47 us, same signature |

Earlier crashes (user log): 1 stock build/host transport/c=16; 2 sorted/zerocopy first request;
3 sorted/host transport ~30 min; 4 new PSU; 5 GFXOFF off; 6 power cap 210 W; 7 new sw stack.
Each of those settings is therefore excluded as the cause. Crash 6 (210 W cap) is consistent with
the mechanism: a power cap limits the *average* over the SMU's control window, not the instantaneous
step from the spin-wait state (3.4 GHz, 1.16 V, ~10 W) into a full-rate kernel; a clock/voltage cap
does limit that step. Under the DPM cap the cards sit at 0.77 V and cannot produce the large step.

Why synthetic loads survive and behemoth does not: the synthetic kernels are either compute-bound
(SMU drops clocks to ~2.9 GHz/0.98 V to hold 300 W) or memory-bound (3.45 GHz/1.16 V but only
200 W). behemoth's weight-streaming + AQLM decode kernels sit at 300-340 W **and** 3.1 GHz/1.05 V at
the same time, in lockstep on both cards 176 times per token, with idle spin-wait gaps in between.
That is the largest and most synchronized current step the two cards can produce, and it only
appears when both cards are at stock V/f. A dedicated PSU that survives 600 W sustained plus
synchronized 600 W pulses but not this pattern points at a transient (di/dt) problem in the cable /
connector path or the PSU's transient response, not at average capacity.

## What this is not

- Not GFXOFF / idle-state firmware wedge (GLM report hypothesis): crash 8 and 9 happened at 300 W
  full load with 200 Hz evidence of the exact moment; nothing idles. The "idle" crash 3 had four
  requests active.
- Not the ROCm/DKMS/SMU stack: identical signature on both stacks.
- Not the zero-copy transport, SCOPE_SYS loads, or the sorted layout: crash 1 and 3 used neither;
  the same binary survives 69 min under cap.
- Not the ROCm/TheRock issue #9299 failure mode as such: there the cards *hang* (SMU/MES no response)
  with the link up and no AER; here the cards are *reset* with a Surprise Down on the link. Shared
  factor with that report: bursty high-V/f load on gfx1201 and the SMU driver-if 0x2e / fw 0x33
  mismatch, so an AMD firmware component cannot be fully excluded, but it does not explain why
  GPU 0 alone at stock never fails and only the pair does.
- Not a single PSU unit: two units, same result, same breakout cable.

## Fixes

### Hardware (the real fix) - for the user
1. Re-cable the GPU PSU: **one native 2x8-pin (or 12V-2x6) cable set per card, each from its own
   PSU output connector**; retire the single breakout that feeds both cards. If the GPU PSU is a
   server PSU + breakout board, each card must have its own 2 (ideally 3) 8-pin leads from separate
   breakout outputs, 16 AWG, short. Check every 8-pin pin for a loose crimp / burnt plastic.
2. If the PSU is multi-rail, put the two cards on different 12 V rails (OCP per rail), or use a
   single-rail PSU >= 1000 W rated for GPU transients (ATX 3.x / PCIe 5.x spec, which requires
   surviving 200 % power excursions for 100 us).
3. Verify with a meter at a card's 8-pin under the behemoth load (Exp. C conditions): the 11.9 V the
   user measured earlier was at steady load; the fault is a transient, so a scope or a min-hold meter
   is needed. If after re-cabling crash 8's condition (stock V/f, `load_bench` c=4) survives 45 min,
   done; also re-run with the old PSU if the coil whine was the PSU's control loop fighting the kHz
   load steps (the whine during decode was the audible version of the same event).
4. Keep the slot power paths as they are; the host side (root ports, slot presence, 3.3 V) never
   showed a fault.

### Software/settings workaround (in place now; resets at every boot)
`~/gpu_vf_cap.sh` on .229 (also safe to call from `~/kernel.sh`):

```
for c in card1 card2; do
  d=/sys/class/drm/$c/device
  echo manual | sudo tee $d/power_dpm_force_performance_level
  echo 1      | sudo tee $d/pp_dpm_sclk          # ~2.3 GHz max, ~0.77 V, <=~190 W observed
done
# revert: echo auto | sudo tee /sys/class/drm/card{1,2}/device/power_dpm_force_performance_level
```

Cost: c=4 aggregate 16.3 tok/s vs ~25-30 tok/s expected at stock (stock c=1 was 13.5 tok/s and c=8
~40 tok/s per DECODE_PERF_REPORT); roughly 40-50 % throughput. A finer cap (e.g. 2.6-2.8 GHz) is not
available at runtime because OverDrive is masked out (`ppfeaturemask=0xfff7bfff` has bit 14 clear,
so `pp_od_clk_voltage` does not exist and `amd-smi set -L sclk max` returns NOT_SUPPORTED); enabling
it needs the module parameter `amdgpu.ppfeaturemask=0xffffbfff` (a boot setting, not done here) and
then `s 1 2700` / `vo -50` in `pp_od_clk_voltage` would allow a gentler cap or an undervolt.

Note: capping only one card is not enough as a safety measure in general (F1 survived 24 min, but
that is a sample of one); cap both until re-cabled.

### Other recommendations
- `behemoth_server` `/health`: read a GPU register / run a 1-token roundtrip so a dead pair is reported.
- Set `power/control=on` for `0000:33:00.1` and `0000:cc:00.1` if the HDA noise is unwanted (cosmetic).
- Leave `~/fastpoll` and `~/gpu_poll2.sh` running (both started via setsid nohup, logs in
  `/tmp/fastpoll.log`, `/tmp/fastpoll_power.log`, `/tmp/gpu_poll2.csv`, `/tmp/gpu_poll2.log`):
  if a drop happens again, `grep -v hb /tmp/fastpoll.log` gives the exact order and timing.
- Not needed: BIOS/VBIOS/firmware flashes, PCIe Gen3 forcing (links were fine before, during and
  after; Gen downshift would not change a 12 V transient), kernel/driver downgrades.
- Optional AMD report: dual R9700 (XFX RX-97XPROAIY), SMU 104.80.0 driver-if mismatch, simultaneous
  Surprise Down of two cards on a shared PSU under lockstep TP load, survivable at DPM level 1.

## Current state of .229 (21:26)
- Power-cycled at 21:19, kernel 7.0.0-38, `~/kernel.sh` applied, both GPUs healthy
  (`SDES-` on both root ports), `~/gpu_vf_cap.sh` applied to both cards.
- `behemoth_server` (pid 3193) on 0.0.0.0:1337, command line unchanged (zerocopy, sorted,
  memory-fraction 0.9, token-budget 1024, prefill-chunk 256, threads 4); healthy; first `load_bench`
  batch c=4 OK (7.9 tok/s including autotune; steady state ~16 tok/s). Load: 160-177 W per card,
  2290 MHz, 0.77 V.
- Pollers: `fastpoll` (pid 3196) and `gpu_poll2.sh` (pid 3195).
- Nothing in the behemoth git tree was modified (HEAD a12a658, clean).
