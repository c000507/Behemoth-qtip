# R9700 OverDrive limit search on 192.168.0.229 (2026-10-25, 21:50-00:05)

Follow-up to `GPU_CRASH_ROOTCAUSE.md`. The goal was the fastest operating point for the two R9700s
that survives behemoth_server (TP=2) load until the shared GPU 12 V feed gets re-cabled.

## Result

**Chosen setting: OverDrive `OD_SCLK_OFFSET -500 MHz`, `OD_VDDGFX_OFFSET 0 mV`, `power1_cap 210 W`,
perf level `auto`, power profile `BOOTUP_DEFAULT`, the same on both cards.**

- Stability: 20.9 min clean (28 c=4 batches) plus another 6 min clean on a repeat run. The 45 min
  final run was skipped on instruction after 6 min with no fault.
- Speed: c=4 aggregate 17.97 tok/s, single request 11.47 tok/s (~87 ms/token).
- Load: ~209 W avg per card (peak 228-237 W), ~2.57-2.61 GHz avg, 0.87 V avg.
- Compared with DPM level 1 (old `~/gpu_vf_cap.sh`): **+10 % c=4 and +16 % single-request.**
- Compared with stock (300 W, which crashes): -9 % c=4 and -3 % single-request.
- Margin: the crash boundary at -500 MHz is between 230 W (survived 20.7 min) and 250 W (crashed
  after 5.7 min). 210 W is the lowest cap the firmware allows. Note that crash 6 happened at 210 W
  with no offset, so the offset is part of what keeps this point stable.

## 1. Boot parameter

**The mask in the brief (`0xffffbfff`) does not enable OverDrive.** OverDrive is
`PP_OVERDRIVE_MASK = 0x4000` (bit 14), and that bit is clear in both `0xfff7bfff` (the default) and
`0xffffbfff`. `0xffffbfff` only adds bit 19 (`PP_GFX_DCS_MASK`). I booted it once to check:
`pp_od_clk_voltage` was still missing. The value in use is the default mask plus the OverDrive bit:

```
/etc/default/grub:
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash iommu=pt amdgpu.ppfeaturemask=0xfff7ffff"
```

`sudo update-grub` was run. Both 7.0.0-38 entries and the 6.17 entry in `/boot/grub/grub.cfg` carry
it. After a BMC cycle: `/proc/cmdline` contains it, `/sys/module/amdgpu/parameters/ppfeaturemask`
= `0xfff7ffff`, and dmesg shows `amdgpu: Overdrive is enabled`. The kernel is now tainted, which is
cosmetic. `ctest --test-dir build-rocm`: 3/3 passed (core, gpu, gpu_transport).

**Revert:** `sudo cp /etc/default/grub.bak-20261025 /etc/default/grub && sudo update-grub`, or delete
` amdgpu.ppfeaturemask=0xfff7ffff` from that line and run update-grub. Then do a BMC ForceOff/On.
After that, OD files are gone and `~/kernel.sh` falls back to `~/gpu_vf_cap.sh` on its own.

## 2. OD interface found (gfx1201, SMU 14.0.2, DKMS 7.1.9)

`/sys/class/drm/card{1,2}/device/pp_od_clk_voltage`:
```
OD_SCLK_OFFSET: 0Mhz        OD_MCLK: 0: 97Mhz 1: 1259MHz      OD_VDDGFX_OFFSET: 0mV
OD_RANGE: SCLK_OFFSET -500..1000 MHz   MCLK 97..1500 MHz   VDDGFX_OFFSET -200..0 mV
```
- Write `s <MHz>` (sclk offset) and `vo <mV>` (voltage offset), then `c` to commit. `r` resets.
  Writes work while the perf level is `auto`.
- There is no OD_PPT. The power limit is set through hwmon `power1_cap`: min 210 W, max = default
  = 300 W.
- The sclk offset only lowers the top of the V/F curve. Under load, -500 MHz still lets the cards
  reach 3.0-3.1 GHz at ~1.03 V in light phases (spin-wait); 4-6 % of loaded samples were above
  2.95 GHz at 210-230 W. Below 2.9 GHz the curve is not limited by the offset at all, so 300 W at
  -500 MHz behaves like stock.
- There is no runtime way to set the maximum clock between 2.35 GHz and the boosted top:
  - `pp_dpm_sclk` (manual) has only the levels 500 / ~2300 / 2350. Level 2 runs at 2.30-2.35 GHz,
    the same as level 1.
  - The `POWER_SAVING` power profile (tested 6 min at -500/210 W) did not remove the boosts and was
    slower (16.8 tok/s), so it was reverted.
  - So, as far as I could test, the top-state boosts cannot be removed without losing the OD gain.

## 3. Results

Same server process and command line as before. Load: `behemoth_load_bench prompt.txt 4 8 96`
batches, run continuously. "c1" is 2 sequential single requests of 96 tokens. "ms/tok" is 1000/c1
tok/s. The server's `inter_token_p50_ms` is a rolling value mixed with c=4 tokens, so it is not
used. Power, clock, temperature and voltage come from `/tmp/gpu_poll2.csv` (5 Hz hwmon, samples with
>60 W), shown as avg/peak, card0 | card1. Settings were identical on both cards.

| Tag | sclk off / vo / cap | Duration | Crash? | c=4 tok/s | c1 tok/s (ms/tok) | Power W | GFX clock MHz | VDDGFX mV | Temp C |
|---|---|---|---|---|---|---|---|---|---|
| L1 (old workaround) | DPM manual level 1 | 6.5 min | no (45 min earlier, Exp E) | 16.4 | 9.92 (101) | 162/194 \| 161/191 | 2289/2358 \| 2288/2369 | 778/806 \| 772/805 | 62/64 \| 63/66 |
| SO500 | -500 / 0 / 300 | 1.6 min of load | **yes** (both cards gone, Surprise Down) | 19.8 | 11.78 (85) | 287/312 \| 286/312 | 2917/3059 \| 2943/3062 | 978/1029 \| 978/1024 | 67/73 \| 67/75 |
| X1 | -500 / -100 / 210 | 3.0 min | compute fault: engine exception, GPUs stayed on bus, server `healthy:false`, restarted | 19.1 | 11.77 (85) | 206/238 \| 205/229 | 2811/3065 \| 2840/3077 | 844/939 \| 845/941 | 61/64 |
| **X2 (chosen)** | **-500 / 0 / 210** | **20.9 min, 28 batches** | **no** | **17.97** | **11.47 (87)** | 209/228 \| 209/237 | 2571/3072 \| 2611/3074 | 866/1030 \| 868/1019 | 61/68 \| 62/69 |
| X3 | -500 / 0 / 250 | 5.7 min | **yes** (same signature) | 18.9 | 11.79 (85) | 248/275 \| 247/277 | 2751/3054 \| 2798/3057 | 923/1030 \| 927/1018 | 63/64 \| 62/63 |
| X4 | -500 / 0 / 230 | 20.7 min, 28 batches | no | 18.45 | 11.67 (86) | 228/266 \| 228/261 | 2665/3092 \| 2713/3100 | 896/1038 \| 901/1038 | 63/63 \| 61/67 |
| FINAL210 | -500 / 0 / 210 (repeat) | 6.0 min (stopped: 45 min run skipped) | no | 17.90 | 11.51 (87) | 209/232 \| 209/228 | 2574/3015 \| 2617/3027 | 866/1017 \| 870/1007 | 62/63 \| 62/67 |
| PS210 | -500 / 0 / 210 + POWER_SAVING profile | 5.9 min | no | 16.8 | 11.23 (89) | 202/228 \| 202/227 | 2625/3043 \| 2671/3042 | 891/1029 \| 895/1009 | 61 \| 59/63 |

There is no full c=4 batch at plain stock: it crashed in 49 s and in 4 min before. SO500 at 300 W
behaves like stock (power-limited at ~2.9 GHz / 0.98 V), so its 19.8 tok/s c=4 and 11.8 tok/s c1
serve as the stock reference. Exp F1 (one card stock) gave 19.5.

How to read it:
- Power cap is the knob that matters. Crash timing went from 1.6 min at 300 W to 5.7 min at 250 W.
  230 W and 210 W were clean for 20 min.
- Throughput depends little on clock: going from 210 to 300 W gains only +10 % c=4.
- An undervolt of -100 mV is not usable. It caused a real compute error, not the power fault.
  -50 mV was not tested because a silent compute error is worse than a crash.
- Crash logs: `~/od_exp/crash_SO500/`, `~/od_exp/crash_X3/` (fastpoll.log, gpu_poll2.csv). Both show
  the same simultaneous Surprise Down on both root ports as crashes 8 and 9.
- Per-run logs and JSON: `~/od_exp/<TAG>.log`, `_c1.json`, `_b.json`.

### Throughput summary (c=4 aggregate / single request)
| Point | c=4 tok/s | vs L1 | single tok/s | vs L1 |
|---|---|---|---|---|
| DPM level 1 (`gpu_vf_cap.sh`) | 16.4 | - | 9.92 | - |
| **OD -500/0/210 W (chosen)** | **17.97** | **+10 %** | **11.47** | **+16 %** |
| OD -500/0/230 W (survived 20 min, smaller margin) | 18.45 | +12.5 % | 11.67 | +18 % |
| stock-like (300 W, crashes) | 19.8 | +21 % | 11.78 | +19 % |

230 W would add only +2.7 % c=4 while sitting 20 W from a cap that crashed in under 6 minutes, so
210 W was chosen.

## 4. Persistence

- `~/gpu_od_apply.sh [SOFF] [VOFF] [CAP_W]`. It defaults to `-500 0 210`. For each of card1 and
  card2 it sets: perf level `auto`, profile 0, `s -500`, `vo 0`, `c`, and `power1_cap`=210 W. It
  exits non-zero if OD is missing. Revert to stock with `~/gpu_od_apply.sh 0 0 300`.
- `~/kernel.sh` has these lines appended (backup in `~/kernel.sh.bak-20261025`):
  `bash ~/gpu_od_apply.sh || bash ~/gpu_vf_cap.sh`
- Tested end to end: BMC ForceOff/On, then `bash ~/kernel.sh`. Both cards came up with -500 MHz,
  0 mV, 210 W, `auto`, BOOTUP_DEFAULT.
- `~/gpu_vf_cap.sh` (DPM level 1) is unchanged and remains the conservative fallback.
- Experiment harness: `~/od_exp/` (`exp.sh`, `an.py`, `start_server.sh`, `warm.sh`, `postboot.sh`).

## 5. State at the end (00:05)

- Both GPUs healthy (vendor 1002, root ports SDES-). OD setting applied through kernel.sh after a
  fresh BMC cycle.
- `behemoth_server` (pid 2939) is on 0.0.0.0:1337 with the unchanged command line, `healthy:true`,
  and already autotuned (warm-up requests sent).
- Pollers are running: `fastpoll` (pid 2935, `/tmp/fastpoll.log`) and `gpu_poll2.sh` (pid 2936,
  `/tmp/gpu_poll2.csv`). No stress load is running.
- No packages, kernel, driver, BIOS or VBIOS were changed. The only boot change is the
  ppfeaturemask parameter. The behemoth tree was not modified.

## Open items
- The real fix is still the per-card native PCIe power cables. After re-cabling, test
  `~/gpu_od_apply.sh 0 0 300` (stock) for 45 min with `load_bench` c=4.
- Optional: a 45+ min soak of the chosen point would confirm the margin; only 20.9 + 6 min were run.

## Update 2026-10-26
- 11:36: -500 MHz / 0 mV / 210 W dropped both cards off the bus during a 20.8k-token prefill (attention
  round 4 benchmark). Poller peaks that day under this setting: 263-330 W and 3.08-3.09 GHz despite the
  210 W cap and -500 MHz offset - the cap/offset limit the average, not the transients.
- 12:10: -500 MHz / -50 mV / 210 W soak (`perf/crash/stress/stab_od50.sh`: 33.8k-token prefill + c=4
  decode, alternating). Rounds 1-5 clean (prefill ~200 s, c=4 38 tok/s, 96 ms/step; median 171-177 W,
  max 306-330 W, vddgfx max ~0.99 V, edge 63-67 C, junction 75-80 C). At ~12:32 (22 min) the server
  returned "Nonfinite model logits" (silent compute corruption), then at ~12:37
  `amdgpu 0000:33:00.0: MES(1) failed to respond to msg=INVALIDATE_TLBS`. Cards stayed on the bus; a
  server restart recovered without a reboot. -50 mV is NOT stable (as -100 mV before).
- New default (`~/kernel.sh`): OD reset to stock (`gpu_od_apply.sh 0 0 300`) then DPM level 1
  (`gpu_vf_cap.sh`, ~2.1-2.3 GHz under load).
