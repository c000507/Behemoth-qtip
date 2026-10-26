# GPU Crash Investigation — dual Radeon AI PRO R9700 falling off the bus (192.168.0.229)

Date: 2026-10-25. Investigator: opencode (GLM-5.3). All commands run read-only per HARD RULES;
no driver/BIOS/PM settings were changed. Logs saved under `cpp_runtime/perf/crash/` on .229 and
`crash_logs/` in this repo.

## Executive summary

**Three incidents, identical signature, third one happened during this investigation.** Both
GPUs wedge internally (ASIC/firmware) after an **idle period**, while the PCIe link, slot power
and the card's own upstream switch port stay fully alive. Everything below the card's PCIe
switch (downstream port + GPU function) stops decoding config space; the host stays up. The
"16-concurrent load" and "SCOPE_SYS zerocopy" circumstances of incidents 1 and 2 are
**coincidental**: incident 3 wedged with `sorted_stream_policy = 0`, `--transport host`, and
**zero requests ever sent** — both GPUs were simply idle. The workload code (SCOPE_SYS buffer
policies, fine-grained host memory, OOB reads) is exonerated for the bus-loss; it is a
card-firmware / driver idle-power-state failure (GFXOFF/deep-sleep entry or residency),
synchronized across both cards because a TP=2 server idles both GPUs at the same instant.
nmoe survived because continuous training/serving never lets the cards go deep-idle.

**Current state of .229 (as of 16:13):** both GPUs dead (incident 3, discovered 16:05:20),
host up, `behemoth_server` pid 4773 alive but wedged (its `/health` returns `healthy:true`
because it does not probe the GPUs). **Power cycle required (user action). Do not run GPU
work.** No reproduction was attempted — the crash already reproduced itself under the safest
possible configuration.

## Machine / stack facts

- Supermicro X12DAi-N6 in Dell T440 chassis, BIOS **1.1a 08/09/2021** (old), Ubuntu 24.04.
- Kernel **6.17.0-35-generic** (the task note said "7.0.x" — actual is 6.17), cmdline
  `ro quiet splash iommu=pt vt.handoff=7`.
- amdgpu DKMS **6.16.13-2341068.24.04** (amdkcl/amdttm backport), ROCm **7.2.4** userspace.
- GPUs: 2x AMD Radeon AI PRO R9700 (gfx1201, Navi 48), device id 7551 rev c0, XFX boards.
  - `0000:33:00.0` (NUMA 0, root port `30:02.0`), `0000:cc:00.0` (NUMA 1, root port `c9:02.0`).
  - Topology per card: root port → card upstream switch port (`31:00.0` / `ca:00.0`, AMD 1478
    "Navi 10 XL Upstream Port") → card downstream port (`32:00.0` / `cb:00.0`) → GPU. The cards
    are Gen5-capable (32 GT/s), running 16 GT/s x16 at the Gen4 root ports (expected downgrade).
  - VBIOS **113-48WD6SHD1-P02** both, SMC/SMU fw **104.75.0** (0x00684b00), MES fw **0x84**
    (both MES pipes), DMUB 0x0A000601. Boot log: `smu driver if version = 0x2e, smu fw if
    version = 0x32` → **"SMU driver if version not matched"** (DKMS driver interface older
    than card firmware), and "MES FW version must be >= 0x82 to enable LR compute workaround".
- amdgpu params: `runpm=-1`, `aspm=-1`, `mes=0 mes_kiq=0` but **`uni_mes=1`**, `gpu_recovery=-1`,
  `pcie_p2p=Y`, ppfeaturemask 0xfff7bfff (GFXOFF enabled).
- ASPM: root ports report **"ASPM not supported"**, `LnkCtl: ASPM Disabled` on root ports and
  card switch ports, L1 substates all off. **ASPM is not in play.**
- Runtime PM: GPU functions `control=auto` but **never runtime-suspended**
  (`runtime_suspended_time=0`, always `active` — amdgpu holds them while the server has them
  open). HDA functions (`33:00.1`, `cc:00.1`) autosuspend to D3hot within seconds of boot and
  stay suspended (~whole-boot suspended time).
- AER: `_OSC: platform does not support [SHPCHotplug AER LTR DPC]` → firmware-first AER; the OS
  logs no AER records, which is why link events are invisible in dmesg. BMC/Redfish skipped per
  task. Sticky `DevSta: CorrErr+ NonFatalErr+` (+UnsupReq on cc side) on the card upstream ports
  show some corrected/non-fatal PCIe events occurred (undated; plausibly boot-time equalization),
  never FatalErr.
- PCIe slots: `Slot #85/#114, PowerLimit 75W`; cards on their own 1200 W PSUs (user voltmeter
  12.2 V idle / ~11.9 V load — inside tolerance).
- IOMMU: `iommu=pt`, 3 DRHDs; no IOMMU/DMAR faults in any incident window.
- Server cmdline (boot 0): `behemoth_server --model /mnt/nvme/models/Behemoth-R1-123B-v2-aqlm-output
  --devices 0,1 --transport host --code-layout sorted --memory-fraction 0.9 --token-budget 1024
  --prefill-chunk 256 --host 0.0.0.0 --port 1337 --threads 4 --profile false` (pid 4773,
  stdout/stderr → `/tmp/beh_server.log`).

## Timeline of the three incidents

Common signature (identical in all three, to the second and to the message):

1. Silent wedge of both GPUs during/after idle (nothing in any log at the actual wedge moment).
2. First *logged* anomaly: both HDA functions fail `D3hot → D0, device inaccessible` +
   `CORB reset timeout#2, CORBRP = 65535` on BOTH cards in the SAME second, plus
   `workqueue: pm_runtime_work hogged CPU`. This is a **discovery**, not the failure: the probe
   is a PCI-level operation that needs no GPU firmware — failing it means the card was already
   wedged.
3. Later (minutes), when the hung process is killed / queues evicted: `MES(0) failed to respond
   to msg=REMOVE_QUEUE`, `GPU reset begin!. Source: 3`, **`device lost from bus!`**, `GPU reset
   end with ret = -19`, `SMU: response:0xFFFFFFFF ... AllowGfxOff?/SetWorkloadMask?/
   EnableSmuFeaturesLow?/TransferTableSmu2Dram?`, sdma ring timeouts, devcoredumps, MES(1)
   WAIT_REG_MEM timeouts, `MES ring buffer is full`, `GPU Recovery Failed: -19`, TTM workers
   blocked in D state. All reads of the card return 0xFFFFFFFF.

### Incident 1 — Sat 2026-10-25 ~11:46 (boot -2, stock group-major build, `--transport host`)

- 11:12:13-14, 11:25:57 — `amdgpu: failed to disable PTL` on both GPUs (PTL = SR-IOV/PSP
  `amdgpu_virt_ptl_request` path, benign noise on bare metal; marks earlier compute sessions).
- 11:46:33 — HDA D3hot→D0 fail on both cards + CORB timeout + pm_runtime_work hogged
  (first logged anomaly; wedge already happened, 8→16-concurrent load-test start was the
  discovery touch).
- 12:01:25 — user attaches gdb to the hung server (ptrace audit record).
- 12:01:33 → 12:03:33 — process kill; full MES/SMU/reset failure cascade on both cards;
  `device lost from bus!` repeatedly; devcoredumps created for card1/card2 (lost at reboot);
  TTM `ttm_bo_delayed_delete` workers blocked >122 s.
- Reboot ~12:26.

### Incident 2 — Sun 2026-10-25 ~15:11 (boot -1, sorted layout, `--transport zerocopy`, policy 24)

- 14:42:31 — `micro_batched` bench caused `[gfxhub] page fault` / `GCVM_L2_PROTECTION_FAULT`
  bursts on GPU0 (TCP client, RW=0 read, MAPPING_ERROR — OOB reads past a workspace; see code
  review). Faults contained by GPUVM; GPU kept working for another ~29 min.
- 15:11:27 — server "Listening" (per `BATCHED_SORTED_REPORT.md`; weights loaded on both GPUs).
- ~15:11:28 — first (c=1) request: autotune kernels = first GPU compute after the post-load
  idle gap; request **hangs immediately**, 0 completion tokens.
- 15:11:31 — HDA D3hot→D0 fail on both cards (same-second both cards, discovery).
- 15:22:01+ — server killed → MES REMOVE_QUEUE timeouts, `device lost from bus!` both,
  reset ret -19; power cycle 15:37.

### Incident 3 — Sun 2026-10-25, boot 0, discovered 16:05:20 **during this investigation**

- 15:37:09 — boot; amdgpu init clean on both GPUs (gfx_v12_0, mes_v12_0, smu_v14_0_0).
- ~15:43 — `behemoth_server` (pid 4773) loaded both ranks, `Listening on http://0.0.0.0:1337;
  transport=pinned-host; P2P disabled`. **No request was ever sent to this server.**
- ≤16:05:20 — both GPUs wedge silently while **completely idle** (no compute, no zerocopy, no
  SCOPE_SYS, policy 0).
- 16:05:20 — HDA D3hot→D0 fail both cards (discovery; no userland session/udev activity at
  that second — the probe is a periodic PME/pm work path).
- 16:08:14-15, 16:09:15 — repeated HDA probe failures (later reported as D3cold).
- 16:08:43/49 — kernel-internal SMU metrics reads (hwmon/firmware-info paths) log
  `SMU: response:0xFFFFFFFF ... TransferTableSmu2Dram?` and `device lost from bus!`.
- Post-mortem state, verified 16:11-16:13 while still wedged:
  - Root ports `30:02.0`/`c9:02.0`: `LnkSta: Speed 16GT/s, Width x16`, `PresDet+`, DLActive —
    **link trained and slot power present**.
  - Card upstream ports `31:00.0`/`ca:00.0`: config space reads fine (0x1002:1478), LnkSta
    16 GT/s x16 — the card's own switch upstream is alive.
  - GPU functions `33:00.0`/`cc:00.0` and card downstream ports `32:00.0`/`cb:00.0`:
    config space reads return **all 0xFF** (vendor ID 0xffff, header type 0x7f).
  - Host unaffected; `behemoth_server` still serves `/health: healthy:true` (false-negative).

This third incident is the controlled experiment the task would have run: stock build settings,
no load, both cards idle → both wedge. It falsifies every workload-dependent hypothesis.

## Configuration findings

- **ASPM**: not supported by root ports, disabled everywhere including card switch L1 substates
  (`L1SubCtl1: all -`). Ruled out.
- **amdgpu runtime PM**: GPUs never runtime-suspended by the driver (held active by the server
  process; `runtime_suspended_time = 0`). Host-side D3/RTD3 is ruled out as the trigger; the
  deep idle state is entered **by the card firmware itself** (GFXOFF / SMU light-sleep are
  card-internal and invisible to PCI runtime PM).
- **HDA D3hot**: normal audio autosuspend since boot; the failing resume attempts are probes of
  already-wedged cards (PipeWire/wireplumber re-enumeration and periodic PM work). Red herring
  in all three incidents — but a reliable early detector.
- **AER**: firmware-first (`_OSC` platform does not support AER), so PCIe-level detail is not
  logged by Linux; BMC SEL not accessible from here (skipped per task).
- **Power**: incident 3 happened at zero load. Link stayed trained and PresDet+ for hours after
  death — an electrical rail collapse would drop the link/PresDet. External PSUs and slot power
  effectively ruled out.
- **Kernel/firmware version skew**: DKMS amdgpu 6.16.13 (built for kernel 6.17) against card
  SMU fw 104.75 with a printed driver-if-version mismatch; MES 0x84; VBIOS P02; BIOS 1.1a
  (2021). nvidia DKMS 580.178.04 also installed (no NVIDIA hardware present; harmless).
- Server banner claims `link=32.0 GT/s` — it prints the max capability, not current speed
  (actual 16 GT/s); cosmetic only.

## Code review (what this runtime does that nmoe does not)

Findings recorded for completeness; **none can explain a bus-level wedge**, per incident 3.

- `src/transport.hip` — two-rank exchange via fine-grained pinned host memory with
  system-scope atomics (`__hip_atomic_* ... __HIP_MEMORY_SCOPE_SYSTEM`) — GPU reads/writes host
  DRAM over PCIe. Used only with `--transport zerocopy` (incident 2). Legal, common pattern;
  cannot touch card SMU/PCIe.
- `src/projection.hip:744` — `constexpr int sorted_stream_policy = 0, sorted_buffer_flags =
  0x31004000;` (commit ed914a7 introduced policy 24; the committed tree has been reverted to
  policy 0, confirmed on disk and in the running server build).
  - Policy 24 = GFX12 `buffer_load` cache-policy immediate: system-scope cache hint
    (SCOPE_SYS) selecting system-coherent caching for the streamed weight reads. It is a shader
    memory-hint field; it cannot alter PCIe link state or SMU. Runs only when policy != 0.
  - Flags `0x31004000` decode (SQ_BUF_RSRC_WORD3): TYPE=0 (buffer), DST_SEL_W=4 (SEL_0, zero
    4th component of b128 loads), 32-bit element size field, top nibble 3 = GFX12
    word3[31:28] reserved/type field per LLVM AMDGPU usage; NUM_RECORDS bound 0x7fffffff with
    in-kernel `valid[]` clamps. Valid descriptor encoding; OOB handling relies on the kernel
    clamps.
  - OOB reads do exist in the bench paths (incident 2, 14:42:31 `micro_batched` gfxhub
    VM_L2_PROTECTION_FAULTs) — these are contained by the GPU page tables and did NOT kill the
    card (GPU ran 29 more minutes and died idle, like the others).
- No persistent kernels; one non-blocking stream per device (`src/model.cpp:297`). Between
  requests (and after load) the GPUs are fully idle → GFXOFF/deep-sleep eligible within ~1 s.
  **This is the behavioral difference vs nmoe's continuous training loop, and matches every
  incident's idle-before-death pattern.**

## Web search

Blocked from this environment (DuckDuckGo empty result set, GitLab freedesktop behind Anubis
bot-check). Substituted with the DKMS source on disk (`/usr/src/amdgpu-6.16.13-2341068.24.04`),
which is authoritative for the local driver: PTL noise from `amdgpu_virt_ptl_request`
(`amdgpu_psp.c`), GFXOFF control file `amdgpu_gfxoff` in `amdgpu_debugfs.c:1631`. Recommend a
human rerun searches for "gfx1201 device lost from bus", "RDNA4 gfxoff hang", "amdgpu unified
MES gfx12 idle" on gitlab.freedesktop.org and ROCm GitHub.

## Ranked hypotheses

1. **Card-firmware wedge in GFXOFF / SMU light-sleep entry or residency (gfx1201, SMU 104.75 /
   MES 0x84 / VBIOS P02), synchronized across both cards by simultaneous TP idle.**
   For: three identical incidents; inc3 zero-load pure-idle death; link+slot power+upstream
   switch alive while downstream wedged (card-internal); host runtime PM never involved; nmoe
   (never idle) unaffected; wedge discovered by PCI-level probes that need no GPU fw (card
   already dead). Against: exact wedge second unobserved; no direct gfxoff telemetry captured
   yet.
2. **Wedge at wake (exit from deep idle on first compute/probe)** — same firmware defect,
   opposite edge. For: inc1 discovered at load start, inc2 at first request. Against: in inc3
   the audio probe failed at PCI level (no SMU involvement) → card was dead before the touch;
   also the 16:08+ metrics reads failing means it never woke at all. Subsumed by 1.
3. **Host amdgpu/DKMS gfx12 driver bug in the idle path** (SMU interface-version mismatch,
   unified MES housekeeping) that poisons the card during idle. For: printed "SMU driver if
   version not matched"; DKMS older than ROCm userspace. Against: nothing in logs shows host
   activity at wedge time. Would still present as 1; fix path differs (update DKMS).
4. **Electrical (slot 3.3 V / external PSU transient).** Against: link stayed trained and
   PresDet+ for hours post-mortem; inc3 at zero power load; no other PCI devices affected;
   host never wavered. Effectively ruled out.
5. **ASPM L1 exit failure.** Ruled out — ASPM not supported/disabled end-to-end.
6. **Workload: SCOPE_SYS policy 24 / zerocopy fine-grained host memory / OOB reads / long
   kernels.** Ruled out as bus-loss cause — incident 3: policy 0, host transport, no requests,
   died anyway. (OOB reads remain a real bug to fix in the bench, but VM-fault-contained.)
7. **PCIe signal integrity / Gen4 link instability.** No support: links stable at 16GT/s x16
   before, during and after death; only sticky corrected/nonfatal bits; fw-first AER hides
   detail but link state itself stayed up.

## Phase 2 (controlled reproduction) — not performed

Not possible and not needed: at investigation start the machine had **already crashed a third
time** under the minimal configuration (policy 0, host transport, idle). Running the proposed
single-GPU SCOPE_SYS bench would have required a power cycle (user action per HARD RULES) and
would test an already-falsified hypothesis. No server process was killed; pid 4773 left as-is
for the user to inspect.

## Recommendations

1. **Power cycle .229** (user). Afterwards GPUs are fresh; host is currently up with dead GPUs.
2. **Instrument for the next wedge** (cheap, read-only):
   - 1 Hz poll of GPU config space + timestamps to a file, e.g.
     `while sleep 1; do date +%s.%N; sudo dd if=/sys/bus/pci/devices/0000:33:00.0/config bs=2 count=1 2>/dev/null | od -An -tx1; done | tee /tmp/gpu0.poll`
     (both GPUs) — pins the wedge second and its relation to idle/wake.
   - Log `cat /sys/kernel/debug/dri/*/amdgpu_gfxoff` (exists in this DKMS,
     `amdgpu_debugfs.c:1631`) and GPU utilisation alongside.
3. **Primary mitigation test — disable GFXOFF** after reboot, run the same idle scenario:
   `echo 0 | sudo tee "/sys/kernel/debug/dri/*/amdgpu_gfxoff"` (dynamic, reversible; if the
   file needs per-card paths, apply to both). If wedges stop → confirmed hypothesis 1 and a
   usable workaround while awaiting fixed firmware.
4. **Band-aid keepalive**: a background thread in `behemoth_server` (or a cron script) issuing a
   tiny kernel/memset every ≤1 s while idle to prevent deep idle. Also validates hypothesis 1
   (no idle → no wedge).
5. Optional: `echo on | sudo tee /sys/bus/pci/devices/0000{33,cc}:00.1/power/control` to keep
   HDA in D0 (removes the failing-probe noise; probes are symptoms, not cause).
6. **Fix `/health`** in behemoth_server to probe the GPUs (read a scratch register / 1-token
   roundtrip); incident 3 showed `healthy:true` hours after both GPUs died.
7. **Stack updates**: BIOS 1.1a (2021) → latest X12DAi-N6; try a newer amdgpu DKMS/ROCm combo
   (driver-if-version mismatch with SMU 104.75 is a plausible contributor); keep the kernel on
   a version the DKMS officially targets.
8. **Report to AMD** with: VBIOS 113-48WD6SHD1-P02, SMU 104.75.0, MES 0x84, DKMS 6.16.13,
   kernel 6.17.0-35, the three journal excerpts, and next-time capture of
   `/sys/class/drm/card*/device/devcoredump/data` **before** power cycling (inc1/inc2 coredumps
   were lost at reboot).
9. Fix the bench OOB reads (inc2 14:42 gfxhub faults from `micro_batched`) independently —
   not the crash cause, but a real bug.
10. Optional power-check (only if hypothesis 1 is falsified by test 3): limit clocks/power via
    `amd-smi`/`rocm-smi` power cap and rerun — but with the link staying up post-mortem, power
    is a low-probability cause.

## Artifacts

- `crash_logs/inc1_boot-2_kernel.log`, `crash_logs/inc2_boot-1_kernel.log` — kernel logs of
  incidents 1-2 (windows 11:35-12:15 and 14:55-15:35).
- `crash_logs/config_snapshot.txt` — sysfs/lspci/dmidecode/DKMS/firmware snapshot (boot 0).
- Boot-0 (incident 3) evidence pulled live from `journalctl -k -b 0` (quoted above).
- On .229: copies under `cpp_runtime/perf/crash/` (report + logs). Pre-existing fault logs:
  `cpp_runtime/perf/results/e2e/gpu_fault_dmesg.txt`,
  `cpp_runtime/perf/results/batched/gpu_fault_dmesg.txt`, `BATCHED_SORTED_REPORT.md` §fault.
