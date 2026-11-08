# SP (scalar) instruction encodings from libnrt builders

Source: objdump of `add_*` instruction builders from aws-neuronx-runtime-lib `libnrt.so` (trn1/sunda), input `/tmp/opencode/addfns.txt`. Every builder zero-fills a 64-byte instruction on the stack, writes fields, then appends via `add_ins`.

All offsets below are byte offsets **within the 64-byte instruction**. `argN` = function argument N in System V AMD64 order (rdi, rsi, rdx, rcx, r8, r9, then stack at rsp+8, +16, ...). Arguments are named only by role inferred from the function name / usage; semantics beyond that are not asserted.

## Conventions

- `add_ins(ctx, pos, ins)` — copies 64B from `ins` to `ctx->base[pos]`; asserts `pos+0x40 <= ctx->cap`; returns `pos+0x40`. All builders return the next position.
- **Header**: byte0 = opcode, byte1 = `tdrv_arch_instr_get_inst_word_len(ctx)` (arch-dependent, 16 in practice), bytes2-3 = 0 unless noted.
- Optional trailing `argK` is a pointer to a u64 that, if non-NULL, is dereferenced into offset 0x4 in most builders (observed at 0x4 or the noted offset). Listed as `0x04 = *argK (opt)`.
- Instruction slot base used by builders varies (rsp, rsp+0x8, rsp+0x10, rsp+0x28); offsets here are always instruction-relative.
- Shift/mask arithmetic observed: `add_notify`/`add_halt_v2_with_notify` write `((arg<<3) & 0x78)` — a 4-bit engine/target id shifted into bits 3..6. `add_mov32_reg` packs stride-4 u32 arrays into byte-packed u64s (`b[i]`, `b[i]<<8`, ...).

## Opcodes seen in this file

| value | name (from NEURON_ISA_TPB_OPCODE) | builders |
|---|---|---|
| 0x49 (73) | MEMSET | add_memset_1d |
| 0x74 (74) | REG_LOAD | add_tensor_scalar_addr |
| 0x79 (121) | EMBEDDING_UPDATE | add_embedding_update |
| 0x7a (122) | (unknown name) | add_load_pool_arguments |
| 0x7c (66) | TENSOR_REDUCE_ARITH_OP | add_1d_max_all_reduce, add_reduce |
| 0x7d (125) | (unknown name) | add_2d_and_all_reduce |
| 0x9f (159) | ENGINE_NOP | add_engine_nop |
| 0xa0 (160) | EVENT_SEMAPHORE | add_evsem, all add_semaphore_* |
| 0xa1 (161) | HALT | add_halt, add_halt_v2_with_notify |
| 0xa2 (162) | DRAIN | add_drain |
| 0xa4 (164) | NOP | add_nop, add_ev_set/wait/clr/wait_and_set, add_debug_info |
| 0xa5 (165) | WRITE | add_wr32, add_wr32_2addr, add_wr32_v3, add_dma_tail_inc(_bcast), add_dma_rearm(_bcast), add_dma_config_base/size |
| 0xa6 (166) | NOTIFY | add_notify, add_custom_notify |
| 0xa7 (167) | MOVE | add_mov32_imm, add_mov32_reg |
| 0xa8 (168) | ALU_OP | add_add32(_imm), add_sub32(_imm), add_mul32(_imm), add_max32, add_min32, add_and, add_iseq_imm, add_bit_or_imm, add_reg_add_u64_imm, add_save_pc_to_regs |
| 0xa9 (169) | COMPARE_BRANCH | add_br_*, add_cmp*_br_* |
| 0xaa (170) | TENSOR_LOAD | add_ldr32_addr/reg, add_ldr16_addr/reg, add_ldr_addr, add_args_table_swap |
| 0xab (171) | TENSOR_STORE | add_str32_reg, add_str32_addr, add_str32_imm_reg, add_str32_imm_addr(_v3) |
| 0xb0 (176) | (unknown name) | add_sema_clear |
| 0xb1 (177) | (unknown name) | add_set_ordering_mode |
| 0xb3 (179) | POLL_SEM | add_poll_sem |
| 0xb5 (181) | (unknown name) | add_br_hint_rel_imm |

---

## MOVE (0xa7)

### add_mov32_imm(ctx, pos, arg3, arg4, arg5)

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0xa7 |
| 0x1 | 1 | inst_word_len |
| 0x4 | 8 | `*arg5` u64 (0 if arg5 NULL) |
| 0xc | 2 | 0x901 |
| 0xe | 1 | 1 |
| 0xf..0x17 | 9 | 0 (explicit) |
| 0x18 | 1 | arg3 (byte) |
| 0x19..0x1f | 7 | 0 (explicit) |
| 0x20 | 4 | arg4 (u32) |
| 0x24..0x3f | 28 | 0 (explicit) |

Uncertain: role of arg3 (byte @0x18) and arg4 (u32 @0x20); neither labeled by disassembly.

### add_mov32_reg(ctx, pos, arg3=count, arg4=ptr array, arg5=ptr array, arg6=ptr opt)

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0xa7 |
| 0x1 | 1 | inst_word_len |
| 0x4 | 8 | `*arg6` (opt) |
| 0xc | 1 | arg3 (count byte; asserts count < 9) |
| 0xd | 2 | 0x0009 (u16) |
| 0xf | 1 | 0 |
| 0x10 | 8 | byte-packed u64 of `arg5[]` (bytes at arg5+0, +4, +8, ... shifted 0,8,16,...) |
| 0x18 | 8 | byte-packed u64 of `arg4[]` (same stride-4 packing via SSE loop) |
| 0x20..0x3f | 32 | 0 |

## ALU_OP (0xa8)

### reg-reg 3-op family (form field 0xe = 0)

`add_add32` (0xc=0x804), `add_sub32` (0xc=0x805), `add_mul32` (0xc=0x806), `add_max32` (0xc=0x808), `add_min32` (0xc=0x809), `add_and` (0xc=0x90d). All: `(ctx, pos, arg3, arg4, arg5, arg6=ptr opt)`.

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0xa8 |
| 0x1 | 1 | inst_word_len |
| 0x4 | 8 | `*arg6` (opt) |
| 0xc | 2 | subop (u16, see per-function above) |
| 0xe | 1 | 0 |
| 0x10 | 1 | arg3 (byte) |
| 0x12 | 1 | arg4 (byte) |
| 0x14 | 1 | arg5 (byte) |

(arg3/arg4/arg5 = dst/src2/src1 registers by role guess only.)

### reg-imm 2-op family (form field 0xe = 1)

`add_add32_imm` (0xc=0x804), `add_sub32_imm` (0xc=0x805), `add_mul32_imm` (0xc=0x806), `add_iseq_imm` (0xc=0x812), `add_bit_or_imm` (0xc=0x90b). All: `(ctx, pos, arg3, arg4, arg5, arg6=ptr opt)`.

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0xa8 |
| 0x1 | 1 | inst_word_len |
| 0x4 | 8 | `*arg6` (opt) |
| 0xc | 2 | subop (u16) |
| 0xe | 1 | 1 |
| 0x10 | 1 | arg3 (byte) |
| 0x14 | 1 | arg4 (byte) |
| 0x18 | 4 | arg5 (u32 imm) |

### add_reg_add_u64_imm(ctx, pos, arg3=ptr opt, arg4, arg5, arg6 u64, arg7, arg8)

Instruction buffer at rsp+8.

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0xa8 |
| 0x1 | 1 | inst_word_len |
| 0x4 | 8 | `*arg3` (opt) |
| 0x14 | 4 | 0x00010104 (u32; bytes 04 01 01 00) |
| 0x18 | 1 | arg4 (byte) |
| 0x19 | 1 | arg5 (byte) |
| 0x1a | 2 | 0 |
| 0x1c | 1 | arg7 (byte, from stack+0x80) |
| 0x1d | 1 | arg8 (byte, from stack+0x88) |
| 0x1e | 2 | 0 |
| 0x20 | 8 | arg6 (u64 imm) |

Uncertain: 0x14 constant meaning (subop in a 64-bit ALU field layout; do not map to the u16 subops above).

### add_save_pc_to_regs(ctx, pos, arg3, arg4, arg5)

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0xa8 |
| 0x1 | 1 | inst_word_len |
| 0x4 | 8 | 0 (explicit) |
| 0xc | 8 | 0x00818000010104 (u64; bytes 04 01 01 00 80 81 00 00) |
| 0x14 | 1 | arg4 (byte) |
| 0x15 | 1 | arg5 (byte) |
| 0x16 | 2 | 0 |
| 0x18 | 8 | arg3 (u64) |

Uncertain: constant at 0xc; per name, saves PC into regs arg4/arg5.

## COMPARE_BRANCH (0xa9) / 0xb5

### add_br_label_id(ctx, pos, arg3=ptr opt, arg4)

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0xa9 |
| 0x1 | 1 | inst_word_len |
| 0x4 | 8 | `*arg3` (opt) |
| 0xc | 1 | 0 |
| 0xe | 1 | 3 |
| 0x30 | 8 | arg4 (u64) |

### add_br_abs(ctx, pos, arg3=ptr opt, arg4)

Same layout as br_label_id but **0xe = 0** (not 3).

### add_br_reg(ctx, pos, arg3, arg4, arg5=ptr opt)

| off | size | value |
|---|---|---|
| 0xc | 1 | 0 |
| 0xe | 1 | 1 |
| 0x22 | 1 | arg3 (byte) |
| 0x23 | 1 | arg4 (byte) |
| 0x4 | 8 | `*arg5` (opt) |

### add_br_rel(ctx, pos, arg3, arg4=ptr opt)

| off | size | value |
|---|---|---|
| 0x3 | 1 | 2 |
| 0xc | 1 | 0 |
| 0xe | 1 | 3 |
| 0x30 | 4 | arg3 (u32 rel) |
| 0x4 | 8 | `*arg4` (opt) |

Note: byte3 = 2 distinguishes this from br_label_id/br_abs (which leave byte3 = 0).

### add_cmp_reg_br_rel(ctx, pos, arg3, arg4, arg5, arg6, arg7, arg8=ptr opt)

| off | size | value |
|---|---|---|
| 0x3 | 1 | 2 |
| 0xc | 1 | arg3 (byte) |
| 0xd | 1 | arg4 (byte) |
| 0xe | 1 | 3 |
| 0x20 | 1 | arg5 (byte) |
| 0x21 | 1 | arg6 (byte) |
| 0x30 | 4 | arg7 (u32 rel, from stack+0x80) |
| 0x4 | 8 | `*arg8` (opt, stack+0x88) |

### add_cmpgt_imm_br_rel(ctx, pos, arg3, arg4, arg5, arg6=ptr opt)

| off | size | value |
|---|---|---|
| 0xc | 2 | 0x806 (u16) |
| 0xe | 1 | 3 |
| 0x10 | 4 | arg4 (u32 imm) |
| 0x20 | 1 | arg3 (byte) |
| 0x30 | 4 | arg5 (u32 rel) |
| 0x4 | 8 | `*arg6` (opt) |

### add_cmplt_imm_br_rel

Identical to cmpgt but **0xc = 0x801**.

### add_cmp_imm_uint8_br_rel(ctx, pos, arg3, arg4, arg5, arg6, arg7, arg8=ptr opt)

| off | size | value |
|---|---|---|
| 0x3 | 1 | 2 |
| 0xc | 1 | arg3 (byte) |
| 0xd | 1 | arg4 (byte) |
| 0xe | 1 | 3 |
| 0x10 | 1 | arg5 (byte imm8) |
| 0x20 | 1 | arg6 (byte) |
| 0x30 | 4 | arg7 (u32 rel, stack+0x80) |
| 0x4 | 8 | `*arg8` (opt, stack+0x88) |

### add_cmp_imm_br_rel(ctx, pos, arg3, arg4, arg5, arg6, arg7, arg8=ptr opt)

Same slots as cmp_imm_uint8 but **0x10 = arg6 as u32** (not byte), and no byte3=2. Layout:

| off | size | value |
|---|---|---|
| 0xc | 1 | arg3 (byte) |
| 0xd | 1 | arg4 (byte) |
| 0xe | 1 | 3 |
| 0x10 | 4 | arg6 (u32 imm; from r9d) |
| 0x20 | 1 | arg5 (byte) |
| 0x30 | 4 | arg7 (u32 rel) |
| 0x4 | 8 | `*arg8` (opt) |

### add_cmpne_imm_br_rel(ctx, pos, arg3, arg4, arg5, arg6=ptr opt)

| off | size | value |
|---|---|---|
| 0xc | 2 | 0x804 (u16) |
| 0xe | 1 | 3 |
| 0x10 | 4 | arg4 (u32 imm) |
| 0x20 | 1 | arg3 (byte) |
| 0x30 | 4 | arg5 (u32 rel) |
| 0x4 | 8 | `*arg6` (opt) |

### add_cmpeq_imm_br_reg_rel(ctx, pos, arg3, arg4, arg5, arg6=ptr opt)

| off | size | value |
|---|---|---|
| 0xc | 2 | 0x803 (u16) |
| 0xe | 1 | 4 |
| 0x10 | 4 | arg4 (u32 imm) |
| 0x20 | 1 | arg3 (byte) |
| 0x22 | 1 | arg5 (byte) |
| 0x4 | 8 | `*arg6` (opt) |

### add_cmpne_imm_br_reg_rel

Identical but **0xc = 0x804**.

### add_cmpeq_imm_br_rel(ctx, pos, arg3, arg4, arg5, arg6=ptr opt)

| off | size | value |
|---|---|---|
| 0xc | 2 | 0x803 (u16) |
| 0xe | 1 | 3 |
| 0x10 | 4 | arg4 (u32 imm) |
| 0x20 | 1 | arg3 (byte) |
| 0x30 | 4 | arg5 (u32 rel) |
| 0x4 | 8 | `*arg6` (opt) |

### add_br_hint_rel_imm(ctx, pos, arg3=ptr opt, arg4, arg5, arg6, arg7, arg8, arg9)

Opcode **0xb5** (not in the known enum excerpt).

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0xb5 |
| 0x1 | 1 | inst_word_len |
| 0x4 | 8 | `*arg3` (opt) |
| 0x14 | 1 | arg4 (byte) |
| 0x15 | 2 | 0x0003 (u16) |
| 0x17 | 1 | 0 |
| 0x18 | 4 | arg5 (u32; asserted to fit in int32 — "label id" by name) |
| 0x1c | 8 | 0 (explicit) |
| 0x24 | 1 | 0 |
| 0x25 | 1 | arg6 (byte) |
| 0x26 | 1 | arg7 (byte, stack+0x80) |
| 0x27 | 1 | arg8 (byte, stack+0x88) |
| 0x28 | 8 | arg9 (u64, stack+0x90) |

## EVENT_SEMAPHORE (0xa0)

Common: 0x20 = op byte, 0x22 = update_mode byte (SEM_* enum values 0x13/0x14/0x15/0x17/0x19 observed), 0x23 = semaphore id byte, 0x24 = u32 compare/mask value, 0x28 = u32 operand value.

| builder | args | 0x20 | 0x21 | 0x22 | 0x23 | 0x24 u32 | 0x28 u32 |
|---|---|---|---|---|---|---|---|
| add_evsem(ctx,pos,arg3=ptr opt,arg4,arg5=ptr opt@0x4) | | — | — | — | — | — | arg4 |
| add_semaphore_inc_val(ctx,pos,arg3,arg4) | | 0 | — | 0x15 | arg3 | — | arg4 |
| add_semaphore_inc(ctx,pos,arg3) | | 0 | — | 0x15 | arg3 | — | 1 |
| add_semaphore_dec(ctx,pos,arg3) | | 0 | — | 0x17 | arg3 | — | 1 |
| add_semaphore_wait(ctx,pos,arg3) | | 4 | arg3 | 0x14 | arg3 | 0 | 1 |
| add_semaphore_set(ctx,pos,arg3,arg4) | | 0 | — | 0x19 | arg3 | 0 | arg4 |
| add_semaphore_wait_eq(ctx,pos,arg3,arg4) | | 1 | arg3 | 0x00 | — | arg4 | — |
| add_semaphore_wait_eq_and_inc(ctx,pos,arg3,arg4) | | 1 | arg3 | 0x13 | arg3 | arg4 | 1 |
| add_semaphore_wait_ge_and_dec(ctx,pos,arg3,arg4) | | 5 | arg3 | 0x14 | arg3 | arg4 | 1 |
| add_semaphore_wait_eq_and_clear(ctx,pos,arg3,arg4) | | 1 | arg3 | 0x19 | arg3 | arg4 | 0 |
| add_semaphore_wait_ge(ctx,pos,arg3,arg4) | | 5 | arg3 | 0x00 | — | arg4 | — |

Notes:
- `add_evsem`: 0x20 = arg3 u64 (full 8 bytes), 0x28 = arg4 u32, 0x4 = `*arg5`. Not the semaphore-op field layout of the others.
- `add_semaphore_wait*` write the sem id to both 0x21 and (where present) 0x23.
- Update-mode mapping (per user-known enum): 0x15 = SEM_ADD_IMM_COMPLETE (21). Others (0x13, 0x14, 0x17, 0x19) unnamed here.

### add_sema_clear(ctx, pos, arg3, arg4, arg5=ptr opt)

Opcode **0xb0** (unknown enum name — not EVENT_SEMAPHORE).

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0xb0 |
| 0x1 | 1 | inst_word_len |
| 0x4 | 8 | `*arg5` (opt) |
| 0xc | 1 | 1 |
| 0xd | 1 | arg3 (byte) |
| 0xe | 1 | arg4 (byte) |

### add_poll_sem(ctx, pos, arg3, arg4, arg5, arg6, arg7=ptr opt)

Opcode **0xb3** (POLL_SEM=179 confirmed).

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0xb3 |
| 0x1 | 1 | inst_word_len |
| 0x4 | 8 | `*arg7` (opt, stack+0x80) |
| 0x14 | 1 | arg3 (byte) |
| 0x15 | 1 | arg5 (byte) |
| 0x16 | 2 | 0x0009 (u16) |
| 0x18 | 8 | arg4 (u64) |
| 0x20 | 8 | arg6 (u64) |

## NOP / NOTIFY (0xa4 / 0xa6 / 0x9f)

### add_nop(ctx, pos, arg3=ptr opt, arg4)

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0xa4 |
| 0xc | 4 | arg4 (u32) |
| 0x4 | 8 | `*arg3` (opt) |

### add_engine_nop(ctx, pos, arg3=ptr opt)

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0x9f |
| 0x4 | 8 | `*arg3` (opt) |

### add_debug_info(ctx, pos, arg3=char*, arg4=ptr opt)

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0xa4 |
| 0xc | 4 | 1 |
| 0x28 | 24 | strncpy(arg3, 24) |

### add_ev_set(ctx, pos, arg3) / add_ev_clr / add_ev_wait / add_ev_wait_and_set

All opcode 0xa4 with 0xc = 1 (u32).

| builder | 0x4 | 0x5 | 0x6 | 0x7 |
|---|---|---|---|---|
| add_ev_set(arg3) | 0 u16 | — | 0x11 | arg3 |
| add_ev_clr(arg3) | 0 u16 | — | 0x12 | arg3 |
| add_ev_wait(arg3) | 0x07 | arg3 | 0 u16 | — |
| add_ev_wait_and_set(arg3, arg4) | 0x07 | arg3 | 0x11 | arg4 |

### add_notify(ctx, pos, arg3, arg4)

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0xa6 |
| 0x1 | 1 | inst_word_len |
| 0x2 | 1 | 9 |
| 0xa | 1 | 9 |
| 0xb | 1 | `(arg3 << 3) & 0x78` |
| 0x2c | 2 | arg4 (u16) |
| 0x2e | 2 | 0x0100 (u16) |

### add_custom_notify(ctx, pos, arg3=ptr opt, arg4, arg5, arg6, arg7)

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0xa6 |
| 0x1 | 1 | inst_word_len |
| 0x2 | 1 | 9 |
| 0x4 | 8 | `*arg3` (opt) |
| 0x24 | 2 | arg4 (u16) |
| 0x26 | 2 | 0x0200 (u16) |
| 0x28 | 4 | arg5 (u32) |
| 0x2c | 2 | arg7 (u16, stack+0x80) |
| 0x2e | 1 | `arg6 ^ 1` (byte) |
| 0x2f | 1 | 1 |

### add_set_ordering_mode(ctx, pos, arg3)

Opcode **0xb1** (unknown enum name).

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0xb1 |
| 0xc | 1 | arg3 (byte) |

## HALT / DRAIN

### add_halt(ctx, pos, arg3=ptr opt)

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0xa1 |
| 0x4 | 8 | `*arg3` (opt) |

### add_halt_v2_with_notify(ctx, pos, arg3=ptr opt, arg4, arg5)

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0xa1 |
| 0x3 | 1 | `(arg5 << 3) & 0x78` |
| 0x4 | 8 | `*arg3` (opt) |
| 0x24 | 2 | arg4 (u16) |
| 0x26 | 2 | 0x0100 (u16) |

### add_drain(ctx, pos, arg3, arg4=ptr opt)

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0xa2 |
| 0x4 | 8 | `*arg4` (opt) |
| 0x28 | 1 | 1 (only if arg3 != 0) |

## WRITE (0xa5)

### add_wr32(ctx, pos, arg3, arg4, arg5=ptr opt) — identical body to add_wr32_v3

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0xa5 |
| 0x1 | 1 | inst_word_len |
| 0x4 | 8 | `*arg5` (opt) |
| 0xc | 1 | 4 (address-width byte) |
| 0x10 | 8 | arg3 (u64 address) |
| 0x20 | 4 | arg4 (u32 value) |

### add_wr32_2addr(ctx, pos, arg3, arg4, arg5, arg6, arg7=ptr opt)

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0xa5 |
| 0x4 | 8 | `*arg7` (opt, stack+0x80) |
| 0xc | 2 | 0x0404 (u16) |
| 0x10 | 8 | arg3 (u64 address #1) |
| 0x18 | 8 | arg5 (u64 address #2) |
| 0x20 | 4 | arg4 (u32 value #1) |
| 0x28 | 4 | arg6 (u32 value #2) |

### add_dma_config_base(ctx, pos, arg3=dma-ctx, arg4=eng_id, arg5=base u64, arg6, arg7=ptr opt)

Addresses computed via `get_dma_queue_base_low_offset(arg3, arg4, arg6)` / `get_dma_queue_base_high_offset(arg3, arg4, arg6)`.

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0xa5 |
| 0x4 | 8 | `*arg7` (opt, stack+0x90) |
| 0x1c | 2 | 0x0404 (u16) |
| 0x20 | 8 | low-offset address (u64) |
| 0x28 | 8 | high-offset address (u64) |
| 0x30 | 4 | arg5 & 0xffffffff (base lo u32) |
| 0x38 | 4 | arg5 >> 32 (base hi u32) |

### add_dma_config_size(ctx, pos, arg3=dma-ctx, arg4=eng_id, arg5=size u32, arg6, arg7=ptr opt)

Addresses via `get_dma_queue_size_offset(arg3, arg4, 1)` / `(arg3, arg4, 0)`.

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0xa5 |
| 0x4 | 8 | `*arg7` (opt) |
| 0x1c | 2 | 0x0404 |
| 0x20 | 8 | size-offset address (u64, variant 1) |
| 0x28 | 8 | size-offset address (u64, variant 0) |
| 0x30 | 4 | arg5 (u32) |
| 0x38 | 4 | arg5 (u32 — same value written twice) |

### add_dma_rearm(ctx, pos, arg3=dma-ctx, arg4=eng_id, arg5=ptr opt)

Addresses via `get_dma_queue_sw_ctrl_offset(arg3, arg4, 1)` / `(arg3, arg4, 0)`; value = `*(u32*)(tdrv_arch_ops + 0x140)` (arch rearm magic).

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0xa5 |
| 0x4 | 8 | `*arg5` (opt) |
| 0xc | 2 | 0x0404 |
| 0x10 | 8 | sw_ctrl offset (variant 0) |
| 0x18 | 8 | sw_ctrl offset (variant 1) |
| 0x20 | 4 | rearm magic u32 |
| 0x28 | 4 | rearm magic u32 |

### add_dma_tail_inc(ctx, pos, arg3=dma-ctx, arg4=eng_id, arg5=val u32, arg6=val u32, arg7=ptr opt)

Addresses via `get_dma_queue_tail_inc_offset(arg3, arg4, 1)` / `(arg3, arg4, 0)`. If either value is 0, emits a single-address variant:

- Both values nonzero: 0xc = 0x0404, 0x10 = tail_inc(0), 0x18 = tail_inc(1), 0x20 = arg6, 0x28 = arg5.
- arg6 == 0: 0xc = 4 (byte), 0x10 = tail_inc(1), 0x20 = arg5.
- arg5 == 0: 0xc = 4 (byte), 0x10 = tail_inc(0), 0x20 = arg6.

### add_dma_tail_inc_bcast / add_dma_rearm_bcast

Same field layouts, but loop over bcast rings: ring count from `tdrv_arch_get_bcast_ring_masks`, per-ring addresses from `get_dma_queue_tail_inc_bcast_offset` / `get_dma_queue_sw_ctrl_bcast_offset` (variant 1 and 0 per ring). Each ring gets its own 0xa5 WRITE instruction appended. Additional validation: cl < 0x10, values < 0x10000.

### add_dma_configs_priority_class_v3(dst_ins, src_ins)

Not a builder: copies `src[0xc]` → `dst[0xc]` if `dst[0xf]==0`, or → `dst[0x3d]` if `dst[0xf]==1`. Documents that the priority/class byte sits at different offsets in v2 vs v3 WRITE encodings.

## TENSOR_LOAD (0xaa) / TENSOR_STORE (0xab)

Register-indirect address form (ldr32_reg, ldr16_reg, str32_reg): address u64 = `0x8000000000000000 | (arg5 & 0xff) | ((arg6 & 0xff) << 8)` (bit63 set + two reg-id bytes).

Direct-address form (ldr/str_addr, str_imm_addr): address u64 taken from pointer arg.

### add_ldr32_addr(ctx, pos, arg3=count, arg4=u32 array, arg5=ptr→addr, arg6, arg7=ptr opt→0x2c)

Instruction buffer at rsp+0x28.

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0xaa |
| 0x1 | 1 | inst_word_len |
| 0x4 | 8 | `*arg7` (opt, stack+0xa0) |
| 0xc | 2 | 0x0009 (u16) |
| 0xe | 1 | arg3 (count byte) |
| 0xf | 1 | 1 |
| 0x10 | 8 | `*arg5` (u64 address) |
| 0x18 | 4 | arg6 (u32) |
| 0x1c | 4 | 0 |
| 0x20..0x37 | ≤24 | packed bytes from `arg4[]` (u32 array, low byte each element, count elements, SSE loop) |

### add_ldr16_addr

Identical to ldr32_addr but **0xc = 0x0005**.

### add_ldr32_reg(ctx, pos, arg3=count, arg4=array, arg5=addr reg byte, arg6=addr reg byte, arg7 u32 (stack+0xa0)→0x18, arg8=ptr opt (stack+0xa8)→0x4)

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0xaa |
| 0x4 | 8 | `*arg8` (opt) |
| 0xc | 2 | 0x0009 |
| 0xe | 1 | arg3 (count) |
| 0xf | 1 | 1 |
| 0x10 | 8 | reg-address u64 (see above; arg5 low byte, arg6 second byte) |
| 0x18 | 4 | arg7 (u32) |
| 0x1c | 4 | 0 |
| 0x20.. | ≤24 | packed bytes from `arg4[]` |

### add_ldr16_reg

Identical but **0xc = 0x0005**.

### add_str32_reg(ctx, pos, ...)

Same layout as ldr32_reg but opcode **0xab**.

### add_str32_addr(ctx, pos, arg3=count, arg4=array, arg5=ptr→addr, arg6, arg7=ptr opt)

Same layout as ldr32_addr but opcode **0xab**.

### add_str32_imm_reg(ctx, pos, arg3=count ≤8, arg4=imm array, arg5=reg byte, arg6=reg byte, arg7 u32 (stack+0xa0)→0x18, arg8=ptr opt (stack+0xa8)→0x4)

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0xab |
| 0x4 | 8 | `*arg8` (opt) |
| 0xc | 2 | 0x0109 (u16) |
| 0xe | 1 | arg3 (count) |
| 0xf | 1 | 1 |
| 0x10 | 8 | reg-address u64 (0x8000000000000000 | arg5 | arg6<<8) |
| 0x18 | 4 | arg7 (u32) |
| 0x1c | 4 | 0 |
| 0x20..0x27 | ≤8 | imm bytes (memcpy from arg4, arg3 bytes) |

### add_str32_imm_addr(ctx, pos, arg3=count ≤8, arg4=imm array, arg5=ptr→addr, arg6, arg7=ptr opt)

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0xab |
| 0x4 | 8 | `*arg7` (opt, stack+0xa0) |
| 0xc | 2 | 0x0109 |
| 0xe | 1 | arg3 (count) |
| 0xf | 1 | 1 |
| 0x10 | 8 | `*arg5` (u64 address) |
| 0x18 | 4 | arg6 (u32) |
| 0x20..0x27 | ≤8 | imm bytes from arg4 |

### add_str32_imm_addr_v3(ctx, pos, arg3=count, arg4=count2, arg5=imm array, arg6=ptr→addr, arg7 u32 (stack+0xa0)→0x18, arg8 u32 (stack+0xa8)→0x1c, arg9=ptr opt (stack+0xb0)→0x4)

Asserts `arg3 * arg4 <= 8`.

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0xab |
| 0x4 | 8 | `*arg9` (opt) |
| 0xc | 2 | 0x0109 |
| 0xe | 1 | arg3 (count) |
| 0xf | 1 | arg4 (count2 — second dim, replaces the constant `1`) |
| 0x10 | 8 | `*arg6` (u64 address) |
| 0x18 | 4 | arg7 (u32) |
| 0x1c | 4 | arg8 (u32) |
| 0x20.. | ≤8 | imm bytes (arg3*arg4 bytes from arg5) |

### add_ldr_addr(ctx, pos, arg3=count, arg4=array, arg5=ptr→addr, arg6, arg7 u32 (stack+0xa0)→0xc, arg8=ptr opt (stack+0xa8)→0x4)

Instruction buffer at rsp (0x0-based).

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0xaa |
| 0x4 | 8 | `*arg8` (opt) |
| 0xc | 1 | arg7 (byte) |
| 0xd | 1 | 0 |
| 0xe | 1 | arg3 (count) |
| 0xf | 1 | 1 |
| 0x10 | 8 | `*arg5` (u64 address) |
| 0x18 | 4 | arg6 (u32) |
| 0x1c | 4 | 0 |
| 0x20.. | ≤24 | packed bytes from arg4 |

Note: different field placement than ldr32_addr (count at 0xe vs 0xe but u16 0x9 field absent; byte at 0xc).

### add_args_table_swap(ctx, pos, arg3)

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0xaa |
| 0x1 | 1 | inst_word_len |
| 0x4 | 8 | 0 (explicit) |
| 0xc | 4 | 0x01020009 (u32; bytes 09 00 02 01) |
| 0x10 | 8 | arg3 (u64) |
| 0x18 | 8 | 1 (u64) |
| 0x20 | 2 | 0x8382 (u16) |

## REG_LOAD (0x74) / misc tensor ops

### add_tensor_scalar_addr(ctx, pos, arg3, arg4, arg5=ptr→0x34, arg6, arg7 u64→0x18, arg8 u64→0x20, arg9..arg15 bytes→0x28..0x2e, arg16=ptr opt→0x4)

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0x74 |
| 0x4 | 8 | `*arg16` (opt, stack+0xc8) |
| 0xc | 8 | arg3 (u64) |
| 0x14 | 4 | arg4 (u32) |
| 0x18 | 8 | arg7 (u64, stack+0x80) |
| 0x20 | 8 | arg8 (u64, stack+0x88) |
| 0x28..0x2e | 7×1 | arg9..arg15 (bytes, stack+0x90..0xc0 step 8) |
| 0x34 | 8 | `*arg5` (u64) |
| 0x3c | 4 | arg6 (u32) |

## REDUCE family (0x7c / 0x7d)

### add_1d_max_all_reduce(ctx, pos, arg3=addr u64 packed, arg4, arg5, arg6=ptr opt)

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0x7c |
| 0x4 | 8 | `*arg6` (opt) |
| 0xc | 4 | arg3 & 0xffffffff |
| 0x10 | 2 | (arg3 >> 32) & 0xffff |
| 0x12 | 4 | 0 |
| 0x16 | 2 | 0 |
| 0x18 | 2 | arg3 >> 48 |
| 0x1a | 8 | 0x909000100010001 (u64) |
| 0x22 | 1 | arg5 (byte) |
| 0x23 | 2 | 0x0002 (u16) |
| 0x2c | 4 | arg4 (u32) |
| 0x30 | 8 | 0 (u64) |
| 0x38 | 8 | 0x01000100010001 (u64) |

Address arg3 is split: lo32 @0xc, mid16 @0x10, hi16 @0x18.

### add_reduce(ctx, pos, arg3=opcode byte, arg4, arg5, arg6, arg7, arg8, arg9=desc 20B (stack+0x80), arg10=desc 20B (stack+0x98), arg11=ptr opt)

Opcode byte is an argument (0x7c when called for max-reduce; others possible).

| off | size | value |
|---|---|---|
| 0x0 | 1 | arg3 (byte opcode) |
| 0x4 | 8 | `*arg11` (opt, stack+0xc0) |
| 0xc | 16 | first 16B of arg9 struct |
| 0x1c | 4 | u32 at arg9+0x10 (overlaps tail of the 16B copy) |
| 0x20 | 1 | arg4 (byte) |
| 0x21 | 1 | arg5 (byte) |
| 0x22 | 1 | arg6 (byte) |
| 0x23 | 1 | arg7 (byte, stack+0xb0) |
| 0x24 | 1 | arg8 (byte, stack+0xb8) |
| 0x2c | 16 | first 16B of arg10 struct |
| 0x3c | 4 | u32 at arg10+0x10 |

### add_2d_and_all_reduce(ctx, pos, arg3=ptr→0xc, arg4, arg5, arg6=ptr opt, arg7, arg8, arg9)

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0x7d |
| 0x4 | 8 | `*arg6` (opt) |
| 0xc | 8 | `*arg3` (u64) |
| 0x14 | 4 | 0 |
| 0x18 | 4 | arg4 (u32) |
| 0x1c | 4 | 0x00010001 (u32) |
| 0x20 | 1 | arg7 (byte, stack+0x80) |
| 0x21 | 1 | arg8 (byte, stack+0x88) |
| 0x22 | 1 | arg9 (byte, stack+0x90) |
| 0x23 | 2 | 0x0004 (u16) |
| 0x2c | 4 | arg5 (u32) |
| 0x30 | 8 | 0 (u64) |
| 0x38 | 8 | 0x01000100010001 (u64) |

## MEMSET (0x49)

### add_memset_1d(ctx, pos, arg3=addr u64, arg4, arg5, arg6=ptr opt)

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0x49 |
| 0x2 | 16 | 0 (explicit) |
| 0x1c | 4 | (arg3 >> 32) & 0xffff |
| 0x20 | 2 | 0x0009 (u16) |
| 0x22 | 1 | arg5 (byte) |
| 0x23 | 4 | 0 |
| 0x27 | 1 | 0 |
| 0x28 | 4 | arg4 (u32) |
| 0x2c | 4 | arg3 & 0xffffffff |
| 0x30 | 2 | (arg3 >> 32) & 0xffff |
| 0x32 | 4 | 0 |
| 0x36 | 2 | 0 |
| 0x38 | 2 | (arg3 >> 32) & 0xffff |
| 0x3a | 4 | 0x00010001 (u32) |
| 0x3e | 2 | 0x0001 (u16) |
| 0x4 | 8 | `*arg6` (opt) |

Uncertain: hi16 of the address repeated at 0x1c, 0x30, 0x38; field roles not labeled by disassembly.

## EMBEDDING_UPDATE (0x79)

### add_embedding_update(ctx, pos, arg3=struct ptr, arg4, arg5=ptr opt)

Fields copied from `arg3` struct (source offset → instruction offset):

| src off | size | dst off |
|---|---|---|
| +0x0c | 1 | 0x0c |
| +0x10 | 8 | 0x10 |
| +0x18 | 8 | 0x20 |
| +0x20 | 4 | 0x28 |
| +0x24 | 8 | 0x2c |
| +0x30 | 4 | 0x18 |
| +0x34 | 2 | 0x1c |
| +0x3c | 2 | 0x38 |

Fixed fields: 0xd = 0x0309 (u16), 0xf = 0x80 (byte), 0x1e = 1 (byte), 0x34 = arg4 (u32), 0x4 = `*arg5` (opt), opcode 0x79.

## LOAD_POOL_ARGUMENTS (0x7a — opcode name unknown in enum excerpt)

### add_load_pool_arguments(ctx, pos, arg3, arg4, arg5, arg6, arg7=ptr opt)

| off | size | value |
|---|---|---|
| 0x0 | 1 | 0x7a |
| 0x4 | 8 | `*arg7` (opt, stack+0x80) |
| 0xe | 1 | 3 |
| 0x20 | 4 | arg3 (u32) |
| 0x24 | 4 | arg3 + 4 (u32) |
| 0x28 | 8 | arg4 (u64) |
| 0x30 | 8 | arg5 (u64) |
| 0x38 | 8 | arg6 (u64) |

## Composite builders (no single fixed encoding)

- **add_dge_bound_check_move_ins** — bounds-check wrapper; emits 1–2 `add_mov32_imm` instructions (second index = first+1; instruction index derived as `vaddr << 6` when a flag bit (0x4) is set in an input struct byte at +0x7). Validates DGE ranges against table entries.
- **add_sync_barrier(ctx, pos, ..., barrier_mask, ...)** — sequence: `add_drain`, then semaphore phase (`tdrv_sync_get_sync_barrier` → barrier id `b`, popcount of masked engine bits `n`):
  - full mask (0x1f): `add_semaphore_inc(b)` + `add_semaphore_wait_eq_and_clear(b, 8-n)`
  - else: `add_semaphore_wait_eq_and_inc(b, n)` (twice when engine count == 5)
  then trailing `add_drain`.
- **add_sema_reset** — loop of `add_sema_clear` over reserved-semaphore id range (bounds from `tdrv_sync_get_num_reserved_semaphores` and TPB engine count).
- **add_wait_before_cc_command** — per-topsp loop: `get_ack_evt_id_for_topsp_engine` → `tdrv_add_ev_wait(evt_id)`. No direct encoding in this function.
- **add_dma_bundle_rearm** — scans DMA bundle descriptor arrays (strided by 0x170 bytes per queue), selects queues by tag/size fields, masks engine bits, then calls `add_dma_rearm_bcast`.
- **add_mem2d_from_pseudo_mem2d** — resolves pseudo mem2d via hash tables (`ht_find`), computes address via `mem_ref_to_addr` or base+stride arithmetic, then emits one instruction: opcode 0xab (TENSOR_STORE) if pseudo opcode 0xcd, 0xaa (TENSOR_LOAD) if 0xce; remaining fields copied from the resolved pseudo descriptor (base at pseudo-0xd0 relative layout), appended at index 0.

## Misc non-builders in the dump

- `add_sconv_object`, `add_converter`, `add_model` — linked-list/object constructors, no instruction encoding.
- `add_dma_packet`, `add_dma_packet_cce` — large DMA descriptor builders (JSON-descriptor domain), not 64B SP instructions.
- `instr_set_halt` / `instr_remove_halt` — set/clear bit7 of instruction byte 2 (halt flag lives in byte2 bit7).
- Helper at dcc8d0: packs `(b0 | b1<<8 | b2<<16 | b3<<24 | r8<<32)` — 5-byte field pack helper (byte2 here = 16-bit field lo/hi split visible in several builders).

## Explicit uncertainties

1. Names for opcodes 0xb0, 0xb1, 0xb5, 0x7a, 0x7d not present in the provided enum excerpt — flag for DWARF cross-check via neuron-dbg.
2. `add_mov32_imm` arg3/arg4 roles unknown.
3. ALU 64-bit form constant 0x00010104 (and save_pc variant 0x00818000010104) not decomposed.
4. EVENT_SEMAPHORE update-mode values 0x13/0x14/0x17/0x19 unnamed here (0x15 = SEM_ADD_IMM_COMPLETE = 21 confirmed).
5. `add_ldr_addr` field layout (byte @0xc, no 0x9 u16) differs from the other ldr/str builders — verified but role of the 0xc byte unknown.
6. Repeated hi16 writes in `add_memset_1d` (0x1c/0x30/0x38) — purpose unknown.
7. `add_ins` "ctx" first argument layout inferred: ctx[0] = base pointer, ctx[8] = capacity end; `inst_word_len` helper called with the same first argument (the stream/ctx handle).
