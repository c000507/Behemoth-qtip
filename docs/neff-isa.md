# Neuron ISA (TPB) - first verified facts

Source: neuron-dbg/libndbg DWARF (aws-neuronx-tools 2.33.38),
validated against ffn1_full.neff + flash_mh_m128_d64_h64.neff (trn1, neuronx-cc 2.26).

## Instruction format (VERIFIED)

- Instruction = 32 bytes. Engines fetch 64B lines (INST_NBYTES=64, INST_NWORDS=16).
- Header struct (NEURON_ISA_TPB_HEADER, 4B): `u8 opcode; u8 inst_word_len; u8 debug_cmd; u8 debug_hint`.
- `inst_word_len` normally 16 (=64B line). Reused as payload field on v2-only opcodes (op 10: 1/2/3/10; op 170: 10; op 128: 0) - opcode-specific meaning.
- Rest of 32B = opcode-specific payload (S#D# operand structs; layouts pending).

## HW geometry constants (VERIFIED from DWARF consts)

| const | value |
|---|---|
| INST_NBYTES | 64 |
| INST_NWORDS | 16 |
| PE_ARRAY_NUM_ROWS | 128 |
| PE_ARRAY_NUM_COLS | 128 |
| NUM_REGISTERS | 64 |
| NUM_ADDR_REGISTERS | 64 |
| NUM_SHAPE_REGISTERS | 4 |
| NUM_DGE_SHAPE_REGISTERS | 4 |
| POOLING_NUM_CHANNELS | 128 |
| DVE_NUM_CHANNELS | 128 |
| ACTIVATION_NUM_CHANNELS | 128 |
| ACTIVATION_NUM_TABLES | 8 |
| STATE_BUF_NUM_PARTITIONS | 128 |
| STATE_BUF_WORD_SIZE | 8 |
| PSUM_BUF_NUM_PARTITIONS | 128 |
| STATE_BUF_PARTITION_SIZE | 0 |
| STATE_BUF_ADDR_SPACE_CLUSTER_SIZE | 0 |
| STATE_BUF_ADDR_SPACE_QUADRANT_SIZE | 0 |
| STATE_BUF_PARTITION_ACTIVE_SIZE | 0 |
| PSUM_BUF_PARTITION_SIZE | 32768 |
| PSUM_BUF_PARTITION_ACTIVE_SIZE | 16384 |
| PSUM_BUF_NUM_BANKS | 16 |
| PSUM_BUF_BANK_SIZE | 1024 |
| TPB_PARTITION_ADDR_MASK | 0 |
| MEM_PATTERN_MAX_NUM_DIM | 4 |
| NUM_EVENTS | 256 |
| EVENT_SIZE | 4 |
| NUM_SEMAPHORES | 256 |
| SEMAPHORE_SIZE | 4 |
| SEMAPHORE_MAX_VAL | -1 |
| PE_ARRAY_NUM_ROW_GROUPS | 4 |
| PE_ARRAY_NUM_COL_GROUPS | 4 |
| REG_SHUFFLE_MAX_NUM_REGS | 32 |
| POOL_BUFFER_MAX_ELEMENTS | 512 |
| NOTIFICATION_NUM_QUEUES | 10 |
| DEBUG_CMD_CONSTS_TRACE_MODE_SHIFT | 0 |
| DEBUG_CMD_CONSTS_TRACE_MODE_SIZE | 2 |
| DEBUG_CMD_CONSTS_USE_INST_QUEUE_IDX_SHIFT | 2 |
| DEBUG_CMD_CONSTS_USE_INST_QUEUE_IDX_SIZE | 1 |
| DEBUG_CMD_CONSTS_QUEUE_IDX_SHIFT | 3 |
| DEBUG_CMD_CONSTS_QUEUE_IDX_SIZE | 4 |
| DEBUG_CMD_CONSTS_HALT_EN_SHIFT | 7 |
| DEBUG_CMD_CONSTS_HALT_EN_SIZE | 1 |
| STATE_BUF_OFFSET | 0 |
| PSUM_BUF_OFFSET | 0 |
| STATE_BUF_SZ | 0 |
| PSUM_BUF_SZ | 0 |
| ENGINE_QUEUE_DEPTH_PE | 64 |
| ENGINE_QUEUE_DEPTH_ACT | 8 |
| ENGINE_QUEUE_DEPTH_DVE | 8 |
| ENGINE_QUEUE_DEPTH_POOL | 8 |
| PE_ENGINE_FREQ_MHZ | 2800 |
| ACT_ENGINE_FREQ_MHZ | 1400 |
| DVE_ENGINE_FREQ_MHZ | 1120 |
| POOL_ENGINE_FREQ_MHZ | 1400 |
| DMA_ENGINE_FREQ_MHZ | 850 |
| DMA_ENGINE_DATA_WIDTH_BYTES | 32 |
| DMA_ENGINE_COUNT_PER_TPB | 16 |
| DMA_QUEUE_COUNT_PER_ENGINE | 16 |
| DMA_CCE_MAX_DTYPE_ELEM_COUNT | 1024 |
| DMA_CCE_MAX_BIT_ACCURATE_ELEM_SIZE | 4096 |
| DMA_MAX_CCE_SOURCE | 16 |
| DMA_MAX_CCE_ELEMENT | 1024 |
| DMA_MAX_TRANSFER_SIZE_PER_DESC_KIB | 64 |
| DMA_MAX_DESC_COUNT_PER_PACKET | 64 |
| HBM_STACK_CAPACITY_GIB | 16 |
| RAND_PHILOX_NUM_STATES | 7 |
| RAND_XORWOW_SINGLE_NUM_STATES | 6 |
| RAND_XORWOW_NUM_STATES | 24 |
| ADDR4_MARKER_SHAPE_REG | 64 |
| ADDR4_MARKER_ADDR_SHAPE_REG | 192 |
| ADDR4_MARKER_IMM | 0 |
| ADDR4_MARKER_ADDR_REG | 128 |
| ADDR4_MARKER_INDIRECT_IMM | 32 |
| ADDR4_MARKER_INDIRECT_REG | 160 |
| ADDR8_MARKER_OFFSET_WIDE_BIT | 4 |
| ADDR8_MARKER_OFFSET_REG_BIT | 8 |
| ADDR8_MARKER_ADDR_TBL_BIT | 32 |
| ADDR8_MARKER_SHAPE_REG_BIT | 64 |
| ADDR8_MARKER_ADDR_REG_BIT | 128 |
| ADDR8_MARKER_IMM | 0 |
| ADDR8_MARKER_SHAPE_REG | 64 |
| ADDR8_MARKER_ADDR_REG | 128 |
| ADDR8_MARKER_ADDR_SHAPE_REG | 192 |
| ADDR8_MARKER_ADDR_TBL_SHAPE_IMM | 32 |
| ADDR8_MARKER_ADDR_TBL_SHAPE_REG | 96 |
| ADDR8_MARKER_ADDR_TBL_OFFSET_REG | 8 |
| ADDR8_MARKER_ADDR_TBL_OFFSET_WIDE | 4 |
| ADDR8_MARKER_MASK | 252 |
| PSEUDO_ADDR8_MARKER_IMM | 0 |
| PSEUDO_ADDR8_MARKER_OFFSET_WIDE_BIT | 4 |
| PSEUDO_ADDR8_MARKER_OFFSET_REG_BIT | 8 |
| PSEUDO_ADDR8_MARKER_VAR_ID_BIT | 16 |
| PSEUDO_ADDR8_MARKER_ADDR_TBL_BIT | 32 |
| PSEUDO_ADDR8_MARKER_SHAPE_REG_BIT | 64 |
| PSEUDO_ADDR8_MARKER_ADDR_REG_BIT | 128 |
| PSEUDO_ADDR8_MARKER_MASK | 252 |
| MATMUL_PSUM_ACCUMULATE_MODE_MULTI_START | 1 |
| MATMUL_PSUM_ACCUMULATE_MODE_MULTI_MID | 0 |
| MATMUL_PSUM_ACCUMULATE_MODE_MULTI_END | 2 |
| MATMUL_PSUM_ACCUMULATE_MODE_SINGLE | 3 |
| RAND_RNG_LFSR_DVE_NUM_STATES | 4 |
| RAND_CVT_LFSR_DVE_NUM_STATES | 4 |
| RAND_CVT_LFSR_POOL_NUM_STATES | 1 |
| RAND_CVT_LFSR_ACT_NUM_STATES | 2 |

## Opcodes (NEURON_ISA_TPB_OPCODE enum, common ISA)

| val | name |
|---|---|

## Observed opcode usage in compiled trn1 graphs (32B slots)

flash_mh (NKI attention) + ffn1_full (XLA FFN):

- PE: LDWEIGHTS(1)+MATMUL(2) pairs, op10 x2 per pair (v2-only), EVENT_SEMAPHORE(160), prologue BRANCH_PREFETCH_HINT(177), epilogue PSEUDO_TENSOR_STORE(204).
- DVE: op10 (len1=10) + STREAM_TRANSPOSE(107) bulk, TENSOR_TENSOR_ARITH(65)/SCALAR_ARITH(67)/REDUCE_ARITH(66), COPY_PREDICATED(114), TENSOR_SCALAR_SELECT(152, len1=72!), op8 (v2).
- Pool: op212 (v2, len1=16) + op128 (len1=0) bulk, EVENT_SEMAPHORE, prologue/epilogue, 126.
- Activation: ACTIVATE(33) len1=16 + TENSOR_LOAD(170) len1=10 alternating, ACTIVATION_TABLE_LOAD(35).
- SP: 177 / 204 / PSEUDO_DMATRIGGER(193).

v2-only opcodes not in common enum: 8, 10, 128, 212 (plus 0 with len1=8/10 padding).

## DTYPE enum (verified)

BFLOAT16=6 FP16=7 FP32=10 FP32R=11 FP8_EXP3=13 FP8_EXP4=14 FP8_EXP5=15 INT8=2 INT16=4 INT32=8 INT64=12 UINT8=3 UINT16=5 UINT32=9 UINT64=1.

## Semaphore/event contract (VERIFIED enums; scheduling semantics TODO)

WAIT_MODE: NONE=0 SEM_EQ_IMM=1 SEM_LT_IMM=2 SEM_LE_IMM=3 SEM_GT_IMM=4 SEM_GE_IMM=5 EVT_SET=6 EVT_SET_THEN_CLEAR=7; REG variants = value+128; INVALID=255; EVT_CLEAR=14 EVT_CLEAR_THEN_SET=15.
UPDATE_MODE: NONE=0 EVT_SET_READ=1 EVT_CLR_READ=2 SEM_INC_READ=3 SEM_DEC_READ=4 SEM_ADD_IMM_READ=5 SEM_SUB_IMM_READ=7 SEM_INC_COMPLETE=19 SEM_DEC_COMPLETE=20 EVT_SET_COMPLETE=17 EVT_CLR_COMPLETE=18 SEM_ADD_IMM_COMPLETE=21 SEM_SUB_IMM_COMPLETE=23 SEM_WR_IMM_READ=9 SEM_WR_IMM_COMPLETE=25; REG variants: ADD 133/149 SUB 135/151 WR 137/153.
NOTIFIC_KIND: NONE=0 EXPLICIT=1 CUSTOM=2.

## DVE ext-ISA tables (in-tar for xla graphs; runtime defaults otherwise)

- default_opcode_table.bin: 2048B = 32 x 64B entries; entry0 u32=0x180.
- default_control_table.bin: 2048B = 32 x 64B; nonzero from entry0 (`5555 0000 0000 1000`).
- default_datapath_table.bin: 8192B = 128 x 64B; nonzero from 0x40 (`0000 1000 ...`).
- act bkt/ctrl bins per function-set (silu etc.) - LUT microcode for activation engine.
- These configure DVE/act field-decode HW. libnrt inits via aws_hal_stpb_sunda_{dve,act,pe,pooling}_hw_decode_table_init.

## Where the remaining payload layouts live

1. neuron-dbg cgo accessors name every field of every struct (get_/set_ per field) - 432 struct families, v2/v3/v4/common + tonga. Member OFFSETS still needed.
2. libnrt.so (unstripped, 48.7k local syms): C++ code comparing `instr->header.opcode == NEURON_ISA_TPB_OPCODE_*` = real decoder; encd_* = NEFF builder (410 fns).
3. libnrtucode_extisa.so exports assembler (nrtucode_ll_builder_*) + nrtucode_get_hwdecode_table - dlopen hangs w/o hw; run on trn1 i-0d79198d32a0df40f.
4. Embedded default_*_table.bin microcode = the field-extraction config itself.
