// neuron_asm: header-only C++20 ISA assembler for AWS Trainium NEFF graphs.
//
// Target: let an LLM drive the Neuron engines (PE / DVE / ACT / Pool / SP)
// at instruction level with typed, checked accessors and loud failures.
//
// Layout per engine stream = sequence of 32B instructions (engines fetch
// 64B = 2 instrs). Every instruction = [opcode u8][word_len u8=16]
// [debug_cmd u8][debug_hint u8][fields...].
//
// Field offsets follow tools/neff-isa-v2-fields.md / neff-isa-fields.json
// (DWARF-extracted). Fields NOT in the field map (empirical passthroughs
// reverse-engineered from compiler oracles) are written via raw_u16/raw_u32
// and named kEmp* constants - keep them central so future decodes can
// replace them one by one.
#pragma once

#include <array>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace nasm {

// ---------------------------------------------------------------- instr --

constexpr size_t kInstrBytes = 32;
constexpr size_t kDescBytes = 64;   // op212 DMA descs = full 64B instr

struct Instr {
    std::array<uint8_t, kDescBytes> b{};

    Instr() = default;
    explicit Instr(uint8_t opcode, uint8_t word_len = 16) {
        b[0] = opcode;
        b[1] = word_len;
    }

    uint8_t opcode() const { return b[0]; }
    // op212 DMA descs are full 64B instructions (word_len 16 = 16 u32 words)
    size_t used() const { return b[0] == 212 ? kDescBytes : kInstrBytes; }

    void raw_u8(size_t off, uint8_t v) { b[off] = v; }
    void raw_u16(size_t off, uint16_t v) {
        b[off] = uint8_t(v & 0xFF);
        b[off + 1] = uint8_t(v >> 8);
    }
    void raw_u32(size_t off, uint32_t v) {
        for (int i = 0; i < 4; ++i) b[off + i] = uint8_t(v >> (8 * i));
    }
    void raw_f32(size_t off, float f) {
        uint32_t v;
        std::memcpy(&v, &f, 4);
        raw_u32(off, v);
    }
    uint16_t get_u16(size_t off) const {
        return uint16_t(b[off]) | uint16_t(b[off + 1]) << 8;
    }
    uint32_t get_u32(size_t off) const {
        return uint32_t(b[off]) | uint32_t(b[off + 1]) << 8 |
               uint32_t(b[off + 2]) << 16 | uint32_t(b[off + 3]) << 24;
    }
};

// ------------------------------------------------------------- opcodes --

// PE engine
constexpr uint8_t OP_PROLOGUE_177 = 177;  // engine preamble (undecoded body)
constexpr uint8_t OP_PAD = 0;             // zero instr / padding
constexpr uint8_t OP_LDWEIGHTS = 1;       // v2.S3_LW
constexpr uint8_t OP_MATMUL = 2;          // v2.S3D3_MM
// DVE engine
constexpr uint8_t OP_DVE = 10;            // generic DVE op (op10 companion)
// ACT engine
constexpr uint8_t OP_ACTIVATE = 33;
constexpr uint8_t OP_ACT_TABLE_LOAD = 35;
constexpr uint8_t OP_LOAD_POOL_ARG = 122;
constexpr uint8_t OP_TENSOR_LOAD = 170;
// Pool engine
constexpr uint8_t OP_POOL_TAIL_EV = 164;
constexpr uint8_t OP_PSEUDO_DMA_DIRECT2D = 212;
// Scalar engine preamble
constexpr uint8_t OP_SP_PROLOGUE_177 = 177;
constexpr uint8_t OP_SP_CTRL_204 = 204;
constexpr uint8_t OP_SP_C1 = 193;         // SP name-carrying instr
// CTRL (event / semaphore notify)
constexpr uint8_t OP_CTRL_EV = 160;

// -------------------------------------------------------- empirical fields
// Offsets reverse-engineered from compiler oracle streams (not in the
// DWARF field map). Grouped here; replace as they get decoded.

namespace emp {
// op10 (DVE generic) companion instruction
constexpr size_t kOp10_u6const = 6;      // u16: small const (shape-dependent)
constexpr size_t kOp10_unk12 = 12;       // u16: 3855 in matmul streams
constexpr size_t kOp10_kind = 14;        // u16: 2 = LW-companion, 16 = MM-companion
constexpr size_t kOp10_ring = 11;        // u8: producer-consumer ring select
constexpr size_t kOp10_dst = 16;         // u32: PSUM dst (raw incl 0x02000000)
constexpr size_t kOp10_step = 20;        // u16
constexpr size_t kOp10_dn = 26;          // u16
constexpr size_t kOp10_tick28 = 28;      // u16: 1
constexpr size_t kOp10_tick30 = 30;      // u16: 1
// LW (S3_LW) engine instr
constexpr size_t kLw_src = 16;           // u32: SB src addr (end-address - 4)
constexpr size_t kLw_step20 = 20;        // u16: 0xFFFF
constexpr size_t kLw_num = 26;           // u16: rows
constexpr size_t kLw_flag28 = 28;        // u8: 1
constexpr size_t kLw_flag30 = 30;        // u8: 1
// MM (S3D3_MM) engine instr
constexpr size_t kMm_a = 16;             // u32: A-pattern src (SB/PSUM)
constexpr size_t kMm_step20 = 20;        // u16: 1
constexpr size_t kMm_anum = 26;          // u16: A rows
constexpr size_t kMm_flag28 = 28;        // u8: 1
constexpr size_t kMm_flag30 = 30;        // u8: 1
// events block (shared, @4..@11)
constexpr size_t kEv_wait_mode = 4;      // u8: 5 = wait
constexpr size_t kEv_wait_idx = 5;       // u8: semaphore index
constexpr size_t kEv_upd_mode = 6;       // u8 (MM upd: 19 -> b7=5)
constexpr size_t kEv_upd_idx = 7;        // u8
constexpr size_t kEv_sem_value = 8;      // u16 (LW/MV wait value / upd 19)
// op160 EV instr: u16@8 = value
constexpr size_t kOp160_value = 8;
// op212 DMA desc: sem << 8 at u16@12
constexpr size_t kP212_sem12 = 12;
constexpr size_t kP212_kind14 = 14;      // u16: 16
}  // namespace emp

// ------------------------------------------------------------- streams --

struct Stream {
    std::vector<uint8_t> bytes;
    void add(const Instr& i) {
        bytes.insert(bytes.end(), i.b.begin(), i.b.begin() + i.used());
    }
    void add_raw(const void* p, size_t n) {
        const uint8_t* q = static_cast<const uint8_t*>(p);
        bytes.insert(bytes.end(), q, q + n);
    }
    void pad32() { bytes.insert(bytes.end(), kInstrBytes, 0); }
    size_t instrs() const { return bytes.size() / kInstrBytes; }
    const uint8_t* data() const { return bytes.data(); }
    size_t size() const { return bytes.size(); }
};

// ------------------------------------------------------------------ SB --

// State-buffer geometry (SB = 256KB; addresses are byte offsets).
// Layout model (from the 27-bucket oracle study, mm_synth.py):
//   top        = 0x30000 (identity region lives just below: to_off = top - rows)
//   x ring     = [x_bottom, tof)  - streamed x chunks, 512B per K-chunk
//   w region   = fresh: fixed base W0 = 0x18000; old: packed under x ring
//   window     = 0x4000.. - SB<->PSUM staging window
constexpr uint32_t SB_TOP = 0x30000;
constexpr uint32_t SB_IDENTITY_MAX_ROWS = 512;
constexpr uint32_t W0_FRESH = 0x18000;   // fresh-compile w region base
constexpr uint32_t WINDOW_BASE = 0x4000;

// --------------------------------------------------------- allocators --

// Semaphore sets (def.json dma_queue semaphore_set contract).
struct SemRange {
    uint8_t first, last;
    bool contains(uint8_t s) const { return s >= first && s <= last; }
};
struct SemSets {
    SemRange pool_dynamic{8, 23};       // qPoolDynamic
    SemRange sp_spill_reload{24, 39};   // qSPSpillReload0
    SemRange io{40, 55};                // qSPIO0
    SemRange pool_io{56, 71};           // qPoolIO0
};

// SB allocator: hands out disjoint regions; throws on overlap.
class SbAlloc {
public:
    struct Region {
        uint32_t base, size;
        std::string name;
    };
    explicit SbAlloc(uint32_t top = SB_TOP) : top_(top) {}

    // Allocate [base, base+size) claiming `size` bytes below `limit`.
    Region take(uint32_t size, const char* name, uint32_t limit = SB_TOP) {
        // first-fit from 0 upward, honoring existing regions
        uint32_t lo = 0;
        for (;;) {
            uint32_t hi = limit;
            for (const auto& r : used_) {
                if (r.base < hi && r.base + r.size > lo) {
                    if (r.base > lo) { hi = r.base; break; }
                    lo = r.base + r.size;
                }
            }
            if (lo + size <= hi && lo + size <= limit) {
                used_.push_back({lo, size, name});
                return {lo, size, name};
            }
            if (lo >= limit) break;
            // advance lo past the blocking region
            bool moved = false;
            for (const auto& r : used_) {
                if (r.base < hi && r.base + r.size > lo && r.base <= lo) {
                    uint32_t nl = r.base + r.size;
                    if (nl > lo) { lo = nl; moved = true; }
                }
            }
            if (!moved) break;
        }
        fail(std::string("SB allocator: no room for ") + name +
             " (" + std::to_string(size) + "B)");
        return {};
    }
    void claim(uint32_t base, uint32_t size, const char* name) {
        for (const auto& r : used_) {
            if (base < r.base + r.size && r.base < base + size)
                fail(std::string("SB allocator: overlap claiming ") + name +
                     " vs " + r.name);
        }
        used_.push_back({base, size, name});
    }
    const std::vector<Region>& regions() const { return used_; }

private:
    [[noreturn]] static void fail(const std::string& m) {
        throw std::runtime_error(m);
    }
    uint32_t top_;
    std::vector<Region> used_;
};

// PSUM allocator: bank blocks of 0x800 within the 0x10000 PSUM space.
class PsumAlloc {
public:
    static constexpr uint32_t kPsumBase = 0x02000000;
    struct Region {
        uint32_t base, size;
        std::string name;
    };
    Region take(uint32_t nbytes, const char* name) {
        // first-fit in 0x800 steps (compiler aligns banks to 0x800)
        uint32_t need = (nbytes + 0x7FF) & ~0x7FFu;
        for (uint32_t off = 0; off + need <= 0x10000; off += 0x800) {
            bool hit = false;
            for (const auto& r : used_) {
                if (off < r.base + r.size && r.base < off + need) { hit = true; break; }
            }
            if (!hit) {
                used_.push_back({off, need, name});
                return {off, need, name};
            }
        }
        throw std::runtime_error(std::string("PSUM allocator: no room for ") + name);
    }
    void claim(uint32_t off, uint32_t nbytes, const char* name) {
        uint32_t need = (nbytes + 0x7FF) & ~0x7FFu;
        for (const auto& r : used_) {
            if (off < r.base + r.size && r.base < off + need)
                throw std::runtime_error(std::string("PSUM allocator: overlap ") + name);
        }
        used_.push_back({off, need, name});
    }

private:
    std::vector<Region> used_;
};

// ------------------------------------------------------------- engines --

// PE engine builder (matmul side: LDWEIGHTS + MATMUL + op10 companions).
class Pe {
public:
    Stream& out;

    explicit Pe(Stream& o) : out(o) {}

    // LDWEIGHTS: stream 128-row chunk from SB src (src = region end - 4).
    // companion op10: byte1 = b1 (1 at even group, 2 at odd), u6 const.
    void ldweights(uint32_t src, uint16_t num, const uint8_t* wait /*2 vals or null*/,
                   uint8_t b1, uint16_t u6) {
        Instr s(OP_LDWEIGHTS);
        if (wait) {
            s.raw_u8(emp::kEv_wait_mode, 5);
            s.raw_u8(emp::kEv_wait_idx, wait[0]);
            s.raw_u16(emp::kEv_sem_value, wait[1]);
        }
        s.raw_u32(emp::kLw_src, src);
        s.raw_u16(emp::kLw_step20, 0xFFFF);
        s.raw_u16(emp::kLw_num, num);
        s.raw_u8(emp::kLw_flag28, 1);
        s.raw_u8(emp::kLw_flag30, 1);
        out.add(s);
        out.add(op10(b1, u6, 2));
    }
    // MATMUL: A-pattern src (u32), dst PSUM raw, dn; upd emits op10 with
    // events update (u16@6 = 19, byte7 = 5) on the ENGINE instr.
    void matmul(uint32_t a, bool upd, uint8_t b1, uint8_t ring, uint32_t dst,
                uint16_t dn, uint16_t u6, uint16_t anum) {
        Instr s(OP_MATMUL);
        if (upd) {
            s.raw_u8(emp::kEv_upd_mode, 19);
            s.raw_u8(emp::kEv_upd_idx, 5);
        }
        s.raw_u32(emp::kMm_a, a);
        s.raw_u16(emp::kMm_step20, 1);
        s.raw_u16(emp::kMm_anum, anum);
        s.raw_u8(emp::kMm_flag28, 1);
        s.raw_u8(emp::kMm_flag30, 1);
        out.add(s);
        Instr o = op10(b1, u6, 16);
        o.b[emp::kOp10_ring] = ring;
        o.raw_u32(emp::kOp10_dst, dst);
        o.raw_u16(emp::kOp10_step, 1);
        o.raw_u16(emp::kOp10_dn, dn);
        o.raw_u16(emp::kOp10_tick28, 1);
        o.raw_u16(emp::kOp10_tick30, 1);
        out.add(o);
    }

    // op160 event/semaphore notify (wait_mode 5, idx = val>>8, u16 value).
    void ev(uint16_t val, uint16_t value = 16) {
        Instr s(OP_CTRL_EV);
        s.b[1] = 16;
        s.raw_u8(emp::kEv_wait_mode, 5);
        s.raw_u8(emp::kEv_wait_idx, uint8_t((val >> 8) & 0xFF));
        s.raw_u16(emp::kOp160_value, value);
        out.add(s);
    }

    void prologue() {
        Instr a(OP_PROLOGUE_177);
        a.b[1] = 16;
        out.add(a);
        out.pad32();
        Instr c(OP_SP_CTRL_204);
        c.b[1] = 16;
        out.add(c);
        out.pad32();
    }

private:
    static Instr op10(uint8_t b1, uint16_t u6, uint16_t kind) {
        Instr o(OP_DVE);
        o.b[1] = b1;
        o.raw_u16(emp::kOp10_u6const, u6);
        o.raw_u16(emp::kOp10_unk12, 3855);
        o.raw_u16(emp::kOp10_kind, kind);
        return o;
    }
};

}  // namespace nasm
