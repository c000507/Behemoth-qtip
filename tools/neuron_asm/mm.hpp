// mm.hpp: C++ port of tools/mm_synth.py (PE0.bin + Pool0.bin synthesis for
// y[M,N] = x[M,K] @ w[K,N]). Byte-exact acceptance: tools/mm_synth_check.py
// oracle hex, via port_gate.cpp.
//
// Everything mirrors the Python synthesizer 1:1 including the literal
// tables (D0_REARM / D1_FIELDS / D2_DRAIN) - those are compiler-oracle
// constants for shape-dependent choices not yet formula-decoded.
#pragma once

#include <cstdint>
#include <cstring>
#include <map>
#include <tuple>
#include <vector>

#include "neuron_asm.hpp"

namespace nmm {

using nasm::Instr;
using nasm::Stream;

constexpr uint32_t PSUM = 0x02000000;
constexpr uint32_t W0 = 0x18000;

struct Shape {
    int M, K, N;
};
inline bool operator<(const Shape& a, const Shape& b) {
    return std::tie(a.M, a.K, a.N) < std::tie(b.M, b.K, b.N);
}

// ------------------------------------------------------------ tables ----

// desc0 rearm literal per bucket (compiler choice)
inline const std::map<Shape, uint32_t>& d0_rearm_tbl() {
    static const std::map<Shape, uint32_t> m = {
        {Shape{64, 384, 128}, 1792},  {Shape{32, 384, 128}, 1664},
        {Shape{32, 128, 384}, 1024},  {Shape{64, 128, 384}, 768},
        {Shape{32, 384, 256}, 2048},  {Shape{64, 512, 384}, 2304},
        {Shape{32, 640, 128}, 3072},  {Shape{64, 1024, 128}, 4608},
        {Shape{32, 128, 128}, 640},   {Shape{32, 128, 256}, 640},
        {Shape{32, 128, 512}, 1024},  {Shape{32, 256, 128}, 1152},
        {Shape{32, 256, 256}, 1536},  {Shape{32, 512, 128}, 2560},
        {Shape{32, 512, 256}, 2560},  {Shape{32, 1024, 128}, 4608},
        {Shape{64, 128, 128}, 768},   {Shape{64, 128, 256}, 1024},
        {Shape{64, 256, 128}, 1280},  {Shape{64, 256, 256}, 1536},
        {Shape{64, 512, 128}, 2304},  {Shape{64, 512, 256}, 2560},
        {Shape{96, 128, 128}, 896},   {Shape{96, 512, 256}, 2432},
        {Shape{96, 640, 192}, 2944},  {Shape{128, 128, 128}, 1024},
    };
    return m;
}

// desc1 w-in: 11 u16 fields @24..@58 + u32 addr40 (compiler literals)
struct D1F {
    uint16_t f[11];
    uint32_t addr40;
};
inline const std::map<Shape, D1F>& d1_fields_tbl() {
    static const std::map<Shape, D1F> m = {
        {Shape{128, 128, 128}, D1F{{512, 0, 1, 128, 1, 0, 4, 1, 0, 128, 1}, 195072}},
        {Shape{32, 1024, 128}, D1F{{0, 1, 512, 8, 128, 512, 0, 0, 4, 8, 128}, 187904}},
        {Shape{32, 128, 128}, D1F{{512, 0, 1, 128, 1, 0, 4, 1, 0, 128, 1}, 195456}},
        {Shape{32, 128, 256}, D1F{{1024, 0, 1, 128, 1, 0, 4, 1, 0, 128, 1}, 194944}},
        {Shape{32, 128, 512}, D1F{{2048, 0, 1, 128, 1, 0, 4, 1, 0, 128, 1}, 193536}},
        {Shape{32, 256, 128}, D1F{{0, 1, 512, 2, 128, 512, 0, 0, 4, 2, 128}, 194432}},
        {Shape{32, 256, 256}, D1F{{0, 2, 1024, 2, 128, 1024, 0, 0, 4, 2, 128}, 193024}},
        {Shape{32, 512, 128}, D1F{{0, 1, 512, 4, 128, 512, 0, 0, 4, 4, 128}, 192000}},
        {Shape{32, 512, 256}, D1F{{0, 2, 1024, 4, 128, 1024, 0, 0, 4, 4, 128}, 189952}},
        {Shape{64, 128, 128}, D1F{{512, 0, 1, 128, 1, 0, 4, 1, 0, 128, 1}, 195328}},
        {Shape{64, 128, 256}, D1F{{1024, 0, 1, 128, 1, 0, 4, 1, 0, 128, 1}, 194560}},
        {Shape{64, 256, 128}, D1F{{0, 1, 512, 2, 128, 512, 0, 0, 4, 2, 128}, 194304}},
        {Shape{64, 256, 256}, D1F{{0, 2, 1024, 2, 128, 1024, 0, 0, 4, 2, 128}, 193024}},
        {Shape{64, 512, 128}, D1F{{0, 1, 512, 4, 128, 512, 0, 0, 4, 4, 128}, 192256}},
        {Shape{64, 512, 256}, D1F{{0, 2, 1024, 4, 128, 1024, 0, 0, 4, 4, 128}, 189952}},
        {Shape{96, 128, 128}, D1F{{512, 0, 1, 128, 1, 0, 4, 1, 0, 128, 1}, 195200}},
        {Shape{96, 512, 256}, D1F{{0, 2, 1024, 4, 128, 1024, 0, 0, 4, 4, 128}, 190080}},
        {Shape{96, 640, 192}, D1F{{32768, 1, 768, 5, 128, 768, 0, 0, 4, 5, 128}, 189824}},
    };
    return m;
}

// desc2 y-out drain literal (SB-side drain window) for fresh shapes
inline const std::map<Shape, uint32_t>& d2_drain_tbl() {
    static const std::map<Shape, uint32_t> m = {
        {Shape{64, 384, 128}, 17152}, {Shape{32, 384, 128}, 16768},
        {Shape{32, 128, 384}, 16896}, {Shape{64, 128, 384}, 16640},
        {Shape{32, 384, 256}, 17024}, {Shape{32, 640, 128}, 17152},
        {Shape{64, 1024, 128}, 18688},
    };
    return m;
}

// ----------------------------------------------------------- helpers ----

inline uint16_t u16clip(uint32_t v) { return uint16_t(v & 0xFFFF); }

// wait pair (semaphore idx, value) -> 2-byte scratch (python tuple parity)
inline const uint8_t* w8(uint8_t idx, uint8_t val) {
    static thread_local uint8_t buf[2];
    buf[0] = idx;
    buf[1] = val;
    return buf;
}

inline bool is_fresh(Shape s) {
    static const std::map<Shape, bool> m = {
        {Shape{32, 384, 128}, true}, {Shape{32, 384, 256}, true},
        {Shape{32, 640, 128}, true}, {Shape{32, 128, 384}, true},
        {Shape{64, 128, 384}, true}, {Shape{48, 256, 128}, true},
        {Shape{48, 384, 128}, true}, {Shape{64, 1024, 128}, true},
        {Shape{64, 384, 128}, true},
    };
    auto it = m.find(s);
    return it != m.end() && it->second;
}

inline bool is_unsupported(Shape s) {
    // 32_640_256 = multi-round w staging; 64_512_384 = degenerate oracle
    return (s.M == 32 && s.K == 640 && s.N == 256) ||
           (s.M == 64 && s.K == 512 && s.N == 384);
}

inline uint32_t to_off(int M, int K, int N) {
    bool big = (N >= 512 || (M <= 64 && N >= 256 && (K >= 256 || M == 64)) ||
                (M == 32 && K >= 512));
    return big ? 0x2FE00 : 0x30000 - 4 * M;
}

inline uint32_t a_base(int M, int K, int N) {
    int NT = (N + 127) / 128;
    if (K == 128) {
        if (NT == 1) return 0x800;
        if (M == 32) return NT >= 3 ? 0x1000 : 0x800;
        return NT >= 3 ? 0x800 : 0x1000;  // M64: NT2=0x1000, NT3=0x800
    }
    if (K >= 1024) return 0x1000;
    if (NT >= 2 && M <= 64) return 0x0;
    if (M == 32 && K >= 512) return 0x1000;
    return 0x800;
}

// final scheme: col | direct | pv
struct Scheme {
    const char* kind;
    std::vector<uint32_t> dsts;
    std::vector<uint16_t> dns;
};

inline Scheme final_scheme(int M, int K, int N) {
    int NT = (N + 127) / 128;
    if (K == 128) {
        if (N <= 256) {
            uint16_t dn = uint16_t((M >= 96 || M == 32) ? N : 128);
            int tiles = (N + dn - 1) / dn;
            Scheme s{"col", {}, {}};
            for (int i = 0; i < tiles; ++i) {
                s.dsts.push_back(PSUM | uint32_t(4 * M * i));
                s.dns.push_back(dn);
            }
            return s;
        }
        if (M == 64) return {"direct", {PSUM | 0}, {uint16_t(N)}};
        Scheme s{"col", {}, {}};
        for (int i = 0; i < NT; ++i) {
            s.dsts.push_back(PSUM | uint32_t(4 * M * i));
            s.dns.push_back(uint16_t(M));
        }
        return s;
    }
    if (M >= 96) return {"direct", {PSUM | 0}, {uint16_t(N)}};
    bool pv = (M == 32 && (K >= 512 || (N >= 256 && K >= 256))) ||
              (M == 64 && (N >= 256 || K >= 1024));
    if (pv) {
        uint32_t base = NT == 1 ? 0x800 : (NT == 2 ? 0x1800 : 0x800);
        Scheme s{"pv", {}, {}};
        for (int t = 0; t < NT; ++t) {
            s.dsts.push_back(PSUM | (base + 0x200 * uint32_t(t)));
            s.dns.push_back(128);
        }
        return s;
    }
    return {"direct", {PSUM | 0}, {uint16_t(N)}};
}

inline std::vector<uint32_t> b_temps(int M, int K, int N) {
    Scheme sc = final_scheme(M, K, N);
    int NT = (N + 127) / 128;
    if (std::string(sc.kind) == "pv") {
        std::vector<uint32_t> v;
        if (NT == 1) {
            v.push_back(PSUM | 0);
        } else {
            for (int t = 0; t < NT; ++t)
                v.push_back(PSUM | uint32_t(0x800 * (1 + t)));
        }
        return v;
    }
    return {};
}

inline bool wring(int M, int K, int N) {
    return (M == 32 && K >= 512 && N <= 128) ||
           (N >= 256 && K >= 256 && M <= 64) || (N >= 256 && M == 64);
}

// w-region base: pool writes w chunk s at addr + 4N*s; PE scr/LW derive.
inline uint32_t desc1_addr(int M, int K, int N) {
    Shape s{M, K, N};
    if (is_fresh(s) || is_unsupported(s)) return W0;
    auto it = d1_fields_tbl().find(s);
    if (it != d1_fields_tbl().end()) return it->second.addr40;
    // d1_fields() interpolation (python parity)
    int AS = K / 128;
    uint32_t tof = to_off(M, K, N);
    uint32_t x_bottom = tof - 512u * uint32_t(AS);
    int kcls = K <= 128 ? 0 : (K <= 512 ? 1 : 2);
    bool wri = wring(M, K, N);
    std::string kind = final_scheme(M, K, N).kind;
    uint32_t Ws = std::max(512u, 4u * uint32_t(N));
    uint32_t extra = Ws > 512 ? 512u : 0u;
    uint32_t addr;
    if (kcls == 0) addr = x_bottom - 4u * uint32_t(N);
    else if (wri) addr = x_bottom - extra - Ws * uint32_t(AS - 1) - 512;
    else if (kind == "col") addr = x_bottom - 4u * uint32_t(N);
    else addr = x_bottom - uint32_t(AS) * 4u * uint32_t(N);
    return addr;
}

inline uint32_t scr_col(int M, int K, int N) {
    uint32_t tof = to_off(M, K, N);
    uint32_t x_bottom = tof - 512u * uint32_t(K / 128);
    if (N >= 512 || (N >= 256 && M >= 64)) return 0x4000;
    return x_bottom - 4u * uint32_t(N);
}

// ---------------------------------------------------------- PE stream ---

// Port of mm_synth.synth: full PE0.bin. Throws for unsupported shapes.
inline std::vector<uint8_t> synth_pe(int M, int K, int N) {
    Shape sh{M, K, N};
    if (is_unsupported(sh)) {
        throw std::runtime_error(
            "shape has no decoded synthesis (structural anomaly) - use oracle template");
    }
    int AS = K / 128;
    int NT = (N + 127) / 128;
    uint32_t tof = to_off(M, K, N);
    Stream out;
    nasm::Pe pe{out};
    pe.prologue();
    pe.ev(0x1805);
    out.pad32();
    uint32_t x_bottom = tof - 512u * uint32_t(AS);
    bool fresh = is_fresh(sh);

    int32_t xsh = 0;
    if (M == 32 && K == 128 && N == 384) xsh = -int32_t(N);
    else if (M == 64 && K == 128 && N == 384) xsh = 4 * M;
    else if (M == 64 && K == 1024 && N == 128) xsh = -4 * M;
    // x-side relocation: LW shift for all three; A/PV scr shift = same set
    uint32_t xsh_ascr = uint32_t(int32_t(tof) + xsh) - tof;  // signed delta
    uint32_t xsh_pvscr = xsh_ascr;

    auto x_lw = [&](int s) -> uint32_t {
        return uint32_t(int32_t(tof) - 512 * (AS - 1 - s) - 4 + xsh);
    };

    uint16_t U6A = uint16_t(0x8000 + M);
    uint16_t U6F = uint16_t(0x80 + 0x1000 * (M / 16));
    uint16_t U6W = 0x8080;
    bool wri = wring(M, K, N);
    Scheme sc = final_scheme(M, K, N);
    uint32_t d1a = desc1_addr(M, K, N);
    std::string kind = sc.kind;
    bool u6w_ok = (wri && !(M == 64 && K == 128 && N > 256)) ||
                  (M == 64 && K >= 1024);
    uint16_t U6B = (u6w_ok ||
                    (kind == "col" && N > 256 && !(M == 64 && K == 128 && N > 256)))
                       ? U6W
                       : U6F;
    std::vector<uint32_t> temps = b_temps(M, K, N);

    auto upd_flag = [](bool u) { return u ? 1 : 0; };

    // ---- A phase: AS chunks x 2 pairs
    uint32_t ab = a_base(M, K, N);
    for (int s = 0; s < AS; ++s) {
        for (int p = 0; p < 2; ++p) {
            int gp = 2 * s + p;
            bool upd = gp == 2 * AS - 1;
            const uint8_t* wait = gp == 0 ? w8(8, 16) : nullptr;
            pe.ldweights(x_lw(s), 128, wait, uint8_t(gp % 2 == 0 ? 1 : 2), U6A);
            pe.matmul(tof + uint32_t(xsh_ascr), upd, 2,
                      uint8_t(p == 0 ? 1 : 2), PSUM | (ab + 4u * uint32_t(M) * uint32_t(s)),
                      uint16_t(M), U6A, uint16_t(M));
            (void)upd_flag;
        }
    }

    // ---- B phase
    pe.ev(0x905, 16);
    out.pad32();
    if (kind == "col") {
        uint16_t lw_num = uint16_t((N <= 128 || (M == 32 && N <= 256)) ? M : 128);
        int ntiles = int(sc.dsts.size());
        int gpt = 2;
        for (int t = 0; t < ntiles; ++t) {
            for (int g = 0; g < gpt; ++g) {
                int gp = t * gpt + g;
                bool upd = gpt > 1 && t == ntiles - 1 && g == gpt - 1;
                const uint8_t* wait = gp == 0 ? w8(4, 1) : nullptr;
                uint32_t scr_t = (fresh && M <= 32) ? 0x4000 : scr_col(M, K, N);
                uint32_t lw_t = ntiles == 1 ? 0x4000 + 4u * uint32_t(M) * (t + 1) - 4
                                            : d1a + 512u * (t + 1) - 4;
                uint16_t dn_t = (M == 64 && N == 256) ? uint16_t(M) : sc.dns[t];
                pe.ldweights(lw_t, lw_num, wait, uint8_t(g == 0 ? 1 : 2), U6B);
                pe.matmul(scr_t, upd, 3,
                          uint8_t(g == 0 ? 1 : (g == gpt - 1 ? 2 : 0)),
                          sc.dsts[t], dn_t, U6B, dn_t);
            }
        }
    } else if (kind == "pv") {
        // wring families: LW = w ring, scr = window
        uint32_t Ws = std::max(512u, 4u * uint32_t(N));
        auto lwbase = [&](int s) -> uint32_t {
            if (fresh || Ws > 512)
                return d1a + 512u + Ws * uint32_t(s) - 4;
            return x_bottom - Ws * uint32_t(AS - 1 - s) - 4;
        };
        for (int ti = 0; ti < NT; ++ti) {
            int t = NT - 1 - ti;  // tiles emitted in reverse order
            uint32_t off = (t == NT - 1) ? 0 : 512;
            for (int s = 0; s < AS; ++s) {
                for (int p = 0; p < 2; ++p) {
                    int gp = ti * 2 * AS + 2 * s + p;
                    bool upd = s == AS - 1 && p == 1;
                    const uint8_t* wait = gp == 0 ? w8(4, 1) : nullptr;
                    pe.ldweights(lwbase(s) + off, 128, wait,
                                 uint8_t(gp % 2 == 0 ? 1 : 2), U6B);
                    pe.matmul(0x4000 + 4u * uint32_t(M) * uint32_t(s), upd, 3,
                              uint8_t((s == 0 && p == 0) ? 1 : ((s == AS - 1 && p == 1) ? 2 : 0)),
                              temps[t], uint16_t(M), U6B, uint16_t(M));
                }
            }
        }
    } else {
        // direct row-mode: LW = window, scr = w chunks
        for (int s = 0; s < AS; ++s) {
            for (int p = 0; p < 2; ++p) {
                int gp = 2 * s + p;
                bool upd = s == AS - 1 && p == 1;
                const uint8_t* wait = gp == 0 ? w8(4, 1) : nullptr;
                uint32_t scr_s = fresh ? W0 + 4u * uint32_t(N) * uint32_t(s)
                                       : x_bottom - 4u * uint32_t(N) * uint32_t(AS - s);
                pe.ldweights(0x4000 + 4u * uint32_t(M) * (s + 1) - 4, uint16_t(M),
                             wait, uint8_t(gp % 2 == 0 ? 1 : 2), U6B);
                pe.matmul(scr_s, upd, 3,
                          uint8_t((s == 0 && p == 0) ? 1 : ((s == AS - 1 && p == 1) ? 2 : 0)),
                          sc.dsts[0], sc.dns[0], U6B, sc.dns[0]);
            }
        }
    }

    // ---- PV phase
    if (kind == std::string("pv") ||
        (kind == std::string("col") && sc.dsts.size() > 1)) {
        uint32_t pv0 = 0x4000 + uint32_t(M == 32 ? K : 2 * K) + 4u * uint32_t(M);
        bool pv_kind = kind == std::string("pv");
        for (int t = 0; t < NT; ++t) {
            for (int p = 0; p < 2; ++p) {
                int gp = 2 * NT * AS + 2 * t + p;
                const uint8_t* wait = nullptr;
                uint8_t wbuf[2];
                if (p == 0 && t == 0) {
                    wbuf[0] = 4; wbuf[1] = 2; wait = wbuf;
                } else if (p == 0 && t == 1 && NT == 2 && pv_kind) {
                    wbuf[0] = 4; wbuf[1] = 3; wait = wbuf;
                }
                bool upd = t == NT - 1 && p == 1;
                uint32_t dst = pv_kind ? sc.dsts[t] : PSUM | (0x800 + 0x200u * uint32_t(t));
                pe.ldweights(pv0 + 4u * uint32_t(M) * uint32_t(t) - 4, uint16_t(M),
                             wait, uint8_t(gp % 2 == 0 ? 1 : 2), U6F);
                pe.matmul(tof + uint32_t(xsh_pvscr), upd, 2,
                          uint8_t(p == 0 ? 1 : 2), dst, 128, U6F, 128);
            }
        }
    }
    return out.bytes;
}

// ---------------------------------------------------------- Pool stream --

// d0_rearm: bucket literal else 4K + 4M
inline uint32_t d0_rearm(int M, int K, int N) {
    auto it = d0_rearm_tbl().find(Shape{M, K, N});
    return it != d0_rearm_tbl().end() ? it->second : 4u * uint32_t(K) + 4u * uint32_t(M);
}

// d1_fields interpolation for shapes outside the oracle table
inline D1F d1_fields_interp(int M, int K, int N) {
    int AS = K / 128;
    uint32_t tof = to_off(M, K, N);
    uint32_t x_bottom = tof - 512u * uint32_t(AS);
    int kcls = K <= 128 ? 0 : (K <= 512 ? 1 : 2);
    D1F f{};
    f.f[0] = uint16_t(kcls == 0 ? 4 * N : 0);
    f.f[1] = uint16_t(AS == 1 ? 0 : (N + 127) / 128);
    f.f[2] = uint16_t(kcls == 0 ? 1 : 4 * N);
    f.f[3] = uint16_t(kcls == 0 ? 128 : AS);
    f.f[4] = uint16_t(kcls == 0 ? 1 : 128);
    f.f[5] = uint16_t(kcls == 0 ? 0 : 4 * N);
    f.f[6] = uint16_t(kcls == 0 ? 4 : 0);
    f.f[7] = uint16_t(kcls == 0 ? 1 : 0);
    f.f[8] = uint16_t(kcls == 0 ? 0 : 4);
    f.f[9] = uint16_t(kcls == 0 ? 128 : AS);
    f.f[10] = uint16_t(kcls == 0 ? 1 : 128);
    uint32_t Ws = std::max(512u, 4u * uint32_t(N));
    uint32_t extra = Ws > 512 ? 512u : 0u;
    if (kcls == 0) f.addr40 = x_bottom - 4u * uint32_t(N);
    else if (wring(M, K, N)) f.addr40 = x_bottom - extra - Ws * uint32_t(AS - 1) - 512;
    else if (std::string(final_scheme(M, K, N).kind) == "col")
        f.addr40 = x_bottom - 4u * uint32_t(N);
    else f.addr40 = x_bottom - uint32_t(AS) * 4u * uint32_t(N);
    return f;
}

// One 64B op212 DMA descriptor.
inline Instr desc212(uint8_t sem, uint16_t w_size, uint16_t x_size,
                     const uint16_t extra[4], uint16_t drain) {
    Instr s(212);
    s.raw_u16(12, uint16_t(sem) << 8);
    s.raw_u16(14, 16);
    s.raw_u16(22, 8192);
    s.raw_u16(24, w_size);
    s.raw_u16(26, extra[0]);
    s.raw_u16(28, 1);
    s.raw_u16(32, extra[1]);
    s.raw_u16(34, 1);
    s.raw_u16(36, w_size);
    s.raw_u16(40, uint16_t(65536 - extra[2]));
    s.raw_u16(42, 2);
    s.raw_u16(46, 4096);
    s.raw_u16(50, 4);
    s.raw_u16(52, 1);
    s.raw_u16(56, extra[3]);
    s.raw_u16(58, 1);
    s.raw_u16(60, x_size);
    s.raw_u16(62, 0x0A0A);
    (void)drain;
    return s;
}

// Port of mm_synth.synth_pool: full Pool0.bin.
inline std::vector<uint8_t> synth_pool(int M, int K, int N) {
    Shape sh{M, K, N};
    if (is_unsupported(sh)) {
        throw std::runtime_error("unsupported shape (multi-round w staging / degenerate oracle)");
    }
    int AS = K / 128;
    int NT = (N + 127) / 128;
    Scheme sc = final_scheme(M, K, N);
    std::string kind = sc.kind;
    bool pv = kind == "pv" || (kind == "col" && sc.dsts.size() > 1);
    Stream out;
    Instr pre(177);
    pre.b[1] = 16;
    out.add(pre);
    out.pad32();
    Instr c(204);
    c.b[1] = 16;
    out.add(c);
    out.pad32();

    // desc0: x in, sem 8
    uint32_t d0r = d0_rearm(M, K, N);
    uint16_t e0[4] = {0, uint16_t(M), uint16_t(d0r), uint16_t(M)};
    out.add(desc212(8, uint16_t(4 * K), uint16_t(4 * K), e0, 0));

    // desc1: w in, sem 9
    uint32_t drain;
    auto d2it = d2_drain_tbl().find(sh);
    if (d2it != d2_drain_tbl().end()) {
        drain = d2it->second;
    } else if (pv) {
        uint32_t pv0 = 0x4000 + uint32_t(M == 32 ? K : 2 * K) + 4u * uint32_t(M);
        drain = pv0 + 4u * uint32_t(M) * uint32_t(NT - 1);
    } else {
        drain = 0x4000 + 4u * uint32_t(M) * uint32_t(AS);
    }
    auto f = d1_fields_tbl().find(sh);
    D1F fld = f != d1_fields_tbl().end() ? f->second : d1_fields_interp(M, K, N);
    uint32_t addr40 = desc1_addr(M, K, N);
    Instr d1(212);
    d1.raw_u16(12, 9 << 8);
    d1.raw_u16(14, 16);
    d1.raw_u16(20, 1);
    d1.raw_u16(22, 8192);
    static const size_t offs[11] = {24, 26, 28, 32, 34, 48, 50, 52, 54, 56, 58};
    for (int i = 0; i < 11; ++i) d1.raw_u16(offs[i], fld.f[i]);
    d1.raw_u32(36, 4u * uint32_t(N));
    d1.raw_u32(40, addr40);
    d1.raw_u16(46, 4096);
    d1.raw_u16(60, uint16_t(4 * N));
    d1.raw_u16(62, 0x0A0A);
    out.add(d1);

    // desc2: y out, sem 10, waits B-done
    uint16_t val = 2;
    if (kind == "pv") val = uint16_t(2 + NT);
    else if (pv || (M == 64 && K >= 1024)) val = 3;
    uint32_t drain32 = drain & 0xFFFF;  // python packs u16@16 (truncate)
    Instr d2(212);
    d2.raw_u16(4, 5);
    d2.raw_u16(5, 4);
    d2.raw_u16(8, val);
    d2.raw_u16(12, 10 << 8);
    d2.raw_u16(14, 16);
    d2.raw_u16(16, uint16_t(drain32));
    d2.raw_u16(22, 4096);
    d2.raw_u16(26, 4);
    d2.raw_u16(28, 1);
    d2.raw_u16(32, uint16_t(M));
    d2.raw_u16(34, 1);
    d2.raw_u16(36, uint16_t(4 * N));
    d2.raw_u16(44, 2);
    d2.raw_u16(46, 8192);
    d2.raw_u16(48, uint16_t(4 * N));
    d2.raw_u16(52, 1);
    d2.raw_u16(56, uint16_t(M));
    d2.raw_u16(58, 1);
    d2.raw_u16(60, uint16_t(4 * N));
    d2.raw_u16(62, 0x0A0A);
    out.add(d2);

    // tail event + pad
    Instr ev(164);
    ev.b[1] = 16;
    ev.raw_u16(4, 5);
    ev.raw_u16(5, 10);
    ev.raw_u16(8, 16);
    out.add(ev);
    out.pad32();
    return out.bytes;
}

}  // namespace nmm
