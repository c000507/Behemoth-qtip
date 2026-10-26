// Shared host-side helpers: numeric conversions, logging, timing.
#pragma once
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

namespace qtip {

inline double now_s() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

void log_set_file(const std::string& path);
void logf(const char* fmt, ...) __attribute__((format(printf, 1, 2)));

[[noreturn]] inline void die(const std::string& msg) {
  std::fprintf(stderr, "fatal: %s\n", msg.c_str());
  std::fflush(stderr);
  std::abort();
}
#define QCHECK(cond, msg)                                                              \
  do {                                                                                 \
    if (!(cond)) ::qtip::die(std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
                             ": " + (msg));                                            \
  } while (0)

// ---- bf16 / fp16 / fp8 conversions (host) -------------------------------------
inline float bf16_to_f32(uint16_t h) {
  uint32_t u = uint32_t(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
inline uint16_t f32_to_bf16(float f) {  // round to nearest even
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return uint16_t((u >> 16) | 0x40);
  u += 0x7fffu + ((u >> 16) & 1u);
  return uint16_t(u >> 16);
}
inline float fp16_to_f32(uint16_t h) {
  uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  float v;
  if (e == 0) v = std::ldexp(float(m), -24);
  else if (e == 31) v = m ? NAN : INFINITY;
  else v = std::ldexp(float(m | 0x400), int(e) - 25);
  return s ? -v : v;
}
inline uint16_t f32_to_fp16(float f) {  // round to nearest even
  uint32_t u;
  std::memcpy(&u, &f, 4);
  uint32_t sign = (u >> 16) & 0x8000u;
  uint32_t a = u & 0x7fffffffu;
  if (a >= 0x7f800000u) return uint16_t(sign | 0x7c00u | (a > 0x7f800000u ? 0x200u : 0u));
  if (a >= 0x477ff000u) return uint16_t(sign | 0x7c00u);  // overflow -> inf
  if (a < 0x38800000u) {                                  // subnormal / zero in fp16
    float af;
    std::memcpy(&af, &a, 4);
    // value / 2^-24, rounded to nearest even
    float q = af * 16777216.0f;
    float r = std::nearbyint(q);
    return uint16_t(sign | uint32_t(r));
  }
  uint32_t e = (a >> 23) - 112;  // rebias 127 -> 15
  uint32_t m = a & 0x7fffffu;
  uint32_t h = (e << 10) | (m >> 13);
  uint32_t rem = m & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;
  return uint16_t(sign | h);
}
inline float round_fp16(float f) { return fp16_to_f32(f32_to_fp16(f)); }
// FP8 E4M3 (OCP "fn": no inf, 0x7f/0xff = NaN, bias 7)
inline float fp8e4m3_to_f32(uint8_t b) {
  uint32_t s = b >> 7, e = (b >> 3) & 0xf, m = b & 7;
  float v;
  if (e == 0) v = std::ldexp(float(m), -9);
  else if (e == 15 && m == 7) v = NAN;
  else v = std::ldexp(float(m | 8), int(e) - 10);
  return s ? -v : v;
}

// splitmix64-based deterministic RNG (sign vectors etc.)
struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x1234567ull) {}
  uint64_t next() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  double uniform() { return double(next() >> 11) * (1.0 / 9007199254740992.0); }
  double normal() {
    double u1 = uniform(), u2 = uniform();
    if (u1 < 1e-300) u1 = 1e-300;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
  }
  float sign() { return (next() >> 63) ? -1.0f : 1.0f; }
};

}  // namespace qtip
