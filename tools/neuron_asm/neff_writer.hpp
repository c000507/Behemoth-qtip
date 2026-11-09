// neff_writer.hpp: dependency-free NEFF container writer.
// Layout: 1024B header + gzip(deflate) of a ustar tar archive.
// Digest A @0xAC = md5(gz stream), each u32 word-swapped (big-endian
// read of little-endian bytes). Digest B not required by the runtime
// (verified on device: zeroed digest B loads + executes fine).
#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace nasm {

// --------------------------------------------------------------- md5 ----
class MD5 {
public:
    MD5() { reset(); }
    void reset() {
        abcd_[0]=0x67452301; abcd_[1]=0xefcdab89; abcd_[2]=0x98badcfe; abcd_[3]=0x10325476;
        len_ = 0;
        buf_ = 0;
    }
    void update(const uint8_t* p, size_t n) {
        len_ += n;
        while (n) {
            if (buf_ == 0 && n >= 64) {
                block(p);
                p += 64;
                n -= 64;
            } else {
                size_t take = std::min(n, size_t(64) - buf_);
                std::memcpy(cbuf_ + buf_, p, take);
                buf_ += take;
                p += take;
                n -= take;
                if (buf_ == 64) { block(cbuf_); buf_ = 0; }
            }
        }
    }
    std::array<uint8_t, 16> digest() {
        uint64_t bits = uint64_t(len_) * 8;
        uint8_t pad = 0x80;
        update(&pad, 1);
        uint8_t z = 0;
        while (buf_ != 56) update(&z, 1);
        uint8_t lenb[8];
        for (int i = 0; i < 8; ++i) lenb[i] = uint8_t(bits >> (8 * i));
        update(lenb, 8);
        std::array<uint8_t, 16> out;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                out[i * 4 + j] = uint8_t(abcd_[i] >> (8 * j));
        return out;
    }

private:
    static uint32_t rot(uint32_t x, int c) { return (x << c) | (x >> (32 - c)); }
    void block(const uint8_t* p) {
        static const uint32_t K[64] = {
            0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a,
            0xa8304613, 0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
            0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340,
            0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
            0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8,
            0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
            0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
            0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
            0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92,
            0xffeff47d, 0x85845dd1, 0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
            0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391};
        static const int S[4][4] = {{7, 12, 17, 22}, {5, 9, 14, 20},
                                    {4, 11, 16, 23}, {6, 10, 15, 21}};
        uint32_t m[16];
        for (int i = 0; i < 16; ++i)
            m[i] = uint32_t(p[i * 4]) | uint32_t(p[i * 4 + 1]) << 8 |
                   uint32_t(p[i * 4 + 2]) << 16 | uint32_t(p[i * 4 + 3]) << 24;
        uint32_t a = abcd_[0], b = abcd_[1], c = abcd_[2], d = abcd_[3];
        for (int i = 0; i < 64; ++i) {
            uint32_t f; int g;
            if (i < 16) { f = (b & c) | (~b & d); g = i; }
            else if (i < 32) { f = (d & b) | (~d & c); g = (5 * i + 1) % 16; }
            else if (i < 48) { f = b ^ c ^ d; g = (3 * i + 5) % 16; }
            else { f = c ^ (b | ~d); g = (7 * i) % 16; }
            uint32_t tmp = d; d = c; c = b;
            int r = S[i / 16][i % 4];
            b = b + rot(a + f + K[i] + m[g], r);
            a = tmp;
        }
        abcd_[0] += a; abcd_[1] += b; abcd_[2] += c; abcd_[3] += d;
        buf_ = 0;
    }
    uint32_t abcd_[4];
    uint64_t len_;
    size_t buf_;
    uint8_t cbuf_[64];
};

// ------------------------------------------------------- stored gzip ----
// Deflate with BTYPE=00 (stored) blocks - valid gzip, no compression.
// Size is fine for graph containers (tens of KB).
inline std::vector<uint8_t> gzip_stored(const std::vector<uint8_t>& raw,
                                        uint32_t mtime = 0) {
    std::vector<uint8_t> out;
    out.push_back(0x1F); out.push_back(0x8B);          // magic
    out.push_back(8);                                   // deflate
    out.push_back(0);                                   // flags
    for (int i = 0; i < 4; ++i) out.push_back(uint8_t(mtime >> (8 * i)));
    out.push_back(0); out.push_back(3);                 // XFL, OS=unix
    size_t i = 0;
    while (i < raw.size()) {
        size_t chunk = std::min(size_t(65535), raw.size() - i);
        bool final = i + chunk >= raw.size();
        out.push_back(uint8_t(final ? 1 : 0));  // BFINAL + BTYPE=00 (stored)
        out.push_back(uint8_t(chunk & 0xFF));
        out.push_back(uint8_t(chunk >> 8));
        out.push_back(uint8_t(~chunk & 0xFF));
        out.push_back(uint8_t((~chunk >> 8) & 0xFF));
        out.insert(out.end(), raw.begin() + i, raw.begin() + i + chunk);
        i += chunk;
    }
    uint32_t crc = 0xFFFFFFFFu;
    for (uint8_t b : raw) {
        crc ^= b;
        for (int k = 0; k < 8; ++k)
            crc = (crc >> 1) ^ (0xEDB88320u & (~((crc & 1) - 1)));
    }
    crc = ~crc;
    uint32_t sz = uint32_t(raw.size());
    for (int k = 0; k < 4; ++k) out.push_back(uint8_t(crc >> (8 * k)));
    for (int k = 0; k < 4; ++k) out.push_back(uint8_t(sz >> (8 * k)));
    return out;
}

// ------------------------------------------------------------- ustar ----
struct TarEntry {
    std::string name;
    std::vector<uint8_t> data;
};

inline void ustar_field(char* p, size_t n, const std::string& v) {
    std::memset(p, 0, n);
    std::memcpy(p, v.c_str(), std::min(v.size(), n - 1));
}
inline void ustar_octal(char* p, size_t n, uint64_t v) {
    std::memset(p, '0', n - 1);
    p[n - 1] = 0;
    std::string s;
    do { s.insert(s.begin(), char('0' + (v % 8))); v /= 8; } while (v);
    std::memcpy(p + n - 1 - s.size(), s.c_str(), s.size());
}

inline std::vector<uint8_t> build_tar(const std::vector<TarEntry>& entries) {
    std::vector<uint8_t> out;
    for (const auto& e : entries) {
        char h[512];
        std::memset(h, 0, 512);
        ustar_field(h + 0, 100, e.name);
        ustar_octal(h + 100, 8, 0644);
        ustar_octal(h + 108, 8, 0);
        ustar_octal(h + 116, 8, 0);
        ustar_octal(h + 124, 12, e.data.size());
        ustar_octal(h + 136, 12, 0);
        std::memcpy(h + 156, "0", 1);      // regular file
        ustar_field(h + 257, 6, "ustar");
        ustar_field(h + 263, 2, "00");
        uint32_t chk = 0;
        for (size_t i = 0; i < 512; ++i) chk += uint8_t(i < 148 ? h[i] : (i < 156 ? 32 : h[i]));
        ustar_octal(h + 148, 8, chk);
        out.insert(out.end(), h, h + 512);
        out.insert(out.end(), e.data.begin(), e.data.end());
        size_t pad = (512 - e.data.size() % 512) % 512;
        out.insert(out.end(), pad, 0);
    }
    out.insert(out.end(), 1024, 0);        // EOF blocks
    return out;
}

// -------------------------------------------------------------- json ----
// Minimal JSON value tree (our schemas are flat / shallow).
struct Json {
    enum Type { NUL, STR, NUM, BOOL, ARR, OBJ } type = NUL;
    std::string s;
    double num = 0;
    bool b = false;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj;  // ordered
    static Json str(const std::string& v) { Json j; j.type = STR; j.s = v; return j; }
    static Json number(double v) { Json j; j.type = NUM; j.num = v; return j; }
    static Json boolean(bool v) { Json j; j.type = BOOL; j.b = v; return j; }
    static Json array() { Json j; j.type = ARR; return j; }
    static Json object() { Json j; j.type = OBJ; return j; }
    void set(const std::string& k, Json v) {
        obj.push_back({k, std::move(v)});
    }
    static Json int_list(const std::vector<uint64_t>& v) {
        Json j = array();
        for (uint64_t x : v) j.arr.push_back(number(double(x)));
        return j;
    }
    void dump(std::string& out) const;
};

inline void jdump_str(std::string& out, const std::string& s) {
    out.push_back('"');
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            default: out.push_back(c);
        }
    }
    out.push_back('"');
}
inline void jnum(std::string& out, double v) {
    if (v == double(int64_t(v)) && std::abs(v) < 1e15)
        out += std::to_string(int64_t(v));
    else {
        char buf[40];
        snprintf(buf, sizeof buf, "%.17g", v);
        out += buf;
    }
}
inline void Json::dump(std::string& out) const {
    switch (type) {
        case NUL: out += "null"; break;
        case STR: jdump_str(out, s); break;
        case NUM: jnum(out, num); break;
        case BOOL: out += b ? "true" : "false"; break;
        case ARR: {
            out.push_back('[');
            for (size_t i = 0; i < arr.size(); ++i) {
                if (i) out.push_back(',');
                arr[i].dump(out);
            }
            out.push_back(']');
            break;
        }
        case OBJ: {
            out.push_back('{');
            for (size_t i = 0; i < obj.size(); ++i) {
                if (i) out.push_back(',');
                jdump_str(out, obj[i].first);
                out.push_back(':');
                obj[i].second.dump(out);
            }
            out.push_back('}');
            break;
        }
    }
}

// ------------------------------------------------------ neff builder ----
struct NeffFile {
    std::string name;
    std::vector<uint8_t> data;
};

struct NeffBuild {
    std::string graph_name = "graph.neff";   // header name field
    std::vector<NeffFile> files;             // tar members (name, bytes)

    void add(const std::string& n, const std::vector<uint8_t>& d) {
        files.push_back({n, d});
    }
    void add_text(const std::string& n, const std::string& text) {
        files.push_back({n, {text.begin(), text.end()}});
    }

    std::vector<uint8_t> finish() const {
        std::vector<TarEntry> entries;
        for (const auto& f : files) entries.push_back({f.name, f.data});
        std::vector<uint8_t> tar = build_tar(entries);
        std::vector<uint8_t> gz = gzip_stored(tar);

        uint8_t hdr[1024];
        std::memset(hdr, 0, 1024);
        auto u32 = [&](size_t off, uint32_t v) {
            for (int i = 0; i < 4; ++i) hdr[off + i] = uint8_t(v >> (8 * i));
        };
        auto u64 = [&](size_t off, uint64_t v) {
            for (int i = 0; i < 8; ++i) hdr[off + i] = uint8_t(v >> (8 * i));
        };
        u32(0x00, 2);                          // version
        u64(0x08, 0x400);                      // payload offset
        u64(0x10, gz.size());                  // payload size
        u64(0x18, 2);
        u32(0xA8, 1);                          // digest count
        MD5 md5;
        md5.update(gz.data(), gz.size());
        auto dg = md5.digest();
        for (int w = 0; w < 4; ++w) {          // u32 word-swap
            for (int i = 0; i < 4; ++i)
                hdr[0xAC + w * 4 + i] = dg[w * 4 + 3 - i];
        }
        std::string nm = graph_name.substr(0, 63);
        std::memcpy(hdr + 0xDC, nm.c_str(), nm.size());

        std::vector<uint8_t> out(hdr, hdr + 1024);
        out.insert(out.end(), gz.begin(), gz.end());
        return out;
    }
};

}  // namespace nasm
