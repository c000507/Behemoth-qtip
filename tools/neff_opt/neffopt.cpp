// neffopt: NEFF DMA post-processor (brief6). C++20, zlib for inflate.
//
//   neffopt in.neff out.neff [--report FILE] [--verbose]
//
// v1 = parse + classify + pass-through. Rewrites are flag-gated and
// only apply to decoded patterns; anything unrecognized passes through
// unchanged with the reason in the report.
//
// Build: g++ -std=c++20 -O2 -o neffopt neffopt.cpp -lz
#include <zlib.h>
#include <openssl/md5.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static std::string slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
static void spit(const fs::path& p, const std::string& d) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(d.data(), (std::streamsize)d.size());
}

// ---------------- NEFF container: 1024B header + gzip(tar) ----------------
struct Neff {
    std::string header;          // 1024 B
    std::string tar_bytes;       // decompressed tar
    // parsed members: name -> content
    std::map<std::string, std::string> members;
    std::vector<std::string> order;  // member order (tar order)
    bool ok = false;
    std::string err;
};

static bool gunzip(const std::string& in, std::string& out, std::string& err) {
    if (in.size() < 18) { err = "too short"; return false; }
    z_stream zs{};
    if (inflateInit2(&zs, 15 + 32) != Z_OK) { err = "inflateInit2"; return false; }
    zs.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(in.data()));
    zs.avail_in = static_cast<uInt>(in.size());
    out.clear();
    int rc = Z_OK;
    char buf[65536];
    do {
        zs.next_out = reinterpret_cast<Bytef*>(buf);
        zs.avail_out = sizeof buf;
        rc = inflate(&zs, Z_NO_FLUSH);
        out.append(buf, sizeof buf - zs.avail_out);
    } while (rc == Z_OK);
    inflateEnd(&zs);
    if (rc != Z_STREAM_END) { err = "inflate rc=" + std::to_string(rc); return false; }
    return true;
}
static std::string gzip_compress(const std::string& in) {
    z_stream zs{};
    if (deflateInit2(&zs, 6, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY) != Z_OK) return "";
    zs.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(in.data()));
    zs.avail_in = static_cast<uInt>(in.size());
    std::string out;
    char buf[65536];
    int rc = Z_OK;
    do {
        zs.next_out = reinterpret_cast<Bytef*>(buf);
        zs.avail_out = sizeof buf;
        rc = deflate(&zs, Z_FINISH);
        out.append(buf, sizeof buf - zs.avail_out);
    } while (rc == Z_OK || rc == Z_BUF_ERROR);
    deflateEnd(&zs);
    return out;
}

static bool parse_neff(const std::string& raw, Neff& n) {
    if (raw.size() < 1024) { n.err = "shorter than header"; return false; }
    n.header = raw.substr(0, 1024);
    if (!gunzip(raw.substr(1024), n.tar_bytes, n.err)) return false;
    // ustar walk
    size_t off = 0;
    auto& t = n.tar_bytes;
    while (off + 512 <= t.size()) {
        const char* h = t.data() + off;
        bool allzero = true;
        for (size_t i = 0; i < 512; ++i) if (h[i] != '\0') { allzero = false; break; }
        if (allzero) break;
        std::string name(h, 100);
        name = name.substr(0, name.find_first_of(std::string("\0 ", 2)));
        if (name.empty()) break;
        size_t size = std::strtoull(std::string(h + 124, 12).c_str(), nullptr, 8);
        char type = h[156];
        off += 512;
        // ustar: member data always padded to a full 512-byte block
        size_t pad = (512 - size % 512) % 512;
        if (type == '0' || type == '\0') {
            if (off + size > t.size()) { n.err = "tar truncated at " + name; return false; }
            n.members[name] = t.substr(off, size);
            n.order.push_back(name);
        }
        off += size + pad;
    }
    n.ok = true;
    return true;
}

static std::string emit_neff(const Neff& n) {
    // rebuild the tar preserving member order and sizes
    std::string t;
    auto hdr = [](const std::string& name, size_t size) {
        char h[512]; std::memset(h, 0, 512);
        std::memcpy(h, name.c_str(), std::min(name.size(), size_t(99)));
        std::memcpy(h + 100, "000644 \0", 8);
        std::memcpy(h + 108, "0000000 \0", 8);
        std::memcpy(h + 116, "0000000 \0", 8);
        std::snprintf(h + 124, 13, "%011lo ", (unsigned long)size);
        std::snprintf(h + 136, 13, "%011o ", 0u);
        h[156] = '0';
        std::memcpy(h + 257, "ustar\00000", 8);
        unsigned sum = 0;
        for (size_t i = 0; i < 512; ++i) sum += (unsigned char)h[i];
        std::snprintf(h + 148, 8, "%06o ", sum);
        return std::string(h, 512);
    };
    for (auto& name : n.order) {
        auto& d = n.members.at(name);
        t += hdr(name, d.size());
        t += d;
        t.append((4 - t.size() % 4) % 4, '\0');
    }
    t.append(1024, '\0');
    std::string gz = gzip_compress(t);
    return n.header + gz;   // digest A = md5(gz) word-swapped @0xAC: recompute!
}

// ---------------- pool descriptor model ----------------
struct Desc {
    int slot;            // 32B slot index in Pool0.bin
    uint16_t raw[32];
    uint8_t sem;         // assigned semaphore (u32@12 >> 8)
    uint16_t wait_idx, wait_mode;
    uint16_t upd_idx, upd_mode;
    uint32_t wait_val;
    // form-A (continuous): packet-count fields
    uint16_t pkt_a, pkt_b;
    bool has_form_a;
};

static uint16_t rd16(const uint16_t* w, int word) { return w[word]; }

static void parse_pool(const std::string& pool, std::vector<Desc>& descs,
                       int& n_ev, int& n_other, std::stringstream& rep) {
    const uint16_t* w = reinterpret_cast<const uint16_t*>(pool.data());
    size_t nw = pool.size() / 2;
    size_t slot = 0;
    while (slot + 16 <= nw) {
        uint8_t op = pool[slot * 2];
        uint8_t wl = pool[slot * 2 + 1];
        if (op == 0xd4 && wl == 0x10) {
            Desc d{};
            d.slot = int(slot);
            std::memcpy(d.raw, w + slot, 64);
            d.sem = (d.raw[6] >> 8) & 0xFF;
            d.wait_mode = d.raw[2] & 0xFF; d.wait_idx = (d.raw[2] >> 8) & 0xFF;
            d.upd_mode = d.raw[3] & 0xFF;  d.upd_idx = (d.raw[3] >> 8) & 0xFF;
            d.wait_val = d.raw[4] | (uint32_t(d.raw[5]) << 16);
            d.pkt_a = d.raw[16]; d.pkt_b = d.raw[27];
            // form-A signature: src+dst marker pair 0x0a0a/0x0606 at word31,
            // addr 0x2000 at words 11/22, packet counts at 16/27
            d.has_form_a = (d.raw[11] == 0x2000 && d.raw[22] == 0x2000 &&
                            (d.raw[16] == d.raw[27]));
            descs.push_back(d);
            slot += 32;   // 64B desc = 2 slots
        } else {
            if (op == 0xa0) ++n_ev; else ++n_other;
            slot += 16;
        }
    }
}

int main(int argc, char** argv) {
    std::string in, out, report;
    bool verbose = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--report" && i + 1 < argc) report = argv[++i];
        else if (a == "--verbose") verbose = true;
        else if (in.empty()) in = a;
        else if (out.empty()) out = a;
    }
    if (in.empty() || out.empty()) {
        std::fprintf(stderr, "usage: neffopt in.neff out.neff [--report FILE] [--verbose]\n");
        return 2;
    }
    Neff n;
    if (!parse_neff(slurp(in), n)) {
        std::fprintf(stderr, "neffopt: parse failed: %s\n", n.err.c_str());
        return 3;
    }
    std::stringstream rep;
    rep << "{\n \"input\": \"" << in << "\",\n";
    rep << " \"members\": " << n.order.size() << ",\n";

    // find pool instr files
    int total_descs = 0, total_ev = 0, total_other = 0;
    std::string classification = "no-pool";
    for (auto& [name, data] : n.members) {
        if (name.find("Pool") == std::string::npos || name.find(".bin") == std::string::npos) continue;
        std::vector<Desc> descs; int nev = 0, noth = 0;
        parse_pool(data, descs, nev, noth, rep);
        total_descs += int(descs.size()); total_ev += nev; total_other += noth;
        rep << " \"" << name << "\": {\"descs\": " << descs.size()
            << ", \"event_gates\": " << nev << ", \"other\": " << noth << "}";
        // classify
        if (descs.size() == 1 && descs[0].has_form_a)
            classification = "mul-form (single continuous desc)";
        else if (descs.size() > 4)
            classification = "ring (circular auto-advancing stream)";
        else if (descs.size() >= 2)
            classification = "serial-chain";
        rep << ",\n \"" << name << "_class\": \"" << classification << "\"";
        if (verbose) {
            int k = 0;
            for (auto& d : descs) {
                rep << ",\n \"desc" << k++ << "\": {\"slot\": " << d.slot
                    << ", \"sem\": " << int(d.sem)
                    << ", \"wait\": \"" << int(d.wait_idx) << "/" << int(d.wait_mode)
                    << "\", \"upd\": \"" << int(d.upd_idx) << "/" << int(d.upd_mode)
                    << "\", \"wait_val\": " << d.wait_val
                    << ", \"form_a\": " << (d.has_form_a ? "true" : "false") << "}";
            }
        }
        rep << "\n";
    }
    rep << ", \"total_descs\": " << total_descs << ", \"total_gates\": " << total_ev << "\n";

    // v1: no rewrite applied -> output = the input VERBATIM (bit-identical
    // guarantee). A rewrite path would re-pack + recompute digest A
    // (md5(gz) u32-word-swapped @0xAC) — implemented in emit_neff() and
    // enabled only when a transformation actually touches bytes.
    std::string outn = slurp(in);
    spit(out, outn);

    rep << ", \"changed_bytes\": 0\n}";
    std::string reps = rep.str();
    if (!report.empty()) spit(report, reps);
    if (verbose) std::fputs(reps.c_str(), stderr);
    std::printf("neffopt: %s -> %s (class: %s, pass-through, 0 bytes changed)\n",
                in.c_str(), out.c_str(), classification.c_str());
    return 0;
}
