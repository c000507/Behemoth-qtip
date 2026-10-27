// Long-context decode attention check: GPU paged_attention vs scalar references.
//  ref_kernel : reference_attention(fp8 P, same partition)  -> kernel correctness
//  ref_exact  : double attention on the ORIGINAL float q/k/v (no FP8 at all) -> total FP8 error
#include "behemoth/gpu.hpp"
#include "behemoth/attention.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <cstring>
#include <cstdlib>
#include <string>

using namespace behemoth;

struct Case {
    AttentionLayout l;
    AttentionData a;
    std::vector<float> fq, fk, fv; // original floats, [row][d] and [loc][d]
    std::vector<BFloat16> qb;      // BF16 copies of the original queries (BF16 Q/P mode)
};

static Case make(std::vector<int> lengths, int seed, bool peaky, std::vector<int> queries) {
    Case c;
    auto &l = c.l;
    auto &a = c.a;
    l.query_heads = 48;
    l.kv_heads = 4;
    l.lengths = lengths;
    int maxlen = *std::max_element(lengths.begin(), lengths.end());
    l.table_stride = int(ceil_div(size_t(maxlen), page_tokens));
    l.physical_pages = 3;
    l.query_offsets = {0};
    for (size_t r = 0; r < lengths.size(); ++r) {
        const int len = lengths[r], nq = queries[std::min(r, queries.size() - 1)];
        l.physical_pages += int(ceil_div(size_t(len), page_tokens));
        for (int i = std::max(0, len - nq); i < len; ++i)
            l.positions.push_back(i); // decode: one query at the last position; prefill: trailing chunk
        l.query_offsets.push_back(int(l.positions.size()));
    }
    std::vector<int> pages(size_t(l.physical_pages));
    std::iota(pages.begin(), pages.end(), 0);
    std::mt19937 rng{unsigned(seed)};
    std::shuffle(pages.begin(), pages.end(), rng);
    l.page_table.assign(lengths.size() * size_t(l.table_stride), -999);
    size_t next = 0;
    for (size_t r = 0; r < lengths.size(); ++r)
        for (int t = 0; t < lengths[r]; t += page_tokens)
            l.page_table[r * size_t(l.table_stride) + size_t(t / page_tokens)] = pages[next++];
    size_t qs = size_t(l.queries()) * 48, ks = size_t(l.physical_pages) * 4 * page_tokens;
    a.q.resize(qs * head_dim);
    a.q_scale.resize(qs);
    a.k.assign(ks * head_dim, 127);
    a.v = a.k;
    a.k_scale.assign(ks, NAN);
    a.v_scale = a.k_scale;
    c.fq.resize(qs * head_dim);
    c.fk.assign(ks * head_dim, 0);
    c.fv.assign(ks * head_dim, 0);
    std::normal_distribution<float> normal(0, 1);
    std::array<float, head_dim> x{};
    auto put = [&](std::vector<uint8_t> &dst, std::vector<float> &sc, std::vector<float> &fl,
                   size_t i) {
        auto q = quantize(x);
        sc[i] = q.scale;
        std::copy(q.values.begin(), q.values.end(), dst.begin() + std::ptrdiff_t(i * head_dim));
        std::copy(x.begin(), x.end(), fl.begin() + std::ptrdiff_t(i * head_dim));
    };
    for (size_t i = 0; i < qs; ++i) {
        for (float &v : x) v = normal(rng);
        put(a.q, a.q_scale, c.fq, i);
    }
    c.qb.resize(c.fq.size());
    for (size_t i = 0; i < c.qb.size(); ++i) c.qb[i] = BFloat16(c.fq[i]);
    for (size_t r = 0; r < lengths.size(); ++r) {
        int len = lengths[r];
        std::uniform_int_distribution<int> pos(1, len - 1);
        std::vector<int> needles = {0, len / 3, len / 2, len - 3, pos(rng), pos(rng)};
        for (int t = 0; t < len; ++t)
            for (int h = 0; h < 4; ++h) {
                int p = l.page_table[r * size_t(l.table_stride) + size_t(t / page_tokens)];
                size_t loc = (size_t(p) * 4 + size_t(h)) * page_tokens + size_t(t % page_tokens);
                bool needle = peaky && std::find(needles.begin(), needles.end(), t) != needles.end();
                // peaky: needle keys aligned with query head (h*12) -> large scores for that group
                const float *qd = &c.fq[(size_t(l.query_offsets[r]) * 48 + size_t(h * 12)) * head_dim];
                for (int d = 0; d < head_dim; ++d)
                    x[size_t(d)] = needle ? (t == 0 ? 1.2f : 0.7f) * qd[d] + 0.3f * normal(rng)
                                          : normal(rng);
                put(a.k, a.k_scale, c.fk, loc);
                for (float &v : x) v = normal(rng) * (t % 3 == 0 ? .125f : 2.f);
                put(a.v, a.v_scale, c.fv, loc);
            }
    }
    return c;
}

static std::vector<float> exact(const Case &c) {
    const auto &l = c.l;
    std::vector<float> out(size_t(l.queries()) * 48 * head_dim);
    for (int r = 0; r < l.requests(); ++r)
        for (int q = l.query_offsets[r]; q < l.query_offsets[r + 1]; ++q)
#pragma omp parallel for schedule(static)
            for (int h = 0; h < 48; ++h) {
                size_t row = size_t(q) * 48 + size_t(h);
                int end = l.positions[size_t(q)] + 1, kh = h / 12;
                std::vector<double> s(static_cast<size_t>(end));
                double m = -INFINITY;
                std::vector<size_t> locs(static_cast<size_t>(end));
                for (int t = 0; t < end; ++t) {
                    int p = l.page_table[size_t(r) * size_t(l.table_stride) + size_t(t / page_tokens)];
                    size_t loc = (size_t(p) * 4 + size_t(kh)) * page_tokens + size_t(t % page_tokens);
                    locs[size_t(t)] = loc;
                    double d = 0;
                    for (int i = 0; i < head_dim; ++i)
                        d += double(c.fq[row * head_dim + size_t(i)]) * double(c.fk[loc * head_dim + size_t(i)]);
                    s[size_t(t)] = d / std::sqrt(double(head_dim));
                    m = std::max(m, s[size_t(t)]);
                }
                double sum = 0;
                std::array<double, head_dim> o{};
                for (int t = 0; t < end; ++t) {
                    double e = std::exp(s[size_t(t)] - m);
                    sum += e;
                    for (int i = 0; i < head_dim; ++i)
                        o[size_t(i)] += e * double(c.fv[locs[size_t(t)] * head_dim + size_t(i)]);
                }
                for (int i = 0; i < head_dim; ++i)
                    out[row * head_dim + size_t(i)] = float(o[size_t(i)] / sum);
            }
    return out;
}

static double relrms(const std::vector<float> &a, const std::vector<float> &b, double *maxabs) {
    double se = 0, sr = 0, mx = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double e = double(a[i]) - double(b[i]);
        se += e * e;
        sr += double(b[i]) * double(b[i]);
        mx = std::max(mx, std::abs(e));
        if (!std::isfinite(a[i])) return INFINITY;
    }
    if (maxabs) *maxabs = mx;
    return std::sqrt(se / std::max(sr, 1e-30));
}

int main(int argc, char **argv) {
    HIP_CHECK(hipSetDevice(0));
    hipStream_t stream{};
    HIP_CHECK(hipStreamCreate(&stream));
    const int reps = argc > 1 ? std::atoi(argv[1]) : 5;
    struct Spec { std::vector<int> lengths; bool peaky; int queries = 1; std::vector<int> per; };
    std::vector<Spec> specs = {
        {{2048}, false}, {{2048}, true}, {{8192}, true}, {{13000}, false}, {{13000}, true},
        {{16384}, true}, {{32768}, true}, {{40000}, true}, {{13000, 517, 8191, 2049}, true},
        {{2048, 2048, 2048, 2048}, true}, {{90}, false}, {{90, 90, 90, 90}, false},
        {{2048}, true, 256}, {{6144}, true, 256}};
    if (argc > 2) { // single spec: lengths "a,b,c" [queries]
        specs.clear();
        Spec s; s.peaky = true;
        for (char *tok = std::strtok(argv[2], ","); tok; tok = std::strtok(nullptr, ",")) s.lengths.push_back(std::atoi(tok));
        s.queries = argc > 3 ? std::atoi(argv[3]) : 1;
        // LA_QUERIES="2048,1,256": queries per request (last value repeats)
        if (const char *v = std::getenv("LA_QUERIES")) {
            std::string list = v;
            for (char *tok = std::strtok(list.data(), ","); tok; tok = std::strtok(nullptr, ","))
                s.per.push_back(std::atoi(tok));
            s.queries = *std::max_element(s.per.begin(), s.per.end());
        }
        specs.push_back(s);
    }
    // A/B against another build or kernel (e.g. BEHEMOTH_ATTN_PREFILL_V1=1) without the slow CPU
    // references: LA_NOREF=1 skips them, LA_DUMP=file writes every GPU output, LA_CMP=file
    // compares every output with the dumped one (rel rms, max abs), LA_PARTS="0,512" sets the
    // multi-query key partitions (default 0).
    const bool noref = std::getenv("LA_NOREF") && std::atoi(std::getenv("LA_NOREF"));
    FILE *dump = std::getenv("LA_DUMP") ? std::fopen(std::getenv("LA_DUMP"), "wb") : nullptr;
    FILE *cmp = std::getenv("LA_CMP") ? std::fopen(std::getenv("LA_CMP"), "rb") : nullptr;
    std::vector<int> multi_parts{0};
    if (const char *v = std::getenv("LA_PARTS")) {
        multi_parts.clear();
        std::string list = v;
        for (char *tok = std::strtok(list.data(), ","); tok; tok = std::strtok(nullptr, ","))
            multi_parts.push_back(std::atoi(tok));
    }
    std::printf("%-26s %5s %5s %6s %10s %10s %10s %10s %9s\n", "lengths", "peaky", "mode", "part",
                "vsKernRef", "maxabs", "vsExact", "refVsEx", "gpu_ms");
    for (auto &sp : specs) {
        auto c = make(sp.lengths, 11, sp.peaky, sp.per.empty() ? std::vector<int>{sp.queries} : sp.per);
        auto ex = noref ? std::vector<float>{} : exact(c);
        GpuMetadata md;
        md.upload(c.l);
        DeviceBuffer<uint8_t> q, k, v;
        DeviceBuffer<float> qs, ks, vs;
        DeviceBuffer<BFloat16> qb;
        q.upload(c.a.q); k.upload(c.a.k); v.upload(c.a.v);
        qs.upload(c.a.q_scale); ks.upload(c.a.k_scale); vs.upload(c.a.v_scale);
        qb.upload(c.qb);
        size_t rows = size_t(c.l.queries()) * 48;
        DeviceBuffer<BFloat16> out(rows * head_dim);
        std::string ls;
        for (int x : sp.lengths) ls += std::to_string(x) + ",";
        if (!sp.per.empty()) {
            ls += " q";
            for (int x : sp.per) ls += std::to_string(x) + ",";
        } else if (sp.queries > 1)
            ls += " q" + std::to_string(sp.queries);
        for (int bf16 = 0; bf16 < 2; ++bf16) {
            AttentionData ref_data = c.a;
            if (bf16) ref_data.q_bf16 = c.qb;
            const int automatic = sp.queries > 1 ? 0 : decode_attention_partition(c.l, 32u << 20);
            std::vector<int> parts = sp.queries > 1 ? multi_parts : std::vector<int>{automatic, 0, 32, 1024};
            for (int part : parts) {
                DeviceBuffer<std::byte> ws(std::max<size_t>(attention_workspace_bytes(c.l, part), 1));
                AttentionDeviceView view{q.data(), qs.data(), k.data(), v.data(), ks.data(), vs.data(), out.data()};
                if (bf16) view.q_bf16 = qb.data();
                paged_attention(md, view, ws.data(), ws.size(), part, stream);
                HIP_CHECK(hipStreamSynchronize(stream));
                auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < reps; ++i)
                    paged_attention(md, view, ws.data(), ws.size(), part, stream);
                HIP_CHECK(hipStreamSynchronize(stream));
                double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
                auto got = out.download();
                std::vector<float> g(got.size());
                for (size_t i = 0; i < g.size(); ++i) g[i] = float(got[i]);
                bool finite = true;
                for (float x : g) finite = finite && std::isfinite(x);
                if (dump) std::fwrite(g.data(), sizeof(float), g.size(), dump);
                if (cmp) {
                    std::vector<float> other(g.size());
                    if (std::fread(other.data(), sizeof(float), other.size(), cmp) != other.size()) {
                        std::printf("LA_CMP: file too short\n");
                        return 1;
                    }
                    double cm = 0, ce = relrms(g, other, &cm);
                    std::printf("%-26s %5s %6d  vsOther %10.2e maxabs %10.2e finite %d %9.3f ms\n", ls.c_str(),
                                bf16 ? "bf16" : "fp8", part, ce, cm, int(finite), ms);
                    std::fflush(stdout);
                }
                if (noref) {
                    if (!cmp)
                        std::printf("%-26s %5s %6d  finite %d %9.3f ms\n", ls.c_str(), bf16 ? "bf16" : "fp8",
                                    part, int(finite), ms);
                    continue;
                }
                auto ref = reference_attention(c.l, ref_data, !bf16, part).output;
                double mx = 0;
                double e1 = relrms(g, ref, &mx), e2 = relrms(g, ex, nullptr), e3 = relrms(ref, ex, nullptr);
                std::printf("%-26s %5d %5s %6d%s %10.2e %10.2e %10.2e %10.2e %9.3f\n", ls.c_str(), int(sp.peaky),
                            bf16 ? "bf16" : "fp8", part, part == automatic ? "*" : " ", e1, mx, e2, e3, ms);
                std::fflush(stdout);
            }
        }
    }
    if (dump) std::fclose(dump);
    if (cmp) std::fclose(cmp);
    return 0;
}
