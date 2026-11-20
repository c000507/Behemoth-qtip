// Determinism stress for one tensor-parallel NEFF with aliased state buffers (the 1-row gdn nonfinite-fault study,
// docs/neuron-reap-nonfinite.md). Loads the NEFF on every rank like layer_bench, binds the state outputs to the state inputs
// (in place, as the engine does), then repeats: restore one state slot from a saved copy, execute, read the residual output
// and that slot back, compare bit-for-bit with the first iteration. Identical inputs => identical outputs unless the NEFF
// execution itself races.
//
// build: g++ -std=c++20 -O2 -pthread -I/opt/aws/neuron/include neuron/cpp/neff_race.cpp \
//            -L/opt/aws/neuron/lib -lnrt -Wl,-rpath,/opt/aws/neuron/lib -o neff_race
// run:   NEURON_RT_VISIBLE_CORES=0-1 NEURON_RT_ROOT_COMM_ID=127.0.0.1:62128 neff_race graph.neff ITERS STATES [peer.neff]
//   env  RACE_SLOT (state slot the int64 slot input points to, default 1), RACE_POS (position input value, default 5: keep = 1),
//        RACE_SLOTS (slots per state tensor, default 9), RACE_PEER_EVERY (peer launches between two launches, default 1),
//        RACE_NORESET=1 (do not restore the slot: the state evolves; then only nonfinite values are counted),
//        RACE_ROWS=R (the NEFF's decode rows: slots SLOT .. SLOT + R - 1 are used and checked), RACE_PEER_NAN=1 (peer f32 inputs NaN),
//        RACE_SPLIT=1 (bind the state outputs to SEPARATE tensors, copy them back to the inputs on the host after each
//        launch: tests whether the in-place aliasing matters), RACE_INPUTS=DIR (raw inputs DIR/in<i>.r<rank>, else a pattern).
#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>

#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

static void check(NRT_STATUS s, const char* what, int rank) {
    if (s != NRT_SUCCESS) { std::fprintf(stderr, "rank %d: %s failed: status %d\n", rank, what, int(s)); std::exit(2); }
}
static int envi(const char* n, int d) { const char* v = std::getenv(n); return v ? std::atoi(v) : d; }

struct Model {
    nrt_model_t* model = nullptr;
    nrt_tensor_info_array_t* info = nullptr;
    nrt_tensor_set_t *in = nullptr, *out = nullptr;
    std::vector<nrt_tensor_t*> state_in, state_out;      // per state
    std::vector<size_t> state_bytes;
    nrt_tensor_t* x_out = nullptr;
    size_t x_bytes = 0;
};

static std::vector<char> slurp(const char* p) {
    std::ifstream f(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

static void load(Model& m, const std::vector<char>& neff, int rank, int ranks, int n_states, bool split, uint32_t seed) {
    check(nrt_load_collectives(neff.data(), neff.size(), rank, -1, uint32_t(rank), uint32_t(ranks), &m.model), "load", rank);
    check(nrt_get_model_tensor_info(m.model, &m.info), "info", rank);
    check(nrt_allocate_tensor_set(&m.in), "set", rank);
    check(nrt_allocate_tensor_set(&m.out), "set", rank);
    int n_inputs = 0;
    for (uint64_t i = 0; i < m.info->tensor_count; ++i) n_inputs += m.info->tensor_array[i].usage == NRT_TENSOR_USAGE_INPUT;
    m.state_in.assign(n_states, nullptr); m.state_out.assign(n_states, nullptr); m.state_bytes.assign(n_states, 0);
    const char* dir = std::getenv("RACE_INPUTS");
    for (int pass = 0; pass < 2; ++pass)
        for (uint64_t i = 0; i < m.info->tensor_count; ++i) {
            const nrt_tensor_info_t& t = m.info->tensor_array[i];
            const bool is_in = t.usage == NRT_TENSOR_USAGE_INPUT;
            if (is_in != (pass == 0)) continue;
            const int idx = std::atoi(t.name + (is_in ? 5 : 6));
            const int sidx = is_in ? idx - (n_inputs - n_states) : idx - 1;
            nrt_tensor_t* dev = nullptr;
            if (!is_in && sidx >= 0 && sidx < n_states && !split) dev = m.state_in[sidx];
            else check(nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, rank, t.size, t.name, &dev), "alloc", rank);
            if (is_in) {
                std::vector<uint8_t> fill(t.size, 0);
                bool from_file = false;
                if (dir) {
                    std::ifstream f(std::string(dir) + "/in" + std::to_string(idx) + ".r" + std::to_string(rank), std::ios::binary);
                    if (f) { f.read((char*)fill.data(), std::streamsize(fill.size())); from_file = size_t(f.gcount()) == fill.size(); }
                }
                if (from_file) {
                } else if (int(t.dtype) == 8) {                      // int64: position / slot indices
                    std::vector<int64_t> v(t.size / 8, idx == 1 ? envi("RACE_POS", 5) : envi("RACE_SLOT", 1));
                    if (idx != 1) for (size_t k = 0; k < v.size(); ++k) v[k] += int64_t(k);   // R rows -> slots SLOT .. SLOT + R - 1
                    std::memcpy(fill.data(), v.data(), t.size);
                } else if (sidx >= 0 && sidx < n_states) {           // f32 state: finite values in [-0.5, 0.5)
                    std::vector<float> v(t.size / 4);
                    for (size_t k = 0; k < v.size(); ++k) v[k] = float(((k * 2654435761u + seed + rank) >> 8) & 0xffff) / 65536.f - 0.5f;
                    std::memcpy(fill.data(), v.data(), t.size);
                } else if (seed && envi("RACE_PEER_NAN", 0)) {   // peer f32 / fp8 inputs all NaN (0xff): stale on-chip bytes become NaN
                    std::memset(fill.data(), 0xff, fill.size());
                } else {
                    for (size_t k = 0; k < fill.size(); ++k) fill[k] = uint8_t(((k * 2654435761u + seed + rank) >> 24) & 0x3f);
                }
                check(nrt_tensor_write(dev, fill.data(), 0, fill.size()), "write", rank);
                check(nrt_add_tensor_to_tensor_set(m.in, t.name, dev), "add in", rank);
                if (sidx >= 0 && sidx < n_states) { m.state_in[sidx] = dev; m.state_bytes[sidx] = t.size; }
            } else {
                check(nrt_add_tensor_to_tensor_set(m.out, t.name, dev), "add out", rank);
                if (sidx >= 0 && sidx < n_states) m.state_out[sidx] = dev;
                if (idx == 0) { m.x_out = dev; m.x_bytes = t.size; }
            }
        }
}

struct Stats { long diff_iters = 0, nonfinite_iters = 0, x_diff = 0, s_diff = 0; double secs = 0; };

static void run(int rank, int ranks, const char* path, const char* peer_path, int iters, int n_states, std::barrier<>& sync, Stats& st) {
    const bool split = envi("RACE_SPLIT", 0), noreset = envi("RACE_NORESET", 0);
    const int sleep_us = envi("RACE_SLEEP_US", 0);   // pause after the peer launch(es), before the next main launch
    const int slot = envi("RACE_SLOT", 1), slots = envi("RACE_SLOTS", 9), peer_every = envi("RACE_PEER_EVERY", 1);
    auto neff = slurp(path);
    Model m, p;
    load(m, neff, rank, ranks, n_states, split, 0);
    if (peer_path) { auto pn = slurp(peer_path); load(p, pn, rank, ranks, 0, false, 77); }
    std::vector<std::vector<uint8_t>> init(n_states), ref(n_states), cur(n_states);
    std::vector<size_t> off(n_states), len(n_states);
    for (int s = 0; s < n_states; ++s) {
        len[s] = m.state_bytes[s] / slots; off[s] = len[s] * slot; len[s] *= envi("RACE_ROWS", 1);
        init[s].resize(len[s]); cur[s].resize(len[s]);
        check(nrt_tensor_read(m.state_in[s], init[s].data(), off[s], len[s]), "read init", rank);
    }
    std::vector<uint8_t> xref(m.x_bytes), x(m.x_bytes);
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) {
        if (!noreset)
            for (int s = 0; s < n_states; ++s) check(nrt_tensor_write(m.state_in[s], init[s].data(), off[s], len[s]), "reset", rank);
        check(nrt_execute(m.model, m.in, m.out), "exec", rank);
        if (peer_path) for (int k = 0; k < peer_every; ++k) check(nrt_execute(p.model, p.in, p.out), "exec peer", rank);
        if (sleep_us) std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
        check(nrt_tensor_read(m.x_out, x.data(), 0, m.x_bytes), "read x", rank);
        for (int s = 0; s < n_states; ++s) {
            check(nrt_tensor_read(m.state_out[s], cur[s].data(), off[s], len[s]), "read s", rank);
            if (split) check(nrt_tensor_write(m.state_in[s], cur[s].data(), off[s], len[s]), "copy back", rank);
        }
        long nf = 0;
        for (size_t k = 0; k < x.size() / 4; ++k) nf += !std::isfinite(((float*)x.data())[k]);
        for (int s = 0; s < n_states; ++s) for (size_t k = 0; k < cur[s].size() / 4; ++k) nf += !std::isfinite(((float*)cur[s].data())[k]);
        if (nf) ++st.nonfinite_iters;
        if (it == 0) { xref = x; for (int s = 0; s < n_states; ++s) ref[s] = cur[s]; continue; }
        if (noreset) { if (nf && st.nonfinite_iters <= 5) std::printf("rank %d iter %d: %ld nonfinite\n", rank, it, nf); continue; }
        long dx = 0, ds = 0; float mx = 0; size_t first = SIZE_MAX; int fs = -1;
        for (size_t k = 0; k < x.size() / 4; ++k) {
            float a = ((float*)x.data())[k], b = ((float*)xref.data())[k];
            if (std::memcmp(&a, &b, 4)) { ++dx; mx = std::max(mx, std::fabs(a - b)); }
        }
        for (int s = 0; s < n_states; ++s)
            for (size_t k = 0; k < cur[s].size() / 4; ++k)
                if (std::memcmp(cur[s].data() + 4 * k, ref[s].data() + 4 * k, 4)) { if (first == SIZE_MAX) { first = k; fs = s; } ++ds; }
        if (dx || ds) {
            ++st.diff_iters; st.x_diff += dx; st.s_diff += ds;
            if (st.diff_iters <= 8) {
                float got = fs >= 0 ? ((float*)cur[fs].data())[first] : 0, want = fs >= 0 ? ((float*)ref[fs].data())[first] : 0;
                std::printf("rank %d iter %d: x diff %ld (max |d| %g), state diff %ld (first: state %d elem %zu got %g want %g), nonfinite %ld\n",
                            rank, it, dx, mx, ds, fs, first == SIZE_MAX ? 0 : first, got, want, nf);
                std::fflush(stdout);
            }
        }
    }
    st.secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    sync.arrive_and_wait();
}

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s graph.neff ITERS STATES [peer.neff]\n", argv[0]); return 1; }
    const int iters = std::atoi(argv[2]), n_states = std::atoi(argv[3]), ranks = envi("RACE_RANKS", 2);
    const char* peer = argc > 4 ? argv[4] : nullptr;
    check(nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", ""), "nrt_init", -1);
    std::barrier<> sync(ranks);
    std::vector<Stats> st(ranks);
    std::vector<std::thread> th;
    for (int r = 0; r < ranks; ++r) th.emplace_back(run, r, ranks, argv[1], peer, iters, n_states, std::ref(sync), std::ref(st[r]));
    for (auto& t : th) t.join();
    for (int r = 0; r < ranks; ++r)
        std::printf("RESULT %s rank %d iters %d diff_iters %ld nonfinite_iters %ld x_diff %ld s_diff %ld ms/iter %.3f\n", argv[1], r, iters,
                    st[r].diff_iters, st[r].nonfinite_iters, st[r].x_diff, st[r].s_diff, st[r].secs * 1e3 / iters);
    nrt_close();
    return 0;
}
