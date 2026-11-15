// CPU stand-in for neuron::NeuronModel (no NeuronCores, no weights): host-side tests of neuron_engine.cpp and of the
// HTTP server (host_test.sh). The "K/V cache" holds the token at every paged position, and a row's logits are a
// deterministic function of the whole sequence read back through the page table: an induction head (the token that
// followed the latest earlier occurrence of the current token gets +8, so prompt-lookup drafts are often right) on
// top of hashed noise in [0, 1). A page the engine failed to fill or restore throws, so preemption and prefix-cache
// bugs change or abort the output instead of passing unnoticed.
// BEHEMOTH_FAKE_RECURRENT=1: a recurrent family (Qwen3.5's gated-delta layers). Every sequence (identified by its first
// page) then also has a recurrent state = (tokens absorbed, their rolling hash), advanced token by token, zeroed at
// position 0; a token whose position is not the state's count, or whose state hash differs from the hash of the
// tokens read back through the page table, throws: a prefix restore at a position without the matching state
// snapshot, a stale state, or a snapshot taken at the wrong position cannot pass unnoticed.
//   BEHEMOTH_FAKE_VOCAB (default: config.json vocab_size, else 131072), BEHEMOTH_FAKE_PAGES (KV pages, 256),
//   BEHEMOTH_FAKE_ROWS (largest decode bucket rows, 8), BEHEMOTH_FAKE_CONTEXT (4096); EOS from config.json (else 2).
#include "neuron_model.hpp"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <thread>

namespace neuron {
namespace {
int env(const char *name, int fallback) {
    const char *v = std::getenv(name);
    return v ? std::atoi(v) : fallback;
}
uint64_t mix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
} // namespace

struct NeuronModel::Impl {
    int vocab = 131072, pages = 256, rows = 8, context = 4096;
    std::vector<int> eos{2};
    std::vector<int> kv; // token at (page, slot), -1: never written
    std::vector<float> logits;
    uint64_t steps = 0;
    struct State {
        uint64_t count = 0, hash = 0;
    };
    std::map<int, State> state; // recurrent mode: per sequence (first page)
    bool recurrent = false;
    int step_ms = 0;
};

NeuronModel::NeuronModel(const ModelOptions &o) : impl_(std::make_unique<Impl>()) {
    auto &m = *impl_;
    if (std::ifstream f(o.model / "config.json"); f) {
        const auto c = nlohmann::json::parse(f);
        m.vocab = c.value("vocab_size", m.vocab);
        if (c.contains("eos_token_id"))
            m.eos = c.at("eos_token_id").is_array() ? c.at("eos_token_id").get<std::vector<int>>()
                                                     : std::vector<int>{c.at("eos_token_id").get<int>()};
    }
    m.vocab = env("BEHEMOTH_FAKE_VOCAB", m.vocab);
    m.pages = env("BEHEMOTH_FAKE_PAGES", m.pages);
    m.rows = env("BEHEMOTH_FAKE_ROWS", m.rows);
    m.context = env("BEHEMOTH_FAKE_CONTEXT", m.context);
    m.kv.assign(size_t(m.pages + 1) * 16, -1);
    m.recurrent = env("BEHEMOTH_FAKE_RECURRENT", 0) != 0;
    m.step_ms = env("BEHEMOTH_FAKE_STEP_MS", 0);
}
NeuronModel::~NeuronModel() = default;
bool NeuronModel::recurrent() const { return impl_->recurrent; }
// BEHEMOTH_FAKE_STATE_SNAPSHOTS=0: a recurrent family without snapshot support (GLM): the engine keeps the cache off
size_t NeuronModel::state_bytes() const { return impl_->recurrent && env("BEHEMOTH_FAKE_STATE_SNAPSHOTS", 1) ? sizeof(Impl::State) : 0; }
void NeuronModel::state_save(int first_page, std::byte *dst) {
    auto it = impl_->state.find(first_page);
    if (it == impl_->state.end())
        throw std::runtime_error("fake model: no recurrent state for the sequence on page " + std::to_string(first_page));
    std::memcpy(dst, &it->second, sizeof(Impl::State));
}
void NeuronModel::state_restore(int first_page, const std::byte *src) { std::memcpy(&impl_->state[first_page], src, sizeof(Impl::State)); }
void NeuronModel::state_drop(int first_page) { impl_->state.erase(first_page); }
int NeuronModel::cache_pages() const { return impl_->pages; }
int NeuronModel::vocab() const { return impl_->vocab; }
int NeuronModel::max_decode_rows() const { return impl_->rows; }
int NeuronModel::decode_bucket_pages(int pages) const { return pages; }
bool NeuronModel::has_shared_verify() const { return false; }
const std::vector<int> &NeuronModel::eos() const { return impl_->eos; }
int NeuronModel::max_context() const { return impl_->context; }
nlohmann::json NeuronModel::last_profile() const { return {{"fake_steps", impl_->steps}}; }
size_t NeuronModel::kv_page_bytes() const { return 16 * sizeof(int); }
void NeuronModel::kv_save(std::span<const int> device, std::span<const int> host, std::byte *arena) {
    for (size_t i = 0; i < device.size(); ++i)
        std::memcpy(arena + size_t(host[i]) * kv_page_bytes(), impl_->kv.data() + size_t(device[i]) * 16, kv_page_bytes());
}
void NeuronModel::kv_restore(std::span<const int> host, std::span<const int> device, const std::byte *arena) {
    for (size_t i = 0; i < device.size(); ++i)
        std::memcpy(impl_->kv.data() + size_t(device[i]) * 16, arena + size_t(host[i]) * kv_page_bytes(), kv_page_bytes());
}

bool NeuronModel::device_argmax_used() const { return false; }
int NeuronModel::device_token(size_t) const { return -1; }
int NeuronModel::mtp_max_drafts() const { return 0; }
int NeuronModel::mtp_max_drafts(int, int) const { return 0; }
std::vector<int> NeuronModel::mtp_draft(std::span<const int>, int, std::span<const int>, int) { return {}; }

std::span<const float> NeuronModel::forward(const behemoth::InferenceBatch &batch, bool) {
    auto &m = *impl_;
    const auto &l = batch.layout;
    ++m.steps;
    if (const int ms = m.step_ms; ms > 0) // a slow device (stream keepalive tests)
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    std::vector<int> request_of(batch.tokens.size());
    auto slot = [&](int r, int position) -> int & {
        const int page = l.page_table.at(size_t(r) * size_t(l.table_stride) + size_t(position / 16));
        if (page < 0 || page > m.pages)
            throw std::runtime_error("fake model: position " + std::to_string(position) + " has no page");
        return m.kv[size_t(page) * 16 + size_t(position % 16)];
    };
    for (int r = 0; r < l.requests(); ++r)
        for (int q = l.query_offsets[size_t(r)]; q < l.query_offsets[size_t(r) + 1]; ++q) {
            request_of[size_t(q)] = r;
            slot(r, l.positions[size_t(q)]) = batch.tokens[size_t(q)];
            if (m.recurrent) {
                const int position = l.positions[size_t(q)], seq = l.page_table.at(size_t(r) * size_t(l.table_stride));
                if (position == 0)
                    m.state[seq] = {};
                auto st = m.state.find(seq);
                if (st == m.state.end() || st->second.count != uint64_t(position))
                    throw std::runtime_error("fake model: recurrent state of the sequence on page " + std::to_string(seq) + " is " +
                                             (st == m.state.end() ? std::string("missing") : "at " + std::to_string(st->second.count)) +
                                             ", token at position " + std::to_string(position));
                st->second.hash = mix(st->second.hash ^ uint64_t(batch.tokens[size_t(q)]));
                ++st->second.count;
            }
        }
    const size_t v = size_t(m.vocab);
    m.logits.resize(batch.sample_rows.size() * v);
    for (size_t s = 0; s < batch.sample_rows.size(); ++s) {
        const int q = batch.sample_rows[s], r = request_of[size_t(q)], position = l.positions[size_t(q)];
        uint64_t h = 0;
        std::vector<int> seq(size_t(position) + 1);
        for (int p = 0; p <= position; ++p) {
            seq[size_t(p)] = slot(r, p);
            if (seq[size_t(p)] < 0)
                throw std::runtime_error("fake model: K/V of position " + std::to_string(p) + " missing");
            h = mix(h ^ uint64_t(seq[size_t(p)]));
        }
        if (m.recurrent && q + 1 == l.query_offsets[size_t(r) + 1]) { // the request's last token: the state is the state after it
            const auto &st = m.state.at(l.page_table.at(size_t(r) * size_t(l.table_stride)));
            if (st.count != uint64_t(position) + 1 || st.hash != h)
                throw std::runtime_error("fake model: recurrent state does not match the sequence at position " + std::to_string(position));
        }
        float *row = m.logits.data() + s * v;
        for (size_t t = 0; t < v; ++t)
            row[t] = float(mix(h + t) >> 40) / float(1 << 24);
        for (int p = position - 1; p >= 0; --p)
            if (seq[size_t(p)] == seq[size_t(position)]) {
                if (size_t(seq[size_t(p) + 1]) < v)
                    row[seq[size_t(p) + 1]] += 8;
                break;
            }
    }
    return m.logits;
}
} // namespace neuron
