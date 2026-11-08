// CPU stand-in for neuron::NeuronModel (no NeuronCores, no weights): host-side tests of neuron_engine.cpp and of the
// HTTP server (host_test.sh). The "K/V cache" holds the token at every paged position, and a row's logits are a
// deterministic function of the whole sequence read back through the page table: an induction head (the token that
// followed the latest earlier occurrence of the current token gets +8, so prompt-lookup drafts are often right) on
// top of hashed noise in [0, 1). A page the engine failed to fill or restore throws, so preemption and prefix-cache
// bugs change or abort the output instead of passing unnoticed.
//   BEHEMOTH_FAKE_VOCAB (default: config.json vocab_size, else 131072), BEHEMOTH_FAKE_PAGES (KV pages, 256),
//   BEHEMOTH_FAKE_ROWS (largest decode bucket rows, 8), BEHEMOTH_FAKE_CONTEXT (4096); EOS from config.json (else 2).
#include "neuron_model.hpp"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
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
}
NeuronModel::~NeuronModel() = default;
bool NeuronModel::recurrent() const { return env("BEHEMOTH_FAKE_RECURRENT", 0) != 0; }
int NeuronModel::cache_pages() const { return impl_->pages; }
int NeuronModel::vocab() const { return impl_->vocab; }
int NeuronModel::max_decode_rows() const { return impl_->rows; }
int NeuronModel::decode_bucket_pages(int pages) const { return pages; }
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

std::span<const float> NeuronModel::forward(const behemoth::InferenceBatch &batch) {
    auto &m = *impl_;
    const auto &l = batch.layout;
    ++m.steps;
    if (const int ms = env("BEHEMOTH_FAKE_STEP_MS", 0); ms > 0) // a slow device (stream keepalive tests)
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
