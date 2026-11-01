#include "behemoth/constraint.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <iostream>

namespace behemoth {
namespace {
bool is_space(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}
constexpr std::string_view open_tag = "<think>", close_tag = "</think>";
void set_bit(TokenMask &m, size_t t) {
    m[t >> 6] |= uint64_t(1) << (t & 63);
}
} // namespace

// ---------------------------------------------------------------------------
// Vocabulary
// ---------------------------------------------------------------------------
Vocabulary::Vocabulary(std::vector<Token> tokens) : tokens_(std::move(tokens)) {
    if (tokens_.size() != size_t(vocabulary_size))
        throw std::invalid_argument("Vocabulary must have 32768 tokens");
    nodes_.push_back(Node{});
    same_.assign(tokens_.size(), -1);
    for (size_t id = 0; id < tokens_.size(); ++id) {
        const Token &t = tokens_[id];
        if (t.special || t.rest.empty())
            continue;
        if (t.rest.find('>') != std::string::npos)
            close_candidates_.push_back(int(id));
        int32_t cur = 0;
        for (unsigned char b : t.rest) {
            int32_t c = nodes_[size_t(cur)].first_child;
            while (c >= 0 && nodes_[size_t(c)].byte != b)
                c = nodes_[size_t(c)].next_sibling;
            if (c < 0) {
                Node n;
                n.byte = b;
                n.next_sibling = nodes_[size_t(cur)].first_child;
                c = int32_t(nodes_.size());
                nodes_.push_back(n);
                nodes_[size_t(cur)].first_child = c;
            }
            cur = c;
        }
        same_[id] = nodes_[size_t(cur)].token;
        nodes_[size_t(cur)].token = int32_t(id);
    }
}

// ---------------------------------------------------------------------------
// ThinkSplitter
// ---------------------------------------------------------------------------
bool ThinkSplitter::feed(std::string_view text, std::string *reasoning, std::string *answer, bool finish) {
    auto out = [](std::string *s, std::string_view v) {
        if (s)
            s->append(v);
    };
    for (char ch : text) {
        const auto c = uint8_t(ch);
        switch (phase) {
        case Lead:
            if (held.empty() && is_space(c)) {
                if (++whitespace > max_lead_whitespace)
                    return false;
                break;
            }
            held.push_back(ch);
            if (open_tag.starts_with(held)) {
                if (held.size() == open_tag.size()) {
                    phase = Think;
                    depth = 1;
                    held.clear();
                }
                break;
            }
            phase = Answer;
            out(answer, held);
            held.clear();
            break;
        case AnswerLead:
            if (is_space(c)) {
                if (++whitespace > max_lead_whitespace)
                    return false;
                break;
            }
            phase = Answer;
            [[fallthrough]];
        case Answer:
            if (answer)
                answer->push_back(ch);
            break;
        case Think: {
            held.push_back(ch);
            std::string_view h(held);
            if (h.ends_with(open_tag) || h.ends_with(close_tag)) {
                const bool opening = h.ends_with(open_tag);
                out(reasoning, h.substr(0, h.size() - (opening ? open_tag : close_tag).size()));
                held.clear();
                if (opening) {
                    if (depth < 255)
                        ++depth;
                } else if (--depth == 0) {
                    phase = AnswerLead;
                    whitespace = 0;
                }
                break;
            }
            // Keep the longest suffix that may still become a tag.
            size_t keep = 0;
            for (auto tag : {open_tag, close_tag})
                for (size_t n = std::min(tag.size() - 1, h.size()); n > keep; --n)
                    if (h.substr(h.size() - n) == tag.substr(0, n)) {
                        keep = n;
                        break;
                    }
            if (keep < held.size()) {
                out(reasoning, h.substr(0, held.size() - keep));
                held.erase(0, held.size() - keep);
            }
            break;
        }
        }
    }
    if (finish && !held.empty()) {
        out(phase == Think ? reasoning : answer, held);
        held.clear();
    }
    return true;
}

// ---------------------------------------------------------------------------
// Grammar engines and cache
// ---------------------------------------------------------------------------
GrammarEngine::GrammarEngine(std::string gbnf, bool json)
    : text(std::move(gbnf)), grammar(grammar::ParseGbnf(text)), matcher(grammar), json_mode(json) {}

void GrammarCache::check(const std::string &gbnf) {
    grammar::Grammar g = grammar::ParseGbnf(gbnf);
    grammar::Matcher m(g); // initial expansion may also reject
    (void)m;
}

std::unique_ptr<GrammarEngine> GrammarCache::acquire(const std::string &gbnf, bool json) {
    {
        std::lock_guard lock(mutex_);
        auto it = free_.find(gbnf);
        if (it != free_.end()) {
            for (auto e = it->second.begin(); e != it->second.end(); ++e)
                if ((*e)->json_mode == json) {
                    auto engine = std::move(*e);
                    it->second.erase(e);
                    return engine;
                }
        }
    }
    return std::make_unique<GrammarEngine>(gbnf, json);
}

void GrammarCache::release(std::unique_ptr<GrammarEngine> engine) {
    // Engines whose memo grew very large are dropped rather than kept warm.
    if (!engine || engine->matcher.num_states() > 200000 || engine->masks.size() > 4096)
        return;
    std::lock_guard lock(mutex_);
    const std::string key = engine->text;
    auto &slot = free_[key];
    if (slot.size() < 4)
        slot.push_back(std::move(engine));
    order_.erase(std::remove(order_.begin(), order_.end(), key), order_.end());
    order_.push_back(key);
    while (order_.size() > max_grammars_) {
        free_.erase(order_.front());
        order_.erase(order_.begin());
    }
}

// ---------------------------------------------------------------------------
// StructuredConstraint
// ---------------------------------------------------------------------------
StructuredConstraint::StructuredConstraint(const Vocabulary &vocab, std::unique_ptr<GrammarEngine> engine,
                                           bool allow_think, GrammarCache *cache)
    : vocab_(vocab), engine_(std::move(engine)), cache_(cache),
      cur_{ThinkSplitter(allow_think), engine_->matcher.initial_state(), {}, true} {}

StructuredConstraint::~StructuredConstraint() {
    if (cache_ && engine_)
        cache_->release(std::move(engine_));
}

bool StructuredConstraint::accepting(int sid, const grammar::PartialUtf8 &p) const {
    return p.n_remain == 0 && engine_->matcher.Accepting(sid);
}

bool StructuredConstraint::complete() const {
    return cur_.split.phase == ThinkSplitter::Answer && accepting(cur_.sid, cur_.partial);
}

bool StructuredConstraint::step_answer_bytes(State &st, std::string_view bytes) {
    auto &m = engine_->matcher;
    for (char ch : bytes) {
        // JSON mode: nothing (in particular no whitespace) follows a complete root value.
        if (engine_->json_mode && is_space(uint8_t(ch)) && accepting(st.sid, st.partial))
            return false;
        if (!m.StepBytes(st.sid, st.partial, &ch, 1))
            return false;
    }
    return true;
}

bool StructuredConstraint::step(State &st, int token) {
    if (token < 0 || token >= vocabulary_size)
        return false;
    const auto &t = vocab_.token(token);
    if (t.special) {
        if (token == eos_token) // only a complete answer may end
            return (st.split.phase == ThinkSplitter::Answer || st.split.phase == ThinkSplitter::AnswerLead) &&
                   accepting(st.sid, st.partial);
        return st.split.phase == ThinkSplitter::Think; // control tokens: reasoning only
    }
    const std::string &piece = st.first ? t.first : t.rest;
    if (st.first)
        st.first = t.first_after;
    if (piece.empty()) // emits nothing: only harmless inside reasoning
        return st.split.phase == ThinkSplitter::Think;
    if (st.split.phase == ThinkSplitter::Answer)
        return step_answer_bytes(st, piece);
    if (st.split.phase == ThinkSplitter::Think && piece.find('>') == std::string::npos &&
        piece.find('<') == std::string::npos && st.split.held.empty())
        return true; // plain reasoning text: no tag can start or end here
    std::string answer;
    if (!st.split.feed(piece, nullptr, &answer))
        return false;
    return answer.empty() || step_answer_bytes(st, answer);
}

bool StructuredConstraint::accepts(int token) {
    if (dead_)
        return token == eos_token;
    State copy = cur_;
    try {
        return step(copy, token);
    } catch (const std::exception &e) {
        std::cerr << "[structured] grammar error: " << e.what() << '\n';
        dead_ = true;
        return token == eos_token;
    }
}

bool StructuredConstraint::accept_first(bool greedy) const {
    return greedy || cur_.split.phase == ThinkSplitter::Think;
}

void StructuredConstraint::scan_all(TokenMask &mask) {
    ++stats_.scans;
    if (cur_.split.phase == ThinkSplitter::Think) {
        // Everything except EOS, minus '>' tokens that would close the block
        // into text the grammar rejects.
        mask.fill(~uint64_t(0));
        mask[size_t(eos_token) >> 6] &= ~(uint64_t(1) << (eos_token & 63));
        for (int id : vocab_.close_candidates()) {
            State copy = cur_;
            if (!step(copy, id))
                mask[size_t(id) >> 6] &= ~(uint64_t(1) << (id & 63));
        }
        if (cur_.first) // first-position pieces differ: check them exactly
            for (int id = 0; id < vocabulary_size; ++id) {
                State copy = cur_;
                if (!step(copy, id))
                    mask[size_t(id) >> 6] &= ~(uint64_t(1) << (id & 63));
            }
        return;
    }
    mask.fill(0);
    for (int id = 0; id < vocabulary_size; ++id) {
        State copy = cur_;
        if (step(copy, id))
            set_bit(mask, size_t(id));
    }
}

void StructuredConstraint::walk(int32_t node, int sid, grammar::PartialUtf8 partial, TokenMask &mask) {
    const auto &nodes = vocab_.nodes();
    auto &m = engine_->matcher;
    const bool json = engine_->json_mode;
    const bool done = json && accepting(sid, partial);
    for (int32_t c = nodes[size_t(node)].first_child; c >= 0; c = nodes[size_t(c)].next_sibling) {
        const auto &nd = nodes[size_t(c)];
        const uint8_t b = nd.byte;
        if (done && is_space(b))
            continue;
        int nsid = sid;
        grammar::PartialUtf8 np = partial;
        if (np.n_remain > 0) {
            if ((b & 0xC0) != 0x80)
                continue;
            np.value = (np.value << 6) | (b & 0x3Fu);
            if (--np.n_remain == 0) {
                nsid = m.Step(sid, np.value);
                np = {};
                if (nsid < 0)
                    continue;
            } else {
                uint32_t lo, hi;
                grammar::PartialRange(np, lo, hi);
                if (!m.MayMatchRange(sid, lo, hi))
                    continue;
            }
        } else if (b < 0x80) {
            nsid = m.Step(sid, b);
            if (nsid < 0)
                continue;
        } else {
            if ((b & 0xE0) == 0xC0)
                np = {b & 0x1Fu, 1, 2};
            else if ((b & 0xF0) == 0xE0)
                np = {b & 0x0Fu, 2, 3};
            else if ((b & 0xF8) == 0xF0 && b <= 0xF4)
                np = {b & 0x07u, 3, 4};
            else
                continue; // stray continuation / invalid lead byte
            uint32_t lo, hi;
            grammar::PartialRange(np, lo, hi);
            if (!m.MayMatchRange(sid, lo, hi))
                continue;
        }
        for (int32_t t = nd.token; t >= 0; t = vocab_.same_piece(t))
            set_bit(mask, size_t(t));
        if (nd.first_child >= 0)
            walk(c, nsid, np, mask);
    }
}

TokenConstraint::Allowed StructuredConstraint::allowed() {
    ++stats_.steps;
    auto summarize = [](const TokenMask &mask, int &count, int &single) {
        count = 0;
        single = -1;
        for (size_t w = 0; w < mask.size(); ++w)
            if (mask[w]) {
                count += std::popcount(mask[w]);
                if (single < 0)
                    single = int(w * 64 + size_t(std::countr_zero(mask[w])));
            }
        if (count != 1)
            single = -1;
    };
    auto forced_eos = [&]() -> Allowed {
        scratch_.fill(0);
        set_bit(scratch_, size_t(eos_token));
        return {&scratch_, 1, eos_token};
    };
    if (dead_)
        return forced_eos();
    try {
        if (cur_.split.phase == ThinkSplitter::Answer && !cur_.first) {
            GrammarEngine::Key key{cur_.sid, cur_.partial.value, cur_.partial.n_remain, cur_.partial.n_total};
            auto it = engine_->masks.find(key);
            if (it != engine_->masks.end()) {
                ++stats_.mask_hits;
                return {&it->second.mask, it->second.count, it->second.single};
            }
            auto start = std::chrono::steady_clock::now();
            if (engine_->masks.size() >= 4096)
                engine_->masks.clear(); // bound memory (4 KiB per entry)
            GrammarEngine::CachedMask cached{};
            walk(0, cur_.sid, cur_.partial, cached.mask);
            if (accepting(cur_.sid, cur_.partial))
                set_bit(cached.mask, size_t(eos_token));
            summarize(cached.mask, cached.count, cached.single);
            ++stats_.mask_builds;
            stats_.mask_build_ms +=
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            if (!cached.count) {
                dead_ = true;
                return forced_eos();
            }
            auto &slot = engine_->masks[key] = cached;
            return {&slot.mask, slot.count, slot.single};
        }
        auto start = std::chrono::steady_clock::now();
        scan_all(scratch_);
        stats_.mask_build_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        int count, single;
        summarize(scratch_, count, single);
        if (!count) {
            dead_ = true;
            return forced_eos();
        }
        return {&scratch_, count, single};
    } catch (const std::exception &e) {
        std::cerr << "[structured] grammar error: " << e.what() << '\n';
        dead_ = true;
        return forced_eos();
    }
}

void StructuredConstraint::advance(int token) {
    if (dead_)
        return;
    try {
        if (!step(cur_, token)) {
            std::cerr << "[structured] token " << token << " violates the constraint; ending the answer\n";
            dead_ = true;
        }
    } catch (const std::exception &e) {
        std::cerr << "[structured] grammar error: " << e.what() << '\n';
        dead_ = true;
    }
}

} // namespace behemoth
