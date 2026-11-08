// Structured outputs on the checkpoint vocabulary (CPU only, no GPU, no weights):
//  1. exactness: along constrained random walks, every step's mask equals accepts()
//     over every vocabulary id, and completed answers validate against the schema;
//  2. cost: per-token time of the constraint calls the engine makes (accepts() for
//     the greedy fast path, allowed() + advance() for masked sampling), cold
//     (fresh grammar) and warm (grammar reused through the GrammarCache).
// Usage: behemoth_structured_bench MODEL_DIRECTORY [walks]
#include "../tests/structured_support.hpp"
#include "behemoth/json_schema.hpp"
#include "behemoth/tokenizer.hpp"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

using namespace behemoth;
using namespace structured_test;
using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

int main(int argc, char **argv) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: behemoth_structured_bench MODEL_DIRECTORY [walks]\n";
            return 2;
        }
        const int walks = argc > 2 ? std::stoi(argv[2]) : 20;
        Tokenizer tokenizer(argv[1]);
        auto t0 = Clock::now();
        Vocabulary vocab(vocabulary_pieces(tokenizer), vocabulary_eos(tokenizer));
        std::cout << "vocabulary: " << vocab.nodes().size() << " trie nodes, " << vocab.close_candidates().size()
                  << " '>' tokens, built in " << ms_since(t0) << " ms\n";
        const std::vector<std::pair<std::string, ojson>> schemas = {
            {"json_object", ojson(nullptr)},
            {"person", ojson::parse(R"({"type":"object","properties":{"name":{"type":"string","maxLength":40},
               "age":{"type":"integer","minimum":0,"maximum":150},"email":{"type":"string","format":"email"},
               "tags":{"type":"array","items":{"enum":["red","green","blue"]},"maxItems":4},"active":{"type":"boolean"}},
               "required":["name","age","tags","active"],"additionalProperties":false})")},
            {"nested", ojson::parse(R"({"$defs":{"item":{"type":"object","properties":{"id":{"type":"integer"},
               "price":{"type":"number","minimum":0,"maximum":1000},"note":{"anyOf":[{"type":"string"},{"type":"null"}]}},
               "required":["id","price","note"],"additionalProperties":false}},"type":"object","properties":{
               "order":{"type":"string","format":"uuid"},"date":{"type":"string","format":"date"},
               "items":{"type":"array","items":{"$ref":"#/$defs/item"},"minItems":1,"maxItems":3}},
               "required":["order","date","items"],"additionalProperties":false})")},
        };
        std::mt19937_64 rng(2026);
        GrammarCache cache;
        bool ok = true;
        for (const auto &[name, schema] : schemas) {
            const std::string g =
                schema.is_null() ? grammar::json_object_grammar() : grammar::json_schema_grammar(schema);
            auto compile0 = Clock::now();
            GrammarCache::check(g);
            const double compile_ms = ms_since(compile0);
            // 1. exactness (2 walks with the full-vocabulary comparison at every step)
            int exact_steps = 0;
            for (int i = 0; i < 2; ++i) {
                StructuredConstraint c(vocab, std::make_unique<GrammarEngine>(g, true), false);
                auto r = walk(vocab, c, false, biased_chooser(vocab, rng), 80, true);
                exact_steps += r.tokens;
            }
            // 2. cost
            int completed = 0, tokens = 0, valid = 0;
            double allowed_ms = 0, accepts_ms = 0, advance_ms = 0, worst_ms = 0, first_ms = 0;
            double cold_allowed_ms = 0;
            int cold_tokens = 0;
            uint64_t builds = 0, hits = 0, scans = 0;
            for (int i = 0; i < walks; ++i) {
                const bool cold = i == 0;
                StructuredConstraint c(vocab, cold ? std::make_unique<GrammarEngine>(g, true) : cache.acquire(g, true),
                                       false, &cache);
                ThinkSplitter split(false);
                std::string reasoning, answer;
                bool first = true, ended = false;
                auto choose = biased_chooser(vocab, rng);
                for (int step = 0; step < 400; ++step) {
                    auto a0 = Clock::now();
                    auto a = c.allowed();
                    const double am = ms_since(a0);
                    auto allowed = ids(*a.mask);
                    int t = a.count == 1 ? a.single : choose(allowed);
                    auto c0 = Clock::now();
                    const bool accepted = c.accepts(t);
                    const double cm = ms_since(c0);
                    auto v0 = Clock::now();
                    c.advance(t);
                    const double vm = ms_since(v0);
                    if (!accepted)
                        throw std::runtime_error("mask allowed a rejected token");
                    allowed_ms += am;
                    accepts_ms += cm;
                    advance_ms += vm;
                    if (step == 0)
                        first_ms += am;
                    if (cold) {
                        cold_allowed_ms += am;
                        ++cold_tokens;
                    }
                    worst_ms = std::max(worst_ms, am + cm + vm);
                    ++tokens;
                    if (t == vocab.eos()) {
                        ended = true;
                        break;
                    }
                    const auto &tok = vocab.token(t);
                    std::string piece = first ? tok.first : tok.rest;
                    if (first)
                        first = tok.first_after;
                    split.feed(piece, &reasoning, &answer);
                }
                builds += c.stats().mask_builds;
                hits += c.stats().mask_hits;
                scans += c.stats().scans;
                if (!ended)
                    continue;
                ++completed;
                try {
                    auto parsed = json::parse(answer);
                    if (schema.is_null() ? parsed.is_object() : validate(schema, parsed, schema))
                        ++valid;
                    else {
                        ok = false;
                        std::cout << "  INVALID (schema): " << answer << '\n';
                    }
                } catch (const std::exception &e) {
                    ok = false;
                    std::cout << "  INVALID (json): " << answer << '\n';
                }
            }
            std::cout << std::fixed << std::setprecision(1) << name << ": grammar " << g.size() << " B compiled in "
                      << compile_ms << " ms; exact mask==accepts over " << exact_steps << " steps; walks "
                      << completed << "/" << walks << " completed, " << valid << " valid\n"
                      << "  per token (all walks): allowed " << allowed_ms * 1000 / tokens << " us, accepts "
                      << accepts_ms * 1000 / tokens << " us, advance " << advance_ms * 1000 / tokens
                      << " us; worst token " << worst_ms << " ms; first token (scan) " << first_ms / walks
                      << " ms\n  cold walk: allowed " << cold_allowed_ms * 1000 / std::max(1, cold_tokens)
                      << " us/token over " << cold_tokens << " tokens; mask builds " << builds << ", cache hits "
                      << hits << ", full scans " << scans << '\n';
        }
        // Think phase: the per-token gate check while reasoning (greedy fast path).
        {
            StructuredConstraint c(vocab, cache.acquire(grammar::json_object_grammar(), true), true, &cache);
            auto think = tokenizer.encode("<think>");
            for (int t : think)
                c.advance(t);
            auto text = tokenizer.encode(" We are creating a JSON object for a fictional person with keys name and "
                                         "age. The name should be a string and the age an integer. <x> -> done.");
            auto t1 = Clock::now();
            int n = 0;
            for (int rep = 0; rep < 50; ++rep)
                for (int t : text) {
                    if (!c.accepts(t))
                        throw std::runtime_error("reasoning token rejected");
                    c.advance(t);
                    ++n;
                }
            const double accept_us = ms_since(t1) * 1000 / n;
            auto t2 = Clock::now();
            auto a = c.allowed();
            const double fallback_ms = ms_since(t2);
            std::cout << "think phase: accepts+advance " << std::setprecision(2) << accept_us
                      << " us/token; masked fallback " << fallback_ms << " ms (" << a.count << " allowed)\n";
            if (c.phase() != ThinkSplitter::Think)
                throw std::runtime_error("left the think block early");
            for (int t : tokenizer.encode("</think>\n\n{\"a\": 1}"))
                if (c.accepts(t))
                    c.advance(t);
                else
                    throw std::runtime_error("answer token rejected after think");
            if (!c.complete() || c.allowed().single != vocab.eos())
                throw std::runtime_error("EOS not forced after the answer");
            std::cout << "think -> answer -> forced EOS: ok\n";
        }
        std::cout << (ok ? "PASS" : "FAIL") << " structured bench\n";
        return ok ? 0 : 1;
    } catch (const std::exception &e) {
        std::cerr << "FAIL " << e.what() << '\n';
        return 1;
    }
}
