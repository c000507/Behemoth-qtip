#pragma once
#include "tokenizer.hpp"
#include <nlohmann/json.hpp>
namespace glm53 {
class Tokenizer;
class InferenceBackend;
} // namespace glm53
namespace behemoth::cuda_runtime {
class TensorModel;
namespace fs = std::filesystem;
using json = nlohmann::json;
using Vec = std::vector<float>;
inline void check(bool ok, const std::string &text) {
    if (!ok)
        throw std::runtime_error(text);
}
class Tokenizer {
    std::unique_ptr<behemoth::Tokenizer> tokenizer_;
    std::unique_ptr<glm53::Tokenizer> bpe_;
    json settings_;
    std::map<int, std::string> added_;

  public:
    explicit Tokenizer(const fs::path &);
    ~Tokenizer();
    std::vector<int> encode(const std::string &text) const;
    std::string decode(const std::vector<int> &, bool special = false) const;
    std::string piece(int, bool &first, bool special = false) const;
    std::string chat(const json &) const;
};
class Model {
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::shared_ptr<glm53::InferenceBackend> backend_;
#if defined(BEHEMOTH_CUDA_FAST)
    std::unique_ptr<TensorModel> tensor_;
#endif
    int position_ = 0;

  public:
    struct Config {
        int vocab = 32768, max_context;
        std::vector<int> eos{2};
    } config;
    // GLM: `tp` GPUs run tensor-parallel unless layer_split asks for the layer pipeline over them.
    explicit Model(const fs::path &, int device = 0, int context = 4096, int tp = 1, int max_batch = 1,
                   bool layer_split = false);
    ~Model();
    Vec step(int token);
    Vec prefill(const std::vector<int> &tokens);
    std::vector<Vec> prefill_batch(const std::vector<std::vector<int>> &tokens);
    std::vector<Vec> step_batch(const std::vector<int> &tokens);
    void reset();
};
} // namespace behemoth::cuda_runtime
