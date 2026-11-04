#include "glm53.hpp"
#include <cmath>
#include <fstream>
namespace glm53 {
void check(bool ok, const std::string &message) {
    if (!ok)
        throw std::runtime_error(message);
}
json read_json(const fs::path &p) {
    std::ifstream f(p);
    check(bool(f), "cannot read " + p.string());
    return json::parse(f);
}
Config::Config(const json &root) {
    const auto &c = root.contains("text_config") ? root.at("text_config") : root;
    check(c.value("model_type", "glm5_next_text") == "glm5_next_text", "expected glm5_next_text config");
    hidden = c.at("hidden_size");
    intermediate = c.at("intermediate_size");
    layers = c.at("num_hidden_layers");
    vocab = c.at("vocab_size");
    heads = c.at("num_attention_heads");
    check(c.value("num_key_value_heads", heads) == heads,
          "GLM53 MLA requires one expanded KV head per query head");
    qrank = c.at("q_lora_rank");
    kvrank = c.at("kv_lora_rank");
    qdim = c.at("qk_nope_head_dim");
    vdim = c.at("v_head_dim");
    check(c.value("qk_rope_head_dim", 0) == 0 && c.value("mla_use_nope", true),
          "GLM53 NoPE text attention required");
    const auto &a = c.at("linear_attn_config");
    linear_heads = a.at("num_heads");
    linear_dim = a.at("head_dim");
    conv = a.at("short_conv_kernel_size");
    gate_lower = a.value("gate_lower_bound", -5.f);
    streams = c.at("hc_mult");
    sinkhorn = c.at("hc_sinkhorn_iters");
    hc_eps = c.at("hc_eps");
    eps = c.at("rms_norm_eps");
    check(c.value("mhc", false), "GLM53 mHC must be enabled");
    check(c.value("hidden_act", "silu") == "silu" && !c.value("attention_bias", false),
          "GLM53 requires bias-free attention and SwiGLU");
    check(c.value("index_kpool_compress", true), "uncompressed GLM53 indexer not supported");
    index_heads = c.at("index_n_heads");
    index_dim = c.at("index_head_dim");
    pool = c.at("index_kpool");
    topk = c.at("index_topk");
    tail = c.value("index_kpool_always_select_tail", true);
    check(tail && pool > 0 && topk >= pool && topk % pool == 0,
          "GLM53 indexer requires a visible tail and pool-aligned top-k");
    attention = c.at("layer_types").get<std::vector<std::string>>();
    mlp = c.value("mlp_layer_types", std::vector<std::string>{});
    if (mlp.empty())
        for (int l = 0; l < layers; ++l)
            mlp.push_back(l < c.value("first_k_dense_replace", layers) ? "dense" : "sparse");
    indexers = c.value("indexer_types", std::vector<std::string>(layers, "full"));
    experts = c.value("n_routed_experts", 0);
    expert_topk = c.value("num_experts_per_tok", 0);
    expert_intermediate = c.value("moe_intermediate_size", 0);
    shared = c.value("n_shared_experts", 0);
    groups = c.value("n_group", 1);
    top_groups = c.value("topk_group", 1);
    norm_topk = c.value("norm_topk_prob", true);
    routed_scale = c.value("routed_scaling_factor", 1.f);
    swiglu_limit = c.value("swiglu_limit", 10.f);
    max_context = c.at("max_position_embeddings");
    bos = c.value("bos_token_id", -1);
    if (c.at("eos_token_id").is_array())
        eos = c.at("eos_token_id").get<std::vector<int>>();
    else
        eos = {c.at("eos_token_id").get<int>()};
    for (int n : {hidden, intermediate, layers, vocab, heads, qrank, kvrank, qdim, vdim, linear_heads,
                  linear_dim, conv, streams, sinkhorn, index_heads, index_dim, pool, topk, max_context})
        check(n > 0, "nonpositive GLM53 dimension");
    check(std::isfinite(eps) && eps > 0 && std::isfinite(hc_eps) && hc_eps > 0 && std::isfinite(gate_lower) &&
              gate_lower < 0 && std::isfinite(swiglu_limit) && swiglu_limit > 0,
          "invalid GLM53 numeric config");
    check(attention.size() == size_t(layers) && mlp.size() == size_t(layers) &&
              indexers.size() == size_t(layers),
          "GLM53 layer lists must match num_hidden_layers");
    bool indexed = false;
    for (int l = 0; l < layers; ++l) {
        check(attention[l] == "linear_attention" || attention[l] == "deepseek_sparse_attention",
              "unknown GLM53 attention type");
        check(mlp[l] == "dense" || mlp[l] == "sparse", "unknown GLM53 MLP type");
        check(indexers[l] == "full" || indexers[l] == "shared", "unknown indexer type");
        if (attention[l] == "deepseek_sparse_attention") {
            check(indexers[l] == "full" || indexed, "shared indexer has no preceding full layer");
            indexed = true;
        }
        if (mlp[l] == "sparse")
            check(c.value("scoring_func", "sigmoid") == "sigmoid" &&
                      c.value("topk_method", "noaux_tc") == "noaux_tc" && std::isfinite(routed_scale) &&
                      routed_scale > 0 && experts > 0 && expert_topk > 0 && expert_topk <= experts &&
                      expert_intermediate > 0 && shared > 0 && groups > 0 && experts % groups == 0 &&
                      experts / groups >= 2 && top_groups > 0 && top_groups <= groups,
                  "invalid MoE geometry");
    }
    for (int id : eos)
        check(id >= 0 && id < vocab, "EOS outside vocabulary");
    check(bos == -1 || (bos >= 0 && bos < vocab), "BOS outside vocabulary");
    check(!eos.empty(), "at least one EOS token is required");
}
json Config::summary() const {
    int dense = 0, kda = 0;
    for (auto &v : mlp)
        dense += v == "dense";
    for (auto &v : attention)
        kda += v == "linear_attention";
    return {{"hidden_size", hidden},
            {"layers", layers},
            {"dense_layers", dense},
            {"moe_layers", layers - dense},
            {"kda_layers", kda},
            {"sparse_mla_layers", layers - kda},
            {"vocab_size", vocab},
            {"mlp_parameters", double(dense) * 3 * hidden * intermediate + double(layers - dense) *
                                                                               (experts + shared) * 3 *
                                                                               hidden * expert_intermediate},
            {"context", max_context}};
}
} // namespace glm53
