// Generates a tiny synthetic qwen3_5_moe checkpoint (random weights, HF safetensors layout) for
// CPU/GPU parity tests of the MoE path: 4 layers (3 linear-attention DeltaNet + 1 full attention),
// hidden 512, 8 experts top-2 + sigmoid-gated shared expert, untied lm_head.
// usage: qwen_make_tiny_moe [out_dir] [seed]
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>
#include <string>
#include <vector>

#include "common.hpp"
#include "safetensors.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

static uint16_t f32_to_bf16_bits(float f) { return qtip::f32_to_bf16(f); }

int main(int argc, char** argv) {
  std::string out = argc >= 2 ? argv[1] : "qwen_tiny_moe";
  unsigned seed = argc >= 3 ? unsigned(std::atoi(argv[2])) : 1234u;
  const int H = 512, L = 4, vocab = 2048, hd = 64, nq = 8, nkv = 2;
  const int nk = 4, kd = 64, nv = 8, vd = 64, conv_k = 4;
  const int key_dim = nk * kd, value_dim = nv * vd, conv_dim = 2 * key_dim + value_dim;
  const int E = 8, topk = 2, F = 128, S = 128;
  const int rot = 16;  // 0.25 * hd
  fs::create_directories(out);

  std::mt19937 rng(seed);
  auto randn = [&](float s) {
    return std::normal_distribution<float>(0.0f, s)(rng);
  };
  auto wmat = [&](int rows, int cols, float scale_mul = 1.0f) {
    std::vector<uint16_t> w(size_t(rows) * size_t(cols));
    float s = scale_mul / std::sqrt(float(cols));
    for (auto& x : w) x = f32_to_bf16_bits(randn(s));
    return w;
  };
  auto near1 = [&](int n, float s) {  // zero-centred norm weights: stored w, kernel uses (1+w)
    std::vector<uint16_t> w(n);
    for (auto& x : w) x = f32_to_bf16_bits(randn(s));
    return w;
  };

  qtip::SafeTensorsWriter w;
  w.add("model.embed_tokens.weight", "BF16", {vocab, H}, wmat(vocab, H, 0.6f).data(),
        size_t(vocab) * H * 2);
  w.add("lm_head.weight", "BF16", {vocab, H}, wmat(vocab, H).data(), size_t(vocab) * H * 2);
  w.add("model.norm.weight", "BF16", {H}, near1(H, 0.02f).data(), size_t(H) * 2);

  // per-layer residual-scale: keep activations bounded through 4 layers
  const float attn_s = 0.5f, mlp_s = 0.5f;
  for (int l = 0; l < L; ++l) {
    std::string p = "model.layers." + std::to_string(l) + ".";
    bool full = (l == L - 1);
    w.add(p + "input_layernorm.weight", "BF16", {H}, near1(H, 0.02f).data(), size_t(H) * 2);
    w.add(p + "post_attention_layernorm.weight", "BF16", {H}, near1(H, 0.02f).data(), size_t(H) * 2);
    if (full) {
      w.add(p + "self_attn.q_proj.weight", "BF16", {nq * 2 * hd, H}, wmat(nq * 2 * hd, H, attn_s).data(),
            size_t(nq) * 2 * hd * H * 2);
      w.add(p + "self_attn.k_proj.weight", "BF16", {nkv * hd, H},
            wmat(nkv * hd, H, attn_s).data(), size_t(nkv) * hd * H * 2);
      w.add(p + "self_attn.v_proj.weight", "BF16", {nkv * hd, H},
            wmat(nkv * hd, H, attn_s).data(), size_t(nkv) * hd * H * 2);
      w.add(p + "self_attn.o_proj.weight", "BF16", {H, nq * hd},
            wmat(H, nq * hd, attn_s).data(), size_t(H) * nq * hd * 2);
      w.add(p + "self_attn.q_norm.weight", "BF16", {hd}, near1(hd, 0.02f).data(), size_t(hd) * 2);
      w.add(p + "self_attn.k_norm.weight", "BF16", {hd}, near1(hd, 0.02f).data(), size_t(hd) * 2);
    } else {
      w.add(p + "linear_attn.in_proj_qkv.weight", "BF16", {conv_dim, H},
            wmat(conv_dim, H, attn_s).data(), size_t(conv_dim) * H * 2);
      w.add(p + "linear_attn.in_proj_z.weight", "BF16", {value_dim, H},
            wmat(value_dim, H, attn_s).data(), size_t(value_dim) * H * 2);
      w.add(p + "linear_attn.in_proj_b.weight", "BF16", {nv, H}, wmat(nv, H).data(), size_t(nv) * H * 2);
      w.add(p + "linear_attn.in_proj_a.weight", "BF16", {nv, H}, wmat(nv, H).data(), size_t(nv) * H * 2);
      w.add(p + "linear_attn.out_proj.weight", "BF16", {H, value_dim},
            wmat(H, value_dim, attn_s).data(), size_t(H) * value_dim * 2);
      {  // conv1d [conv_dim, 1, K]: mild low-pass, positive centre
        std::vector<uint16_t> cw(size_t(conv_dim) * conv_k);
        float base[4] = {0.05f, 0.1f, 0.25f, 0.6f};
        for (int c = 0; c < conv_dim; ++c)
          for (int j = 0; j < conv_k; ++j)
            cw[size_t(c) * conv_k + j] = f32_to_bf16_bits(base[j] + randn(0.02f));
        w.add(p + "linear_attn.conv1d.weight", "BF16", {conv_dim, 1, conv_k}, cw.data(), cw.size() * 2);
      }
      {  // A_log / dt_bias f32: decay ~ exp(-exp(A_log)*softplus(a+dt_bias)) in a sane range
        std::vector<float> alog(nv), dtb(nv);
        for (int i = 0; i < nv; ++i) {
          alog[size_t(i)] = std::log(1.0f + 0.1f * i);
          dtb[size_t(i)] = 0.01f * float(i);
        }
        w.add(p + "linear_attn.A_log", "F32", {nv}, alog.data(), alog.size() * 4);
        w.add(p + "linear_attn.dt_bias", "F32", {nv}, dtb.data(), dtb.size() * 4);
      }
      w.add(p + "linear_attn.norm.weight", "BF16", {vd}, near1(vd, 0.02f).data(), size_t(vd) * 2);
    }
    // MoE block
    w.add(p + "mlp.gate.weight", "BF16", {E, H}, wmat(E, H).data(), size_t(E) * H * 2);
    for (int e = 0; e < E; ++e) {
      std::string ep = p + "mlp.experts." + std::to_string(e) + ".";
      w.add(ep + "gate_proj.weight", "BF16", {F, H}, wmat(F, H, mlp_s).data(), size_t(F) * H * 2);
      w.add(ep + "up_proj.weight", "BF16", {F, H}, wmat(F, H, mlp_s).data(), size_t(F) * H * 2);
      w.add(ep + "down_proj.weight", "BF16", {H, F}, wmat(H, F, mlp_s).data(), size_t(H) * F * 2);
    }
    w.add(p + "mlp.shared_expert.gate_proj.weight", "BF16", {S, H}, wmat(S, H, mlp_s).data(),
          size_t(S) * H * 2);
    w.add(p + "mlp.shared_expert.up_proj.weight", "BF16", {S, H}, wmat(S, H, mlp_s).data(),
          size_t(S) * H * 2);
    w.add(p + "mlp.shared_expert.down_proj.weight", "BF16", {H, S}, wmat(H, S, mlp_s).data(),
          size_t(H) * S * 2);
    w.add(p + "mlp.shared_expert_gate.weight", "BF16", {1, H}, wmat(1, H).data(), size_t(H) * 2);
  }
  size_t bytes = w.write(out + "/model.safetensors");

  json layer_types = {"linear_attention", "linear_attention", "linear_attention", "full_attention"};
  json text = {
      {"hidden_size", H},
      {"num_hidden_layers", L},
      {"vocab_size", vocab},
      {"rms_norm_eps", 1e-6},
      {"num_attention_heads", nq},
      {"num_key_value_heads", nkv},
      {"head_dim", hd},
      {"attn_output_gate", true},
      {"linear_num_key_heads", nk},
      {"linear_key_head_dim", kd},
      {"linear_num_value_heads", nv},
      {"linear_value_head_dim", vd},
      {"linear_conv_kernel_dim", conv_k},
      {"layer_types", layer_types},
      {"mlp_only_layers", json::array()},
      {"num_experts", E},
      {"num_experts_per_tok", topk},
      {"moe_intermediate_size", F},
      {"shared_expert_intermediate_size", S},
      {"shared_expert_gated", true},
      {"max_position_embeddings", 8192},
      {"rope_parameters", {{"rope_theta", 1e7}, {"partial_rotary_factor", 0.25}, {"rope_type", "default"}}},
      {"model_type", "qwen3_5_moe_text"},
  };
  json cfg = {
      {"architectures", {"Qwen3_5MoeForCausalLM"}},
      {"model_type", "qwen3_5_moe"},
      {"text_config", text},
      {"tie_word_embeddings", false},
      {"torch_dtype", "bfloat16"},
  };
  {
    std::ofstream f(out + "/config.json");
    f << cfg.dump(2) << "\n";
  }
  {
    json idx = {{"metadata", {{"total_size", bytes}}},
                {"weight_map", {{"model.embed_tokens.weight", "model.safetensors"},
                                {"lm_head.weight", "model.safetensors"},
                                {"model.norm.weight", "model.safetensors"}}}};
    // fill weight_map from the writer's tensor list
    json wm = json::object();
    for (auto& n : w.names()) wm[n] = "model.safetensors";
    idx["weight_map"] = wm;
    std::ofstream f(out + "/model.safetensors.index.json");
    f << idx.dump(2) << "\n";
  }
  std::printf("tiny qwen3_5_moe written to %s (%.1f MB, seed %u)\n", out.c_str(), bytes / 1e6, seed);
  return 0;
}
