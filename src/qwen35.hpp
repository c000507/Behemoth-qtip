// Qwen3.5 (hybrid Gated DeltaNet + gated full attention, dense/MoE MLP) config, checkpoint reading and a
// CPU fp32 reference forward. Math follows nmoe/docs/qwen35_model_math.md (HF transformers 5.12 reference).
#pragma once
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "safetensors.hpp"

namespace qtip {

struct QwenConfig {
  int hidden = 0, inter = 0, layers = 0, vocab = 0;
  float eps = 1e-6f, theta = 1e7f;
  // full attention
  int nq = 0, nkv = 0, hd = 0, rot_dim = 0;  // rot_dim 0 -> full head rotation
  bool attn_output_gate = true;
  // linear attention (Gated DeltaNet)
  int lin_nk = 0, lin_kd = 0, lin_nv = 0, lin_vd = 0, conv_k = 0;
  // MoE (0 experts -> dense MLP)
  int n_experts = 0, topk = 0, moe_inter = 0, shared_inter = 0;
  bool shared_gate = true;  // shared expert gated by sigmoid(shared_expert_gate)
  bool tied = false;
  std::vector<uint8_t> full_attn;  // per layer: 1 = full attention, 0 = linear
  std::string raw_json;
  std::string prefix;  // safetensors prefix of the text model ("model." / "model.language_model." / "")

  int key_dim() const { return lin_nk * lin_kd; }
  int value_dim() const { return lin_nv * lin_vd; }
  int conv_dim() const { return 2 * key_dim() + value_dim(); }
  bool moe() const { return n_experts > 0; }
  bool is_full(int layer) const { return full_attn[size_t(layer)] != 0; }
  int lin_group() const { return lin_nv / lin_nk; }  // v-heads per k-head
};
// Parses config.json (dense "qwen3_5" or MoE "qwen3_5_moe", nested text_config); QCHECKs on anything
// unsupported. Does NOT touch the Behemoth path in model.hpp.
QwenConfig load_qwen_config(const std::string& dir);
// Detects (cheaply, from config.json alone) whether dir is a Qwen3.5 text model.
bool is_qwen35_dir(const std::string& dir);

// Linear tensors of one Qwen3.5 layer kept as bf16 row-major [out][in] (original dtype preserved where
// fp32: a_log/dt_bias/router bias). Weight dtype BF16 or F8_E4M3+scale checkpoints both supported;
// everything is normalised to bf16 on read.
struct QwenLayer {
  int idx = -1;
  bool full = false;   // full attention vs linear (delta) layer
  bool moe = false;
  // full-attention projections
  std::vector<uint16_t> w_q, w_k, w_v, w_o;  // q: [nq*2*hd][hidden] (q|gate interleaved per head)
  std::vector<uint16_t> q_norm, k_norm;      // zc rmsnorm [hd]
  // linear-attention projections
  std::vector<uint16_t> w_in_qkv, w_in_z, w_in_ba, w_out;  // ba: [2*nv][hidden] (b rows then a rows)
  std::vector<uint16_t> conv_w;                             // [conv_dim * conv_k]
  std::vector<float> a_log, dt_bias;                        // [nv] fp32
  std::vector<uint16_t> dn_norm;                            // gated rmsnorm weight [vd]
  // dense MLP
  std::vector<uint16_t> w_gate, w_up, w_down;
  // MoE
  std::vector<uint16_t> w_router;                 // [n_experts][hidden] bf16
  std::vector<float> w_router_bias;               // [n_experts] fp32 (optional, empty if absent)
  std::vector<std::vector<uint16_t>> w_exp_gate, w_exp_up, w_exp_down;  // per expert
  std::vector<uint16_t> w_sh_gate, w_sh_up, w_sh_down, w_sh_gate_w;     // shared expert + its gate [1][hidden]
  // norms
  std::vector<uint16_t> ln_in, ln_post;  // zc [hidden] bf16
  double read_s = 0;
};
std::shared_ptr<QwenLayer> read_qwen_layer(const SafeTensorsReader& st, const QwenConfig& cfg, int idx);

// LRU cache with background prefetch (same pattern as LayerCache in model.hpp).
class QwenLayerCache {
 public:
  QwenLayerCache(const SafeTensorsReader& st, const QwenConfig& cfg, int capacity);
  ~QwenLayerCache();
  std::shared_ptr<QwenLayer> get(int idx);
  void prefetch(int idx);

 private:
  void worker();
  const SafeTensorsReader& st_;
  QwenConfig cfg_;
  int cap_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::map<int, std::shared_ptr<QwenLayer>> have_;
  std::map<int, long> last_use_;
  std::vector<int> queue_;
  std::map<int, bool> loading_;
  long tick_ = 0;
  bool stop_ = false;
  std::thread th_;
};

// bf16 buffer -> fp32 buffer
std::vector<float> bf16_to_f32_vec(const std::vector<uint16_t>& v);

// ------------------------------------------------------------ CPU reference forward (fp32)
// Embeddings table [vocab][hidden] bf16, optional separate lm_head [vocab][hidden] (null -> tied).
// tokens: [T]. Writes logits [T][vocab] fp32. capture_layer: if >= 0, writes the residual-stream input
// of that layer (i.e. the activations after `capture_layer` layers) for layer T dogfooding tests.
struct QwenCpuRefs {
  const std::vector<uint16_t>* embed = nullptr;
  const std::vector<uint16_t>* lm_head = nullptr;
};
// One dense (non-MoE) layer on the residual stream, fp32 in place; delta state starts at zero.
void qwen_cpu_layer(const QwenConfig& cfg, const QwenLayer& l, float* x, int T, int nthreads = 32);
void qwen_cpu_forward(const QwenConfig& cfg, const SafeTensorsReader& st,
                      const std::vector<std::shared_ptr<QwenLayer>>& layers, const QwenCpuRefs& refs,
                      const int32_t* tokens, int T, float* logits, int nthreads = 32,
                      int capture_layer = -1, std::vector<float>* captured = nullptr);

}  // namespace qtip
