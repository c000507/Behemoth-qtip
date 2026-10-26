// Behemoth (Mistral-Large style) layer loading from the FP8-Dynamic checkpoint and the bf16 forward pass used for
// Hessian collection (inputs of q/k/v, o, gate/up, down).
#pragma once
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "gpu.hpp"
#include "safetensors.hpp"

namespace qtip {

struct ModelConfig {
  int hidden = 12288, inter = 28672, nq = 96, nkv = 8, hd = 128, layers = 88, vocab = 32768;
  float eps = 1e-5f, theta = 1e6f;
  std::string raw_json;  // original config.json
};
ModelConfig load_model_config(const std::string& dir);

// linear order inside a layer
enum Lin { LQ = 0, LK, LV, LO, LGate, LUp, LDown, kNumLin };
extern const char* kLinName[kNumLin];  // "self_attn.q_proj", ...

struct LayerHost {
  int idx = -1;
  std::vector<uint8_t> w[kNumLin];   // fp8 e4m3 [out][in]
  std::vector<uint16_t> sc[kNumLin]; // bf16 per-output-channel scale [out]
  std::vector<uint16_t> ln_in, ln_post;  // bf16 [hidden]
  int out_dim[kNumLin] = {}, in_dim[kNumLin] = {};
  double read_s = 0;
};
std::shared_ptr<LayerHost> read_layer(const SafeTensorsReader& st, const ModelConfig& cfg, int idx);

// Thread-safe LRU cache of host layers with background prefetch (the checkpoint is on NFS).
class LayerCache {
 public:
  LayerCache(const SafeTensorsReader& st, const ModelConfig& cfg, int capacity);
  ~LayerCache();
  std::shared_ptr<LayerHost> get(int idx);
  void prefetch(int idx);

 private:
  void worker();
  const SafeTensorsReader& st_;
  const ModelConfig& cfg_;
  int cap_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::map<int, std::shared_ptr<LayerHost>> have_;
  std::map<int, long> last_use_;
  std::vector<int> queue_;
  std::map<int, bool> loading_;
  long tick_ = 0;
  bool stop_ = false;
  std::thread th_;
};

// Hessian accumulators (sum of x x^T, fp32) for one layer on one GPU.
struct HessAcc {
  DBuf<float> h[4];  // qkv, o, gateup, down
  long count = 0;    // tokens
  int dim[4] = {};
  int blocks[4] = {1, 1, 1, 1};  // block partition used for the triangular accumulation
};
extern const char* kHessName[4];  // "qkv", "o", "gateup", "down"

class LayerRunner {
 public:
  LayerRunner(int device, const ModelConfig& cfg, int max_rows, int seq_len);
  ~LayerRunner();
  void upload(const LayerHost& L);  // fp8 -> bf16 on device (exact), scales -> fp32
  // In-place forward of nseq sequences (host bf16 [nseq][T][hidden]); if acc: accumulate Hessians.
  void forward(uint16_t* host_x, long nseq, HessAcc* acc);
  void alloc_hess(HessAcc& a);
  void zero_hess(HessAcc& a);
  void symmetrize_hess(HessAcc& a);  // fill the blocks skipped by the triangular accumulation
  hipStream_t stream() const { return s_; }
  Gemm& gemm() { return *gemm_; }
  int device() const { return dev_; }
  double gemm_flops = 0;  // accumulated GEMM flops (for throughput reporting)

 private:
  void forward_batch(long rows, HessAcc* acc);
  int dev_;
  ModelConfig cfg_;
  int max_rows_, T_;
  hipStream_t s_;
  std::unique_ptr<Gemm> gemm_;
  // weights
  DBuf<uint16_t> wqkv_, wo_, wgu_, wd_, ln_in_, ln_post_;
  DBuf<float> sqkv_, so_, sgu_, sd_;
  DBuf<uint8_t> stage8_;
  // activations
  DBuf<uint16_t> x_, h_, attn_, act_, Qp_, Kp_, Vp_, P_;
  DBuf<float> qkv_, y_, gu_, S_, O_;
  HBuf<uint16_t> pin_;
};

}  // namespace qtip
