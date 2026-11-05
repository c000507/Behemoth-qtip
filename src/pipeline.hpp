// qtip_quant pipeline: calibration forward + Hessian collection (per GPU token halves), QTIP quantization
// and the adaptive mixed-format mode (per-layer QTIP-2bit vs NVFP4 selection by an output-quality metric).
#pragma once
#include <atomic>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "common.hpp"
#include "gpu.hpp"
#include "model.hpp"
#include "nvfp4.hpp"
#include "nlohmann/json.hpp"
#include "quantizer.hpp"
#include "qtip_core.hpp"
#include "safetensors.hpp"

namespace qtip {

using json = nlohmann::json;

struct Options {
  std::string mode = "run";  // run | collect | quantize | assemble | adaptive
  std::string model = "/mnt/evo/Behemoth-R1-123B-v2-FP8-Dynamic";
  std::string tokens;        // int32 [nseq][seq_len]
  std::string out;           // output model dir
  std::string work;          // progress, logs, Hessians
  std::string hess_dir;      // default <work>/hess
  int layer_begin = 0, layer_end = -1;
  std::vector<int> gpus = {0, 1};
  long ntok = 4194304;
  int seq_len = 2048;
  int batch_rows = 4096;
  int backlog = 2;
  bool keep_hessians = false;
  bool fp8_kv = true;
  bool kv_explicit = false;  // --kv-format was passed (BF16 sources default to qtip)
  uint64_t seed = 0;
  std::string dump_acts;  // write the first sequence of GPU 0 after every chain layer (bf16)
  // ---- adaptive mode (--mode adaptive)
  double kl_threshold = 0.05;              // metric value above which a layer falls back to NVFP4
  std::string kl_metric = "lmkl";          // lmkl (mean token KL of next-token logits) | mse (normalized)
  int kl_tokens = 512;                     // probe document length for the metric
  std::string adaptive_next_input = "original";  // original | quantized
  bool nvfp4_calibrate = true;             // Hessian-aware (LDLQ-style) rounding of NVFP4 fallback layers
  bool nvfp4_scale_search = true;          // per-16-block E4M3 scale search for NVFP4 fallback layers
  bool nvfp4_compare = false;              // also measure the naive / search-only / calibrated variants (report only)
  long logit_tile = 0;                     // lm_head vocab rows per logits GEMM (0 = auto, ~256 MB fp32 tile)
  bool cache_teacher_logits = true;        // original-input chaining has the same teacher at every layer
  QuantOpts q;
};

struct LayerState {
  enum St { kNone, kPartial, kReady, kQuantizing, kDone } st = kNone;
  std::vector<std::vector<float>> H;  // combined (normalised) Hessians in RAM when kReady (may be empty: on disk)
  std::vector<std::vector<float>> partial;
  long partial_count = 0;
  int partial_parts = 0;
};

class Pipeline {
 public:
  Pipeline(const Options& o);
  void run();
  void assemble();

 private:
  // progress
  void load_progress();
  void save_progress_locked();
  std::string hess_path(int L) const;
  std::string layer_path(int L) const;
  // tasks
  void worker(int wi);
  void chain_step(int wi, LayerRunner& R, int L, bool collect);
  void quantize_layer(int wi, LayerRunner* R, hipStream_t s, Gemm& g, int L);
  void deposit_partial(int L, HessAcc& acc, LayerRunner& R);
  std::vector<std::vector<float>> take_hessians(int L);
  // shared QTIP group quantization of one layer; fills jl like quantize_layer does
  std::vector<LinearOut> run_qtip_groups(int L, std::vector<std::vector<float>>&& H,
                                         const std::shared_ptr<LayerHost>& LH, const QuantOpts& q, json& jl);
  // write one layer file in the chosen format ("qtip" uses outs, "nvfp4" uses nvs; fp8-preserved K/V
  // tensors come from LH in both cases). Returns bytes written.
  size_t write_layer_file(int L, const LayerHost& LH, const char* format, const std::vector<LinearOut>* outs,
                          const std::vector<Nvfp4Tensor>* nvs);
  void finish_layer(int L, json& jl, double secs, int dev, const char* format);

  // ---- adaptive mode
  struct ChosenLayer {
    int layer = -1;
    bool quantized = false;                  // next chain layer input from these weights?
    std::vector<std::vector<uint16_t>> w;    // [kNumLin] bf16 weights, empty = keep original FP8
  };
  void eval_runners_create(int device);
  void eval_runners_destroy();
  void adaptive_load_head();                 // final norm + lm_head (device), probe token ids
  void probe_advance_to(int L);              // advance the probe to the input of layer L
  std::vector<uint16_t> adaptive_layer_forward(const std::shared_ptr<LayerHost>& LH,
                                               const std::vector<LinearOut>* qtip_out,
                                               const std::vector<Nvfp4Tensor>* nv);  // eval runner forward
  std::vector<float> adaptive_logits(int L, std::vector<uint16_t> h);  // original layers L+1.. + head
  double kl_mean_token(const std::vector<float>& p_logits, const std::vector<float>& q_logits) const;
  double nmse_layer(const std::vector<uint16_t>& y_q, const std::vector<uint16_t>& y_o) const;
  void adaptive_decide(int wi, LayerRunner* chain, hipStream_t qs, Gemm& qg, int L);
  void adaptive_report_write_locked();
  bool adaptive_prev_decided(int L) const;  // chain gate for adaptive lockstep

  Options o_;
  SafeTensorsReader st_;
  ModelConfig cfg_;
  LayerCache cache_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::map<int, LayerState> layers_;
  std::set<int> done_;          // quantized (from progress)
  std::set<int> hess_on_disk_;  // complete Hessians on disk
  json progress_;
  // chains
  std::vector<int> chain_pos_;
  int chain_end_ = 0;           // chains must reach this layer (exclusive)
  std::set<int> need_collect_;
  std::vector<std::unique_ptr<uint16_t[]>> act_;
  std::vector<std::unique_ptr<uint16_t[]>> act_in_;  // adaptive quantized-next-input: input snapshot
  std::vector<long> act_nseq_;
  std::atomic<int> failures_{0};
  // adaptive state (worker 0 / gpus[0])
  bool adaptive_ = false;
  std::vector<int32_t> probe_tok_;
  std::vector<uint16_t> probe_;      // [kl_tokens][hidden] bf16, input of layer probe_at_
  int probe_at_ = -1;
  std::unique_ptr<LayerRunner> a_orig_, a_eval_;
  bool head_loaded_ = false;
  std::vector<uint16_t> head_w_, norm_w_;
  DBuf<uint16_t> d_head_, d_norm_;
  DBuf<uint16_t> d_x_;
  DBuf<float> d_logits_;
  std::vector<float> teacher_logits_;       // original model logits, independent of candidate layer
  ChosenLayer chosen_;
  json report_;                     // adaptive_report.json content
  std::map<int, std::string> layer_format_;  // decisions (for report/assemble)
};

}  // namespace qtip
