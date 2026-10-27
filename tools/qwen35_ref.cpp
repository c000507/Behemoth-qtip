// CPU fp32 reference forward for a dense Qwen3.5 checkpoint; prints logits (optionally saves .bin).
// usage: qwen35_ref <model_dir> [tokens_file: int32 raw] [T] [logits_out.bin] [layers_out.bin]
// layers_out.bin: fp32 [layers][T][hidden] residual after each layer (parity debugging).
// Without a tokens file, tokenizes nothing: uses the ids from tools/qwen35_hf_ref.py's prompt baked in.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#include "common.hpp"
#include "qwen35.hpp"

using namespace qtip;

int main(int argc, char** argv) {
  QCHECK(argc >= 2, "usage: qwen35_ref <model_dir> [tokens_file] [T] [logits_out.bin]");
  std::string dir = argv[1];
  QwenConfig cfg = load_qwen_config(dir);
  QCHECK(!cfg.moe(), "MoE checkpoints not supported by the CPU reference yet");
  SafeTensorsReader st(dir);
  std::vector<int32_t> toks;
  int T = 16;
  if (argc >= 3) {
    std::ifstream f(argv[2], std::ios::binary | std::ios::ate);
    QCHECK(f, "cannot open tokens file");
    size_t sz = size_t(f.tellg()) / 4;
    T = argc >= 4 ? std::atoi(argv[3]) : int(sz);
    QCHECK(T >= 1 && size_t(T) <= sz, "T out of range");
    toks.resize(size_t(T));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(toks.data()), std::streamsize(size_t(T) * 4));
  } else {
    if (argc >= 4) T = std::atoi(argv[3]);
    // "The quick brown fox jumps over the lazy dog." via Qwen BPE (from tools/qwen35_hf_ref.py)
    toks = {2288, 4062, 13771, 30135, 517, 33984, 511, 8411, 11658, 13};
    T = std::min(T, int(toks.size()));
    toks.resize(size_t(T));
  }
  std::vector<std::shared_ptr<QwenLayer>> layers;
  for (int L = 0; L < cfg.layers; ++L) layers.push_back(read_qwen_layer(st, cfg, L));
  std::vector<float> logits(size_t(T) * cfg.vocab), all_layers;
  double t0 = now_s();
  qwen_cpu_forward(cfg, st, layers, QwenCpuRefs{}, toks.data(), T, logits.data(), 64, -1, nullptr,
                   argc >= 6 ? &all_layers : nullptr);
  std::printf("forward: %d tokens in %.1fs (%.0f tok/s)\n", T, now_s() - t0, T / (now_s() - t0));
  for (int t = 0; t < T; ++t) {
    int am = 0;
    for (int v = 1; v < cfg.vocab; ++v)
      if (logits[size_t(t) * cfg.vocab + v] > logits[size_t(t) * cfg.vocab + am]) am = v;
    std::printf("argmax[%d] = %d\n", t, am);
  }
  if (argc >= 5) {
    std::ofstream f(argv[4], std::ios::binary);
    f.write(reinterpret_cast<const char*>(logits.data()), std::streamsize(logits.size() * 4));
  }
  if (argc >= 6) {
    std::ofstream f(argv[5], std::ios::binary);
    f.write(reinterpret_cast<const char*>(all_layers.data()), std::streamsize(all_layers.size() * 4));
  }
  return 0;
}
