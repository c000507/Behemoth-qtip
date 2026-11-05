// qtip_quant CLI: option parsing only; the pipeline lives in pipeline.cpp.
#include <cstdio>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include "common.hpp"
#include "pipeline.hpp"

using namespace qtip;

static void usage() {
  std::printf(
      "usage: qtip_quant --out DIR --work DIR [--tokens FILE] [options]\n"
      "  --mode run|collect|quantize|assemble|adaptive  (default run: collect Hessians + quantize)\n"
      "  --model DIR          FP8-Dynamic checkpoint (default /mnt/evo/Behemoth-R1-123B-v2-FP8-Dynamic)\n"
      "  --tokens FILE        calibration tokens, raw int32 [nseq][seq_len]\n"
      "  --ntok N             calibration tokens to use (default 4194304)\n"
      "  --seq-len N          sequence length of the token file (default 2048)\n"
      "  --layers A:B         layer range [A, B) (default all)\n"
      "  --gpus 0,1           HIP devices (each GPU owns a token half and runs its own layer chain)\n"
      "  --batch-rows N       tokens per forward batch (default 4096, rounded up to whole sequences)\n"
      "  --hess-dir DIR       Hessian files (default <work>/hess)\n"
      "  --backlog N          max layers with collected-but-unquantized Hessians (default 2, ~5.1 GB each)\n"
      "  --keep-hessians      do not delete Hessians after a layer is quantized\n"
      "  --kv-format fp8|qtip K/V weight format (default fp8 for FP8-Dynamic sources, qtip for BF16 sources; fp8 on a BF16 source converts K/V to FP8)\n"
      "  --vit-batch N        trellises per Viterbi launch (default 1024; 260 KB scratch each)\n"
      "  --sigma-reg X        Hessian damping (default 1e-2)   --scale-override X (default 0.9)\n"
      "  --ldl-threads N      CPU threads per fp64 LDL (default 16)\n"
      "  --dump-dir DIR       write W / reconstructed hatW (fp32) per linear for validation\n"
      "  --seed N             sign-vector seed (default 0)\n"
      "  --dump-acts DIR      debug: dump sequence 0 (GPU 0) activations after each chain layer\n"
      "adaptive mode (--mode adaptive; see docs/ADAPTIVE.md):\n"
      "  --kl-threshold X     per-layer metric value above which the layer is written as NVFP4 (default 0.05)\n"
      "  --kl-metric lmkl|mse layer quality metric (default lmkl: mean token KL of the next-token\n"
      "                      distribution via final norm + lm_head, remaining layers original; mse:\n"
      "                      normalized MSE of the layer output)\n"
      "  --kl-tokens N        probe document length for the metric (default 512)\n"
      "  --adaptive-next-input original|quantized  source of the next layer's input activations\n"
      "  --nvfp4-calibrate on|off   Hessian-aware LDLQ rounding of NVFP4 fallback layers (default on)\n"
      "  --nvfp4-scale-search on|off  per-16-block E4M3 scale search for NVFP4 layers (default on)\n"
      "  --nvfp4-compare      also evaluate naive / search-only / calibrated NVFP4 and record their metrics\n"
      "                      (default original: Hessians/KL use original-model activations, as the\n"
      "                      classic pipeline; quantized: chain flows through the chosen layers)\n");
}

static Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      QCHECK(i + 1 < argc, "missing value for " + a);
      return argv[++i];
    };
    if (a == "--mode") o.mode = next();
    else if (a == "--model") o.model = next();
    else if (a == "--tokens") o.tokens = next();
    else if (a == "--out") o.out = next();
    else if (a == "--work") o.work = next();
    else if (a == "--hess-dir") o.hess_dir = next();
    else if (a == "--ntok") o.ntok = std::stol(next());
    else if (a == "--seq-len") o.seq_len = std::stoi(next());
    else if (a == "--batch-rows") o.batch_rows = std::stoi(next());
    else if (a == "--backlog") o.backlog = std::stoi(next());
    else if (a == "--keep-hessians") o.keep_hessians = true;
    else if (a == "--kv-format") {
      auto value = next();
      QCHECK(value == "fp8" || value == "qtip", "--kv-format must be fp8 or qtip");
      o.fp8_kv = value == "fp8";
      o.kv_explicit = true;
    }
    else if (a == "--vit-batch") o.q.vit_batch = std::stoi(next());
    else if (a == "--sigma-reg") o.q.sigma_reg = std::stod(next());
    else if (a == "--scale-override") o.q.scale_override = std::stod(next());
    else if (a == "--ldl-threads") o.q.ldl_threads = std::stoi(next());
    else if (a == "--dump-dir") o.q.dump_dir = next();
    else if (a == "--seed") o.seed = std::stoull(next());
    else if (a == "--dump-acts") o.dump_acts = next();
    else if (a == "--kl-threshold") o.kl_threshold = std::stod(next());
    else if (a == "--kl-metric") o.kl_metric = next();
    else if (a == "--kl-tokens") o.kl_tokens = std::stoi(next());
    else if (a == "--adaptive-next-input") o.adaptive_next_input = next();
    else if (a == "--nvfp4-calibrate" || a == "--nvfp4-scale-search") {
      const std::string v = next();
      QCHECK(v == "on" || v == "off", a + " expects on|off");
      (a == "--nvfp4-calibrate" ? o.nvfp4_calibrate : o.nvfp4_scale_search) = v == "on";
    } else if (a == "--nvfp4-compare") o.nvfp4_compare = true;
    else if (a == "--layers") {
      std::string v = next();
      auto c = v.find(':');
      QCHECK(c != std::string::npos, "--layers A:B");
      o.layer_begin = std::stoi(v.substr(0, c));
      o.layer_end = std::stoi(v.substr(c + 1));
    } else if (a == "--gpus") {
      o.gpus.clear();
      std::stringstream ss(next());
      std::string t;
      while (std::getline(ss, t, ',')) o.gpus.push_back(std::stoi(t));
    } else if (a == "-h" || a == "--help") {
      usage();
      std::exit(0);
    } else {
      usage();
      die("unknown option " + a);
    }
  }
  QCHECK(o.mode == "run" || o.mode == "collect" || o.mode == "quantize" || o.mode == "assemble" ||
             o.mode == "adaptive",
         "--mode must be run, collect, quantize, assemble or adaptive");
  QCHECK(!o.out.empty() && !o.work.empty(), "--out and --work are required");
  QCHECK(o.seq_len > 0 && o.batch_rows > 0, "--seq-len and --batch-rows must be positive");
  long batch = ((long(o.batch_rows) + o.seq_len - 1) / o.seq_len) * o.seq_len;
  QCHECK(batch <= INT32_MAX, "forward batch is too large");
  o.batch_rows = int(batch);
  if (o.hess_dir.empty()) o.hess_dir = o.work + "/hess";
  return o;
}

int main(int argc, char** argv) {
  Options o = parse(argc, argv);
  Pipeline p(o);
  if (o.mode == "assemble") p.assemble();
  else p.run();
  return 0;
}
