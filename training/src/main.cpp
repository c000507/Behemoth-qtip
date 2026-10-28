#include "train.hpp"
#include <cmath>
#include <iostream>
#include <set>
#include <sstream>
#include <sys/resource.h>
int main(int argc, char **argv) {
    try {
        // Do not let a crash dump hundreds of GB of registered model/checkpoint RAM.
        const rlimit core_limit{0, 0};
        setrlimit(RLIMIT_CORE, &core_limit);
        qat::Options o;
        std::set<std::string> seen;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            qat::require(seen.insert(a).second, "duplicate option " + a);
            if (a == "--help") {
                std::cout << R"(Native QTIP fine-tuning (no Python, PyTorch or libtorch).
qtip_finetune --model MODEL --data PACKED --work CHECKPOINTS [options]
  --devices 0,1|cpu      GPU list (default: CPU verification)
  --out DIRECTORY       Export an HF-compatible QTIP model
  --resume PATH         Native checkpoint directory or latest.json
  --rows N              MLP recompute chunk (1024)
  --loss-rows N         Vocabulary loss chunk (128)
  --max-tokens N        Single epoch token budget (500000)
  --warmup-steps N       Warmup steps (10)
  --lr X --norm-lr X    Parameter / norm learning rates (1e-4 / 1e-5)
  --norm-decay X        Norm-only AdamW decay (.01)
  --wscale-lr X         Wscale log-ratio learning rate (default: --lr)
  --clip X              Gradient norm clip (1): global limit, or default group limit
  --clip-mode M         group (default): clip su/sv/wscale/norm groups separately;
                        global: one norm over all parameters
  --clip-su X --clip-sv X --clip-wscale X --clip-norm X
                        Per-group max norm in group mode (default --clip; 0 off)
  --checkpoint-every N  Checkpoint interval (10, 0 disables periodic)
  --stop-after-steps N  Stop after absolute step N; checkpoint for resume
  --arena-gb X          Fixed GPU allocation budget (24 decimal GB)
  --eval-only           Evaluate; --work optional
  --reference           CPU FP32 math for gradient verification
)";
                return 0;
            }
            if (a == "--eval-only") {
                o.eval = true;
                continue;
            }
            if (a == "--reference") {
                o.rounding = false;
                continue;
            }
            qat::require(i + 1 < argc, "missing value for " + a);
            std::string v = argv[++i];
            auto number = [&]() {
                size_t n = 0;
                auto x = std::stoull(v, &n);
                qat::require(n == v.size() && !v.starts_with('-'), "invalid unsigned value " + v);
                return x;
            };
            auto real = [&]() {
                size_t n = 0;
                auto x = std::stof(v, &n);
                qat::require(n == v.size() && std::isfinite(x), "invalid number " + v);
                return x;
            };
            if (a == "--model")
                o.model = v;
            else if (a == "--data")
                o.data = v;
            else if (a == "--work")
                o.work = v;
            else if (a == "--out")
                o.output = v;
            else if (a == "--resume")
                o.resume = v;
            else if (a == "--devices") {
                o.devices.clear();
                std::stringstream s(v);
                std::string item;
                std::set<int> ids;
                while (std::getline(s, item, ',')) {
                    size_t n = 0;
                    int d = item == "cpu" ? -1 : std::stoi(item, &n);
                    qat::require(item == "cpu" || (n == item.size() && d >= 0), "invalid device");
                    qat::require(ids.insert(d).second, "duplicate device");
                    o.devices.push_back(d);
                }
            } else if (a == "--rows" || a == "--loss-rows") {
                auto n = number();
                qat::require(n > 0 && n <= INT32_MAX, "row count out of range");
                if (a == "--rows")
                    o.rows = int(n);
                else
                    o.loss_rows = int(n);
            } else if (a == "--max-tokens")
                o.max_tokens = number();
            else if (a == "--warmup-steps")
                o.warmup = number();
            else if (a == "--checkpoint-every")
                o.checkpoint_every = number();
            else if (a == "--stop-after-steps")
                o.stop_after = number();
            else if (a == "--lr")
                o.lr = real();
            else if (a == "--norm-lr")
                o.norm_lr = real();
            else if (a == "--norm-decay")
                o.decay = real();
            else if (a == "--clip")
                o.clip = real();
            else if (a == "--wscale-lr") {
                o.wscale_lr = real();
                qat::require(o.wscale_lr > 0, "--wscale-lr must be positive");
            } else if (a == "--clip-mode") {
                qat::require(v == "group" || v == "global", "--clip-mode must be group or global");
                o.clip_mode = v == "group" ? qat::Options::ClipMode::group : qat::Options::ClipMode::global;
            } else if (a == "--clip-su" || a == "--clip-sv" || a == "--clip-wscale" || a == "--clip-norm") {
                auto x = real();
                qat::require(x >= 0, a + " must be non-negative");
                auto g = a == "--clip-su"   ? qat::Group::su
                         : a == "--clip-sv" ? qat::Group::sv
                         : a == "--clip-wscale" ? qat::Group::wscale
                                                : qat::Group::norm;
                o.group_clip[int(g)] = x;
            }
            else if (a == "--arena-gb") {
                double gb = real();
                qat::require(gb > 0 && gb <= 1024, "arena GB out of range");
                o.arena_bytes = size_t(gb * 1e9);
            } else
                throw std::runtime_error("unknown option " + a);
        }
        return qat::train(o);
    } catch (const std::exception &e) {
        std::cerr << "qtip_finetune: " << e.what() << '\n';
        return 1;
    }
}
