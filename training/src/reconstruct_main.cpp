#include "train.hpp"
#include <iostream>
#include <set>
int main(int argc, char **argv) {
    try {
        qat::ReconstructionOptions o;
        std::set<std::string> seen;
        for (int i = 1; i < argc; ++i) {
            std::string key = argv[i];
            if (key == "--help") {
                std::cout << R"(Teacher-guided sequential block-output reconstruction (no cross-entropy).
qtip_reconstruct --model MIXED_QTIP --teacher ORIGINAL_FP8 --data CALIBRATION
  --validation-data HELDOUT --work WORK --out NEW_MODEL
  --device 0|cpu             Student device (default 0)
  --teacher-device 1|cpu     Teacher device (default 1)
  --samples 4 --validation-samples 1
  --steps 64 --eval-every 8  Steps per block; best held-out step is retained
  --lr 1e-4 --norm-lr 1e-5 --clip 1
  --rows 1024               MLP chunk rows; attention uses the full prepared sequence
  --arena-gb 24 --teacher-arena-gb 12
  --layers A:B              Optimize [A,B), replay prefixes (default all)
  --resume                  Resume a completed-block checkpoint in WORK
K/V FP8 weights/scales and QTIP trellis codes remain frozen. CPU is for tiny tests.
)";
                return 0;
            }
            qat::require(seen.insert(key).second, "duplicate option " + key);
            if (key == "--resume") { o.resume = true; continue; }
            qat::require(i + 1 < argc, "missing value for " + key);
            std::string value = argv[++i];
            auto parse_integer = [](const std::string &s) { size_t n = 0; int v = std::stoi(s, &n); qat::require(n == s.size(), "invalid integer"); return v; };
            auto integer = [&] { return parse_integer(value); };
            auto real = [&] { size_t n = 0; float v = std::stof(value, &n); qat::require(n == value.size(), "invalid number"); return v; };
            if (key == "--model") o.model = value;
            else if (key == "--teacher") o.teacher = value;
            else if (key == "--data") o.data = value;
            else if (key == "--validation-data") o.validation_data = value;
            else if (key == "--work") o.work = value;
            else if (key == "--out") o.output = value;
            else if (key == "--device") o.device = value == "cpu" ? -1 : integer();
            else if (key == "--teacher-device") o.teacher_device = value == "cpu" ? -1 : integer();
            else if (key == "--samples") o.samples = integer();
            else if (key == "--validation-samples") o.validation_samples = integer();
            else if (key == "--steps") o.steps = integer();
            else if (key == "--eval-every") o.eval_every = integer();
            else if (key == "--rows") o.rows = integer();
            else if (key == "--lr") o.lr = real();
            else if (key == "--norm-lr") o.norm_lr = real();
            else if (key == "--clip") o.clip = real();
            else if (key == "--arena-gb" || key == "--teacher-arena-gb") {
                float gb = real();
                qat::require(std::isfinite(gb) && gb > .256f && gb <= 1024, "invalid arena GB");
                (key == "--arena-gb" ? o.arena_bytes : o.teacher_arena_bytes) = size_t(double(gb) * 1e9);
            } else if (key == "--layers") {
                auto colon = value.find(':');
                qat::require(colon != std::string::npos, "--layers expects A:B");
                o.layer_begin = parse_integer(value.substr(0, colon));
                o.layer_end = parse_integer(value.substr(colon + 1));
            } else qat::require(false, "unknown option " + key);
        }
        qat::require(!o.model.empty() && !o.teacher.empty() && !o.data.empty() && !o.validation_data.empty() &&
                         !o.work.empty() && !o.output.empty(),
                     "required: --model --teacher --data --validation-data --work --out");
        return qat::reconstruct(o);
    } catch (const std::exception &e) {
        std::cerr << "qtip_reconstruct: " << e.what() << '\n';
        return 1;
    }
}
