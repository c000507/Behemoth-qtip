#include "bitnet.hpp"
#include "safetensors.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>

namespace {
using namespace bitnet;
double lr_at(const Options &o, int step, int total) {
    if (step <= o.warmup)
        return double(step) / double(std::max(o.warmup, 1));
    const double t = double(step - o.warmup) / double(std::max(total - o.warmup, 1));
    const double c = 0.5 * (1 + std::cos(M_PI * std::min(t, 1.0)));
    return o.min_lr_ratio + (1 - o.min_lr_ratio) * c;
}
json evaluate(Trainer &tr, Model &m, const qat::Dataset &ds, int rows, int max_rows, cudaStream_t s) {
    double ce = 0, count = 0;
    const size_t limit = max_rows > 0 ? std::min<size_t>(size_t(max_rows), ds.count) : ds.count;
    for (size_t i = 0; i < limit; i += size_t(rows)) {
        std::vector<qat::Batch> rb;
        for (size_t r = i; r < std::min(i + size_t(rows), limit); ++r)
            rb.push_back(ds.get(r));
        Batch b = upload_batch(rb, s);
        auto rep = tr.evaluate(m, b, nullptr);
        ce += rep.ce, count += rep.count;
    }
    const double mean = count ? ce / count : 0;
    return {{"rows", limit}, {"tokens", count}, {"ce", mean}, {"ppl", std::exp(mean)}};
}
} // namespace

int main(int argc, char **argv) {
    try {
        Options o;
        bool no_teacher = false;
        fs::path grad_dump, init;
        long start_row = 0;
        std::set<std::string> seen;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            require(seen.insert(a).second, "duplicate option " + a);
            if (a == "--help") {
                std::cout << R"(Native BitNet b1.58 distillation (C++20/CUDA, no Python).
bitnet_train --model HF_BF16_MODEL --data PACKED [--work DIR] [options]
  --work DIR            Checkpoint directory (ckpt-N/, latest.json)
  --resume DIR          Resume from a checkpoint directory (or its latest.json)
  --init DIR            Initialise latent weights/norms from a checkpoint (fresh optimizer, new stage)
  --start-row N         First dataset row of a new run (0)
  --export DIR          Export the packed ternary checkpoint at the end
  --export-only         Only export (--resume required)
  --eval-only           Evaluate teacher (unless --no-teacher) and student on --eval-data/--data
  --eval-data PACKED    Held-out dataset for --eval-only and end-of-run evaluation
  --eval-rows N         Rows evaluated (0 = all)
  --rows N              Packed rows per micro-batch (1); --accum N micro-batches per step (1)
  --loss-rows N         Vocabulary chunk rows (256)
  --max-tokens N        Token budget for this run (1000000); --max-steps N overrides
  --warmup-steps N      Linear warmup steps (20); --min-lr-ratio X cosine floor (0.1)
  --lr X --norm-lr X    Latent weight / norm learning rates (1e-4 / 1e-4)
  --decay X --clip X    Weight decay (0) and gradient clip norm (1; 0 disables)
  --temperature T       Distillation temperature (2)
  --ce-w X --kl-w X     Loss weights: cross-entropy (1), logits KL (1; 0 disables teacher)
  --attn-w X --attn-layer L
                        MiniLM-style Q/K/V relation distillation at layer L (0 = off)
  --attn-rope           Use post-RoPE Q/K for the relation loss (default pre-RoPE)
  --no-teacher          Continued training without a teacher (stage 2)
  --no-quant            Dense bf16 student (sanity check); --no-subln disables SubLN
  --checkpoint-every N  Checkpoint interval in steps (50); --stop-after-steps N
  --grad-dump FILE      One micro-batch (rows 0..) forward/backward; writes FILE (.f32 flat gradient)
                        and FILE.json with the losses, for reference cross-checks
  --trace-tokens FILE   JSON {"tokens": [...]} forward trace -> --trace-out FILE (.f32 [T, vocab])
  --device N            CUDA device (0); --seed N (1); --log-every N (1)
)";
                return 0;
            }
            auto flag = [&](const char *name, bool &target) {
                if (a == name) {
                    target = true;
                    return true;
                }
                return false;
            };
            if (flag("--eval-only", o.eval_only) || flag("--export-only", o.export_only) ||
                flag("--no-quant", o.no_quant) || flag("--no-subln", o.no_subln) ||
                flag("--attn-rope", o.attn_distill_rope) || flag("--no-teacher", no_teacher))
                continue;
            require(i + 1 < argc, "missing value for " + a);
            std::string v = argv[++i];
            auto num = [&]() { return std::stol(v); };
            auto real = [&]() { return std::stof(v); };
            if (a == "--model")
                o.model = v;
            else if (a == "--data")
                o.data = v;
            else if (a == "--work")
                o.work = v;
            else if (a == "--resume")
                o.resume = v;
            else if (a == "--init")
                init = v;
            else if (a == "--start-row")
                start_row = num();
            else if (a == "--export")
                o.export_dir = v;
            else if (a == "--eval-data")
                o.eval_data = v;
            else if (a == "--eval-rows")
                o.eval_rows = int(num());
            else if (a == "--rows")
                o.rows = int(num());
            else if (a == "--accum")
                o.accum = int(num());
            else if (a == "--loss-rows")
                o.loss_rows = int(num());
            else if (a == "--max-tokens")
                o.max_tokens = num();
            else if (a == "--max-steps")
                o.max_steps = int(num());
            else if (a == "--warmup-steps")
                o.warmup = int(num());
            else if (a == "--min-lr-ratio")
                o.min_lr_ratio = real();
            else if (a == "--lr")
                o.lr = real();
            else if (a == "--norm-lr")
                o.norm_lr = real();
            else if (a == "--decay")
                o.decay = real();
            else if (a == "--clip")
                o.clip = real();
            else if (a == "--temperature")
                o.temperature = real();
            else if (a == "--ce-w")
                o.ce_w = real();
            else if (a == "--kl-w")
                o.kl_w = real();
            else if (a == "--attn-w")
                o.attn_w = real();
            else if (a == "--attn-layer")
                o.attn_layer = int(num());
            else if (a == "--checkpoint-every")
                o.checkpoint_every = int(num());
            else if (a == "--stop-after-steps")
                o.stop_after = int(num());
            else if (a == "--grad-dump")
                grad_dump = v;
            else if (a == "--trace-tokens")
                o.trace_tokens = v;
            else if (a == "--trace-out")
                o.trace_out = v;
            else if (a == "--device")
                o.device = int(num());
            else if (a == "--seed")
                o.seed = int(num());
            else if (a == "--log-every")
                o.log_every = int(num());
            else
                require(false, "unknown option " + a);
        }
        require(!o.model.empty(), "--model required");
        require(o.rows > 0 && o.accum > 0 && o.loss_rows > 0 && o.temperature > 0, "invalid options");
        gpu::check(cudaSetDevice(o.device), "cudaSetDevice");
        cudaStream_t s;
        gpu::check(cudaStreamCreate(&s), "stream");
        const json config = qat::read_json(o.model / "config.json");
        const Geometry geo = geometry(config);
        qtip::SafeTensorsReader reader(o.model.string());
        Model student(geo, !o.no_quant, !o.no_subln, true);
        State state;
        if (!o.resume.empty()) {
            fs::path dir = o.resume;
            if (dir.filename() == "latest.json")
                dir = dir.parent_path() / qat::read_json(dir).at("checkpoint").get<std::string>();
            // Embeddings/head are frozen and come from the source model.
            student.load_hf(reader, s);
            state = load_checkpoint(dir, student, s);
            std::cerr << json({{"resumed", dir.string()}, {"step", state.step}, {"tokens", state.tokens}}) << "\n";
        } else {
            student.load_hf(reader, s);
            if (!init.empty()) {
                fs::path dir = init;
                if (dir.filename() == "latest.json")
                    dir = dir.parent_path() / qat::read_json(dir).at("checkpoint").get<std::string>();
                State prev = load_checkpoint(dir, student, s);
                gpu::check(cudaMemset(student.m.p, 0, student.m.bytes()), "memset");
                gpu::check(cudaMemset(student.v.p, 0, student.v.bytes()), "memset");
                std::cerr << json({{"initialised_from", dir.string()}, {"step", prev.step}}) << "\n";
            }
            state.position = size_t(start_row);
        }
        const bool want_teacher = !no_teacher && !o.export_only && (o.kl_w > 0 || o.attn_w > 0 || o.eval_only ||
                                                                     !o.trace_tokens.empty());
        std::unique_ptr<Model> teacher;
        if (want_teacher) {
            teacher = std::make_unique<Model>(geo, false, false, false);
            teacher->load_hf(reader, s);
        }
        Trainer trainer(student, teacher.get(), o, s);
        json info = {{"source", o.model.string()}, {"step", state.step}, {"tokens", state.tokens},
                     {"temperature", o.temperature}, {"kl_w", o.kl_w}, {"attn_w", o.attn_w},
                     {"attn_layer", o.attn_layer}};
        if (!o.trace_tokens.empty()) {
            require(!o.trace_out.empty(), "--trace-out required");
            auto tj = qat::read_json(o.trace_tokens);
            qat::Batch b;
            b.tokens = tj.at("tokens").get<std::vector<int32_t>>();
            const size_t T = b.tokens.size();
            b.labels.assign(T, -100), b.segments.assign(T, 0), b.positions.resize(T);
            for (size_t t = 0; t < T; ++t)
                b.positions[t] = int32_t(t);
            b.boundaries();
            Batch db = upload_batch({b}, s);
            std::vector<float> logits;
            trainer.evaluate(student, db, nullptr, &logits);
            gpu::check(cudaStreamSynchronize(s), "sync");
            std::ofstream f(o.trace_out, std::ios::binary);
            f.write(reinterpret_cast<const char *>(logits.data()), std::streamsize(logits.size() * 4));
            require(bool(f), "cannot write trace");
            if (teacher) {
                std::vector<float> tl;
                trainer.evaluate(*teacher, db, nullptr, &tl);
                gpu::check(cudaStreamSynchronize(s), "sync");
                std::ofstream tf(o.trace_out.string() + ".teacher", std::ios::binary);
                tf.write(reinterpret_cast<const char *>(tl.data()), std::streamsize(tl.size() * 4));
            }
            std::cout << json({{"trace", o.trace_out.string()}, {"rows", T}, {"vocab", geo.vocab}}) << "\n";
            return 0;
        }
        if (o.export_only) {
            require(!o.export_dir.empty(), "--export required");
            export_model(o.export_dir, o.model, student, info);
            std::cout << json({{"exported", o.export_dir.string()}}) << "\n";
            return 0;
        }
        if (o.eval_only) {
            const fs::path ep = o.eval_data.empty() ? o.data : o.eval_data;
            require(!ep.empty(), "--eval-data or --data required");
            qat::Dataset ds(ep, false);
            json out = {{"student", evaluate(trainer, student, ds, o.rows, o.eval_rows, s)}};
            if (teacher)
                out["teacher"] = evaluate(trainer, *teacher, ds, o.rows, o.eval_rows, s);
            std::cout << out.dump() << "\n";
            return 0;
        }
        if (!grad_dump.empty()) {
            require(!o.data.empty(), "--data required");
            qat::Dataset gds(o.data, false);
            std::vector<qat::Batch> rb;
            for (int r = 0; r < o.rows; ++r)
                rb.push_back(gds.get(size_t(r)));
            Batch b = upload_batch(rb, s);
            trainer.zero_grad();
            auto rep = trainer.step(b);
            auto grads = trainer.gradient();
            std::ofstream f(grad_dump, std::ios::binary);
            f.write(reinterpret_cast<const char *>(grads.data()), std::streamsize(grads.size() * 4));
            require(bool(f), "cannot write gradient dump");
            json out = {{"loss_ce", rep.ce / rep.count}, {"loss_kl", rep.kl / rep.count}, {"loss_attn", rep.attn},
                        {"targets", rep.count}, {"layout", student.layout()}, {"rows", b.rows}};
            qat::durable_json(grad_dump.string() + ".json", out);
            std::cout << out.dump() << "\n";
            return 0;
        }
        require(!o.data.empty() && !o.work.empty(), "--data and --work required for training");
        qat::Dataset ds(o.data, true);
        const long tokens_per_step = long(ds.length) * o.rows * o.accum;
        const int total_steps = o.max_steps > 0 ? o.max_steps : int(std::max<long>(1, o.max_tokens / tokens_per_step));
        std::cerr << json({{"geometry", student.layout()},
                           {"dataset", {{"rows", ds.count}, {"seqlen", ds.length}}},
                           {"tokens_per_step", tokens_per_step},
                           {"total_steps", total_steps},
                           {"teacher", bool(teacher)},
                           {"quant", student.quant},
                           {"subln", student.subln}})
                  << "\n";
        fs::create_directories(o.work);
        std::ofstream log(o.work / "train.jsonl", std::ios::app);
        auto checkpoint = [&]() {
            const std::string name = "ckpt-" + std::to_string(state.step);
            save_checkpoint(o.work / name, student, state, o, info);
            qat::durable_json(o.work / "latest.json", {{"checkpoint", name}, {"step", state.step}});
        };
        int step = state.step;
        while (step < total_steps && state.position + size_t(o.rows) <= ds.count) {
            if (o.stop_after > 0 && step >= o.stop_after)
                break;
            const auto t0 = std::chrono::steady_clock::now();
            trainer.zero_grad();
            LossReport rep;
            for (int a = 0; a < o.accum && state.position + size_t(o.rows) <= ds.count; ++a) {
                std::vector<qat::Batch> rb;
                for (int r = 0; r < o.rows; ++r)
                    rb.push_back(ds.get(state.position++));
                Batch b = upload_batch(rb, s);
                auto r = trainer.step(b);
                rep.ce += r.ce, rep.kl += r.kl, rep.attn += r.attn, rep.count += r.count;
                state.tokens += long(b.rows);
            }
            ++step;
            const double gn = trainer.grad_norm() / o.accum;
            require(std::isfinite(gn), "non-finite gradient");
            float scale = 1.f / float(o.accum);
            if (o.clip > 0 && gn > o.clip)
                scale *= float(o.clip / gn);
            const double f = lr_at(o, step, total_steps);
            trainer.optimizer_step(float(o.lr * f), float(o.norm_lr * f), scale, step);
            gpu::check(cudaStreamSynchronize(s), "sync");
            state.step = step;
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (step % o.log_every == 0 || step == total_steps) {
                json rec = {{"step", step},
                            {"loss_ce", rep.count ? rep.ce / rep.count : 0},
                            {"loss_kl", rep.count ? rep.kl / rep.count : 0},
                            {"loss_attn", rep.attn / o.accum},
                            {"grad_norm", gn},
                            {"clip_scale", scale * o.accum},
                            {"lr", o.lr * f},
                            {"tokens", state.tokens},
                            {"sec", sec},
                            {"tok_s", tokens_per_step / sec}};
                std::cout << rec.dump() << std::endl;
                log << rec.dump() << "\n";
                log.flush();
            }
            if (o.checkpoint_every > 0 && step % o.checkpoint_every == 0)
                checkpoint();
        }
        if (state.step > 0)
            checkpoint();
        info["step"] = state.step, info["tokens"] = state.tokens;
        if (!o.eval_data.empty()) {
            qat::Dataset ev(o.eval_data, false);
            json out = {{"student", evaluate(trainer, student, ev, o.rows, o.eval_rows, s)}};
            if (teacher)
                out["teacher"] = evaluate(trainer, *teacher, ev, o.rows, o.eval_rows, s);
            std::cout << json({{"eval", out}}).dump() << std::endl;
            log << json({{"eval", out}, {"step", state.step}}).dump() << "\n";
        }
        if (!o.export_dir.empty()) {
            export_model(o.export_dir, o.model, student, info);
            std::cout << json({{"exported", o.export_dir.string()}}) << "\n";
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
