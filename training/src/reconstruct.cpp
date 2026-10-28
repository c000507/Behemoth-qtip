#include "train.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>

namespace qat {
namespace {
struct Sample {
    Batch batch;
    bool validation = false;
    std::vector<float> teacher_input, student_input, target;
};
Mat input(Replica &r, const Sample &s, bool teacher) {
    r.ops->synchronize(); // Select this replica's device on the shared host thread.
    // embedding uploads document boundaries/positions. The actual block input
    // is the separately propagated source or quantized-prefix activation.
    { auto metadata = r.ops->embedding(r.embedding, s.batch); }
    auto x = r.ops->act(int(s.batch.tokens.size()), r.model.config.n);
    r.ops->upload(x, teacher ? s.teacher_input : s.student_input);
    return x;
}
std::vector<float> values(Ops &ops, Mat x) {
    std::vector<float> out(x.size());
    ops.download(x, out);
    return out;
}
double loss(Replica &r, Layer &block, const Sample &s, int rows, bool backward) {
    auto x = input(r, s, false);
    auto target = r.ops->act(x.rows, x.cols);
    r.ops->upload(target, s.target);
    auto y = block.forward(x, s.batch, rows);
    auto dy = backward ? r.ops->alloc(y.rows, y.cols) : Mat{};
    double mse = r.ops->mse(y, target, s.batch.segments, dy);
    y = {};
    target = {};
    if (backward) (void)block.backward(x, dy, s.batch, rows);
    r.ops->synchronize();
    return mse;
}
double evaluate(Replica &r, Layer &block, const std::vector<Sample> &samples, int rows, bool validation) {
    double weighted = 0, count = 0;
    for (const auto &s : samples) {
        if (s.validation != validation) continue;
        double valid = double(std::count_if(s.batch.segments.begin(), s.batch.segments.end(), [](int i) { return i >= 0; }));
        weighted += loss(r, block, s, rows, false) * valid;
        count += valid;
    }
    require(count > 0, "reconstruction evaluation has no valid tokens");
    return weighted / count;
}
// Only this block's continuous parameters move. K/V bytes/scales and trellis
// assignments stay frozen. Wscale is optimized as a log ratio, as in QAT.
struct BlockAdam {
    std::vector<Param> active;
    std::vector<float> first, second;
    std::map<size_t, std::pair<float, float>> scales; // offset -> (initial scale, log ratio)
    const ReconstructionOptions &options;
    BlockAdam(const Model &m, int layer, const std::vector<float> &p, const ReconstructionOptions &o)
        : first(p.size()), second(p.size()), options(o) {
        std::set<size_t> ids;
        for (const auto &[name, id] : m.layers[layer].params) { (void)name; ids.insert(id); }
        for (size_t id : ids) {
            active.push_back(m.params[id]);
            if (m.params[id].group == Group::wscale)
                scales.emplace(m.params[id].offset, std::pair(p[m.params[id].offset], 0.f));
        }
    }
    void step(std::vector<float> &p, std::vector<float> g, int t) {
        std::array<double, group_count> norm{};
        for (const auto &a : active)
            for (size_t i = a.offset; i < a.offset + a.size; ++i) {
                if (a.group == Group::wscale) g[i] *= p[i];
                require(std::isfinite(g[i]), "nonfinite reconstruction gradient");
                norm[int(a.group)] += double(g[i]) * g[i];
            }
        std::array<float, group_count> factor{};
        for (int i = 0; i < group_count; ++i)
            factor[i] = options.clip > 0 && norm[i] > double(options.clip) * options.clip
                ? float(options.clip / std::sqrt(norm[i])) : 1.f;
        float b1 = 1 - std::pow(.9f, float(t)), b2 = 1 - std::pow(.999f, float(t));
        for (const auto &a : active)
            for (size_t i = a.offset; i < a.offset + a.size; ++i) {
                float v = g[i] * factor[int(a.group)];
                first[i] = .9f * first[i] + .1f * v;
                second[i] = .999f * second[i] + .001f * v * v;
                float update = (a.norm ? options.norm_lr : options.lr) * (first[i] / b1) /
                               (std::sqrt(second[i] / b2) + 1e-8f);
                if (a.group == Group::wscale) {
                    auto &s = scales.at(i);
                    s.second -= update;
                    p[i] = s.first * std::exp(s.second);
                    require(p[i] > 0, "reconstruction Wscale underflow");
                } else p[i] -= update;
                require(std::isfinite(p[i]), "reconstruction parameter overflow");
            }
    }
};
void checkpoint(const fs::path &work, json &progress, int next, const std::vector<float> &p) {
    auto name = "parameters-" + std::to_string(next) + ".f32";
    auto temp = work / (name + ".tmp");
    {
        std::ofstream f(temp, std::ios::binary);
        f.write(reinterpret_cast<const char *>(p.data()), std::streamsize(p.size() * 4));
        f.close();
        require(bool(f), "reconstruction checkpoint write failed");
    }
    sync_file(temp);
    fs::rename(temp, work / name);
    auto previous = progress.value("parameters", std::string());
    progress["parameters"] = name;
    progress["parameters_sha256"] = sha256(work / name);
    progress["next_layer"] = next;
    durable_json(work / "reconstruction.json", progress);
    if (!previous.empty() && previous != name) {
        std::error_code ec;
        fs::remove(work / previous, ec);
    }
}
} // namespace
int reconstruct(const ReconstructionOptions &o) {
    require(o.device >= -1 && o.teacher_device >= -1 && o.samples > 0 && o.validation_samples > 0 &&
                o.steps > 0 && o.eval_every > 0 && o.rows > 0 && o.lr > 0 && o.norm_lr > 0 && o.clip >= 0 &&
                std::isfinite(o.lr) && std::isfinite(o.norm_lr) && std::isfinite(o.clip), "invalid reconstruction options");
    require(!o.work.empty() && !o.output.empty() && !fs::exists(o.output), "new --output and --work are required");
    Dataset calibration(o.data), validation(o.validation_data, false);
    require(validation.metadata.at("split") == "heldout", "validation must be prepared with --split heldout");
    require(calibration.identity != validation.identity && size_t(o.samples) <= calibration.count &&
                size_t(o.validation_samples) <= validation.count, "invalid or overlapping reconstruction datasets");
    std::set<int> training;
    for (int id : calibration.metadata.at("training_conversations")) training.insert(id);
    const auto evaluated = validation.metadata.at("evaluated_conversations").get<std::vector<int>>();
    require(!training.empty() && !evaluated.empty(), "datasets need explicit train/held-out conversation IDs");
    for (int id : evaluated) require(!training.contains(id), "reconstruction train/validation conversations overlap");
    Config geometry(read_json(o.model / "config.json"));
    if (o.device < 0 || o.teacher_device < 0)
        require(geometry.n <= 1024 && std::max(calibration.length, validation.length) <= 256,
                "CPU reconstruction is for tiny verification models only");
    Model student(o.model), teacher(o.teacher, true);
    require(teacher.fp8_source, "--teacher must be the original FP8 source checkpoint");
    const auto &c = student.config, &t = teacher.config;
    require(c.n == t.n && c.inter == t.inter && c.heads == t.heads && c.kv == t.kv && c.dim == t.dim &&
                c.layers == t.layers && c.vocab == t.vocab && c.theta == t.theta && c.eps == t.eps,
            "teacher/student geometry mismatch");
    require(student.embedding == teacher.embedding && student.head == teacher.head, "teacher/student frozen embedding/head mismatch");
    for (const auto &name : {"tokenizer.model", "tokenizer_config.json"})
        if (fs::exists(student.path / name) && fs::exists(teacher.path / name))
            require(sha256(student.path / name) == sha256(teacher.path / name), "teacher/student tokenizer mismatch");
    if (calibration.metadata.contains("tokenizer_model_sha256"))
        require(sha256(student.path / "tokenizer.model") == calibration.metadata.at("tokenizer_model_sha256").get<std::string>() &&
                    calibration.metadata.at("tokenizer_model_sha256") == validation.metadata.at("tokenizer_model_sha256"),
                "reconstruction tokenizer differs from prepared data");
    int end = o.layer_end < 0 ? c.layers : o.layer_end;
    require(o.layer_begin >= 0 && o.layer_begin < end && end <= c.layers, "invalid reconstruction layer range");
    auto work = fs::absolute(o.work).lexically_normal();
    for (auto p = work; !p.empty() && p != p.parent_path(); p = p.parent_path())
        require(p != student.path && p != teacher.path, "work directory cannot be inside a source model");
    fs::create_directories(work);
    json signature = {{"format", 1}, {"objective", "sequential_teacher_block_mse"},
        {"student", student.identity}, {"teacher", teacher.identity}, {"calibration", calibration.identity},
        {"validation", validation.identity}, {"samples", o.samples}, {"validation_samples", o.validation_samples},
        {"steps", o.steps}, {"eval_every", o.eval_every}, {"rows", o.rows}, {"lr", o.lr},
        {"norm_lr", o.norm_lr}, {"clip", o.clip}, {"layer_begin", o.layer_begin}, {"layer_end", end},
        {"backend", o.device < 0 ? "cpu" : "hip"}, {"teacher_backend", o.teacher_device < 0 ? "cpu" : "hip"}};
    json progress = {{"signature", signature}, {"next_layer", 0}, {"blocks", json::array()}};
    std::vector<float> parameters = student.initial;
    int next = 0;
    if (o.resume) {
        progress = read_json(work / "reconstruction.json");
        require(progress.at("signature") == signature, "reconstruction resume signature mismatch");
        auto name = progress.at("parameters").get<std::string>();
        require(fs::path(name).filename() == name, "unsafe reconstruction checkpoint path");
        require(fs::file_size(work / name) == parameters.size() * 4 &&
                    sha256(work / name) == progress.at("parameters_sha256").get<std::string>(), "invalid reconstruction parameter checkpoint");
        std::ifstream f(work / name, std::ios::binary);
        f.read(reinterpret_cast<char *>(parameters.data()), std::streamsize(parameters.size() * 4));
        require(bool(f), "reconstruction checkpoint read failed");
        for (float v : parameters) require(std::isfinite(v), "nonfinite reconstruction checkpoint");
        next = progress.at("next_layer");
        require(next >= 0 && next <= end, "invalid reconstruction resume layer");
    } else require(!fs::exists(work / "reconstruction.json"), "work already has reconstruction state; use --resume");
    int length = int(std::max(calibration.length, validation.length));
    // Block reconstruction never offloads an 88-layer activation tape.
    Replica s(student, o.device, length, o.arena_bytes, true, false);
    Replica source(teacher, o.teacher_device, length, o.teacher_arena_bytes, true, false);
    s.ops->synchronize();
    s.ops->upload(s.parameters, parameters);
    std::vector<Sample> samples;
    for (bool heldout : {false, true}) {
        auto &dataset = heldout ? validation : calibration;
        int count = heldout ? o.validation_samples : o.samples;
        for (int i = 0; i < count; ++i) {
            Sample sample;
            sample.batch = dataset.get(size_t(i));
            sample.validation = heldout;
            auto x = s.ops->embedding(s.embedding, sample.batch);
            sample.student_input = values(*s.ops, x);
            sample.teacher_input = sample.student_input;
            samples.push_back(std::move(sample));
        }
    }
    for (int l = 0; l < end; ++l) {
        source.ops->synchronize();
        source.ops->prefetch(l, 0, false);
        source.ops->acquire(0);
        {
            Layer block(*source.ops, teacher, l, source.parameters, source.gradients);
            for (auto &sample : samples) {
                auto x = input(source, sample, true);
                auto y = block.forward(x, sample.batch, o.rows);
                sample.target = values(*source.ops, y);
            }
        }
        source.ops->release(0);
        s.ops->synchronize();
        s.ops->prefetch(l, 0, false);
        s.ops->acquire(0);
        {
            Layer block(*s.ops, student, l, s.parameters, s.gradients);
            if (l >= o.layer_begin && l >= next) {
                double before = evaluate(s, block, samples, o.rows, true), best_loss = before;
                auto best = parameters;
                int best_step = 0;
                BlockAdam optimizer(student, l, parameters, o);
                std::cout << json({{"layer", l}, {"step", 0}, {"validation_mse", before}}).dump() << std::endl;
                for (int step = 1; step <= o.steps; ++step) {
                    s.ops->zero(s.gradients);
                    double fit = loss(s, block, samples[size_t((step - 1) % o.samples)], o.rows, true);
                    optimizer.step(parameters, s.grad_values(), step);
                    s.ops->upload(s.parameters, parameters);
                    block.refresh_parameters();
                    json record = {{"layer", l}, {"step", step}, {"calibration_mse", fit}};
                    if (step % o.eval_every == 0 || step == o.steps) {
                        double validation_loss = evaluate(s, block, samples, o.rows, true);
                        record["validation_mse"] = validation_loss;
                        if (validation_loss < best_loss) { best_loss = validation_loss; best = parameters; best_step = step; }
                        record["best_step"] = best_step;
                    }
                    std::cout << record.dump() << std::endl;
                }
                parameters = std::move(best);
                s.ops->upload(s.parameters, parameters);
                block.refresh_parameters();
                json result = {{"layer", l}, {"validation_mse_before", before}, {"validation_mse_after", best_loss},
                    {"calibration_mse_after", evaluate(s, block, samples, o.rows, false)},
                    {"best_step", best_step}, {"accepted", best_step != 0}};
                progress["blocks"].push_back(result);
                std::cout << result.dump() << std::endl;
            }
            for (auto &sample : samples) {
                auto x = input(s, sample, false);
                auto y = block.forward(x, sample.batch, o.rows);
                sample.student_input = values(*s.ops, y);
                sample.teacher_input = std::move(sample.target);
            }
        }
        s.ops->release(0);
        if (l >= next) checkpoint(work, progress, l + 1, parameters);
    }
    student.export_model(o.output, parameters, progress);
    return 0;
}
} // namespace qat
