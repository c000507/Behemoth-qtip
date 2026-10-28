#include "safetensors.hpp"
#include "train.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fcntl.h>
#include <fstream>
#include <future>
#include <iostream>
#include <numeric>
#include <unistd.h>

namespace qat {
json Options::signature(const Model &m, const Dataset &d) const {
    json schema = json::array();
    for (auto &p : m.params)
        schema.push_back({p.name, p.offset, p.size, p.norm});
    return {{"format", 3},
            {"model", m.identity},
            {"data", d.identity},
            {"schema", schema},
            {"devices", devices.size()},
            {"rounding", rounding},
            {"rows", rows},
            {"loss_rows", loss_rows},
            {"lr", lr},
            {"norm_lr", norm_lr},
            {"decay", decay},
            {"clip", clip},
            {"warmup", warmup},
            {"max_tokens", max_tokens},
            {"backend", devices.front() < 0 ? "cpu" : "hip"}};
}
Replica::Replica(const Model &m, int device, int seqlen, size_t arena, bool rounding)
    : model(m), ops(device < 0 ? cpu_ops(rounding) : hip_ops(device, arena)) {
    require(m.initial.size() <= INT32_MAX, "too many trainable parameters");
    int n = int(m.initial.size());
    parameters = ops->alloc(1, n);
    gradients = ops->alloc(1, n, true);
    first = ops->alloc(1, n, true);
    second = ops->alloc(1, n, true);
    ops->upload(parameters, m.initial);
    embedding = ops->frozen(m.embedding, m.config.vocab, m.config.n);
    head = ops->frozen(m.head, m.config.vocab, m.config.n);
    ops->configure_streaming(m, seqlen);
}
std::vector<float> Replica::values() {
    std::vector<float> v(parameters.size());
    ops->download(parameters, v);
    return v;
}
std::vector<float> Replica::grad_values() {
    std::vector<float> v(gradients.size());
    ops->download(gradients, v);
    return v;
}
double Replica::run(const Batch &b, const Options &o, size_t denominator, bool backward) {
    auto &op = *ops;
    op.synchronize();
    op.zero(gradients);
    auto x = op.embedding(embedding, b);
    int layers = model.config.layers;
    op.prefetch(0, 0, false);
    if (layers > 1)
        op.prefetch(1, 1, false);
    for (int l = 0; l < layers; ++l) {
        int slot = l % 2;
        op.acquire(slot);
        if (backward)
            op.checkpoint(l, x, true);
        {
            Layer layer(op, model, l, parameters, gradients);
            x = layer.forward(x, b, o.rows);
        }
        op.release(slot);
        if (l + 2 < layers)
            op.prefetch(l + 2, slot, false);
    }
    auto &fp = model.params[model.final_norm];
    auto w = op.alloc(1, int(fp.size));
    op.copy(w, parameters.col(int(fp.offset), int(fp.offset + fp.size)));
    op.round(w);
    auto norm = op.rms(x, w, model.config.eps);
    Mat dxn;
    if (backward)
        dxn = op.alloc(x.rows, x.cols);
    double loss = 0;
    for (int a = 0; a < x.rows; a += o.loss_rows) {
        int e = std::min(a + o.loss_rows, x.rows);
        auto logits = op.mm(norm.y.row(a, e), head);
        loss += op.cross_entropy(logits, std::span(b.labels).subspan(a, e - a),
                                 float(std::max(size_t(1), denominator)), backward);
        if (backward) {
            auto d = op.mm(logits, head, true);
            op.copy(dxn.row(a, e), d);
        }
    }
    if (!backward)
        return loss;
    auto dx = op.rms_backward(x, norm, w, dxn, gradients.col(int(fp.offset), int(fp.offset + fp.size)));
    x = {};
    norm = {};
    dxn = {};
    w = {};
    op.prefetch(layers - 1, (layers - 1) % 2, true);
    if (layers > 1)
        op.prefetch(layers - 2, (layers - 2) % 2, true);
    for (int l = layers - 1; l >= 0; --l) {
        int slot = l % 2;
        op.acquire(slot);
        auto input = op.checkpoint(l, {}, false);
        {
            Layer layer(op, model, l, parameters, gradients);
            dx = layer.backward(input, dx, b, o.rows);
        }
        op.release(slot);
        if (l >= 2)
            op.prefetch(l - 2, slot, true);
    }
    op.synchronize();
    return loss;
}
void save_checkpoint(const fs::path &root, Replica &r, const Progress &p, const json &signature) {
    fs::create_directories(root);
    // Keep one snapshot: the previous one is removed only after the new one is published.
    std::string previous;
    if (fs::exists(root / "latest.json"))
        previous = read_json(root / "latest.json").at("path").get<std::string>();
    std::string name = "step-" + std::to_string(p.step) + "-" +
                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    auto temp = root / (name + ".tmp"), dest = root / name;
    fs::create_directory(temp);
    try {
        qtip::SafeTensorsWriter writer;
        auto save = [&](const char *name, Mat m) {
            std::vector<float> v(m.size());
            r.ops->download(m, v);
            for (float x : v)
                require(std::isfinite(x), "nonfinite checkpoint");
            writer.add(name, "F32", {int64_t(v.size())}, v.data(), v.size() * 4);
        };
        save("parameters", r.parameters);
        save("adam_m", r.first);
        save("adam_v", r.second);
        auto state = temp / "state.safetensors";
        writer.write(state.string());
        int fd = open(state.c_str(), O_RDONLY);
        require(fd >= 0, "checkpoint open");
        int rc = fsync(fd);
        close(fd);
        require(rc == 0, "checkpoint fsync");
        durable_json(temp / "metadata.json", {{"signature", signature},
                                              {"step", p.step},
                                              {"position", p.position},
                                              {"tokens", p.tokens},
                                              {"supervised", p.supervised},
                                              {"state_sha256", sha256(state)}});
        sync_directory(temp);
        fs::rename(temp, dest);
        sync_directory(root);
        durable_json(root / "latest.json", {{"path", name}});
    } catch (...) {
        fs::remove_all(temp);
        throw;
    }
    if (!previous.empty() && previous != name && previous.find('/') == std::string::npos &&
        fs::is_directory(root / previous)) {
        std::error_code ec;
        fs::remove_all(root / previous, ec);
    }
}
Progress load_checkpoint(const fs::path &input, const std::vector<std::unique_ptr<Replica>> &replicas,
                         const json &signature) {
    require(input.extension() != ".pt",
            "Python .pt checkpoints are unsupported. Start from an exported model or a native checkpoint.");
    auto root = input;
    if (fs::is_regular_file(root)) {
        auto name = read_json(root).at("path").get<std::string>();
        require(fs::path(name).filename() == name, "unsafe checkpoint path");
        root = root.parent_path() / name;
    } else if (fs::exists(root / "latest.json")) {
        auto name = read_json(root / "latest.json").at("path").get<std::string>();
        require(fs::path(name).filename() == name, "unsafe checkpoint path");
        root /= name;
    }
    auto meta = read_json(root / "metadata.json");
    require(meta.at("signature") == signature, "checkpoint model/data/schedule/backend differs");
    require(sha256(root / "state.safetensors") == meta.at("state_sha256").get<std::string>(),
            "checkpoint checksum mismatch");
    behemoth::SafeTensorFile f(root / "state.safetensors");
    for (auto name : {"parameters", "adam_m", "adam_v"}) {
        auto v = f.floats(name);
        require(f.dtype(name) == "F32" && v.size() == replicas.front()->parameters.size(),
                "checkpoint tensor mismatch");
        std::vector<float> data(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            data[i] = v[i];
            require(std::isfinite(data[i]), "nonfinite checkpoint state");
        }
        for (auto &r : replicas)
            r->ops->upload(std::string(name) == "parameters" ? r->parameters
                           : std::string(name) == "adam_m"   ? r->first
                                                             : r->second,
                           data);
    }
    return {meta.at("step"), meta.at("position"), meta.at("tokens"), meta.at("supervised")};
}
int train(const Options &o) {
    require(!o.model.empty() && !o.data.empty(), "--model and --data required");
    require(o.rows > 0 && o.loss_rows > 0 && o.max_tokens > 0 && !o.devices.empty(),
            "invalid training options");
    require(o.devices.front() < 0 || o.rounding, "--reference is a CPU verification option");
    require(o.lr > 0 && o.norm_lr > 0 && o.decay >= 0 && o.clip >= 0, "invalid optimizer settings");
    require(o.eval || !o.work.empty(), "--work required for training");
    if (o.devices.front() < 0)
        require(o.devices.size() == 1 && Config(read_json(o.model / "config.json")).n <= 1024,
                "CPU backend is for small verification models only");
#ifndef QTIP_TRAIN_HIP
    require(o.devices.front() < 0, "HIP backend not built: configure -DQTIP_TRAIN_HIP=ON");
#endif
    require(o.output.empty() || !fs::exists(o.output), "export destination exists");
    require(o.eval || !fs::exists(o.work / "latest.json") || !o.resume.empty(),
            "work directory has a checkpoint; use --resume or a new --work");
    Model m(o.model);
    Dataset d(o.data, !o.eval);
    require(o.eval || o.max_tokens >= d.length, "token budget must cover one packed sequence");
    size_t count = std::min(d.count, size_t(o.max_tokens / d.length));
    size_t total_steps = (count + o.devices.size() - 1) / o.devices.size();
    std::vector<std::unique_ptr<Replica>> replicas;
    for (int device : o.devices) {
        require((device < 0) == (o.devices.front() < 0), "cannot mix CPU and GPU");
        replicas.push_back(std::make_unique<Replica>(m, device, int(d.length), o.arena_bytes, o.rounding));
    }
    auto signature = o.signature(m, d);
    Progress p;
    if (!o.resume.empty())
        p = load_checkpoint(o.resume, replicas, signature);
    require(p.position <= count, "checkpoint position outside training budget");
    if (o.eval) {
        double loss = 0;
        size_t targets = 0;
        for (size_t i = 0; i < d.count; ++i) {
            auto b = d.get(i);
            targets += b.targets();
            loss += replicas[0]->run(b, o, 1, false);
        }
        require(targets > 0, "dataset has no supervised tokens");
        std::cout << json({{"loss", loss / targets}, {"supervised", targets}}).dump() << '\n';
        return 0;
    }
    while (p.position < count && (!o.stop_after || p.step < o.stop_after)) {
        size_t active = std::min(replicas.size(), count - size_t(p.position));
        std::vector<Batch> b;
        size_t targets = 0, used = active * d.length;
        for (size_t i = 0; i < active; ++i) {
            b.push_back(d.get(p.position + i));
            targets += b.back().targets();
        }
        if (targets == 0) {
            p.position += active;
            p.tokens += used;
            std::cout
                << json({{"position", p.position}, {"tokens", p.tokens}, {"skipped_no_targets", true}}).dump()
                << std::endl;
            continue;
        }
        auto start = std::chrono::steady_clock::now();
        std::vector<std::future<double>> jobs;
        for (size_t i = 0; i < active; ++i)
            jobs.push_back(
                std::async(std::launch::async, [&, i] { return replicas[i]->run(b[i], o, targets, true); }));
        double loss = 0;
        for (auto &j : jobs)
            loss += j.get();
        require(std::isfinite(loss), "nonfinite loss; checkpoint unchanged");
        std::vector<float> gradient(m.initial.size(), 0);
        for (size_t i = 0; i < active; ++i) {
            auto g = replicas[i]->grad_values();
            for (size_t j = 0; j < g.size(); ++j)
                gradient[j] += g[j];
        }
        auto stats = gradient_stats(m.params, gradient);
        double norm = stats.norm;
        float factor = o.clip > 0 ? float(std::min(1., double(o.clip) / (norm + 1e-6))) : 1.f;
        for (float &x : gradient)
            x *= factor;
        float rate =
            p.step < o.warmup
                ? float(double(p.step + 1) / o.warmup)
                : float(.5 *
                        (1 + std::cos(3.141592653589793 * double(p.step - o.warmup) /
                                      std::max(uint64_t(1), uint64_t(total_steps) -
                                                                std::min(uint64_t(total_steps), o.warmup)))));
        ++p.step;
        p.position += active;
        p.tokens += used;
        p.supervised += targets;
        for (auto &r : replicas) {
            r->ops->upload(r->gradients, gradient);
            r->ops->adam(r->parameters, r->gradients, r->first, r->second, m.params, p.step, o.lr * rate,
                         o.norm_lr * rate, o.decay, 0);
            r->ops->synchronize();
        }
        double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        json record = {{"step", p.step},
                       {"position", p.position},
                       {"tokens", p.tokens},
                       {"loss", loss / targets},
                       {"grad_norm", norm},
                       {"lr", o.lr * rate},
                       {"seconds", seconds},
                       {"tokens_per_second", used / seconds},
                       {"device_peak_bytes", replicas[0]->ops->peak_bytes()}};
        record.update(stats.describe(m.params));
        std::cout << record.dump() << std::endl;
        if (o.checkpoint_every && p.step % o.checkpoint_every == 0)
            save_checkpoint(o.work, *replicas[0], p, signature);
    }
    save_checkpoint(o.work, *replicas[0], p, signature);
    if (!o.output.empty())
        m.export_model(o.output, replicas[0]->values());
    return 0;
}
} // namespace qat
