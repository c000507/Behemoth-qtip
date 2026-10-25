#include "behemoth/gpu.hpp"
#include <algorithm>
#include <iostream>
#include <random>

using namespace behemoth;
int main(int argc, char **argv) {
    try {
        if (argc < 2) {
            std::cerr << "Usage: behemoth_projection_bench MODEL_DIRECTORY [ROWS=1] [LAYER=0] "
                         "[DEVICE=0] [LAYOUT=both|all|row-major|group-major|sorted] "
                         "[SEARCH=conservative|exhaustive]\n";
            return 2;
        }
        int rows = argc > 2 ? std::stoi(argv[2]) : 1, layer = argc > 3 ? std::stoi(argv[3]) : 0,
            device = argc > 4 ? std::stoi(argv[4]) : 0;
        if (rows < 1 || rows > 1024)
            throw std::invalid_argument("Rows must be 1..1024");
        std::vector<CodeLayout> layouts;
        std::string requested = argc > 5 ? argv[5] : "both";
        if (requested == "both")
            layouts = {CodeLayout::RowMajor, CodeLayout::GroupMajor};
        else if (requested == "all")
            layouts = {CodeLayout::RowMajor, CodeLayout::GroupMajor, CodeLayout::Sorted};
        else
            layouts = {parse_code_layout(requested)};
        auto search = parse_projection_search(argc > 6 ? argv[6] : "conservative");
        HIP_CHECK(hipSetDevice(device));
        verify_device();
        Checkpoint checkpoint(argv[1]);
        ProjectionRunner runner;
        runner.set_search(search);
        ProjectionWorkspace workspace;
        hipStream_t stream{};
        HIP_CHECK(hipStreamCreate(&stream));
        hipEvent_t begin{}, end{};
        HIP_CHECK(hipEventCreate(&begin));
        HIP_CHECK(hipEventCreate(&end));
        nlohmann::json reports = nlohmann::json::array();
        bool passed = true;
        for (const auto &[name, column] :
             std::vector<std::pair<std::string, bool>>{{"self_attn.q_proj", true},
                                                       {"self_attn.k_proj", true},
                                                       {"self_attn.v_proj", true},
                                                       {"self_attn.o_proj", false},
                                                       {"mlp.gate_proj", true},
                                                       {"mlp.up_proj", true},
                                                       {"mlp.down_proj", false}}) {
            auto host = checkpoint.projection(layer, name, 0, column);
            std::mt19937 rng(29);
            std::normal_distribution<float> normal(0, .1f);
            std::vector<BFloat16> values(size_t(rows) * size_t(host.in));
            for (auto &v : values)
                v = BFloat16(normal(rng));
            DeviceBuffer<BFloat16> input;
            input.upload(values);
            DeviceBuffer<float> output(size_t(rows) * size_t(host.out));
            std::vector<float> baseline;
            for (auto layout : layouts) {
                GpuProjection projection;
                // "sorted" benchmarks keep the group-major codes as well so that
                // every mode (and the fused baseline) runs on the same weight.
                projection.upload(host, layout == CodeLayout::Sorted ? CodeLayout::GroupMajor
                                                                      : layout);
                if (layout == CodeLayout::Sorted)
                    projection.upload_sorted(host);
                auto execute = [&](ProjectionMode mode) {
                    runner.set_mode(mode);
                    runner.run(projection, input.data(), output.data(), rows, workspace, stream,
                               column);
                };
                execute(ProjectionMode::Fused);
                HIP_CHECK(hipStreamSynchronize(stream));
                if (baseline.empty())
                    baseline = output.download();
                struct Record {
                    ProjectionMode mode;
                    nlohmann::json report;
                    int runs = 1;
                    std::vector<double> samples;
                };
                std::vector<Record> active;
                auto warm = [&](ProjectionMode mode) {
                    nlohmann::json report = {{"projection", name},
                                             {"rows", rows},
                                             {"in", host.in},
                                             {"out", host.out},
                                             {"mode", projection_mode_name(mode)},
                                             {"code_layout", code_layout_name(layout)},
                                             {"apply_row_scale", column},
                                             {"logical_code_bytes", projection.codes.bytes()},
                                             {"sorted_layout_bytes",
                                              projection.sorted_idx.bytes() +
                                                  projection.sorted_planes.bytes() +
                                                  projection.sorted_segment_codes.bytes() +
                                                  projection.sorted_tiles.bytes()},
                                             {"sorted_dummy_fraction",
                                              projection.has_sorted()
                                                  ? double(projection.sorted_dummies) /
                                                        double(projection.sorted_entries)
                                                  : 0.},
                                             {"weight_identity", projection.tuning_identity}};
                    try {
                        HIP_CHECK(hipMemsetAsync(output.data(), 0xff, output.bytes(), stream));
                        for (int i = 0; i < 3; ++i)
                            execute(mode);
                        HIP_CHECK(hipStreamSynchronize(stream));
                        const auto error = compare_attention(output.download(), baseline);
                        passed &= error.pass;
                        report["cross_path_pass"] = error.pass;
                        report["max_abs_error"] = error.max_abs;
                        report["partial_workspace_bytes"] = workspace.partial.bytes();
                        report["status"] = "measured";
                        active.push_back({mode, std::move(report), 1, {}});
                    } catch (const ProjectionUnavailable &e) {
                        report["status"] = "unsupported";
                        report["reason"] = e.what();
                        reports.push_back(std::move(report));
                    }
                };
                // Warm every forced path before tuning auto. Then compare all
                // paths interleaved, rather than running auto only at the start.
                for (auto mode : projection_modes())
                    if (mode != ProjectionMode::Auto)
                        warm(mode);
                warm(ProjectionMode::Auto);
                const auto tuning = runner.selection(projection, rows, column);
                const auto selected =
                    parse_projection_mode(tuning.at("selected").get<std::string>());
                auto measure = [&](ProjectionMode mode, int runs) {
                    execute(mode); // same self-warm policy, outside the timing interval
                    HIP_CHECK(hipEventRecord(begin, stream));
                    for (int i = 0; i < runs; ++i)
                        execute(mode);
                    HIP_CHECK(hipEventRecord(end, stream));
                    HIP_CHECK(hipEventSynchronize(end));
                    float elapsed = 0;
                    HIP_CHECK(hipEventElapsedTime(&elapsed, begin, end));
                    const double ms = double(elapsed) / runs;
                    if (!std::isfinite(ms) || ms <= 0)
                        throw std::runtime_error("Invalid projection benchmark event duration");
                    return ms;
                };
                for (auto &record : active) {
                    const double pilot = measure(record.mode, 1);
                    record.runs = int(std::clamp(std::ceil(.5 / pilot), 1., 32.));
                }
                for (size_t round = 0; round < 7; ++round)
                    for (size_t i = 0; i < active.size(); ++i) {
                        const size_t position = round % 2 ? active.size() - 1 - i : i;
                        auto &record = active[((round / 2) + position) % active.size()];
                        record.samples.push_back(measure(record.mode, record.runs));
                    }
                auto auto_it = std::find_if(active.begin(), active.end(), [](const Record &r) {
                    return r.mode == ProjectionMode::Auto;
                });
                auto forced_it = std::find_if(active.begin(), active.end(),
                                              [&](const Record &r) { return r.mode == selected; });
                if (auto_it == active.end() || forced_it == active.end())
                    throw std::logic_error("Selected auto path was not benchmarked");
                const int paired_runs = std::max(auto_it->runs, forced_it->runs);
                std::vector<double> auto_samples, forced_samples;
                for (int pair = 0; pair < 3; ++pair) {
                    auto_samples.push_back(measure(ProjectionMode::Auto, paired_runs));
                    forced_samples.push_back(measure(selected, paired_runs));
                    forced_samples.push_back(measure(selected, paired_runs));
                    auto_samples.push_back(measure(ProjectionMode::Auto, paired_runs));
                }
                const double auto_ms = timing_median(auto_samples),
                             forced_ms = timing_median(forced_samples);
                auto_it->report["auto_tuning"] = tuning;
                auto_it->report["selected_mode"] = projection_mode_name(selected);
                auto_it->report["same_kernel_comparison"] = {
                    {"mode", projection_mode_name(selected)},
                    {"batch_runs", paired_runs},
                    {"auto_samples_ms", auto_samples},
                    {"forced_samples_ms", forced_samples},
                    {"auto_median_ms", auto_ms},
                    {"forced_median_ms", forced_ms},
                    {"relative_auto_overhead", auto_ms / forced_ms - 1},
                    {"order", "ABBA"}};
                for (auto &record : active) {
                    const double median = timing_median(record.samples);
                    record.report["median_ms"] = median;
                    record.report["batch_runs"] = record.runs;
                    record.report["samples_ms"] = record.samples;
                    record.report["min_ms"] =
                        *std::min_element(record.samples.begin(), record.samples.end());
                    record.report["max_ms"] =
                        *std::max_element(record.samples.begin(), record.samples.end());
                    record.report["logical_code_GBps"] =
                        double(projection.codes.bytes()) / (median * 1e6);
                    reports.push_back(std::move(record.report));
                }
            }
        }
        runner.set_mode(ProjectionMode::Auto);
        hipDeviceProp_t props{};
        HIP_CHECK(hipGetDeviceProperties(&props, device));
        std::cout
            << nlohmann::json(
                   {{"gpu", props.name},
                    {"arch", props.gcnArchName},
                    {"layer", layer},
                    {"rank_shard", 0},
                    {"search", projection_search_name(search)},
                    {"results", reports},
                    {"dispatch_plans", runner.profile()},
                    {"kernel_resources", runner.resources()},
                    {"baseline_layout", code_layout_name(layouts.front())},
                    {"note",
                     "All forced paths warm before auto tuning. Result medians use seven "
                     "interleaved rounds, self-warm calls and adaptive batches targeting 0.5 ms "
                     "(1..32 calls). Timings and autotuning both include activation quantization "
                     "and reconstruction where applicable. Auto also has a paired ABBA comparison "
                     "against its exact selected forced kernel. Logical code GB/s is tensor "
                     "bytes/time, not measured HBM bandwidth. Independent scalar checks are in "
                     "behemoth_gpu_tests."}})
                   .dump(2)
            << '\n';
        HIP_CHECK(hipEventDestroy(begin));
        HIP_CHECK(hipEventDestroy(end));
        HIP_CHECK(hipStreamDestroy(stream));
        return passed ? 0 : 1;
    } catch (const std::exception &e) {
        std::cerr << "Projection benchmark failed: " << e.what() << '\n';
        return 1;
    }
}
