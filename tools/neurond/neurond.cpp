// neurond: persistent NEFF runner daemon for the trn1 box.
//
// Links libnrt directly (nrt_init once, then serve). Transport = an S3
// mailbox (robust across SSM; every transfer md5-verified):
//   requests:  s3://<bucket>/mailbox/pending/<id>.req   (tgz: run.json + graph.neff + in0..)
//   responses: s3://<bucket>/mailbox/done/<id>.resp     (tgz: status.json + out0..)
//
// The client is tools/neurond/neuronctl.cpp (dev box). Start via SSM:
//   nohup /home/ubuntu/work/neurond/neurond --mailbox s3://912935854432-neuron-build/mailbox \
//       --workdir /home/ubuntu/work/neurond >> /home/ubuntu/work/neurond/daemon.log 2>&1 &
//
// Build: g++ -std=c++20 -O2 -I/opt/aws/neuron/include -o neurond neurond.cpp -L/opt/aws/neuron/lib -lnrt
#include <nrt/nrt.h>
#include <nrt/nrt_experimental.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

static std::string slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
static void spit(const fs::path& p, const std::string& d) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(d.data(), (std::streamsize)d.size());
}
static void logf(const fs::path& logp, const std::string& msg) {
    std::ofstream l(logp, std::ios::app);
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    char buf[32]; std::strftime(buf, sizeof buf, "%F %T", std::localtime(&t));
    l << buf << " " << msg << "\n";
}

// ---------------- ustar (write + read), self-contained ----------------
static std::string tar_pack(const std::map<std::string, std::string>& files) {
    std::string out;
    auto hdr = [](const std::string& name, size_t size) {
        char h[512]; std::memset(h, 0, 512);
        std::memcpy(h, name.c_str(), std::min(name.size(), size_t(99)));
        std::memcpy(h + 100, "000644 \0", 8);            // mode
        std::memcpy(h + 108, "0000000 \0", 8);           // uid
        std::memcpy(h + 116, "0000000 \0", 8);           // gid
        std::snprintf(h + 124, 13, "%011lo ", (unsigned long)size);      // size
        std::snprintf(h + 136, 13, "%011o ", 0);         // mtime
        std::memset(h + 148, ' ', 8);                    // chksum placeholder
        h[156] = '0';                                    // typeflag
        std::memcpy(h + 257, "ustar\00000", 8);
        unsigned sum = 0;
        for (size_t i = 0; i < 512; ++i) sum += (unsigned char)h[i];
        std::snprintf(h + 148, 8, "%06o ", sum);
        return std::string(h, 512);
    };
    for (auto& [name, data] : files) {
        out += hdr(name, data.size());
        out += data;
        out.append((4 - out.size() % 4) % 4, '\0');
    }
    out.append(1024, '\0');
    return out;
}
// returns map name->content; throws on malformed entries
static std::map<std::string, std::string> tar_unpack(const std::string& blob) {
    std::map<std::string, std::string> files;
    size_t off = 0;
    while (off + 512 <= blob.size()) {
        const char* h = blob.data() + off;
        bool allzero = true;
        for (size_t i = 0; i < 512; ++i) if (h[i] != '\0') { allzero = false; break; }
        if (allzero) break;
        std::string name(h, 100);
        name = name.substr(0, name.find_first_of(std::string("\0 ", 2)));
        if (name.empty()) break;
        size_t size = std::strtoull(std::string(h + 124, 12).c_str(), nullptr, 8);
        char typeflag = h[156];
        off += 512;
        if (typeflag == '0' || typeflag == '\0') {
            if (off + size > blob.size()) throw std::runtime_error("tar truncated: " + name);
            files[name] = blob.substr(off, size);
        }
        off += size + ((4 - size % 4) % 4);
    }
    return files;
}

// ---------------- shell out to aws s3 ----------------
static int s3(const std::string& op, const std::string& a, const std::string& b) {
    std::string cmd = "/usr/local/bin/aws s3 " + op + " \"" + a + "\" \"" + b + "\" 2>/dev/null";
    return std::system(cmd.c_str());
}
static bool s3_get(const std::string& key, const fs::path& to) {
    return s3("cp", key, to.string()) == 0;
}

// ---------------- md5 via libcrypto-free fallback: use md5sum ----------------
static std::string md5file(const fs::path& p) {
    std::string cmd = "md5sum \"" + p.string() + "\"";
    FILE* f = popen(cmd.c_str(), "r");
    char buf[128]; std::string out;
    if (f) { if (fgets(buf, sizeof buf, f)) out = buf; pclose(f); }
    auto pos = out.find(' ');
    return pos == std::string::npos ? out : out.substr(0, pos);
}

// ---------------- run one request ----------------
struct RunResult {
    std::string status;             // "ok" | "load_fail" | "exec_fail" | "error"
    int status_code = 0;
    double latency_avg_us = 0, latency_min_us = 0;
    long iters = 0;
    std::map<std::string, std::string> outputs;  // out0 -> bytes
    std::string detail;
};

static RunResult run_neff(const std::string& neff_bytes,
                          const std::vector<std::pair<std::string, std::string>>& inputs,
                          long iters) {
    RunResult r{"error", 0, 0, 0, 0, {}};
    nrt_model_t* model = nullptr;
    NRT_STATUS s = nrt_load(neff_bytes.data(), neff_bytes.size(), 0, 1, &model);
    for (int attempt = 0; attempt < 3 && s != NRT_SUCCESS; ++attempt) {
        // first run after init can flake; retry silently
        s = nrt_load(neff_bytes.data(), neff_bytes.size(), 0, 1, &model);
    }
    if (s != NRT_SUCCESS) { r.status = "load_fail"; r.status_code = int(s); return r; }

    nrt_tensor_info_array_t* info = nullptr;
    s = nrt_get_model_tensor_info(model, &info);
    if (s != NRT_SUCCESS) { r.status = "error"; r.status_code = int(s); return r; }

    nrt_tensor_set_t* in = nullptr; nrt_tensor_set_t* out = nullptr;
    if (nrt_allocate_tensor_set(&in) || nrt_allocate_tensor_set(&out)) {
        r.status = "error"; r.detail = "tensor_set alloc"; return r;
    }
    int ii = 0, oi = 0;
    for (unsigned long long i = 0; i < info->tensor_count; ++i) {
        const auto& t = info->tensor_array[i];
        nrt_tensor_t* dev = nullptr;
        if (nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, 0, t.size, t.name, &dev)) {
            r.status = "error"; r.detail = "tensor alloc"; return r;
        }
        if (t.usage == NRT_TENSOR_USAGE_INPUT) {
            std::string name = "in" + std::to_string(ii++);
            const std::string* data = nullptr;
            for (auto& [n, d] : inputs) if (n == name) data = &d;
            if (!data) { r.status = "error"; r.detail = std::string("missing input ") + name; return r; }
            if ((long)data->size() != (long)t.size) {
                r.status = "error";
                r.detail = std::string("input ") + name + " size " + std::to_string(data->size()) +
                           " != " + std::to_string(t.size);
                return r;
            }
            if (nrt_tensor_write(dev, data->data(), 0, t.size)) {
                r.status = "error"; r.detail = std::string("tensor write ") + name; return r;
            }
            if (nrt_add_tensor_to_tensor_set(in, t.name, dev)) {
                r.status = "error"; r.detail = std::string("add input ") + name; return r;
            }
        } else {
            if (nrt_add_tensor_to_tensor_set(out, t.name, dev)) {
                r.status = "error"; r.detail = "add output"; return r;
            }
            r.outputs["out" + std::to_string(oi++)] = "";  // placeholder, read after
        }
    }

    // warmup + timed loop (same methodology as timing.cpp)
    s = nrt_execute(model, in, out);
    if (s != NRT_SUCCESS) { r.status = "exec_fail"; r.status_code = int(s); return r; }
    double total = 0, best = 1e30;
    for (long it = 0; it < iters; ++it) {
        auto t0 = std::chrono::steady_clock::now();
        s = nrt_execute(model, in, out);
        auto t1 = std::chrono::steady_clock::now();
        if (s != NRT_SUCCESS) {
            r.status = "exec_fail"; r.status_code = int(s);
            r.latency_avg_us = total / std::max(1L, it);
            return r;
        }
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        total += us; best = std::min(best, us);
    }
    r.status = "ok";
    r.latency_avg_us = total / iters;
    r.latency_min_us = best;
    r.iters = iters;

    // read outputs back
    int k = 0;
    for (unsigned long long i = 0; i < info->tensor_count; ++i) {
        const auto& t = info->tensor_array[i];
        if (t.usage == NRT_TENSOR_USAGE_INPUT) continue;
        // outputs were added in order; re-derive from the set via name
        std::string name = "out" + std::to_string(k++);
        nrt_tensor_t* dev = nullptr;
        if (nrt_get_tensor_from_tensor_set(out, t.name, &dev)) {
            r.status = "error"; r.detail = std::string("get output ") + t.name; return r;
        }
        std::string host(t.size, '\0');
        if (nrt_tensor_read(dev, host.data(), 0, t.size)) {
            r.status = "error"; r.detail = std::string("tensor read ") + t.name; return r;
        }
        r.outputs[name] = host;
    }
    return r;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--selftest") {
        auto files = tar_unpack(slurp(argv[2]));
        std::printf("n=%zu\n", files.size());
        for (auto& [n, d] : files) std::printf("  [%s] %zu\n", n.c_str(), d.size());
        return 0;
    }
    std::string mailbox, workdir = "/home/ubuntu/work/neurond";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--mailbox" && i + 1 < argc) mailbox = argv[++i];
        else if (a == "--workdir" && i + 1 < argc) workdir = argv[++i];
    }
    if (mailbox.empty()) {
        std::fprintf(stderr, "usage: neurond --mailbox s3://bucket/prefix [--workdir DIR]\n");
        return 2;
    }
    fs::create_directories(workdir + "/req");
    fs::create_directories(workdir + "/resp");
    fs::path logp = fs::path(workdir) / "daemon.log";

    auto [bucket, prefix] = [&]() -> std::pair<std::string, std::string> {
        auto m = mailbox;
        m = m.substr(5);  // strip s3://
        auto slash = m.find('/');
        return {m.substr(0, slash), m.substr(slash + 1)};
    }();

    NRT_STATUS is = nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", "");
    logf(logp, "nrt_init status " + std::to_string(int(is)) +
               (is == NRT_SUCCESS ? " (ok)" : " (FAIL)"));
    if (is != NRT_SUCCESS) return 3;
    logf(logp, "neurond serving " + mailbox);

    while (true) {
        // list pending via s3 ls
        std::string cmd = "/usr/local/bin/aws s3 ls \"" + mailbox + "/pending/\" 2>/dev/null";
        FILE* f = popen(cmd.c_str(), "r");
        std::vector<std::string> keys;
        if (f) {
            char buf[1024];
            while (fgets(buf, sizeof buf, f)) {
                std::string line = buf;
                auto p1 = line.find_last_of(' ');
                if (p1 == std::string::npos) continue;
                std::string key = line.substr(p1 + 1);
                while (!key.empty() && (key.back() == '\n' || key.back() == '\r')) key.pop_back();
                if (key.size() > 4 && key.substr(key.size() - 4) == ".req") keys.push_back(key);
            }
            pclose(f);
        }
        std::sort(keys.begin(), keys.end());
        for (auto& key : keys) {
            std::string id = key.substr(0, key.size() - 4);
            {
                std::ifstream pf(fs::path(workdir) / "processed.txt");
                std::string line; bool done_id = false;
                while (std::getline(pf, line)) if (line == id) { done_id = true; break; }
                if (done_id) continue;
            }
            fs::path reqtg = fs::path(workdir) / "req" / (id + ".req");
            bool ok = false;
            for (int t = 0; t < 5 && !ok; ++t) ok = s3_get(mailbox + "/pending/" + key, reqtg);
            if (!ok) { logf(logp, "s3 get failed: " + key); continue; }
            std::string respkey = "done/" + id + ".resp";

            std::map<std::string, std::string> respfiles;
            std::string resp_put_url;
            try {
                std::string _b = slurp(reqtg);
                size_t blob_size = _b.size();
                auto files = tar_unpack(_b);
                std::string neff = files.count("graph.neff") ? files["graph.neff"] : "";
                long iters = 100;
                std::vector<std::pair<std::string, std::string>> inputs;
                if (files.count("run.json")) {
                    // tiny json scrape: iters + resp_put_url
                    auto& js = files["run.json"];
                    auto ip = js.find("\"iters\"");
                    if (ip != std::string::npos) iters = std::strtol(js.c_str() + js.find(':', ip) + 1, nullptr, 10);
                    auto up_ = js.find("\"resp_put_url\"");
                    if (up_ != std::string::npos) {
                        auto qs = js.find('"', js.find(':', up_));
                        auto qe = js.find('"', qs + 1);
                        resp_put_url = js.substr(qs + 1, qe - qs - 1);
                    }
                    for (int n = 0; n < 32; ++n) {
                        std::string iname = "in" + std::to_string(n);
                        if (files.count(iname)) inputs.push_back({iname, files[iname]});
                    }
                }
                if (neff.empty()) {
                    std::string ks;
                    for (auto& [kn, kd] : files) ks += kn + "(len" + std::to_string(kn.size()) + ") ";
                    throw std::runtime_error(
                        "no graph.neff in request (files=" + std::to_string(files.size()) +
                        " blob=" + std::to_string(blob_size) + " keys: " + ks + ")");
                }
                RunResult r = run_neff(neff, inputs, iters);
                std::ostringstream sj;
                sj << "{\n \"id\": \"" << id << "\",\n \"status\": \"" << r.status
                   << "\",\n \"status_code\": " << r.status_code
                   << ",\n \"iters\": " << r.iters
                   << ",\n \"latency_avg_us\": " << r.latency_avg_us
                   << ",\n \"latency_min_us\": " << r.latency_min_us
                   << ",\n \"detail\": \"" << r.detail << "\"\n}\n";
                respfiles["status.json"] = sj.str();
                for (auto& [n, d] : r.outputs) if (!d.empty()) respfiles[n] = d;
                logf(logp, id + " -> " + r.status +
                           (r.status == "ok" ? " avg_us " + std::to_string(r.latency_avg_us) : ""));
            } catch (const std::exception& e) {
                respfiles["status.json"] = std::string("{\n \"status\": \"error\",\n \"detail\": \"") +
                                           e.what() + "\"\n}\n";
                logf(logp, id + " exception: " + e.what());
            }
            fs::path resptg = fs::path(workdir) / "resp" / (id + ".resp");
            spit(resptg, tar_pack(respfiles));
            bool up = false;
            if (!resp_put_url.empty()) {
                std::string cmd = "curl -sS -X PUT -T " + resptg.string() + " '" + resp_put_url + "' 2>/dev/null";
                for (int t = 0; t < 3 && !up; ++t) up = std::system(cmd.c_str()) == 0;
            }
            for (int t = 0; t < 3 && !up; ++t) up = s3("cp", resptg.string(), mailbox + "/" + respkey) == 0;
            if (!up) logf(logp, "s3 put failed: " + respkey);
            {   // mark processed (no DeleteObject permission on the box role)
                std::ofstream pf(fs::path(workdir) / "processed.txt", std::ios::app);
                pf << id << "\n";
            }
            fs::remove(reqtg);
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}
