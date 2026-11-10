// neurond: persistent NEFF runner daemon for the trn1 box.
//
// Architecture (v2, fault-isolated): the PARENT owns only the S3
// mailbox; every request is executed in a FORKED CHILD (execv of this
// same binary with --child). A device fault (e.g. NRT_EXEC_UNIT
// 101 "EU in limbo, cannot recover") or an exec hang dies with the
// child; the parent answers with an error response and keeps serving.
// The child re-inits nrt per request (~2.5 s) - the price of
// isolation; still far cheaper/flakier-free than per-run SSM scripts.
//
// Transport = S3 mailbox (md5 by retry + status):
//   requests:  <mailbox>/pending/<id>.req  (tgz: run.json + graph.neff + in0..)
//   responses: <mailbox>/done/<id>.resp    (tgz: status.json + out0..)
// The box role is PutObject/DeleteObject-DENIED on this bucket, so the
// response goes back via a CLIENT-PRESIGNED PUT URL carried in
// run.json (daemon: curl -T). Pending requests can never be deleted:
// the daemon dedupes via workdir/processed.txt.
//
// Start (box): nohup ./neurond --mailbox s3://... --workdir /home/ubuntu/work/neurond >> daemon.log 2>&1 &
// Restart after an instance stop/start: bash /home/ubuntu/work/neurond/start.sh
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
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
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
        std::memcpy(h + 100, "000644 \0", 8);
        std::memcpy(h + 108, "0000000 \0", 8);
        std::memcpy(h + 116, "0000000 \0", 8);
        std::snprintf(h + 124, 13, "%011lo ", (unsigned long)size);
        std::snprintf(h + 136, 13, "%011o ", 0u);
        h[156] = '0';
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

// ---------------- child: run one NEFF, write outputs + status ----------------
// argv: <self> --child REQDIR OUTDIR   (REQDIR = unpacked request files)
// writes OUTDIR/status.json + OUTDIR/out<i>; exit 0 on any completed run
// (including exec_fail - the status json carries the NRT code), exit 3
// only on internal errors. Never hangs longer than the graphs itself:
// nrt_execute returns on device timeout.
static int child_main(const std::string& reqdir, const std::string& outdir) {
    auto write_status = [&](const std::string& status, int code, double avg,
                            double mn, long iters, const std::string& detail) {
        std::ostringstream sj;
        sj << "{\n \"status\": \"" << status << "\",\n \"status_code\": " << code
           << ",\n \"iters\": " << iters << ",\n \"latency_avg_us\": " << avg
           << ",\n \"latency_min_us\": " << mn
           << ",\n \"detail\": \"" << detail << "\"\n}\n";
        spit(fs::path(outdir) / "status.json", sj.str());
    };
    long iters = 100;
    {
        std::ifstream rf(fs::path(reqdir) / "run.json");
        std::string js((std::istreambuf_iterator<char>(rf)), std::istreambuf_iterator<char>());
        auto ip = js.find("\"iters\"");
        if (ip != std::string::npos)
            iters = std::strtol(js.c_str() + js.find(':', ip) + 1, nullptr, 10);
    }
    std::string neff = slurp(fs::path(reqdir) / "graph.neff");
    std::vector<std::pair<std::string, std::string>> inputs;
    for (int n = 0; n < 64; ++n) {
        fs::path p = fs::path(reqdir) / ("in" + std::to_string(n));
        if (fs::exists(p)) inputs.push_back({"in" + std::to_string(n), slurp(p)});
    }
    NRT_STATUS s = nrt_init(NRT_FRAMEWORK_TYPE_NO_FW, "", "");
    if (s != NRT_SUCCESS) { write_status("init_fail", int(s), 0, 0, 0, ""); return 3; }

    nrt_model_t* model = nullptr;
    s = nrt_load(neff.data(), neff.size(), 0, 1, &model);
    for (int attempt = 0; attempt < 3 && s != NRT_SUCCESS; ++attempt)
        s = nrt_load(neff.data(), neff.size(), 0, 1, &model);
    if (s != NRT_SUCCESS) { write_status("load_fail", int(s), 0, 0, 0, ""); return 0; }

    nrt_tensor_info_array_t* info = nullptr;
    if (nrt_get_model_tensor_info(model, &info)) { write_status("error", 0, 0, 0, 0, "tensor_info"); return 0; }
    nrt_tensor_set_t* in = nullptr; nrt_tensor_set_t* out = nullptr;
    if (nrt_allocate_tensor_set(&in) || nrt_allocate_tensor_set(&out)) {
        write_status("error", 0, 0, 0, 0, "tensor_set"); return 0;
    }
    int ii = 0, oi = 0;
    std::vector<std::string> onames;
    for (unsigned long long i = 0; i < info->tensor_count; ++i) {
        const auto& t = info->tensor_array[i];
        nrt_tensor_t* dev = nullptr;
        if (nrt_tensor_allocate(NRT_TENSOR_PLACEMENT_DEVICE, 0, t.size, t.name, &dev)) {
            write_status("error", 0, 0, 0, 0, "tensor_alloc"); return 0;
        }
        if (t.usage == NRT_TENSOR_USAGE_INPUT) {
            std::string name = "in" + std::to_string(ii++);
            const std::string* data = nullptr;
            for (auto& [n, d] : inputs) if (n == name) data = &d;
            if (!data || (long)data->size() != (long)t.size) {
                write_status("error", 0, 0, 0, 0,
                             "input " + name + " missing/size-mismatch");
                return 0;
            }
            if (nrt_tensor_write(dev, data->data(), 0, t.size) ||
                nrt_add_tensor_to_tensor_set(in, t.name, dev)) {
                write_status("error", 0, 0, 0, 0, "input " + name); return 0;
            }
        } else {
            if (nrt_add_tensor_to_tensor_set(out, t.name, dev)) {
                write_status("error", 0, 0, 0, 0, "add output"); return 0;
            }
            onames.push_back(t.name);
        }
    }
    s = nrt_execute(model, in, out);
    if (s != NRT_SUCCESS) { write_status("exec_fail", int(s), 0, 0, 0, "warmup"); return 0; }
    double total = 0, best = 1e30;
    for (long it = 0; it < iters; ++it) {
        auto t0 = std::chrono::steady_clock::now();
        s = nrt_execute(model, in, out);
        auto t1 = std::chrono::steady_clock::now();
        if (s != NRT_SUCCESS) {
            write_status("exec_fail", int(s), total / std::max(1L, it), best, it, "");
            return 0;
        }
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        total += us; best = std::min(best, us);
    }
    int k = 0;
    for (auto& name : onames) {
        nrt_tensor_t* dev = nullptr;
        if (nrt_get_tensor_from_tensor_set(out, name.c_str(), &dev)) {
            write_status("error", 0, 0, 0, iters, "get output " + name); return 0;
        }
        // size from info (outputs were iterated in order)
        size_t sz = 0;
        for (unsigned long long i = 0; i < info->tensor_count; ++i)
            if (name == info->tensor_array[i].name) sz = info->tensor_array[i].size;
        std::string host(sz, '\0');
        if (nrt_tensor_read(dev, host.data(), 0, sz)) {
            write_status("error", 0, 0, 0, iters, "read " + name); return 0;
        }
        spit(fs::path(outdir) / ("out" + std::to_string(k++)), host);
    }
    write_status("ok", 0, total / iters, best, iters, "");
    return 0;
}

// ---------------- s3 helpers ----------------
static int s3(const std::string& op, const std::string& a, const std::string& b) {
    std::string cmd = "/usr/local/bin/aws s3 " + op + " \"" + a + "\" \"" + b + "\" 2>/dev/null";
    return std::system(cmd.c_str());
}
static bool s3_get(const std::string& key, const fs::path& to) {
    for (int t = 0; t < 5; ++t) if (s3("cp", key, to.string()) == 0) return true;
    return false;
}

int main(int argc, char** argv) {
    if (argc > 2 && std::string(argv[1]) == "--selftest") {
        auto files = tar_unpack(slurp(argv[2]));
        std::printf("n=%zu\n", files.size());
        for (auto& [n, d] : files) std::printf("  [%s] %zu\n", n.c_str(), d.size());
        return 0;
    }
    if (argc > 3 && std::string(argv[1]) == "--child") {
        return child_main(argv[2], argv[3]);
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
    fs::create_directories(workdir + "/run");
    fs::path logp = fs::path(workdir) / "daemon.log";

    logf(logp, "neurond serving " + mailbox + " (fault-isolated child mode)");

    while (true) {
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
            if (!s3_get(mailbox + "/pending/" + key, reqtg)) {
                logf(logp, "s3 get failed: " + key);
                continue;
            }
            std::string respkey = "done/" + id + ".resp";
            std::map<std::string, std::string> respfiles;
            std::string resp_put_url;

            try {
                auto files = tar_unpack(slurp(reqtg));
                if (files.count("run.json")) {
                    auto& js = files["run.json"];
                    auto up_ = js.find("\"resp_put_url\"");
                    if (up_ != std::string::npos) {
                        auto qs = js.find('"', js.find(':', up_));
                        auto qe = js.find('"', qs + 1);
                        resp_put_url = js.substr(qs + 1, qe - qs - 1);
                    }
                }
                // unpack request into run/<id>/
                fs::path rdir = fs::path(workdir) / "run" / id;
                fs::create_directories(rdir);
                for (auto& [n, d] : files) spit(rdir / n, d);
                fs::path odir = fs::path(workdir) / "run" / (id + "_out");
                fs::remove_all(odir);
                fs::create_directories(odir);

                // fork + exec a child; hard timeout 240 s
                pid_t pid = fork();
                if (pid == 0) {
                    std::string self = "/proc/self/exe";
                    std::string a1 = "--child", a2 = rdir.string(), a3 = odir.string();
                    char* const av[] = {&self[0], &a1[0], &a2[0], &a3[0], nullptr};
                    execv(self.c_str(), av);
                    _exit(127);
                }
                int st = -1; bool timed_out = false;
                for (int sec = 0; sec < 240; ++sec) {
                    auto wr = waitpid(pid, &st, WNOHANG);
                    if (wr == pid) break;
                    if (wr < 0) break;
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                    if (sec == 239) {
                        timed_out = true;
                        kill(pid, SIGKILL);
                        waitpid(pid, &st, 0);
                    }
                }
                if (timed_out || !(fs::exists(odir / "status.json"))) {
                    respfiles["status.json"] =
                        "{\n \"status\": \"error\",\n \"detail\": \"child timeout or crash\"\n}\n";
                    logf(logp, id + " child " + (timed_out ? "TIMEOUT" : "crash"));
                } else {
                    respfiles["status.json"] = slurp(odir / "status.json");
                    for (int n = 0; n < 64; ++n) {
                        fs::path p = odir / ("out" + std::to_string(n));
                        if (fs::exists(p)) respfiles["out" + std::to_string(n)] = slurp(p);
                    }
                    logf(logp, id + " -> " +
                               (respfiles["status.json"].find("\"ok\"") != std::string::npos
                                    ? "ok" : "fail"));
                }
                fs::remove_all(rdir);
                fs::remove_all(odir);
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
            {
                std::ofstream pf(fs::path(workdir) / "processed.txt", std::ios::app);
                pf << id << "\n";
            }
            fs::remove(reqtg);
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}
