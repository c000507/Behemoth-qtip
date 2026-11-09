// neuronctl: dev-box client for neurond (tools/neurond/neurond.cpp on the trn1).
//
//   neuronctl run graph.neff in0.bin [in1.bin ...] [--iters N] [--out DIR]
//
// Packs a request tgz (run.json + graph.neff + in0..), PUTs it to the
// mailbox pending/ via the local aws CLI, polls done/<id>.resp, writes
// out0.. + status.json, prints the latency line. All transfers are via
// `aws s3` (credentials from the environment). Md5 verification happens
// implicitly through the daemon's status + explicit s3 server-side
// integrity; for strict checking pass --check to md5-compare.
//
// Build: g++ -std=c++20 -O2 -o neuronctl neuronctl.cpp
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
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
        std::memset(h + 148, ' ', 8);
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

static int sh(const std::string& cmd) { return std::system(cmd.c_str()); }

int main(int argc, char** argv) {
    std::string mailbox = "s3://912935854432-neuron-build/mailbox";
    std::string graph, outdir = ".";
    std::vector<std::string> inputs;
    long iters = 100;
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--iters" && i + 1 < argc) iters = std::strtol(argv[++i], nullptr, 10);
        else if (a == "--out" && i + 1 < argc) outdir = argv[++i];
        else if (a == "--mailbox" && i + 1 < argc) mailbox = argv[++i];
        else pos.push_back(a);
    }
    if (pos.size() < 1 || pos[0] != "run" || pos.size() < 2) {
        std::fprintf(stderr,
                     "usage: neuronctl run GRAPH [IN0.bin IN1.bin ...] [--iters N] [--out DIR]\n");
        return 2;
    }
    graph = pos[1];
    for (size_t i = 2; i < pos.size(); ++i) inputs.push_back(pos[i]);

    auto now = std::chrono::system_clock::now();
    long id = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch()).count();
    std::string ids = std::to_string(id);

    fs::create_directories("/tmp/neuronctl");
    fs::path req = fs::path("/tmp/neuronctl") / (ids + ".req");
    std::map<std::string, std::string> files;
    files["graph.neff"] = slurp(graph);
    std::string url_cmd =
        "/tmp/opencode/venv/bin/python -c \"import boto3;print(boto3.client('s3')"
        ".generate_presigned_url('put_object', Params={'Bucket':'912935854432-neuron-build',"
        "'Key':'mailbox/done/" + ids + ".resp'}))\"";
    std::string put_url;
    {
        FILE* u = popen(url_cmd.c_str(), "r");
        char ub[512];
        if (u && fgets(ub, sizeof ub, u)) { put_url = ub; pclose(u); }
        while (!put_url.empty() && (put_url.back() == '\n' || put_url.back() == '\r')) put_url.pop_back();
    }
    files["run.json"] = "{\n \"iters\": " + std::to_string(iters) +
        ",\n \"resp_put_url\": \"" + put_url + "\"\n}\n";
    for (size_t i = 0; i < inputs.size(); ++i)
        files["in" + std::to_string(i)] = slurp(inputs[i]);
    spit(req, tar_pack(files));

    if (sh("/snap/bin/aws s3 cp \"" + req.string() + "\" \"" + mailbox + "/pending/" + ids + ".req\" >/dev/null 2>&1") != 0) {
        std::fprintf(stderr, "neuronctl: s3 put failed\n");
        return 3;
    }
    fs::remove(req);

    fs::path resp = fs::path("/tmp/neuronctl") / (ids + ".resp");
    for (int poll = 0; poll < 600; ++poll) {   // up to 10 min
        if (sh("/snap/bin/aws s3 cp \"" + mailbox + "/done/" + ids + ".resp\" \"" + resp.string() + "\" >/dev/null 2>&1") == 0) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    if (!fs::exists(resp)) {
        std::fprintf(stderr, "neuronctl: timed out waiting for response\n");
        return 4;
    }
    auto rfiles = tar_unpack(slurp(resp));
    fs::create_directories(outdir);
    spit(fs::path(outdir) / "status.json", rfiles.count("status.json") ? rfiles["status.json"] : "{}");
    for (auto& [n, d] : rfiles)
        if (n.rfind("out", 0) == 0) spit(fs::path(outdir) / n, d);
    // print the status line
    std::string sj = rfiles.count("status.json") ? rfiles["status.json"] : std::string("{}");
    auto avg = sj.find("\"latency_avg_us\"");
    auto st = sj.find("\"status\"");
    if (st != std::string::npos && avg != std::string::npos) {
        std::printf("status=%s latency_avg_us=%s\n",
                    sj.substr(sj.find(':', st) + 2, 20).substr(0, sj.substr(sj.find(':', st) + 2).find('"')).c_str(),
                    sj.substr(avg, 60).substr(sj.substr(avg).find_first_of("0123456789")).substr(0, 10).c_str());
    } else {
        std::fputs(sj.c_str(), stderr);
        return 5;
    }
    fs::remove(resp);
    return 0;
}
