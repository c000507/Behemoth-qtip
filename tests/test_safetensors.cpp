// SafeTensorsReader on hostile / corrupt checkpoints: every malformed header or shard path must stop with a
// QCHECK (abort with a message), never read out of bounds or outside the model directory. CPU only.
#include <sys/wait.h>
#include <unistd.h>

#include <csignal>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "safetensors.hpp"

using namespace qtip;
namespace fs = std::filesystem;
static int g_fail = 0;
#define EXPECT(c, ...)                              \
  do {                                              \
    if (!(c)) {                                     \
      std::printf("  FAIL(line %d): ", __LINE__);   \
      std::printf(__VA_ARGS__);                     \
      std::printf("\n");                            \
      ++g_fail;                                     \
    }                                               \
  } while (0)

// A safetensors file with header `hdr` (its length prefix may be overridden) and `data` bytes after it.
static void write_file(const fs::path& path, const std::string& hdr, size_t data, uint64_t hlen = UINT64_MAX) {
  std::ofstream f(path, std::ios::binary);
  if (hlen == UINT64_MAX) hlen = hdr.size();
  f.write(reinterpret_cast<const char*>(&hlen), 8);
  f << hdr << std::string(data, '\0');
}

// Opens `dir` in a child process; true when the reader stopped with QCHECK's abort (not a crash or success).
static bool rejected(const fs::path& dir) {
  std::fflush(stdout);
  pid_t pid = fork();
  if (pid == 0) {
    std::freopen("/dev/null", "w", stderr);
    SafeTensorsReader r(dir.string());
    for (auto& kv : r.tensors()) (void)r.read(kv.first);
    _exit(0);
  }
  int status = 0;
  waitpid(pid, &status, 0);
  return WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
}

int main() {
  const fs::path root = fs::temp_directory_path() / ("qtip_safetensors_" + std::to_string(getpid()));
  fs::create_directories(root / "model");
  const fs::path model = root / "model", file = model / "model.safetensors";
  const std::string ok = R"({"w":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}})";

  write_file(file, ok, 12);
  {
    SafeTensorsReader r(model.string());
    EXPECT(r.read("w").size() == 12, "valid file reads 12 bytes");
  }
  write_file(file, R"({"w":{"dtype":"BF16","shape":[2,3],"data_offsets":[]}})", 12);
  EXPECT(rejected(model), "empty data_offsets");
  write_file(file, R"({"w":{"dtype":"BF16","shape":[2,3],"data_offsets":[12,0]}})", 12);
  EXPECT(rejected(model), "reversed data_offsets");
  write_file(file, R"({"w":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,1200]}})", 12);
  EXPECT(rejected(model), "offsets past the end of the file");
  write_file(file, R"({"w":{"dtype":"BF16","shape":[2,4],"data_offsets":[0,12]}})", 12);
  EXPECT(rejected(model), "shape does not match the byte count");
  write_file(file, R"({"w":{"dtype":"BF16","shape":[4294967296,4294967296],"data_offsets":[0,12]}})", 12);
  EXPECT(rejected(model), "overflowing shape");
  write_file(file, ok, 12, uint64_t(1) << 40);
  EXPECT(rejected(model), "header length past the end of the file");

  // shard paths from the index stay inside the model directory
  fs::remove(file);
  write_file(root / "outside.safetensors", ok, 12);
  for (const char* shard : {"../outside.safetensors", "/etc/passwd", "sub/model.safetensors", ".."}) {
    std::ofstream(model / "model.safetensors.index.json")
        << R"({"weight_map":{"w":")" << shard << R"("}})";
    EXPECT(rejected(model), "shard path %s", shard);
  }

  fs::remove_all(root);
  std::printf(g_fail ? "FAILED (%d)\n" : "PASS safetensors\n", g_fail);
  return g_fail ? 1 : 0;
}
