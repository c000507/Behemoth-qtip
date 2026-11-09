#include "safetensors.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

#include "common.hpp"

namespace qtip {
namespace fs = std::filesystem;
using json = nlohmann::json;

SafeTensorsReader::SafeTensorsReader(const std::string& model_dir) : dir_(model_dir) {
  fs::path idx = fs::path(model_dir) / "model.safetensors.index.json";
  if (fs::exists(idx)) {
    std::ifstream f(idx);
    json j = json::parse(f);
    std::map<std::string, int> files;
    for (auto& kv : j.at("weight_map").items()) {
      const std::string file = kv.value().get<std::string>();
      // shards live in the model directory itself: no absolute paths, no "..", no subdirectories
      QCHECK(!file.empty() && fs::path(file).filename() == file && file != "." && file != "..",
             "unsafe shard path in weight_map: " + file);
      files[file] = 1;
    }
    for (auto& kv : files) parse_file(kv.first);
  } else {
    parse_file("model.safetensors");
  }
}
SafeTensorsReader::~SafeTensorsReader() {
  for (auto& kv : fds_) ::close(kv.second);
}

// Element width of a safetensors dtype, 0 when unknown (its byte count is then only bounds-checked).
static uint64_t dtype_width(const std::string& d) {
  if (d == "F64" || d == "I64" || d == "U64") return 8;
  if (d == "F32" || d == "I32" || d == "U32") return 4;
  if (d == "F16" || d == "BF16" || d == "I16" || d == "U16") return 2;
  if (d == "F8_E4M3" || d == "F8_E5M2" || d == "F8_E8M0" || d == "I8" || d == "U8" || d == "BOOL") return 1;
  return 0;
}

// Every offset, length and shape in the header is checked against the file before it is used: a corrupt or
// hostile checkpoint fails here with an error instead of driving allocations or reads out of bounds.
void SafeTensorsReader::parse_file(const std::string& file) {
  std::string path = (fs::path(dir_) / file).string();
  std::ifstream f(path, std::ios::binary);
  QCHECK(f, "cannot open " + path);
  std::error_code ec;
  const uint64_t size = fs::file_size(path, ec);
  QCHECK(!ec && size >= 8, "invalid safetensors file " + path);
  uint64_t hlen = 0;
  f.read(reinterpret_cast<char*>(&hlen), 8);
  QCHECK(f && hlen <= 100000000 && hlen <= size - 8, "invalid safetensors header length in " + path);
  std::string hdr(hlen, '\0');
  f.read(hdr.data(), std::streamsize(hlen));
  QCHECK(f, "truncated safetensors header in " + path);
  json j = json::parse(hdr);
  QCHECK(j.is_object(), "safetensors header is not an object in " + path);
  const uint64_t data = size - 8 - hlen;
  for (auto& kv : j.items()) {
    if (kv.key() == "__metadata__") continue;
    TensorInfo t;
    t.dtype = kv.value().at("dtype").get<std::string>();
    t.shape = kv.value().at("shape").get<std::vector<int64_t>>();
    auto offs = kv.value().at("data_offsets").get<std::vector<uint64_t>>();
    QCHECK(offs.size() == 2 && offs[0] <= offs[1] && offs[1] <= data,
           "tensor offsets out of bounds: " + kv.key() + " in " + path);
    if (const uint64_t width = dtype_width(t.dtype)) {
      uint64_t count = 1;
      for (int64_t s : t.shape) {
        QCHECK(s >= 0 && (s == 0 || count <= UINT64_MAX / uint64_t(s)), "invalid shape: " + kv.key());
        count *= uint64_t(s);
      }
      QCHECK(count <= UINT64_MAX / width && count * width == offs[1] - offs[0],
             "tensor byte count does not match its shape: " + kv.key() + " in " + path);
    }
    t.begin = 8 + hlen + offs[0];
    t.end = 8 + hlen + offs[1];
    t.file = file;
    tensors_[kv.key()] = t;
  }
}

const TensorInfo& SafeTensorsReader::info(const std::string& name) const {
  auto it = tensors_.find(name);
  QCHECK(it != tensors_.end(), "tensor not found: " + name);
  return it->second;
}

int SafeTensorsReader::fd(const std::string& file) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = fds_.find(file);
  if (it != fds_.end()) return it->second;
  std::string path = (fs::path(dir_) / file).string();
  int d = ::open(path.c_str(), O_RDONLY);
  QCHECK(d >= 0, "open " + path);
  fds_[file] = d;
  return d;
}

void SafeTensorsReader::read_into(const std::string& name, void* dst, size_t dst_bytes) const {
  const TensorInfo& t = info(name);
  QCHECK(dst_bytes >= t.bytes(), "buffer too small for " + name);
  int d = fd(t.file);
  size_t done = 0, total = t.bytes();
  char* p = static_cast<char*>(dst);
  while (done < total) {
    size_t chunk = std::min<size_t>(total - done, size_t(64) << 20);
    ssize_t r = ::pread(d, p + done, chunk, off_t(t.begin + done));
    QCHECK(r > 0, "pread failed for " + name);
    done += size_t(r);
  }
}

std::vector<uint8_t> SafeTensorsReader::read(const std::string& name) const {
  std::vector<uint8_t> v(info(name).bytes());
  read_into(name, v.data(), v.size());
  return v;
}

void SafeTensorsWriter::add(const std::string& name, const std::string& dtype, const std::vector<int64_t>& shape,
                            const void* data, size_t bytes) {
  Entry e;
  e.dtype = dtype;
  e.shape = shape;
  e.data.assign(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + bytes);
  entries_[name] = std::move(e);
}

std::vector<std::string> SafeTensorsWriter::names() const {
  std::vector<std::string> v;
  for (auto& kv : entries_) v.push_back(kv.first);
  return v;
}

size_t SafeTensorsWriter::write(const std::string& path, const std::map<std::string, std::string>& metadata) const {
  json h = json::object();
  if (!metadata.empty()) {
    json m = json::object();
    for (auto& kv : metadata) m[kv.first] = kv.second;
    h["__metadata__"] = m;
  }
  uint64_t off = 0;
  for (auto& kv : entries_) {
    h[kv.first] = {{"dtype", kv.second.dtype}, {"shape", kv.second.shape},
                   {"data_offsets", {off, off + kv.second.data.size()}}};
    off += kv.second.data.size();
  }
  std::string hs = h.dump();
  while ((hs.size() + 8) % 8) hs.push_back(' ');
  std::string tmp = path + ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary);
    QCHECK(f, "cannot write " + tmp);
    uint64_t hl = hs.size();
    f.write(reinterpret_cast<const char*>(&hl), 8);
    f.write(hs.data(), std::streamsize(hs.size()));
    for (auto& kv : entries_) f.write(reinterpret_cast<const char*>(kv.second.data.data()),
                                      std::streamsize(kv.second.data.size()));
    f.flush();
    QCHECK(f.good(), "write failed: " + tmp);
  }
  fs::rename(tmp, path);
  return size_t(8 + hs.size() + off);
}

}  // namespace qtip
