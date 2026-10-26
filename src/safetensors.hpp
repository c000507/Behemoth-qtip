// Minimal safetensors reader (sharded via model.safetensors.index.json) and writer.
#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace qtip {

struct TensorInfo {
  std::string dtype;  // "BF16", "F8_E4M3", "F32", "F16", "I16", "I64", ...
  std::vector<int64_t> shape;
  uint64_t begin = 0, end = 0;  // absolute file offsets
  std::string file;
  size_t bytes() const { return size_t(end - begin); }
  int64_t numel() const {
    int64_t n = 1;
    for (auto s : shape) n *= s;
    return n;
  }
};

class SafeTensorsReader {
 public:
  explicit SafeTensorsReader(const std::string& model_dir);
  ~SafeTensorsReader();
  bool has(const std::string& name) const { return tensors_.count(name) != 0; }
  const TensorInfo& info(const std::string& name) const;
  void read_into(const std::string& name, void* dst, size_t dst_bytes) const;
  std::vector<uint8_t> read(const std::string& name) const;
  const std::map<std::string, TensorInfo>& tensors() const { return tensors_; }

 private:
  void parse_file(const std::string& path);
  int fd(const std::string& file) const;
  std::string dir_;
  std::map<std::string, TensorInfo> tensors_;
  mutable std::mutex mu_;
  mutable std::map<std::string, int> fds_;
};

class SafeTensorsWriter {
 public:
  // data is copied
  void add(const std::string& name, const std::string& dtype, const std::vector<int64_t>& shape, const void* data,
           size_t bytes);
  // atomically writes path (tmp + rename); returns total bytes
  size_t write(const std::string& path, const std::map<std::string, std::string>& metadata = {{"format", "pt"}}) const;
  std::vector<std::string> names() const;

 private:
  struct Entry {
    std::string dtype;
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;
  };
  std::map<std::string, Entry> entries_;
};

}  // namespace qtip
