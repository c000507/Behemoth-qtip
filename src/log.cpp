#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <string>

#include "common.hpp"

namespace qtip {
static std::mutex g_log_mu;
static FILE* g_log_file = nullptr;
void log_set_file(const std::string& path) {
  std::lock_guard<std::mutex> lk(g_log_mu);
  if (g_log_file) std::fclose(g_log_file);
  g_log_file = std::fopen(path.c_str(), "a");
}
void logf(const char* fmt, ...) {
  char buf[4096];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  std::time_t t = std::time(nullptr);
  char ts[32];
  std::strftime(ts, sizeof ts, "%F %T", std::localtime(&t));
  std::lock_guard<std::mutex> lk(g_log_mu);
  std::fprintf(stdout, "[%s] %s\n", ts, buf);
  std::fflush(stdout);
  if (g_log_file) {
    std::fprintf(g_log_file, "[%s] %s\n", ts, buf);
    std::fflush(g_log_file);
  }
}
}  // namespace qtip
