// gpustress2: synchronized multi-GPU load shapes (one process, one thread per device, host barrier).
// usage: gpustress2 MODE SECONDS DEV[,DEV...] [light_ms=20 heavy_ms=1]
//  MODE: spike   = light memory-streaming phase (clocks boost to max, low power) then ALL devices start a heavy
//                  FMA burst at the same instant (barrier) -> max synchronous di/dt at max V/f
//        membound= pure VRAM streaming (reference: what V/f does a memory-bound kernel reach)
//        syncpulse = all devices idle together, then all heavy together (period = light+heavy ms)
#include <hip/hip_runtime.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#define CK(x) do{hipError_t e=(x); if(e!=hipSuccess){fprintf(stderr,"HIP error %s at %s:%d\n",hipGetErrorString(e),__FILE__,__LINE__); exit(3);} }while(0)
__global__ void heavy(float *out, int iters, float seed) {
  float a = seed + threadIdx.x, b = 1.0001f, c = 0.9999f, d = seed * 0.5f;
  __shared__ float lds[1024]; lds[threadIdx.x] = a;
  for (int i = 0; i < iters; ++i) { a = fmaf(a, b, c); b = fmaf(b, c, d); c = fmaf(c, d, a); d = fmaf(d, a, b); lds[(threadIdx.x * 7 + i) & 1023] += a; }
  if (a + b + c + d + lds[threadIdx.x] == 12345.678f) out[blockIdx.x] = a;
}
__global__ void stream(const float4 *__restrict__ src, float *out, size_t n4) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; float4 acc = {0,0,0,0};
  for (; i < n4; i += gridDim.x * (size_t)blockDim.x) { float4 v = src[i]; acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w; }
  if (acc.x + acc.y + acc.z + acc.w == 1.5f) out[0] = acc.x;
}
struct Barrier { std::mutex m; std::condition_variable cv; int n, count = 0, gen = 0; Barrier(int n):n(n){}
  void wait(){ std::unique_lock<std::mutex> l(m); int g = gen; if (++count == n) { count = 0; ++gen; cv.notify_all(); } else cv.wait(l, [&]{ return g != gen; }); } };
int main(int argc, char **argv) {
  if (argc < 4) { fprintf(stderr, "usage: gpustress2 MODE SECONDS DEV[,DEV..] [light_ms heavy_ms]\n"); return 2; }
  std::string mode = argv[1]; double secs = atof(argv[2]);
  std::vector<int> devs; { std::string s = argv[3]; size_t p = 0; while (p <= s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); devs.push_back(atoi(s.substr(p, q - p).c_str())); p = q + 1; } }
  double light_ms = argc > 4 ? atof(argv[4]) : 20, heavy_ms = argc > 5 ? atof(argv[5]) : 1;
  printf("mode=%s secs=%.0f devs=%zu light=%.2fms heavy=%.2fms\n", mode.c_str(), secs, devs.size(), light_ms, heavy_ms); fflush(stdout);
  Barrier bar((int)devs.size()); std::atomic<long> cycles{0};
  auto t0 = std::chrono::steady_clock::now();
  auto el = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
  std::vector<std::thread> th;
  for (int dev : devs) th.emplace_back([&, dev] {
    CK(hipSetDevice(dev)); hipDeviceProp_t p{}; CK(hipGetDeviceProperties(&p, dev));
    int blocks = p.multiProcessorCount * 8; hipStream_t s; CK(hipStreamCreate(&s));
    float *out; CK(hipMalloc(&out, 1 << 20)); const size_t VB = 1ull << 30; float *vb; CK(hipMalloc(&vb, VB)); CK(hipMemset(vb, 0, VB));
    long n = 0;
    while (el() < secs) {
      if (mode == "membound") { stream<<<blocks * 4, 256, 0, s>>>((const float4 *)vb, out, VB / 16); if ((n & 3) == 0) CK(hipStreamSynchronize(s)); }
      else {
        auto ps = std::chrono::steady_clock::now();
        auto ms_since = [&] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ps).count(); };
        if (mode == "spike") { while (ms_since() < light_ms) { stream<<<blocks * 4, 256, 0, s>>>((const float4 *)vb, out, VB / 64); CK(hipStreamSynchronize(s)); } }
        else { CK(hipStreamSynchronize(s)); std::this_thread::sleep_for(std::chrono::microseconds((long)(light_ms * 1000))); }
        bar.wait(); // all devices launch heavy at the same instant
        auto hs = std::chrono::steady_clock::now();
        while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - hs).count() < heavy_ms) { heavy<<<blocks, 1024, 0, s>>>(out, 400, (float)n); CK(hipStreamSynchronize(s)); }
      }
      ++n; if (dev == devs[0]) cycles = n;
    }
    CK(hipDeviceSynchronize());
  });
  double rep = 0; while (el() < secs) { std::this_thread::sleep_for(std::chrono::seconds(1)); if (el() > rep + 30) { rep += 30; printf("t=%.0fs cycles=%ld\n", el(), cycles.load()); fflush(stdout); } }
  for (auto &t : th) t.join(); printf("done cycles=%ld\n", cycles.load()); return 0;
}
