// gpustress: controllable-shape GPU load for crash discrimination.
// usage: gpustress DEV MODE SECONDS [period_ms=75 duty_pct=50]
//  MODE: burst  = kHz-rate short heavy kernels, host sync after each (decode-like load steps)
//        steady = long heavy kernels back-to-back, no host sync (constant load)
//        pulse  = period_ms cycle: duty% heavy compute, rest idle (max di/dt)
//        pcie   = pinned-host <-> device memcpy loop (D2H+H2D 64 MB) + zero-copy host reads
//        mixed  = burst compute + pcie copies interleaved
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#define CK(x) do{hipError_t e=(x); if(e!=hipSuccess){fprintf(stderr,"HIP error %s at %s:%d\n",hipGetErrorString(e),__FILE__,__LINE__); exit(3);} }while(0)
__global__ void heavy(float *out, int iters, float seed) {
  float a = seed + threadIdx.x, b = 1.0001f, c = 0.9999f, d = seed * 0.5f;
  __shared__ float lds[1024];
  lds[threadIdx.x] = a;
  for (int i = 0; i < iters; ++i) {
    a = fmaf(a, b, c); b = fmaf(b, c, d); c = fmaf(c, d, a); d = fmaf(d, a, b);
    lds[(threadIdx.x * 7 + i) & 1023] += a;
  }
  if (a + b + c + d + lds[threadIdx.x] == 12345.678f) out[blockIdx.x] = a;
}
__global__ void readhost(const float4 *__restrict__ src, float *out, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; float4 acc = {0,0,0,0};
  for (; i < n; i += gridDim.x * (size_t)blockDim.x) { float4 v = src[i]; acc.x += v.x; acc.y += v.y; acc.z+=v.z; acc.w+=v.w; }
  if (acc.x + acc.y + acc.z + acc.w == 1.5f) out[0] = acc.x;
}
int main(int argc, char **argv) {
  if (argc < 4) { fprintf(stderr, "usage: gpustress DEV MODE SECONDS [period_ms duty_pct]\n"); return 2; }
  int dev = atoi(argv[1]); std::string mode = argv[2]; double secs = atof(argv[3]);
  int period = argc > 4 ? atoi(argv[4]) : 75, duty = argc > 5 ? atoi(argv[5]) : 50;
  CK(hipSetDevice(dev));
  hipDeviceProp_t p{}; CK(hipGetDeviceProperties(&p, dev));
  printf("dev %d %s CUs=%d mode=%s secs=%.0f\n", dev, p.name, p.multiProcessorCount, mode.c_str(), secs); fflush(stdout);
  float *out; CK(hipMalloc(&out, 1 << 20));
  const size_t HB = 64u << 20; float *hbuf = nullptr, *dbuf = nullptr;
  CK(hipHostMalloc(&hbuf, HB, hipHostMallocMapped | hipHostMallocCoherent)); CK(hipMalloc(&dbuf, HB));
  float *hdev = nullptr; CK(hipHostGetDevicePointer((void **)&hdev, hbuf, 0));
  hipStream_t s; CK(hipStreamCreate(&s));
  int blocks = p.multiProcessorCount * 8;
  auto t0 = std::chrono::steady_clock::now(); long n = 0, report = 0;
  auto elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
  while (elapsed() < secs) {
    if (mode == "burst") { heavy<<<blocks, 1024, 0, s>>>(out, 2000, n); CK(hipStreamSynchronize(s)); }
    else if (mode == "steady") { heavy<<<blocks, 1024, 0, s>>>(out, 200000, n); if ((n & 7) == 0) CK(hipStreamSynchronize(s)); }
    else if (mode == "pulse") {
      auto ps = std::chrono::steady_clock::now(); double on = period * duty / 100.0;
      while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ps).count() < on) { heavy<<<blocks, 1024, 0, s>>>(out, 2000, n); CK(hipStreamSynchronize(s)); }
      std::this_thread::sleep_for(std::chrono::milliseconds(period) - (std::chrono::steady_clock::now() - ps));
    } else if (mode == "pcie") {
      CK(hipMemcpyAsync(dbuf, hbuf, HB, hipMemcpyHostToDevice, s)); CK(hipMemcpyAsync(hbuf, dbuf, HB, hipMemcpyDeviceToHost, s));
      readhost<<<blocks, 256, 0, s>>>((const float4 *)hdev, out, HB / 16); CK(hipStreamSynchronize(s));
    } else if (mode == "mixed") {
      heavy<<<blocks, 1024, 0, s>>>(out, 2000, n); CK(hipMemcpyAsync(hbuf, dbuf, 1 << 20, hipMemcpyDeviceToHost, s));
      readhost<<<blocks, 256, 0, s>>>((const float4 *)hdev, out, (1 << 20) / 16); CK(hipMemcpyAsync(dbuf, hbuf, 1 << 20, hipMemcpyHostToDevice, s)); CK(hipStreamSynchronize(s));
    } else { fprintf(stderr, "bad mode\n"); return 2; }
    ++n;
    if (elapsed() > report + 30) { report += 30; printf("t=%.0fs iters=%ld (%.1f/s)\n", elapsed(), n, n / elapsed()); fflush(stdout); }
  }
  CK(hipDeviceSynchronize()); printf("done iters=%ld\n", n); return 0;
}
