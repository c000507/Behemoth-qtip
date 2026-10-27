#pragma once
#include "train.hpp"
#include <hip/hip_runtime.h>
namespace qat::gpu {
struct View {
    float *p;
    int rows, cols, ld;
};
inline View view(Mat m) { return {m.p, m.rows, m.cols, m.ld}; }
void binary(View, View, View, int, bool, hipStream_t);
void cast(View, uint16_t *, bool, hipStream_t);
void rounded(View, hipStream_t);
void reduce(View, View, View, bool, hipStream_t);
void rms(View, View, View, View, float, hipStream_t);
void rms_back(View, View, View, View, View, View, hipStream_t);
void rope(View, View, int, int, float, const int32_t *, bool, hipStream_t);
void swiglu(View, View, View, hipStream_t);
void swiglu_back(View, View, View, View, View, hipStream_t);
void embedding(View, const uint16_t *, const int32_t *, int, hipStream_t);
void loss(View, const int32_t *, float *, float, bool, hipStream_t);
void adam(float *, const float *, float *, float *, size_t, bool, uint64_t, float, float, hipStream_t);
void flash(View, View, View, View, View, int, int, const int32_t *, hipStream_t);
void flash_back(View, View, View, View, View, View, View, View, View, int, int, const int32_t *, hipStream_t);
} // namespace qat::gpu
