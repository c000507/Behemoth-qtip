cmake_minimum_required(VERSION 3.25)
project(qtip_quant LANGUAGES CXX HIP)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_HIP_ARCHITECTURES gfx1201 CACHE STRING "AMD GPU architecture")
set(QTIP_CPU_ARCH "icelake-server" CACHE STRING "-march for the host fp64 Cholesky (AVX-512)")

find_package(hip REQUIRED CONFIG)
find_package(hipblaslt REQUIRED CONFIG)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(Threads REQUIRED)

# host-only core (codebook, hadamard reference, reference viterbi, packing, fp64 LDL)
add_library(qtip_core STATIC src/qtip_core.cpp src/log.cpp src/safetensors.cpp)
target_include_directories(qtip_core PUBLIC src third_party)
target_compile_options(qtip_core PRIVATE -O3 -march=${QTIP_CPU_ARCH} -Wall -Wextra)
target_link_libraries(qtip_core PUBLIC OpenMP::OpenMP_CXX Threads::Threads)

# GPU kernels + hipBLASLt wrapper + quantizer + model forward
add_library(qtip_gpu STATIC src/kernels.hip src/viterbi.hip src/gemm.cpp src/staging.cpp src/quantizer.cpp src/model.cpp)
target_include_directories(qtip_gpu PUBLIC src third_party)
target_link_libraries(qtip_gpu PUBLIC qtip_core hip::host roc::hipblaslt)
target_compile_options(qtip_gpu PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-mno-wavefrontsize64 -O3>
                                        $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=${QTIP_CPU_ARCH} -Wall>)

add_executable(qtip_quant src/main.cpp)
target_link_libraries(qtip_quant PRIVATE qtip_gpu)
target_compile_options(qtip_quant PRIVATE -O3 -march=${QTIP_CPU_ARCH} -Wall)

add_executable(qtip_tests_cpu tests/test_cpu.cpp)
target_link_libraries(qtip_tests_cpu PRIVATE qtip_core)
add_executable(qtip_tests_gpu tests/test_gpu.cpp)
target_link_libraries(qtip_tests_gpu PRIVATE qtip_gpu)

enable_testing()
add_test(NAME cpu COMMAND qtip_tests_cpu ${CMAKE_SOURCE_DIR}/tests/refvec)
add_test(NAME gpu COMMAND qtip_tests_gpu)
