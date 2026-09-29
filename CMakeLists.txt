cmake_minimum_required(VERSION 3.24)

# CUDA 13.x rejects MSVC 14.50 / VS 2026 unless this is set. Harmless elsewhere.
# Must be set before project() so compiler identification succeeds.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

project(flash_attention_from_scratch LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)

add_library(flash_attn STATIC
  src/reference/attention_reference.cpp
  src/v1_naive/qk.cu
  src/v1_naive/softmax.cu
  src/v1_naive/pv.cu
  src/v2_tiled/attention_tiled.cu
  src/v3_online_softmax/online_softmax.cu
  src/v4_flash/flash_fwd.cu
  src/v5_shared_memory/flash_shared.cu
  src/v6_warp/flash_warp.cu
  src/v7_tensorcore/flash_mma.cu
  src/v8_flash2/flash2.cu
  src/backward/flash_bwd.cu
)

target_include_directories(flash_attn PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(flash_attn PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-lineinfo;--expt-relaxed-constexpr>
)

target_link_libraries(flash_attn PUBLIC CUDA::cudart)

add_executable(flash_bench src/cli/bench_main.cpp)
target_link_libraries(flash_bench PRIVATE flash_attn)
target_include_directories(flash_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(flash_check src/cli/check_main.cpp)
target_link_libraries(flash_check PRIVATE flash_attn)
target_include_directories(flash_check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
