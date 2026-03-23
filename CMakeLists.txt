cmake_minimum_required(VERSION 3.18)
project(four_stage LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Test executable
add_executable(four_stage_test test_kernel.cu)
target_include_directories(four_stage_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(four_stage_test PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)
