cmake_minimum_required(VERSION 3.18)
# Header-only CUTLASS fetch uses FetchContent_Populate (not full add_subdirectory).
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()
project(HiTile LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Optional CUDA (Linux / machine with nvcc). AppleClang without toolkit: host-only targets.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(HITILE_HAS_CUDA ON)
else()
  set(HITILE_HAS_CUDA OFF)
  message(STATUS "HiTile: CUDA not found — skipping device targets (hitile_device_test, hitile_cute_smoke)")
endif()

# -----------------------------------------------------------------------------
# CUTLASS / CuTe (check phase 0.1): fixed version, header-only include
# -----------------------------------------------------------------------------
option(HITILE_FETCH_CUTLASS "Fetch CUTLASS with CMake FetchContent (needs network once)" ON)
set(HITILE_CUTLASS_GIT_TAG "v3.6.0"
    CACHE STRING "CUTLASS Git tag (see https://github.com/NVIDIA/cutlass/tags)")

if(HITILE_FETCH_CUTLASS)
  include(FetchContent)
  FetchContent_Declare(
    cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
    GIT_TAG        ${HITILE_CUTLASS_GIT_TAG}
    GIT_SHALLOW    TRUE
  )
  FetchContent_GetProperties(cutlass)
  if(NOT cutlass_POPULATED)
    FetchContent_Populate(cutlass)
  endif()
  set(HITILE_CUTLASS_INCLUDE_DIR "${cutlass_SOURCE_DIR}/include" CACHE PATH "CUTLASS headers (CuTe)")
else()
  if(NOT DEFINED HITILE_CUTLASS_INCLUDE_DIR OR NOT HITILE_CUTLASS_INCLUDE_DIR)
    message(FATAL_ERROR "Set HITILE_CUTLASS_INCLUDE_DIR to CUTLASS .../include when HITILE_FETCH_CUTLASS=OFF")
  endif()
endif()

message(STATUS "HiTile: CUTLASS/CuTe include: ${HITILE_CUTLASS_INCLUDE_DIR}")

# Default arch when CUDA is enabled (override: -DCMAKE_CUDA_ARCHITECTURES=80 ...)
if(HITILE_HAS_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 90 CACHE STRING "CUDA virtual architecture(s)")
  endif()
endif()

# Header-only library: #include <four_stage/...>
add_library(four_stage INTERFACE)
target_include_directories(four_stage INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${HITILE_CUTLASS_INCLUDE_DIR}>
)
if(HITILE_HAS_CUDA)
  target_compile_options(four_stage INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
endif()

# Host-only constexpr tests (no GPU required)
add_executable(hitile_compile_only test/compile_only.cpp)
target_link_libraries(hitile_compile_only PRIVATE four_stage)

if(HITILE_HAS_CUDA)
  # Device smoke test (requires CUDA)
  add_executable(hitile_device_test test/kernel.cu)
  target_link_libraries(hitile_device_test PRIVATE four_stage)
  set_target_properties(hitile_device_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
  )

  # CuTe / CUTLASS smoke (check phase 0.1)
  add_executable(hitile_cute_smoke test/cute_smoke.cu)
  target_link_libraries(hitile_cute_smoke PRIVATE four_stage)
  set_target_properties(hitile_cute_smoke PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
  )
endif()
