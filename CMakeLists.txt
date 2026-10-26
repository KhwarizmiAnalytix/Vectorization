# Standalone Vectorization; reusable through add_subdirectory or FetchContent.
cmake_minimum_required(VERSION 3.22)
cmake_policy(SET CMP0069 NEW)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(Vectorization VERSION 1.0.0 LANGUAGES CXX C)
  include(CTest)
  enable_testing()
  option(BUILD_SHARED_LIBS "Build shared libraries" ON)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
endif()
list(PREPEND CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/Cmake/tools"
  "${CMAKE_CURRENT_SOURCE_DIR}/Cmake/flags"
  "${CMAKE_CURRENT_SOURCE_DIR}/Cmake/packages")
include("${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/CMakeLists.txt")

# Default CPU backend (host-aware): x86_64 -> avx2, aarch64 -> neon, else -> no
# This logic is duplicated in include/CMakeLists.txt but needed here for Scalars setup
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _xsigma_root_proc)
if (_xsigma_root_proc MATCHES "^(x86_64|amd64|i686|i386)$")
  set(_xsigma_default_cpu_backend "avx2")
elseif(_xsigma_root_proc MATCHES "^(aarch64|arm64)$")
  set(_xsigma_default_cpu_backend "neon")
else()
  set(_xsigma_default_cpu_backend "no")
endif()

set(VECTORIZATION_CPU_BACKEND "${_xsigma_default_cpu_backend}" CACHE STRING
    "CPU SIMD backend: no, sse, avx, avx2, avx512, neon, sve")
set(VECTORIZATION_GPU_BACKEND "none" CACHE STRING "GPU backend: none, cuda, hip, metal")

# Configure math backends before adding Scalars: it owns their dependencies.
foreach(_feature IN ITEMS SLEEF SVML MKL ACCELERATE)
  option(VECTORIZATION_ENABLE_${_feature} "Enable ${_feature} math backend" OFF)
  set(SCALARS_ENABLE_${_feature} "${VECTORIZATION_ENABLE_${_feature}}"
      CACHE BOOL "Enable ${_feature} math backend through Scalars" FORCE)
endforeach()
if(NOT TARGET Memory::Memory)
  set(MEMORY_ENABLE_TESTING OFF CACHE BOOL "Dependency tests disabled" FORCE)
  set(MEMORY_ENABLE_BENCHMARK OFF CACHE BOOL "Dependency benchmarks disabled" FORCE)
  if(NOT DEFINED MEMORY_GPU_BACKEND)
    set(MEMORY_GPU_BACKEND "${VECTORIZATION_GPU_BACKEND}" CACHE STRING "Memory GPU backend")
  endif()
  add_subdirectory("${XSIGMA_THIRDPARTY_DIR}/Memory" "${CMAKE_CURRENT_BINARY_DIR}/ThirdParty/Memory" EXCLUDE_FROM_ALL)
endif()
if(NOT TARGET Scalars::Scalars)
  set(SCALARS_ENABLE_TESTING OFF CACHE BOOL "Dependency tests disabled" FORCE)
  set(SCALARS_ENABLE_BENCHMARK OFF CACHE BOOL "Dependency benchmarks disabled" FORCE)
  set(SCALARS_CPU_BACKEND "${VECTORIZATION_CPU_BACKEND}" CACHE STRING "Scalars CPU backend" FORCE)
  set(SCALARS_GPU_BACKEND "${VECTORIZATION_GPU_BACKEND}" CACHE STRING "Scalars GPU backend" FORCE)
  add_subdirectory("${XSIGMA_THIRDPARTY_DIR}/Scalars" "${CMAKE_CURRENT_BINARY_DIR}/ThirdParty/Scalars" EXCLUDE_FROM_ALL)
endif()
add_subdirectory(include)
