# =============================================================================
# Library/Vectorization — CMakeLists.txt
#
# Provides the Vectorization library: simd<T> register abstractions, expression templates, and
# architecture-specific kernels for SSE, AVX, AVX2, AVX-512, NEON, and SVE, plus CUDA/HIP/Metal eval.
#
# VECTORIZATION_CPU_BACKEND (cache: no|sse|avx|avx2|avx512|neon|sve) selects the CPU SIMD backend.
# VECTORIZATION_GPU_BACKEND (cache: none|cuda|hip|metal) selects the GPU backend (independent of CPU type).
# Both can be active simultaneously — e.g. CPU_TYPE=avx2 + GPU_TYPE=cuda.
# Optional VECTORIZATION_ENABLE_SVML adds SVML headers.
#
# C++ preprocessor defines: sse → HAS_SSE=1; avx/avx2 → HAS_AVX=1; avx2 → HAS_AVX2=1;
# avx512 → HAS_AVX512=1; cuda → HAS_CUDA=1; hip → HAS_HIP=1; metal → HAS_METAL=1.
# VECTORIZATION_HAS_SVML=1/0   — SVML active in C++ code
# VECTORIZATION_HAS_GTEST=1/0  — Google Test available to Vectorization tests
# =============================================================================
cmake_minimum_required(VERSION 3.16)

# =============================================================================
# SECTION 1 — Cache
# options (VECTORIZATION_CPU_BACKEND, VECTORIZATION_GPU_BACKEND, tooling) and backend selectors
#
# include(utils) runs compiler feature checks (SIMD flags, exception_ptr); results are published to
# CACHE for Core (VECTORIZATION_COMPILER_FLAGS, PROJECT_*).
#
# VECTORIZATION_CPU_BACKEND selects the CPU SIMD backend (one of: no, sse, avx, avx2, avx512, neon, sve).
# VECTORIZATION_GPU_BACKEND selects the GPU backend (one of: none, cuda, hip, metal).
# Both flags are independent — you can combine any CPU backend with any GPU backend.
# =============================================================================
# Host-aware default CPU SIMD: x86 family -> avx2, aarch64 -> neon, anything else -> no.
# Matches the _xsigma_vec_is_x86 / _xsigma_vec_is_aarch64 processor regexes in
# Cmake/utils.cmake, so the default we pick here is exactly what that module will go on to
# validate. Falling back to "no" (rather than guessing "avx2") on unrecognized processors
# (ppc64le, riscv64, wasm32, 32-bit arm, s390x, ...) avoids a spurious "unsupported, falling
# back to no" WARNING later at the auto-downgrade check below.
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _xsigma_vec_proc_default)
if(_xsigma_vec_proc_default MATCHES "^(x86_64|amd64|i686|i386)$")
  set(_xsigma_vectorization_cpu_default "avx2")
elseif(_xsigma_vec_proc_default MATCHES "^(aarch64|arm64)$")
  set(_xsigma_vectorization_cpu_default "neon")
else()
  set(_xsigma_vectorization_cpu_default "no")
endif()

set(VECTORIZATION_CPU_BACKEND "${_xsigma_vectorization_cpu_default}"
    CACHE STRING "CPU SIMD backend. Options: no, sse, avx, avx2, avx512, neon, sve"
)
set_property(
  CACHE VECTORIZATION_CPU_BACKEND
  PROPERTY STRINGS
           no
           sse
           avx
           avx2
           avx512
           neon
           sve
)
if(NOT VECTORIZATION_CPU_BACKEND MATCHES "^(no|sse|avx|avx2|avx512|neon|sve)$")
  message(
    FATAL_ERROR
      "VECTORIZATION_CPU_BACKEND must be one of: no, sse, avx, avx2, avx512, neon, sve (got '${VECTORIZATION_CPU_BACKEND}')"
  )
endif()

# GPU backend. Can be set from the command line: -DVECTORIZATION_GPU_BACKEND=cuda|hip|metal|none
set(VECTORIZATION_GPU_BACKEND "none" CACHE STRING
                                           "GPU backend for Vectorization: none, cuda, hip, metal"
)
set_property(CACHE VECTORIZATION_GPU_BACKEND PROPERTY STRINGS none cuda hip metal)
if(NOT VECTORIZATION_GPU_BACKEND MATCHES "^(none|cuda|hip|metal)$")
  message(
    FATAL_ERROR
      "VECTORIZATION_GPU_BACKEND must be one of: none, cuda, hip, metal (got '${VECTORIZATION_GPU_BACKEND}')"
  )
endif()
if(VECTORIZATION_GPU_BACKEND STREQUAL "metal" AND NOT APPLE)
  message(FATAL_ERROR "VECTORIZATION_GPU_BACKEND=metal is only supported on Apple platforms.")
endif()

include(lto)
xsigma_lto_compute_default(_lto_default PROFILE PERFORMANCE)
set(VECTORIZATION_LTO_MODE "${_lto_default}"
    CACHE STRING "LTO mode for Vectorization: off | thin | full | ipo | auto"
)
set_property(CACHE VECTORIZATION_LTO_MODE PROPERTY STRINGS off thin full ipo auto)
option(VECTORIZATION_ENABLE_COVERAGE "Enable coverage instrumentation for Vectorization" OFF)
option(VECTORIZATION_ENABLE_TESTING "Build Vectorization test suite" ON)
option(VECTORIZATION_ENABLE_EXAMPLES "Build Vectorization example programs" OFF)
option(VECTORIZATION_ENABLE_GTEST "Enable GoogleTest for Vectorization" ON)
option(VECTORIZATION_ENABLE_BENCHMARK "Enable Google Benchmark for Vectorization" ON)

# Clang/GCC: append -march=native for extra instruction scheduling / tuning on the build host.
# Improves runtime on this machine; the binary may not run on older CPUs. MSVC ignores this (no flag
# added).
option(USE_NATIVE_ARCH "Prefer host CPU instruction tuning (-march=native, Clang/GCC only)" OFF)

option(VECTORIZATION_ENABLE_SVML "Enable Intel SVML short vector math library" OFF)
mark_as_advanced(VECTORIZATION_ENABLE_SVML)

# MKL VML backend — statically linked Intel MKL; mutually exclusive with SVML/SLEEF math paths.
# Enable by setting VECTORIZATION_ENABLE_MKL=ON and ensuring MKLROOT is set or MKL is
# discoverable via find_package(MKL CONFIG).
option(VECTORIZATION_ENABLE_MKL "Enable Intel MKL VML vector math backend (x86 only, static link)"
       OFF
)
mark_as_advanced(VECTORIZATION_ENABLE_MKL)

option(VECTORIZATION_ENABLE_SLEEF
       "Enable SLEEF SIMD math library (SSE/AVX/AVX2/AVX512/NEON); mutually exclusive with SVML"
       OFF
)
mark_as_advanced(VECTORIZATION_ENABLE_SLEEF)

option(VECTORIZATION_ENABLE_ACCELERATE "Enable Apple Accelerate vForce on AArch64 NEON" OFF)
mark_as_advanced(VECTORIZATION_ENABLE_ACCELERATE)

# LibTorch (PyTorch C++ frontend) comparison tests/benchmarks — auto-probed via
# CMAKE_PREFIX_PATH when enabled. Off by default so a plain configure never pays the
# find_package(Torch) cost or fails when no LibTorch install is present.
option(VECTORIZATION_ENABLE_LIBTORCH
       "Enable LibTorch (PyTorch C++) comparison tests and benchmarks" OFF
)

# Logging is required — Vectorization log/check/throw macros go through the Logging
# library. Logging is pure third-party (ThirdParty/CMakeLists.txt); added on demand.
xsigma_add_logging()
if(NOT TARGET Logging::Logging)
  message(
    FATAL_ERROR
      "Vectorization requires Logging::Logging. xsigma_add_logging() should have provided it — is ThirdParty/Logging initialized?"
  )
endif()

# Memory is required — allocator, data_ptr, data_view, and device_enum live there.
if(NOT TARGET Memory::Memory)
  message(
    FATAL_ERROR
      "Vectorization requires Memory::Memory. Configure Memory in this tree (full build or --project.vectorization, which now enables Memory)."
  )
endif()

# Profiler presence — detected from the CMake target; not a user option.
# Lets kernel entry points call PROFILER_SCOPE from `<profiler.h>`.
# Profiler is pure third-party (ThirdParty/CMakeLists.txt).
xsigma_add_profiler()
if(TARGET Profiler::Profiler)
  set(_vec_has_profiler 1)
else()
  set(_vec_has_profiler 0)
endif()

option(VECTORIZATION_ENABLE_ICECC "Use Icecream distributed compiler for Vectorization" OFF)
mark_as_advanced(VECTORIZATION_ENABLE_ICECC)
option(VECTORIZATION_ENABLE_CACHE "Enable compiler caching for the Vectorization target" ON)
mark_as_advanced(VECTORIZATION_ENABLE_CACHE)
option(VECTORIZATION_ENABLE_CLANGTIDY "Enable clang-tidy static analysis for Vectorization" OFF)
mark_as_advanced(VECTORIZATION_ENABLE_CLANGTIDY)
option(VECTORIZATION_ENABLE_FIX "Enable clang-tidy fix options for Vectorization" OFF)
mark_as_advanced(VECTORIZATION_ENABLE_FIX)
option(VECTORIZATION_ENABLE_IWYU "Enable include-what-you-use analysis for Vectorization" OFF)
mark_as_advanced(VECTORIZATION_ENABLE_IWYU)
option(VECTORIZATION_ENABLE_SANITIZER "Build Vectorization with sanitizer support (Clang/GCC only)"
       OFF
)
set(VECTORIZATION_SANITIZER_TYPE "address"
    CACHE STRING
          "Sanitizer type for Vectorization. Options: address, undefined, thread, memory, leak"
)
set_property(
  CACHE VECTORIZATION_SANITIZER_TYPE PROPERTY STRINGS address undefined thread memory leak
)
mark_as_advanced(VECTORIZATION_ENABLE_SANITIZER VECTORIZATION_SANITIZER_TYPE)
option(VECTORIZATION_ENABLE_SPELL "Enable check-only spell checking for Vectorization" OFF)
mark_as_advanced(VECTORIZATION_ENABLE_SPELL)
option(VECTORIZATION_ENABLE_VALGRIND "Execute Vectorization test suite with Valgrind" OFF)
mark_as_advanced(VECTORIZATION_ENABLE_VALGRIND)
set(VECTORIZATION_LINKER_CHOICE "default"
    CACHE STRING "Linker for Vectorization: default, lld, mold, gold, lld-link"
)
set_property(CACHE VECTORIZATION_LINKER_CHOICE PROPERTY STRINGS default lld mold gold lld-link)
mark_as_advanced(VECTORIZATION_LINKER_CHOICE)
set(VECTORIZATION_CACHE_BACKEND "none"
    CACHE STRING
          "Compiler cache backend for Vectorization. Options: none, ccache, sccache, buildcache"
)
set_property(CACHE VECTORIZATION_CACHE_BACKEND PROPERTY STRINGS none ccache sccache buildcache)
mark_as_advanced(VECTORIZATION_CACHE_BACKEND)
set(VECTORIZATION_CXX_STANDARD "20" CACHE STRING
                                          "C++ standard for Vectorization. Options: 11 14 17 20 23"
)
set_property(CACHE VECTORIZATION_CXX_STANDARD PROPERTY STRINGS 11 14 17 20 23)
if(NOT VECTORIZATION_CXX_STANDARD MATCHES "^(11|14|17|20|23)$")
  message(FATAL_ERROR "VECTORIZATION_CXX_STANDARD must be one of: 11 14 17 20 23")
endif()

# SVE vector register width. The current backend/sve/ implementation only supports 128 bits.
# Set to 256 or 512 once those backends are extended to support scalable widths.
set(VECTORIZATION_SVE_VECTOR_BITS "128" CACHE STRING
                                              "SVE vector register width in bits: 128, 256, 512"
)
set_property(CACHE VECTORIZATION_SVE_VECTOR_BITS PROPERTY STRINGS 128 256 512)
if(NOT VECTORIZATION_SVE_VECTOR_BITS MATCHES "^(128|256|512)$")
  message(FATAL_ERROR "VECTORIZATION_SVE_VECTOR_BITS must be 128, 256, or 512")
endif()

# Manual-unroll factor for the vectorized hot loops in expressions_evaluator.h (run/accumulate/
# hmin/hmax): how many independent simd<T> registers are processed per software loop iteration.
# 1 (default) reproduces today's single-register-per-step codegen exactly. Values >1 give the
# compiler N independent accumulator/temp chains per iteration instead of one, which hides the
# latency of FMA/div/transcendental ops (and, for reductions, breaks the serial dependency chain
# through the accumulator) at the cost of more registers and code size.
set(VECTORIZATION_PACKET_SIZE
    "1"
    CACHE
      STRING
      "SIMD registers processed per unrolled loop iteration in the expression evaluator (1,2,3,...)"
)
if(NOT VECTORIZATION_PACKET_SIZE MATCHES "^[1-9][0-9]*$")
  message(
    FATAL_ERROR
      "VECTORIZATION_PACKET_SIZE must be a positive decimal integer (got '${VECTORIZATION_PACKET_SIZE}')"
  )
endif()
if(VECTORIZATION_PACKET_SIZE GREATER 16)
  message(
    FATAL_ERROR "VECTORIZATION_PACKET_SIZE must be at most 16 (got '${VECTORIZATION_PACKET_SIZE}')"
  )
endif()

# =============================================================================
# SECTION 3 — CMake
# module includes
# =============================================================================
include(CheckCXXSourceCompiles)
include(CheckCXXSourceRuns)
include(CheckSymbolExists)

include(compiler_checks)
include(checks)
include(build_type)
include(platform)
include(compile_definitions)
include(CMakeDependentOption)
include(cache)
include(utils)

# Auto-downgrade to "no" if the requested CPU SIMD tier is unsupported
# (e.g. ARM64 has no x86 SSE/AVX).  HIP/CUDA are GPU backends — skip this check for them.
if(NOT VECTORIZATION_CPU_BACKEND STREQUAL "no" AND NOT VECTORIZATION)
  message(
    WARNING
      "VECTORIZATION_CPU_BACKEND '${VECTORIZATION_CPU_BACKEND}' is not supported by this compiler/architecture. "
      "Falling back to VECTORIZATION_CPU_BACKEND=no."
  )
  set(VECTORIZATION_CPU_BACKEND "no"
      CACHE STRING "CPU SIMD backend. Options: no, sse, avx, avx2, avx512, neon, sve" FORCE
  )
endif()

# Other libraries must not include Vectorization CMake modules; expose probes for Core et al.
if(HAS_EXCEPTION_PTR)
  set(PROJECT_HAS_EXCEPTION_PTR 1 CACHE INTERNAL "XSigma: std::exception_ptr (Vectorization utils)"
                                        FORCE
  )
else()
  set(PROJECT_HAS_EXCEPTION_PTR 0 CACHE INTERNAL "XSigma: std::exception_ptr (Vectorization utils)"
                                        FORCE
  )
endif()
foreach(_simd IN ITEMS SSE AVX AVX2 AVX512 NEON SVE)
  if(HAS_${_simd})
    set(PROJECT_${_simd} 1 CACHE INTERNAL "XSigma: SIMD (Vectorization utils)" FORCE)
  else()
    set(PROJECT_${_simd} 0 CACHE INTERNAL "XSigma: SIMD (Vectorization utils)" FORCE)
  endif()
endforeach()
set(VECTORIZATION_COMPILER_FLAGS "${VECTORIZATION_COMPILER_FLAGS}"
    CACHE INTERNAL "XSigma: SIMD flags from Vectorization utils" FORCE
)

message("-- SIMD compiler flags: ${VECTORIZATION_COMPILER_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${VECTORIZATION_COMPILER_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${VECTORIZATION_COMPILER_FLAGS}")

# =============================================================================
# SECTION 2 —
# VECTORIZATION_CPU_BACKEND / VECTORIZATION_GPU_BACKEND → VECTORIZATION_HAS_* defines and backend trees
# =============================================================================
if(VECTORIZATION_ENABLE_SANITIZER)
  include(sanitize)
endif()
if(VECTORIZATION_ENABLE_SPELL)
  include(spell)
endif()
if(VECTORIZATION_ENABLE_VALGRIND)
  include(valgrind)
endif()
include(helper_macros)

if(VECTORIZATION_ENABLE_COVERAGE)
  include(coverage)
elseif(MSVC)
  string(APPEND CMAKE_EXE_LINKER_FLAGS " /DEBUG:FULL")
  string(APPEND CMAKE_SHARED_LINKER_FLAGS " /DEBUG:FULL")
endif()

# --- CPU backend → VECTORIZATION_HAS_* (one-hot, driven by VECTORIZATION_CPU_BACKEND) ---
set(VECTORIZATION_COMPILE_DEFINITIONS "")

if(VECTORIZATION_CPU_BACKEND STREQUAL "sse")
  list(APPEND VECTORIZATION_COMPILE_DEFINITIONS VECTORIZATION_HAS_SSE=1)
else()
  list(APPEND VECTORIZATION_COMPILE_DEFINITIONS VECTORIZATION_HAS_SSE=0)
endif()

# AVX2 implies AVX; set once to avoid duplicate -DVECTORIZATION_HAS_AVX on the command line.
if(VECTORIZATION_CPU_BACKEND STREQUAL "avx" OR VECTORIZATION_CPU_BACKEND STREQUAL "avx2")
  list(APPEND VECTORIZATION_COMPILE_DEFINITIONS VECTORIZATION_HAS_AVX=1)
else()
  list(APPEND VECTORIZATION_COMPILE_DEFINITIONS VECTORIZATION_HAS_AVX=0)
endif()

if(VECTORIZATION_CPU_BACKEND STREQUAL "avx2")
  list(APPEND VECTORIZATION_COMPILE_DEFINITIONS VECTORIZATION_HAS_AVX2=1)
else()
  list(APPEND VECTORIZATION_COMPILE_DEFINITIONS VECTORIZATION_HAS_AVX2=0)
endif()

if(VECTORIZATION_CPU_BACKEND STREQUAL "avx512")
  list(APPEND VECTORIZATION_COMPILE_DEFINITIONS VECTORIZATION_HAS_AVX512=1)
else()
  list(APPEND VECTORIZATION_COMPILE_DEFINITIONS VECTORIZATION_HAS_AVX512=0)
endif()

if(VECTORIZATION_CPU_BACKEND STREQUAL "neon")
  list(APPEND VECTORIZATION_COMPILE_DEFINITIONS VECTORIZATION_HAS_NEON=1)
else()
  list(APPEND VECTORIZATION_COMPILE_DEFINITIONS VECTORIZATION_HAS_NEON=0)
endif()

if(VECTORIZATION_CPU_BACKEND STREQUAL "sve")
  list(APPEND VECTORIZATION_COMPILE_DEFINITIONS VECTORIZATION_HAS_SVE=1)
else()
  list(APPEND VECTORIZATION_COMPILE_DEFINITIONS VECTORIZATION_HAS_SVE=0)
endif()

# --- GPU backend → VECTORIZATION_HAS_* (driven by VECTORIZATION_GPU_BACKEND) ---
if(VECTORIZATION_GPU_BACKEND STREQUAL "cuda")
  list(APPEND VECTORIZATION_COMPILE_DEFINITIONS VECTORIZATION_HAS_CUDA=1 VECTORIZATION_HAS_HIP=0
       VECTORIZATION_HAS_METAL=0
  )
elseif(VECTORIZATION_GPU_BACKEND STREQUAL "hip")
  list(APPEND VECTORIZATION_COMPILE_DEFINITIONS VECTORIZATION_HAS_CUDA=0 VECTORIZATION_HAS_HIP=1
       VECTORIZATION_HAS_METAL=0
  )
elseif(VECTORIZATION_GPU_BACKEND STREQUAL "metal")
  list(APPEND VECTORIZATION_COMPILE_DEFINITIONS VECTORIZATION_HAS_CUDA=0 VECTORIZATION_HAS_HIP=0
       VECTORIZATION_HAS_METAL=1
  )
else()
  list(APPEND VECTORIZATION_COMPILE_DEFINITIONS VECTORIZATION_HAS_CUDA=0 VECTORIZATION_HAS_HIP=0
       VECTORIZATION_HAS_METAL=0
  )
endif()

# vectorization_tag_gpu_expression_sources(<files...> [RESULT_LANGUAGE <var>])
#
# A .cpp that composes and evaluates a vectorization expression against a CUDA/HIP tensor
# (`c = a + b`) instantiates the templated __global__ kernel in expressions_evaluator_gpu.h
# for that exact expression type, so it must be compiled in device-language mode
# (CMake source LANGUAGE CUDA/HIP) for __CUDACC__/__HIPCC__ to be defined and that dispatch
# branch to actually compile in — otherwise it silently falls through to the CPU path (see
# expressions_evaluator.h's check_cpu_reachable()). GPU backend selection is a single,
# whole-build, mutually exclusive choice (VECTORIZATION_GPU_BACKEND), so a file tagged this
# way still compiles its CPU fallback path unconditionally in the same object; this is purely
# additive over plain CXX, never a second/duplicate compile of the file.
#
# Centralizes the if(cuda)/elseif(hip) pairing so call sites (Testing/Cxx/CMakeLists.txt:
# TestTensorGpu.cpp, BenchmarkTensorGpu.cpp) don't each re-derive it. A no-op for backend
# "none"/"metal": Metal is ordinary Objective-C++ gated by VECTORIZATION_HAS_METAL, not
# __CUDACC__/__HIPCC__, so it needs no device-language tagging. Pass RESULT_LANGUAGE <var> to
# receive "CUDA"/"HIP"/"" when a caller needs to branch further on which one was applied.
function(vectorization_tag_gpu_expression_sources)
  cmake_parse_arguments(_vtg "" "RESULT_LANGUAGE" "" ${ARGN})
  set(_vtg_lang "")
  if(VECTORIZATION_GPU_BACKEND STREQUAL "cuda")
    set_source_files_properties(${_vtg_UNPARSED_ARGUMENTS} PROPERTIES LANGUAGE CUDA)
    set(_vtg_lang "CUDA")
  elseif(VECTORIZATION_GPU_BACKEND STREQUAL "hip")
    set_source_files_properties(${_vtg_UNPARSED_ARGUMENTS} PROPERTIES LANGUAGE HIP)
    set(_vtg_lang "HIP")
  endif()
  if(_vtg_RESULT_LANGUAGE)
    set(${_vtg_RESULT_LANGUAGE} "${_vtg_lang}" PARENT_SCOPE)
  endif()
endfunction()

compile_definition(VECTORIZATION_COMPILE_DEFINITIONS VECTORIZATION_ENABLE_SVML)
compile_definition(VECTORIZATION_COMPILE_DEFINITIONS VECTORIZATION_ENABLE_SLEEF)
compile_definition(VECTORIZATION_COMPILE_DEFINITIONS VECTORIZATION_ENABLE_ACCELERATE)
compile_definition(VECTORIZATION_COMPILE_DEFINITIONS VECTORIZATION_ENABLE_MKL)
compile_definition(VECTORIZATION_COMPILE_DEFINITIONS VECTORIZATION_ENABLE_GTEST)
list(APPEND VECTORIZATION_COMPILE_DEFINITIONS VECTORIZATION_HAS_PROFILER=${_vec_has_profiler})
# VECTORIZATION_VECTORIZED: 1 when any CPU SIMD backend is active (GPU does not affect this).
if(VECTORIZATION_CPU_BACKEND MATCHES "^(sse|avx|avx2|avx512|neon|sve)$")
  list(APPEND VECTORIZATION_COMPILE_DEFINITIONS VECTORIZATION_VECTORIZED=1)
else()
  list(APPEND VECTORIZATION_COMPILE_DEFINITIONS VECTORIZATION_VECTORIZED=0)
endif()
list(APPEND VECTORIZATION_COMPILE_DEFINITIONS
     VECTORIZATION_PACKET_SIZE=${VECTORIZATION_PACKET_SIZE}
)

set(VECTORIZATION_DEPENDENCY_COMPILE_DEFINITIONS "")
list(APPEND VECTORIZATION_DEPENDENCY_COMPILE_DEFINITIONS ${VECTORIZATION_COMPILE_DEFINITIONS})
set(VECTORIZATION_DEPENDENCY_LINK_DIRS "")

# Build/link ThirdParty SLEEF for AArch64 NEON when requested.
if(VECTORIZATION_ENABLE_SLEEF)
  set(_tpdir "${XSIGMA_THIRDPARTY_DIR}")
  set(_sleef_build_dir "${CMAKE_BINARY_DIR}/ThirdParty/sleef_build")
  set(_sleef_install_prefix "${CMAKE_BINARY_DIR}/ThirdParty/install")

  # SLEEF requires a non-default install prefix even for subdirectory builds.
  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${_sleef_install_prefix}"
        CACHE PATH "Install path prefix, prepended onto install directories." FORCE
    )
    set(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT FALSE)
  endif()

  set(SLEEF_BUILD_TESTS OFF CACHE BOOL "Disable SLEEF tests" FORCE)
  set(SLEEF_BUILD_DFT OFF CACHE BOOL "Disable SLEEF DFT library" FORCE)
  set(SLEEF_BUILD_SCALAR_LIB OFF CACHE BOOL "Disable SLEEF scalar library" FORCE)
  set(SLEEF_BUILD_QUAD OFF CACHE BOOL "Disable SLEEF quad library" FORCE)
  set(SLEEF_BUILD_GNUABI_LIBS OFF CACHE BOOL "Disable SLEEF GNU ABI variants" FORCE)
  set(SLEEF_DISABLE_OPENMP ON CACHE BOOL "Disable SLEEF OpenMP usage" FORCE)
  set(SLEEF_BUILD_INLINE_HEADERS OFF CACHE BOOL "Disable SLEEF inline headers" FORCE)
  set(SLEEF_ENABLE_TESTER OFF CACHE BOOL "Disable SLEEF tester binaries" FORCE)
  set(SLEEF_ENABLE_TESTER4 OFF CACHE BOOL "Disable SLEEF tester4 binaries" FORCE)
  # SLEEF 3.x gates OpenSSL behind SLEEF_DISABLE_SSL (the old SLEEF_ENABLE_SSL name no
  # longer exists). Without this, a system OpenSSL makes src/common register
  # test_psha/test_psha_capi in CTest even though SLEEF_BUILD_TESTS is OFF — those
  # phantom third-party tests then fail as "Not Run" in XSigma's ctest run.
  set(SLEEF_DISABLE_SSL ON CACHE BOOL "Disable SLEEF SSL-backed tester checks" FORCE)
  set(SLEEF_SHOW_CONFIG OFF CACHE BOOL "Suppress SLEEF config output" FORCE)
  set(SLEEF_BUILD_SHARED_LIBS OFF CACHE BOOL "Build SLEEF as static library" FORCE)

  add_third_party_library(
    sleef
    LIBRARY_TYPE
    STATIC
    SOURCE_DIR
    "${_tpdir}/sleef"
    BINARY_DIR
    "${_sleef_build_dir}"
    XSIGMA_TARGETS
    "Sleef::sleef=sleef"
    INCLUDE_DIR
    "${CMAKE_BINARY_DIR}/include"
    EXCLUDE_FROM_ALL
    AUTO_ADD_TO_DEPENDENCIES
    OFF
  )

  list(APPEND VECTORIZATION_DEPENDENCY_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/include")
  list(APPEND VECTORIZATION_DEPENDENCY_INCLUDE_DIRS "${_sleef_build_dir}/include")
  if(TARGET sleef)
    list(APPEND VECTORIZATION_DEPENDENCY_LIBS sleef)
  endif()
  list(APPEND VECTORIZATION_DEPENDENCY_LINK_DIRS "${_sleef_install_prefix}/lib")
  # sleef.h uses SLEEF_STATIC_LIBS to suppress __declspec(dllimport) on Windows.
  # The generated sleef-config.h defines it, but sleef.h does not include that header,
  # so we propagate the definition explicitly to all consumers.
  list(APPEND VECTORIZATION_DEPENDENCY_COMPILE_DEFINITIONS SLEEF_STATIC_LIBS)

endif()

# Detect and configure Intel MKL VML (x86 only, statically linked) when requested.
if(VECTORIZATION_ENABLE_MKL)
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _mkl_proc)
  if(NOT _mkl_proc MATCHES "^(x86_64|amd64|i686|i386)$")
    message(WARNING "VECTORIZATION_ENABLE_MKL=ON is only supported on x86_64; disabling.")
    set(VECTORIZATION_ENABLE_MKL OFF
        CACHE BOOL "Enable Intel MKL VML vector math backend (x86 only, static link)" FORCE
    )
    list(REMOVE_ITEM VECTORIZATION_COMPILE_DEFINITIONS "VECTORIZATION_HAS_MKL=1")
    list(APPEND VECTORIZATION_COMPILE_DEFINITIONS "VECTORIZATION_HAS_MKL=0")
    list(REMOVE_ITEM VECTORIZATION_DEPENDENCY_COMPILE_DEFINITIONS "VECTORIZATION_HAS_MKL=1")
    list(APPEND VECTORIZATION_DEPENDENCY_COMPILE_DEFINITIONS "VECTORIZATION_HAS_MKL=0")
  else()
    # Sequential (single-threaded) VML — no mandatory OpenMP runtime dependency.
    set(MKL_THREADING "SEQ" CACHE STRING "MKL threading flavour used by Vectorization" FORCE)
    # Use shared MKL libraries (pip-installed MKL ships .so only, no .a archives).
    # Set ENABLE_STATIC_MKL=ON manually if you have static archives available.
    set(ENABLE_STATIC_MKL OFF CACHE BOOL "Link MKL statically for Vectorization" FORCE)

    find_package(MKL QUIET)

    if(MKL_FOUND)
      message(
        STATUS "Vectorization: MKL VML found (version ${MKL_VERSION}). Include: ${MKL_INCLUDE_DIR}"
      )

      # Umbrella INTERFACE target — wraps include dir, library list, and the
      # --start-group / --end-group pair needed on Linux to resolve circular
      # references between the three static MKL archives.
      if(NOT TARGET MKL::MKL)
        add_library(MKL::MKL INTERFACE IMPORTED GLOBAL)
        target_include_directories(MKL::MKL INTERFACE "${MKL_INCLUDE_DIR}")
        # --start-group / --end-group is only needed for static archives to resolve
        # circular references.  With shared libraries it is harmless but unnecessary.
        if(UNIX AND NOT APPLE AND ENABLE_STATIC_MKL)
          target_link_libraries(
            MKL::MKL INTERFACE -Wl,--start-group ${MKL_LIBRARIES} -Wl,--end-group ${CMAKE_DL_LIBS}
          )
        else()
          target_link_libraries(MKL::MKL INTERFACE ${MKL_LIBRARIES} ${CMAKE_DL_LIBS})
        endif()
      endif()
    else()
      message(
        WARNING "VECTORIZATION_ENABLE_MKL=ON but MKL was not found — disabling MKL VML backend. "
                "Set MKLROOT or INTEL_MKL_DIR to your MKL installation root "
                "(e.g. /opt/intel/oneapi/mkl/latest)."
      )
      set(VECTORIZATION_ENABLE_MKL OFF
          CACHE BOOL "Enable Intel MKL VML vector math backend (x86 only, static link)" FORCE
      )
      # Patch the compile-definition lists that were populated before this detection block.
      list(REMOVE_ITEM VECTORIZATION_COMPILE_DEFINITIONS "VECTORIZATION_HAS_MKL=1")
      list(APPEND VECTORIZATION_COMPILE_DEFINITIONS "VECTORIZATION_HAS_MKL=0")
      list(REMOVE_ITEM VECTORIZATION_DEPENDENCY_COMPILE_DEFINITIONS "VECTORIZATION_HAS_MKL=1")
      list(APPEND VECTORIZATION_DEPENDENCY_COMPILE_DEFINITIONS "VECTORIZATION_HAS_MKL=0")
    endif()
  endif()
  unset(_mkl_proc)
endif()

# =============================================================================
# SECTION 4 — Source
# discovery with SIMD-tier filtering
# =============================================================================
set(CMAKE_CXX_STANDARD ${VECTORIZATION_CXX_STANDARD})

file(GLOB_RECURSE headers LIST_DIRECTORIES false "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
                                                 "${CMAKE_CURRENT_SOURCE_DIR}/*.inl"
)
file(GLOB_RECURSE sources LIST_DIRECTORIES false "${CMAKE_CURRENT_SOURCE_DIR}/*.cxx"
                                                 "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)
file(GLOB_RECURSE templates LIST_DIRECTORIES false "${CMAKE_CURRENT_SOURCE_DIR}/*.hxx")

list(FILTER sources EXCLUDE REGEX ".*/Testing/.*")
list(FILTER headers EXCLUDE REGEX ".*/Testing/.*")
list(FILTER templates EXCLUDE REGEX ".*/Testing/.*")

# Exclude CPU SIMD backend trees not selected by VECTORIZATION_CPU_BACKEND.
if(NOT VECTORIZATION_CPU_BACKEND STREQUAL "sse")
  list(FILTER headers EXCLUDE REGEX ".*/backend/cpu/sse/.*")
  list(FILTER sources EXCLUDE REGEX ".*/backend/cpu/sse/.*")
  list(FILTER templates EXCLUDE REGEX ".*/backend/cpu/sse/.*")
endif()
if(NOT VECTORIZATION_CPU_BACKEND MATCHES "^(avx|avx2)$")
  list(FILTER headers EXCLUDE REGEX ".*/backend/cpu/avx/.*")
  list(FILTER sources EXCLUDE REGEX ".*/backend/cpu/avx/.*")
  list(FILTER templates EXCLUDE REGEX ".*/backend/cpu/avx/.*")
endif()
if(NOT VECTORIZATION_CPU_BACKEND STREQUAL "avx512")
  list(FILTER headers EXCLUDE REGEX ".*/backend/cpu/avx512/.*")
  list(FILTER sources EXCLUDE REGEX ".*/backend/cpu/avx512/.*")
  list(FILTER templates EXCLUDE REGEX ".*/backend/cpu/avx512/.*")
endif()
if(NOT VECTORIZATION_CPU_BACKEND STREQUAL "neon" AND NOT VECTORIZATION_CPU_BACKEND STREQUAL "sve")
  list(FILTER headers EXCLUDE REGEX ".*/backend/cpu/neon/.*")
  list(FILTER sources EXCLUDE REGEX ".*/backend/cpu/neon/.*")
  list(FILTER templates EXCLUDE REGEX ".*/backend/cpu/neon/.*")
endif()
if(NOT VECTORIZATION_CPU_BACKEND STREQUAL "sve")
  list(FILTER headers EXCLUDE REGEX ".*/backend/cpu/sve/.*")
  list(FILTER sources EXCLUDE REGEX ".*/backend/cpu/sve/.*")
  list(FILTER templates EXCLUDE REGEX ".*/backend/cpu/sve/.*")
endif()

# Note: backend/gpu/{float,double}/simd.h is shared by CUDA and HIP (identical
# scalar semantics) and is only ever pulled in under VECTORIZATION_ON_GPU_DEVICE
# (backend/simd.h), so no VECTORIZATION_GPU_BACKEND-based tree exclusion is needed here.

# Exclude x86 svml.h when neither SVML nor SLEEF is active.
# NEON svml.h is handled separately — it always compiles for NEON (has scalar fallbacks).
if(NOT VECTORIZATION_ENABLE_SVML AND NOT VECTORIZATION_ENABLE_SLEEF)
  list(FILTER headers EXCLUDE REGEX ".*/backend/cpu/(sse|avx|avx512)/svml\\.h$")
endif()

# Always exclude NEON svml.h when not targeting NEON.
if(NOT VECTORIZATION_CPU_BACKEND STREQUAL "neon")
  list(FILTER headers EXCLUDE REGEX ".*/backend/cpu/neon/svml\\.h$")
endif()

# Exclude the MKL backend tree when MKL is not enabled.
if(NOT VECTORIZATION_ENABLE_MKL)
  list(FILTER headers EXCLUDE REGEX ".*/backend/cpu/mkl/.*")
  list(FILTER sources EXCLUDE REGEX ".*/backend/cpu/mkl/.*")
endif()

# expressions_functors_mkl.h is a stub (content merged into expressions_functors.h).
# Exclude it from the build unconditionally to keep the source list clean.
list(FILTER headers EXCLUDE REGEX ".*/expressions_functors_mkl\\.h$")

# =============================================================================
# SECTION 5 — Target
# definition
# =============================================================================
add_library(Vectorization ${sources} ${headers} ${templates})
xsigma_target_apply_lto(Vectorization VECTORIZATION)
if(VECTORIZATION_ENABLE_SANITIZER)
  xsigma_configure_sanitizer(VECTORIZATION)
endif()
xsigma_find_linker(${VECTORIZATION_LINKER_CHOICE} Vectorization VECTORIZATION_LTO_MODE)
xsigma_target_apply_cache(Vectorization VECTORIZATION_ENABLE_CACHE VECTORIZATION_CACHE_BACKEND)
xsigma_target_optional_icecc(Vectorization VECTORIZATION_ENABLE_ICECC)

# --- DLL-export macros (VECTORIZATION_API in source-tree vectorization_export.h) ---
if(BUILD_SHARED_LIBS)
  target_compile_definitions(
    Vectorization PUBLIC VECTORIZATION_SHARED_DEFINE PRIVATE VECTORIZATION_BUILDING_DLL
  )
  if(COMMAND xsigma_set_target_output_directories)
    xsigma_set_target_output_directories(Vectorization)
  endif()
else()
  target_compile_definitions(Vectorization PUBLIC VECTORIZATION_STATIC_DEFINE)
endif()

add_library(Vectorization::Vectorization ALIAS Vectorization)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/Testing/VectorizationTest.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/Testing/VectorizationTest.h" @ONLY
)

set_target_properties(
  Vectorization PROPERTIES CXX_STANDARD ${VECTORIZATION_CXX_STANDARD} CXX_STANDARD_REQUIRED ON
                           CXX_EXTENSIONS OFF
)

# ---- HIP GPU backend -------------------------------------------------------
if(VECTORIZATION_GPU_BACKEND STREQUAL "hip")
  # HIP/ROCm is Unix-only in this project — see Library/Memory/Cmake/hip.cmake and
  # Docs/CUDA_HIP_REMEDIATION_PLAN.md. Fail fast on Windows rather than attempting an
  # untested find_package(hip)/enable_language(HIP) path.
  if(WIN32)
    message(
      FATAL_ERROR
        "VECTORIZATION_GPU_BACKEND=hip is not supported on Windows in this project (HIP/ROCm is "
        "Unix-only here). Use VECTORIZATION_GPU_BACKEND=cuda on Windows, or build on Linux for HIP."
    )
  endif()

  # Prefer ROCm HIP; fall back to CUDA when ROCM_PATH is absent.
  if(NOT DEFINED CMAKE_HIP_COMPILER AND DEFINED ENV{ROCM_PATH})
    set(CMAKE_HIP_COMPILER "$ENV{ROCM_PATH}/bin/hipcc" CACHE FILEPATH "HIP compiler")
  endif()

  # Locate the HIP package (hip::host / hip::device imported targets) so the
  # Vectorization target links HIP even when Memory's HIP finder has not run
  # (e.g. MEMORY_GPU_BACKEND=none). Memory::Memory is always linked, but that
  # does not pull HIP unless Memory itself was configured with HIP.
  include(rocm_prefix)
  xsigma_setup_rocm_prefix()
  find_package(hip QUIET)
  if(NOT hip_FOUND)
    message(
      WARNING "Vectorization: VECTORIZATION_GPU_BACKEND=hip but the HIP package was not found "
              "via find_package(hip). Set ROCM_PATH or CMAKE_PREFIX_PATH to your ROCm install. "
              "Falling back to enable_language(HIP) alone; linking may fail unless the HIP "
              "runtime is otherwise reachable (e.g. transitively via Memory::Memory)."
    )
  endif()

  enable_language(HIP)

  set(VECTORIZATION_GPU_ARCHITECTURES "gfx900;gfx906;gfx908;gfx90a;gfx1030;gfx1100"
      CACHE STRING "HIP GPU architectures (gfxXXX for AMD, smXX for CUDA offload via hipcc)"
  )
  # CI passes -DCMAKE_HIP_ARCHITECTURES=gfx90a (no device at configure time).
  # Honor that instead of compiling every gfx* default.
  if(CMAKE_HIP_ARCHITECTURES AND NOT CMAKE_HIP_ARCHITECTURES STREQUAL "native")
    set(VECTORIZATION_GPU_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
        CACHE STRING "HIP GPU architectures (gfxXXX for AMD, smXX for CUDA offload via hipcc)"
              FORCE
    )
  endif()
  set_target_properties(
    Vectorization PROPERTIES HIP_STANDARD ${VECTORIZATION_CXX_STANDARD}
                             HIP_ARCHITECTURES "${VECTORIZATION_GPU_ARCHITECTURES}"
  )
  target_compile_options(Vectorization PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>)
  if(TARGET hip::host)
    target_link_libraries(Vectorization PUBLIC hip::host)
  endif()
  if(TARGET hip::device)
    target_link_libraries(Vectorization PRIVATE hip::device)
  endif()
  message(
    STATUS "Vectorization: HIP GPU backend enabled (archs: ${VECTORIZATION_GPU_ARCHITECTURES})"
  )
endif()

# ---- CUDA GPU backend -------------------------------------------------------
if(VECTORIZATION_GPU_BACKEND STREQUAL "cuda")
  enable_language(CUDA)

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # enable_language reloads CMakeCUDACompiler.cmake here. With the forced
    # Clang compiler its empty implicit include list shadows Memory's cached
    # fallback. Restore it in this scope (and re-cache) so system TBB's
    # /usr/include is not emitted ahead of libstdc++, breaking cmath's
    # include_next <math.h>.
    list(APPEND CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
    list(REMOVE_DUPLICATES CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES)
    set(CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES "${CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES}"
        CACHE INTERNAL "CUDA implicit host include directories" FORCE
    )
  endif()

  # Prefer arches already chosen by Memory/cuda.cmake; otherwise the shared
  # Turing+ default (sm_70 removed in CUDA 12.8; CUDA 13.x starts at sm_75).
  # FORCE evicts any stale cache entry that still contains sm_70.
  if(DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL ""
     AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "native"
  )
    set(_vectorization_cuda_archs "${CMAKE_CUDA_ARCHITECTURES}")
  else()
    if(DEFINED XSIGMA_CUDA_ARCHITECTURES_DEFAULT)
      set(_vectorization_cuda_archs "${XSIGMA_CUDA_ARCHITECTURES_DEFAULT}")
    else()
      set(_vectorization_cuda_archs "75;80;86;89;90")
    endif()
  endif()
  set(VECTORIZATION_CUDA_ARCHITECTURES "${_vectorization_cuda_archs}"
      CACHE STRING "CUDA compute capability list (e.g. 75;80;86 for Turing/Ampere/Ada)" FORCE
  )
  unset(_vectorization_cuda_archs)
  set_target_properties(
    Vectorization
    PROPERTIES CUDA_STANDARD ${VECTORIZATION_CXX_STANDARD}
               CUDA_STANDARD_REQUIRED ON
               CUDA_ARCHITECTURES "${VECTORIZATION_CUDA_ARCHITECTURES}"
               CUDA_SEPARABLE_COMPILATION OFF
  )
  if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
    target_compile_options(Vectorization PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-ffast-math>)
    # CMAKE_CUDA_COMPILER_FORCED skips CUDA_STANDARD → -std= emission.
    target_compile_options(
      Vectorization PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-std=c++${VECTORIZATION_CXX_STANDARD}>
    )
  else()
    target_compile_options(
      Vectorization PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -Xcompiler=-fPIC>
    )
  endif()
  message(
    STATUS
      "Vectorization: CUDA GPU backend enabled (archs: ${VECTORIZATION_CUDA_ARCHITECTURES}, compiler: ${CMAKE_CUDA_COMPILER_ID})"
  )
endif()

# ---- Metal GPU backend ------------------------------------------------------
# Unlike CUDA/HIP, Metal has no CMake "language"/unified host+device compiler pass — MSL
# kernel source (backend/gpu/metal/kernels.metal) is embedded as a string at configure
# time and compiled at runtime via newLibraryWithSource:, so this block only needs
# Objective-C++ (for the .mm dispatch/allocator glue) plus the Metal/Foundation
# frameworks. See backend/gpu/metal/metal_dispatch.mm and
# expressions/expressions_evaluator_metal.h for the rest of the design.
if(VECTORIZATION_GPU_BACKEND STREQUAL "metal")
  enable_language(OBJCXX)
  set(CMAKE_OBJCXX_STANDARD ${VECTORIZATION_CXX_STANDARD})
  set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

  find_library(METAL_FRAMEWORK Metal REQUIRED)
  find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

  # Embed kernels.metal as a C++ string literal (kMetalKernelSource) — see
  # metal_kernels_source.h.in for the template this fills in.
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/backend/gpu/metal/kernels.metal" METAL_KERNEL_SOURCE)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/backend/gpu/metal/metal_kernels_source.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/backend/gpu/metal/metal_kernels_source.h" @ONLY
  )

  target_sources(Vectorization PRIVATE backend/gpu/metal/metal_dispatch.mm)
  set_source_files_properties(
    backend/gpu/metal/metal_dispatch.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc"
  )
  target_link_libraries(Vectorization PUBLIC ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})

  message(STATUS "Vectorization: Metal GPU backend enabled (framework: ${METAL_FRAMEWORK})")
endif()

if(MSVC)
  target_compile_options(Vectorization PRIVATE /WX)
  if(EXISTS "${PROJECT_SOURCE_DIR}/PRETORIAN.natvis")
    target_sources(Vectorization PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/PRETORIAN.natvis>)
  endif()
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND VECTORIZATION_CPU_BACKEND STREQUAL "sse")
    target_compile_definitions(Vectorization PUBLIC VECTORIZATION_CLANG_CL)
  endif()
else()
  target_compile_options(Vectorization PRIVATE -include cstdlib)
endif()

# Apply SIMD flags as PUBLIC target options so all consumers inherit the correct ISA.
if(VECTORIZATION_COMPILER_FLAGS)
  if(MSVC)
    separate_arguments(_vec_simd_flags_list WINDOWS_COMMAND "${VECTORIZATION_COMPILER_FLAGS}")
  else()
    separate_arguments(_vec_simd_flags_list UNIX_COMMAND "${VECTORIZATION_COMPILER_FLAGS}")
  endif()
  target_compile_options(Vectorization PUBLIC ${_vec_simd_flags_list})
endif()

# =============================================================================
# SECTION 6 — Include
# directories and link libraries
# =============================================================================
target_include_directories(
  Vectorization
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Testing>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/Testing>
         $<INSTALL_INTERFACE:include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

if(VECTORIZATION_DEPENDENCY_INCLUDE_DIRS)
  target_include_directories(Vectorization PUBLIC ${VECTORIZATION_DEPENDENCY_INCLUDE_DIRS})
  message(STATUS "Vectorization: extra include dirs: ${VECTORIZATION_DEPENDENCY_INCLUDE_DIRS}")
endif()

target_link_libraries(Vectorization PUBLIC Logging::Logging)
target_link_libraries(Vectorization PUBLIC Memory::Memory)

if(_vec_has_profiler)
  target_link_libraries(Vectorization PUBLIC Profiler::Profiler)
endif()

if(VECTORIZATION_DEPENDENCY_LIBS)
  target_link_libraries(Vectorization PUBLIC ${VECTORIZATION_DEPENDENCY_LIBS})
  message(STATUS "Vectorization: linked dependencies: ${VECTORIZATION_DEPENDENCY_LIBS}")
endif()

if(VECTORIZATION_DEPENDENCY_LINK_DIRS)
  target_link_directories(Vectorization PUBLIC ${VECTORIZATION_DEPENDENCY_LINK_DIRS})
  message(STATUS "Vectorization: dependency link dirs: ${VECTORIZATION_DEPENDENCY_LINK_DIRS}")
endif()

# Link Apple Accelerate framework when vForce is enabled.
if(VECTORIZATION_ENABLE_ACCELERATE AND APPLE)
  target_link_libraries(Vectorization PUBLIC "-framework Accelerate")
  message(STATUS "Vectorization: linked Apple Accelerate framework (vForce).")
endif()

if(VECTORIZATION_ENABLE_SLEEF AND TARGET Sleef::sleef)
  target_link_libraries(Vectorization PUBLIC Sleef::sleef)
endif()

# Link ThirdParty SVML and stage its runtime binaries next to the output library.
if(VECTORIZATION_ENABLE_SVML AND TARGET SVML::SVML)
  target_link_libraries(Vectorization PUBLIC SVML::SVML)
  if(TARGET SVML::IRC)
    target_link_libraries(Vectorization PUBLIC SVML::IRC)
  endif()
  if(TARGET SVML::INTLC)
    target_link_libraries(Vectorization PUBLIC SVML::INTLC)
  endif()
  foreach(_svml_rt IN LISTS SVML_RUNTIME_LIBRARIES)
    if(_svml_rt)
      add_custom_command(
        TARGET Vectorization
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_svml_rt}"
                "$<TARGET_FILE_DIR:Vectorization>"
        COMMENT "Staging SVML runtime: ${_svml_rt}"
        VERBATIM
      )
    endif()
  endforeach()
  unset(_svml_rt)
  message(
    STATUS "Vectorization: linked SVML::SVML (ThirdParty); runtime files: ${SVML_RUNTIME_LIBRARIES}"
  )
endif()

# Link Intel MKL VML when VECTORIZATION_ENABLE_MKL=ON.
# The MKL::MKL umbrella target was created above (find_package + target setup).
if(VECTORIZATION_ENABLE_MKL)
  if(MKL_FOUND AND TARGET MKL::MKL)
    target_link_libraries(Vectorization PUBLIC MKL::MKL)
    message(STATUS "Vectorization: linked MKL VML via MKL::MKL.")
  else()
    message(FATAL_ERROR "VECTORIZATION_ENABLE_MKL=ON but the MKL::MKL target is not available. "
                        "Set MKLROOT or INTEL_MKL_DIR to your MKL installation root."
    )
  endif()
endif()

if(VECTORIZATION_DEPENDENCY_COMPILE_DEFINITIONS)
  target_compile_definitions(Vectorization PUBLIC ${VECTORIZATION_DEPENDENCY_COMPILE_DEFINITIONS})
  message(
    STATUS "Vectorization: compile definitions: ${VECTORIZATION_DEPENDENCY_COMPILE_DEFINITIONS}"
  )
endif()

if(VECTORIZATION_ENABLE_CLANGTIDY)
  include(clang_tidy)
  xsigma_target_clang_tidy(Vectorization ${VECTORIZATION_ENABLE_FIX})
endif()

if(VECTORIZATION_ENABLE_IWYU)
  include(iwyu)
  xsigma_apply_iwyu(Vectorization)
endif()

xsigma_module_create_filters("${CMAKE_CURRENT_SOURCE_DIR}")

# =============================================================================
# SECTION 7 —
# Installation
# =============================================================================
install(
  TARGETS Vectorization
  EXPORT VectorizationTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES
  DESTINATION include
)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
        DESTINATION include/Vectorization
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hxx" PATTERN "*.inl"
        PATTERN "Testing" EXCLUDE PATTERN "Cmake" EXCLUDE)

# =============================================================================
# SECTION 8 — Testing
# =============================================================================
if(VECTORIZATION_ENABLE_TESTING)
  add_subdirectory(Testing/Cxx)
endif()

if(VECTORIZATION_ENABLE_VALGRIND AND VECTORIZATION_ENABLE_TESTING)
  xsigma_apply_valgrind_timeouts()
endif()

# =============================================================================
# SECTION 9 — Summary
# message
# =============================================================================
message("")
message("******** Vectorization module (CMake flags) ********")
message("  CPU backend          : ${VECTORIZATION_CPU_BACKEND}")
message("  GPU backend          : ${VECTORIZATION_GPU_BACKEND}")
message("  Packet size          : ${VECTORIZATION_PACKET_SIZE}")
message("  Svml                 : ${VECTORIZATION_ENABLE_SVML}")
message("  Sleef                : ${VECTORIZATION_ENABLE_SLEEF}")
message("  Accelerate (vForce)  : ${VECTORIZATION_ENABLE_ACCELERATE}")
message("  Mkl VML              : ${VECTORIZATION_ENABLE_MKL}")
message("  LibTorch             : ${VECTORIZATION_ENABLE_LIBTORCH}")
message("  Memory               : required (Memory::Memory)")
message("  Logging              : required (Logging::Logging)")
message("  Profiler (HAS)       : ${_vec_has_profiler}")
message("  Cxx standard         : ${VECTORIZATION_CXX_STANDARD}")
message("  Cache                : ${VECTORIZATION_ENABLE_CACHE}")
message("  Cache backend        : ${VECTORIZATION_CACHE_BACKEND}")
message("  Icecc                : ${VECTORIZATION_ENABLE_ICECC}")
xsigma_print_lto_linker_summary(VECTORIZATION)
message("  Coverage             : ${VECTORIZATION_ENABLE_COVERAGE}")
message("  Testing              : ${VECTORIZATION_ENABLE_TESTING}")
message("  Gtest                : ${VECTORIZATION_ENABLE_GTEST}")
message("  Benchmark            : ${VECTORIZATION_ENABLE_BENCHMARK}")

message("  Clang-tidy           : ${VECTORIZATION_ENABLE_CLANGTIDY}")
message("  Fix                  : ${VECTORIZATION_ENABLE_FIX}")
message("  Iwyu                 : ${VECTORIZATION_ENABLE_IWYU}")
message("  Sanitizer            : ${VECTORIZATION_ENABLE_SANITIZER}")
message("  Sanitizer type       : ${VECTORIZATION_SANITIZER_TYPE}")
message("  Spell                : ${VECTORIZATION_ENABLE_SPELL}")
message("  Valgrind             : ${VECTORIZATION_ENABLE_VALGRIND}")
message("")
message("Build:")
message("  C++ standard             : ${CMAKE_CXX_STANDARD}")
message("  C flags                  : ${CMAKE_C_FLAGS}")
message("  CXX flags                : ${CMAKE_CXX_FLAGS}")
message("  Shared linker flags      : ${CMAKE_SHARED_LINKER_FLAGS}")
message("")
