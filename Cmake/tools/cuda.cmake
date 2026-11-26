include_guard(GLOBAL)

# =============================================================================
# CUDA GPU backend for Vectorization
#
# Included from the top-level CMakeLists.txt before any dependency is added.
# A language enabled in a subdirectory is invisible to its siblings, so CUDA
# must be enabled here for CMake's own rules (CUDA_ARCHITECTURES,
# CUDA_STANDARD, the toolkit path, MSVC_RUNTIME_LIBRARY) to apply to every
# target: ours, Memory's, Scalars' and LinearAlgebra's.
#
# Vectorization uses LANGUAGE CUDA tagging on .cpp expression-template
# sources (vectorization_tag_gpu_expression_sources()) rather than .cu files.
# =============================================================================

# Honour any existing CMAKE_CUDA_ARCHITECTURES, fall through to
# XSIGMA_CUDA_ARCHITECTURES_DEFAULT (set by a shared fleet helper), otherwise the
# Turing+ fleet default. sm_70 was removed in CUDA 12.8; CUDA 13.x starts at sm_75.
if(DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL ""
   AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
  set(_vec_cuda_archs "${CMAKE_CUDA_ARCHITECTURES}")
elseif(DEFINED XSIGMA_CUDA_ARCHITECTURES_DEFAULT)
  set(_vec_cuda_archs "${XSIGMA_CUDA_ARCHITECTURES_DEFAULT}")
else()
  set(_vec_cuda_archs "75;80;86;89;90")
endif()

set(VECTORIZATION_CUDA_ARCHITECTURES "${_vec_cuda_archs}"
    CACHE STRING "CUDA compute capability list (e.g. 75;80;86 for Turing/Ampere/Ada)" FORCE
)
unset(_vec_cuda_archs)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "${VECTORIZATION_CUDA_ARCHITECTURES}")
endif()

# Clang compiles CUDA whenever it is the C++ compiler; otherwise nvcc.
# A CUDACXX environment variable (e.g. CUDACXX=nvcc to pair nvcc with a Clang
# C++ compiler) takes precedence, as it does in CMake itself. Visual Studio
# generators compile CUDA only through NVIDIA's MSBuild customization (nvcc),
# so they keep nvcc even with the ClangCL toolset.
if(NOT CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX}
   AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
   AND NOT CMAKE_GENERATOR MATCHES "^Visual Studio")
  set(CMAKE_CUDA_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "CUDA compiler (Clang)")
endif()

# CMake identifies Clang as a CUDA compiler by compiling a probe for a fixed GPU
# architecture. Releases before 3.31.10 only probe sm_52 and older, which
# CUDA 13 no longer supports, so identification fails.
find_package(CUDAToolkit REQUIRED)
if(CMAKE_CUDA_COMPILER STREQUAL CMAKE_CXX_COMPILER AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
   AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL 13.0
   AND CMAKE_VERSION VERSION_LESS 3.31.10)
  message(
    FATAL_ERROR
      "Clang as the CUDA compiler with CUDA ${CUDAToolkit_VERSION} requires CMake 3.31.10 or "
      "newer (found ${CMAKE_VERSION}). Install one, e.g. python3 -m pip install 'cmake>=3.31.10'."
  )
endif()

# Visual Studio 2026's MSVC STL (14.5x) rejects CUDA older than 13.4 with
# STL1002, and CUDA 13.2's cudafe++ crashes (0xC0000005) while reporting it, so
# CMake's compiler identification fails without a useful message. nvcc is the
# CUDA compiler unless Clang compiles CUDA itself (see above).
if(MSVC_VERSION GREATER_EQUAL 1950 AND CUDAToolkit_VERSION VERSION_LESS 13.4
   AND NOT CMAKE_CUDA_COMPILER STREQUAL CMAKE_CXX_COMPILER)
  message(
    FATAL_ERROR
      "CUDA ${CUDAToolkit_VERSION} does not support the MSVC ${MSVC_VERSION} STL, which needs "
      "CUDA 13.4 or newer. Install CUDA 13.4+, or build with the MSVC 14.4x (v143) tools: "
      "-T v143,cuda=<CUDA root> or -T ClangCL,version=14.44,cuda=<CUDA root> with a Visual "
      "Studio generator (Scripts/setup.py selects these when v143 is installed), or a "
      "'vcvars64.bat -vcvars_ver=14.44' environment with Ninja."
  )
endif()

# The ClangCL toolset needs MSBuild fixes before NVIDIA's CUDA customization
# works with it (see the props file). Try-compile projects, including CMake's
# CUDA compiler check, need them too.
if(CMAKE_GENERATOR MATCHES "^Visual Studio" AND CMAKE_VS_PLATFORM_TOOLSET STREQUAL "ClangCL")
  if(CMAKE_VS_GLOBALS MATCHES "(^|;)ForceImportBeforeCppTargets=")
    message(
      WARNING
        "CMAKE_VS_GLOBALS already sets ForceImportBeforeCppTargets; import "
        "${CMAKE_CURRENT_LIST_DIR}/clangcl_cuda_msbuild.props from it, or ClangCL CUDA builds fail."
    )
  else()
    list(APPEND CMAKE_VS_GLOBALS
         "ForceImportBeforeCppTargets=${CMAKE_CURRENT_LIST_DIR}/clangcl_cuda_msbuild.props")
    list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES CMAKE_VS_GLOBALS)
  endif()
endif()

if(WIN32 AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"
   AND CMAKE_CUDA_COMPILER STREQUAL CMAKE_CXX_COMPILER AND NOT CMAKE_USER_MAKE_RULES_OVERRIDE_CUDA)
  set(CMAKE_USER_MAKE_RULES_OVERRIDE_CUDA "${CMAKE_CURRENT_LIST_DIR}/clang_cuda_windows_rules.cmake")
endif()

enable_language(CUDA)

# ---------------------------------------------------------------------------
# vectorization_apply_cuda_flags(<target>)
# Apply CUDA language properties and per-target compile options.
# Mirrors linalg_apply_cuda_flags() and PyTorch's torch_cuda_compile_options.
# ---------------------------------------------------------------------------
function(vectorization_apply_cuda_flags target)
  set_target_properties(${target} PROPERTIES
      CUDA_STANDARD ${VECTORIZATION_CXX_STANDARD}
      CUDA_STANDARD_REQUIRED ON
      CUDA_ARCHITECTURES "${VECTORIZATION_CUDA_ARCHITECTURES}"
      CUDA_SEPARABLE_COMPILATION OFF)
  if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-ffast-math>)
  else()
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -Xcompiler=-fPIC>)
  endif()
  # nvcc treats constexpr functions (std::array accessors, lambdas) as host-only
  # and warns (#20013-D/#20015-D) when our __host__ __device__ expression code
  # calls them; Clang CUDA allows these calls by default, and LinearAlgebra
  # already passes the same flag for its kernels.
  target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr>)
  target_compile_options(${target} INTERFACE
      $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr>)
  if(MSVC)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
    # CUDA 13's CCCL headers, which our public headers include, reject cl.exe's
    # traditional preprocessor (#error in cuda/std/__cccl/preprocessor.h), so
    # nvcc consumers such as the CUDA-tagged tests need it too.
    target_compile_options(${target} INTERFACE
        $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-Xcompiler=/Zc:preprocessor>)
    # nvcc's own front end does not see the cl.exe warning policy, so apply its
    # equivalents for CUDA consumers: #1394-D is C4251 (field without a DLL
    # interface, which the exporting headers already disable for cl.exe) and
    # #940-D is C4715 (missing return, disabled project-wide in platform.cmake).
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-diag-suppress=940,1394>)
    target_compile_options(${target} INTERFACE
        $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:-diag-suppress=940,1394>)
  endif()
endfunction()
