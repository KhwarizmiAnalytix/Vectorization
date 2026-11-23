include_guard(GLOBAL)

# =============================================================================
# CUDA GPU backend for Vectorization
#
# Mirrors LinearAlgebra's Cmake/cuda.cmake and PyTorch cmake/Cuda.cmake.
# Called from CMakeLists.txt when VECTORIZATION_GPU_BACKEND=cuda; see
# vectorization_apply_cuda_flags() for per-target usage.
#
# Vectorization uses LANGUAGE CUDA tagging on .cpp expression-template
# sources (vectorization_tag_gpu_expression_sources()) rather than .cu files,
# so this module only needs enable_language(CUDA) and arch setup — no
# find_package(CUDAToolkit) for cublas/cusolver here.
# =============================================================================

# Arch selection: honour any existing CMAKE_CUDA_ARCHITECTURES (e.g. from a
# parent project's Memory/cuda.cmake), fall through to XSIGMA_CUDA_ARCHITECTURES_DEFAULT
# (set by a shared fleet helper), otherwise the Turing+ fleet default.
# sm_70 was removed in CUDA 12.8; CUDA 13.x starts at sm_75.
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

enable_language(CUDA)

# With a forced Clang compiler, enable_language(CUDA) reloads
# CMakeCUDACompiler.cmake and its empty implicit-include list shadows the
# previously cached fallback. Restore it so system TBB's /usr/include is not
# emitted ahead of libstdc++, breaking cmath's include_next <math.h>.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  list(APPEND CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
  list(REMOVE_DUPLICATES CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES)
  set(CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES "${CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES}"
      CACHE INTERNAL "CUDA implicit host include directories" FORCE
  )
endif()

# Language-standard flag for Clang as the CUDA compiler (see below). clang-cl
# ignores a GNU -std= with a warning, and given /clang:-std= as well as the
# -std:c++NN CMake emits it reports the latter as unused, so use clang-cl's
# own spelling.
if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  if(VECTORIZATION_CXX_STANDARD GREATER_EQUAL 23)
    set(VECTORIZATION_CLANG_CUDA_STD_FLAG "-std:c++latest")
  else()
    set(VECTORIZATION_CLANG_CUDA_STD_FLAG "-std:c++${VECTORIZATION_CXX_STANDARD}")
  endif()
else()
  set(VECTORIZATION_CLANG_CUDA_STD_FLAG "-std=c++${VECTORIZATION_CXX_STANDARD}")
endif()

# ---------------------------------------------------------------------------
# vectorization_apply_cuda_flags(<target>)
# Apply CUDA language properties and per-target compile options.
# Mirrors linalg_apply_cuda_flags() and PyTorch's torch_cuda_compile_options.
# CMAKE_CUDA_COMPILER_FORCED skips CUDA_STANDARD → -std= emission for Clang,
# so the explicit -std= flag is required in that case.
# ---------------------------------------------------------------------------
function(vectorization_apply_cuda_flags target)
  set_target_properties(${target} PROPERTIES
      CUDA_STANDARD ${VECTORIZATION_CXX_STANDARD}
      CUDA_STANDARD_REQUIRED ON
      CUDA_ARCHITECTURES "${VECTORIZATION_CUDA_ARCHITECTURES}"
      CUDA_SEPARABLE_COMPILATION OFF)
  if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-ffast-math>
        $<$<COMPILE_LANGUAGE:CUDA>:${VECTORIZATION_CLANG_CUDA_STD_FLAG}>)
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
