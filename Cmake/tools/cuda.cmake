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
        $<$<COMPILE_LANGUAGE:CUDA>:-std=c++${VECTORIZATION_CXX_STANDARD}>)
  else()
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -Xcompiler=-fPIC>)
  endif()
  if(MSVC)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
  endif()
endfunction()
