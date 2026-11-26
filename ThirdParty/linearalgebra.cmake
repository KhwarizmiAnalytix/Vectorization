# Reuse LinearAlgebra's dependency setup; matrix decompositions, solvers, and
# BLAS/LAPACK backends belong to it.
function(xsigma_add_linearalgebra_product)
    if(TARGET LinearAlgebra::LinearAlgebra)
        return()
    endif()

    set(_linalg_src "${VECTORIZATION_THIRDPARTY_DIR}/LinearAlgebra")
    if(NOT EXISTS "${_linalg_src}/CMakeLists.txt")
        message(FATAL_ERROR "ThirdParty/LinearAlgebra is missing. Initialize the submodule:\n"
                            "  git submodule update --init --recursive ThirdParty/LinearAlgebra")
    endif()

    set(LINALG_ENABLE_TESTING OFF CACHE BOOL "Dependency tests disabled" FORCE)
    set(LINALG_ENABLE_BENCHMARK OFF CACHE BOOL "Dependency benchmarks disabled" FORCE)
    set(LINALG_BUILD_EXAMPLES OFF CACHE BOOL "Dependency examples disabled" FORCE)

    # Scalars has already resolved unsupported ISA requests to its scalar fallback.
    # LinearAlgebra's current API uses selectors, not ENABLE_VECTORIZATION or
    # the legacy CUBLAS/ROCBLAS/METAL switches.
    set(LINALG_CPU_BACKEND "${VECTORIZATION_CPU_BACKEND}" CACHE STRING "Configured by Vectorization" FORCE)
    set(LINALG_GPU_BACKEND "${VECTORIZATION_GPU_BACKEND}" CACHE STRING "Configured by Vectorization" FORCE)
    unset(LINALG_ENABLE_VECTORIZATION CACHE)
    foreach(_legacy_gpu IN ITEMS CUBLAS ROCBLAS METAL)
        set(LINALG_ENABLE_${_legacy_gpu} OFF)
        unset(LINALG_ENABLE_${_legacy_gpu} CACHE)
    endforeach()

    # Build LinearAlgebra's kernels for the SMs Scalars resolved (which honors
    # CMAKE_CUDA_ARCHITECTURES). Left alone, LinearAlgebra falls back to
    # "native", which CMake cannot resolve with Clang as the CUDA compiler
    # ("no NVIDIA GPU was detected") and which would differ from Scalars' list.
    if(VECTORIZATION_GPU_BACKEND STREQUAL "cuda" AND SCALARS_CUDA_ARCHITECTURES)
        set(LINALG_CUDA_ARCHITECTURES "${SCALARS_CUDA_ARCHITECTURES}"
            CACHE STRING "Configured by Vectorization" FORCE)
    endif()

    # --- CPU BLAS backends: forward from Vectorization public flags ---
    foreach(_linalg_feature IN ITEMS MKL BLAS BLIS)
        if(DEFINED VECTORIZATION_ENABLE_${_linalg_feature})
            set(LINALG_ENABLE_${_linalg_feature} "${VECTORIZATION_ENABLE_${_linalg_feature}}"
                CACHE BOOL "Configured by Vectorization" FORCE)
        endif()
    endforeach()

    # Only LinearAlgebra's NATIVE LU honors this (BLAS/MKL getrf always
    # pivots). Without partial pivoting, LU is numerically unstable for a
    # general matrix and fails outright on a zero diagonal that a row swap
    # would avoid, so forward Vectorization's default (ON) rather than
    # inheriting the dependency's default (OFF). Bazel mirrors this with
    # --define=linalg_lu_pivoting=true in .bazelrc.
    set(LINALG_LU_PIVOTING "${VECTORIZATION_ENABLE_LU_PIVOTING}"
        CACHE BOOL "Configured by Vectorization" FORCE)

    add_subdirectory("${_linalg_src}" "${CMAKE_BINARY_DIR}/ThirdParty/LinearAlgebra_build" EXCLUDE_FROM_ALL)

    # Windows Clang-CUDA needs the same two fixes as our CUDA-tagged tests
    # (Testing/Cxx/CMakeLists.txt) for LinearAlgebra's .cu kernels: NOMINMAX,
    # since the CUDA headers pull in windows.h's min/max macros, and
    # -fno-cuda-host-device-constexpr, since MSVC's C++20 std::lerp otherwise
    # makes "call to 'fma' is ambiguous" against Clang's device overloads.
    if(WIN32 AND VECTORIZATION_GPU_BACKEND STREQUAL "cuda" AND CMAKE_CUDA_COMPILER_ID STREQUAL "Clang"
       AND TARGET LinearAlgebra)
        target_compile_definitions(LinearAlgebra PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:NOMINMAX>)
        target_compile_options(LinearAlgebra
            PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xclang -fno-cuda-host-device-constexpr>")
    endif()
    # Its CUDA kernels get /Zc:preprocessor as nvcc's -Xcompiler=/Zc:preprocessor,
    # which clang-cl ignores with a warning (see Cmake/dependencies.cmake).
    vectorization_unwrap_cuda_xcompiler_options(LinearAlgebra)
endfunction()
