# Reuse LinearAlgebra's dependency setup; matrix decompositions, solvers, and
# BLAS/LAPACK backends belong to it.
function(xsigma_add_linearalgebra_product)
    if(TARGET LinearAlgebra::LinearAlgebra)
        return()
    endif()

    set(_linalg_src "${XSIGMA_THIRDPARTY_DIR}/LinearAlgebra")
    if(NOT EXISTS "${_linalg_src}/CMakeLists.txt")
        message(FATAL_ERROR "ThirdParty/LinearAlgebra is missing. Initialize the submodule:\n"
                            "  git submodule update --init --recursive ThirdParty/LinearAlgebra")
    endif()

    set(LINALG_ENABLE_TESTING OFF CACHE BOOL "Dependency tests disabled" FORCE)
    set(LINALG_ENABLE_BENCHMARK OFF CACHE BOOL "Dependency benchmarks disabled" FORCE)
    set(LINALG_BUILD_EXAMPLES OFF CACHE BOOL "Dependency examples disabled" FORCE)

    # When no CPU SIMD backend is active the simd<T> types that LinearAlgebra's
    # vectorized scalar GEMM instantiates are not compiled in, so disable that
    # path to avoid build failures with VECTORIZATION_CPU_BACKEND=no.
    if(VECTORIZATION_CPU_BACKEND STREQUAL "no")
        set(LINALG_ENABLE_VECTORIZATION OFF CACHE BOOL "Configured by Vectorization" FORCE)
    else()
        set(LINALG_ENABLE_VECTORIZATION ON  CACHE BOOL "Configured by Vectorization" FORCE)
    endif()

    # --- CPU BLAS backends: forward from Vectorization public flags ---
    foreach(_linalg_feature IN ITEMS MKL BLAS BLIS)
        if(DEFINED VECTORIZATION_ENABLE_${_linalg_feature})
            set(LINALG_ENABLE_${_linalg_feature} "${VECTORIZATION_ENABLE_${_linalg_feature}}"
                CACHE BOOL "Configured by Vectorization" FORCE)
        endif()
    endforeach()

    # --- GPU backend → GPU BLAS library (cuda→cuBLAS, hip→hipBLAS, metal→MetalBLAS) ---
    if(VECTORIZATION_GPU_BACKEND STREQUAL "cuda")
        set(LINALG_ENABLE_CUBLAS  ON  CACHE BOOL "Configured by Vectorization" FORCE)
        set(LINALG_ENABLE_ROCBLAS OFF CACHE BOOL "Configured by Vectorization" FORCE)
        set(LINALG_ENABLE_METAL   OFF CACHE BOOL "Configured by Vectorization" FORCE)
    elseif(VECTORIZATION_GPU_BACKEND STREQUAL "hip")
        set(LINALG_ENABLE_CUBLAS  OFF CACHE BOOL "Configured by Vectorization" FORCE)
        set(LINALG_ENABLE_ROCBLAS ON  CACHE BOOL "Configured by Vectorization" FORCE)
        set(LINALG_ENABLE_METAL   OFF CACHE BOOL "Configured by Vectorization" FORCE)
    elseif(VECTORIZATION_GPU_BACKEND STREQUAL "metal")
        set(LINALG_ENABLE_CUBLAS  OFF CACHE BOOL "Configured by Vectorization" FORCE)
        set(LINALG_ENABLE_ROCBLAS OFF CACHE BOOL "Configured by Vectorization" FORCE)
        set(LINALG_ENABLE_METAL   ON  CACHE BOOL "Configured by Vectorization" FORCE)
    else()
        set(LINALG_ENABLE_CUBLAS  OFF CACHE BOOL "Configured by Vectorization" FORCE)
        set(LINALG_ENABLE_ROCBLAS OFF CACHE BOOL "Configured by Vectorization" FORCE)
        set(LINALG_ENABLE_METAL   OFF CACHE BOOL "Configured by Vectorization" FORCE)
    endif()

    add_subdirectory("${_linalg_src}" "${CMAKE_BINARY_DIR}/ThirdParty/LinearAlgebra_build" EXCLUDE_FROM_ALL)
endfunction()
