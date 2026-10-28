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

    # --- CPU BLAS backends: forward from Vectorization public flags ---
    foreach(_linalg_feature IN ITEMS MKL BLAS BLIS)
        if(DEFINED VECTORIZATION_ENABLE_${_linalg_feature})
            set(LINALG_ENABLE_${_linalg_feature} "${VECTORIZATION_ENABLE_${_linalg_feature}}"
                CACHE BOOL "Configured by Vectorization" FORCE)
        endif()
    endforeach()

    add_subdirectory("${_linalg_src}" "${CMAKE_BINARY_DIR}/ThirdParty/LinearAlgebra_build" EXCLUDE_FROM_ALL)
endfunction()
