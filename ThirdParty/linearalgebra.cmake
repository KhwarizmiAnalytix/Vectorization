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
    add_subdirectory("${_linalg_src}" "${CMAKE_BINARY_DIR}/ThirdParty/LinearAlgebra_build" EXCLUDE_FROM_ALL)
endfunction()
