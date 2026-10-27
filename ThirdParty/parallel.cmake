# Parallel — sibling XSigma repo, compiled from ThirdParty/Parallel.
# Called by Vectorization when VECTORIZATION_ENABLE_PARALLEL=ON, before
# if(TARGET Parallel::Parallel) — idempotent.
function(xsigma_add_parallel_product)
    if(TARGET Parallel::Parallel)
        return()
    endif()

    set(_parallel_src "${XSIGMA_THIRDPARTY_DIR}/Parallel")
    if(NOT EXISTS "${_parallel_src}/CMakeLists.txt")
        message(FATAL_ERROR "ThirdParty/Parallel is missing. Initialize the submodule:\n"
                            "  git submodule update --init --recursive ThirdParty/Parallel")
    endif()

    set(PARALLEL_ENABLE_TESTING   OFF CACHE BOOL "Dependency tests disabled"      FORCE)
    set(PARALLEL_ENABLE_BENCHMARK OFF CACHE BOOL "Dependency benchmarks disabled"  FORCE)
    set(PARALLEL_ENABLE_EXAMPLES  OFF CACHE BOOL "Dependency examples disabled"    FORCE)

    add_subdirectory("${_parallel_src}" "${CMAKE_BINARY_DIR}/ThirdParty/Parallel_build" EXCLUDE_FROM_ALL)
endfunction()
