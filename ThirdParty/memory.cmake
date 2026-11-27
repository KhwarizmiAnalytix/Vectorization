# Memory — sibling XSigma repo, compiled from ThirdParty/Memory. Vectorization
# (the top-level project) owns it; nested projects reach it via xsigma_add_memory().
# Option forwarding happens in Cmake/dependencies.cmake before any subproject
# is added, so it is in place whichever caller runs first.
function(xsigma_add_memory_product)
    if(TARGET Memory::Memory)
        return()
    endif()
    set(_memory_src "${VECTORIZATION_THIRDPARTY_DIR}/Memory")
    if(NOT EXISTS "${_memory_src}/CMakeLists.txt")
        message(FATAL_ERROR "ThirdParty/Memory is missing. Initialize the submodule:\n"
                            "  git submodule update --init --recursive ThirdParty/Memory")
    endif()
    # Tensor allocation and expression evaluation must use the same GPU runtime.
    set(MEMORY_GPU_BACKEND "${VECTORIZATION_GPU_BACKEND}" CACHE STRING "Memory GPU backend" FORCE)
    add_subdirectory("${_memory_src}" "${VECTORIZATION_THIRDPARTY_BINARY_DIR}/Memory" EXCLUDE_FROM_ALL)
endfunction()
