# Reuse Logging's dependency setup; fmt and logging backends belong to it.
function(xsigma_add_logging_product)
    if(TARGET Logging::Logging)
        return()
    endif()

    set(_logging_src "${XSIGMA_THIRDPARTY_DIR}/Logging")
    if(NOT EXISTS "${_logging_src}/CMakeLists.txt")
        message(FATAL_ERROR "ThirdParty/Logging is missing. Initialize the submodule:\n"
                            "  git submodule update --init --recursive ThirdParty/Logging")
    endif()

    set(LOGGING_ENABLE_TESTING OFF CACHE BOOL "Dependency tests disabled" FORCE)
    set(LOGGING_ENABLE_BENCHMARK OFF CACHE BOOL "Dependency benchmarks disabled" FORCE)
    set(LOGGING_ENABLE_EXAMPLES OFF CACHE BOOL "Dependency examples disabled" FORCE)
    # Preserve Vectorization's native logging default; allow caller overrides.
    set(LOGGING_BACKEND "NATIVE" CACHE STRING "Logging backend")
    set(LOGGING_ENABLE_MAGICENUM OFF CACHE BOOL "Enable magic_enum in Logging")
    add_subdirectory("${_logging_src}" "${CMAKE_BINARY_DIR}/ThirdParty/Logging_build" EXCLUDE_FROM_ALL)
endfunction()
