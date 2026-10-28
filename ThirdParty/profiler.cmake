# Profiler is a sibling XSigma repo (KhwarizmiAnalytix/Profiler), not a vendored
# third-party library — it is never bundled or built from source here. Resolved
# exclusively via find_package(Profiler), independent of VECTORIZATION_ENABLE_EXTERNAL
# (that flag governs vendored ThirdParty/ copies, which Profiler no longer is).
# Install it first (cmake --install) and point CMAKE_PREFIX_PATH at it. Optional:
# callers gate on TARGET Profiler::Profiler, so a missing install just disables
# VECTORIZATION_HAS_PROFILER rather than failing the configure.

function(xsigma_add_profiler_product)
    if(TARGET Profiler::Profiler)
        return()
    endif()

    find_package(Profiler CONFIG QUIET)
    if(TARGET Profiler::Profiler)
        message(STATUS "Vectorization: using Profiler::Profiler (find_package)")
    else()
        message(STATUS "Vectorization: Profiler not found via find_package; "
                        "VECTORIZATION_HAS_PROFILER will be 0. Install it and set "
                        "CMAKE_PREFIX_PATH to enable it.")
    endif()
endfunction()
