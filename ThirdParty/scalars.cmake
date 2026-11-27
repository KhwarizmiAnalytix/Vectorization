# Scalars — sibling XSigma repo, compiled from ThirdParty/Scalars. Vectorization
# (the top-level project) owns it; nested projects such as LinearAlgebra reach
# it via xsigma_add_scalars(). Option forwarding happens in
# Cmake/dependencies.cmake before any subproject is added.
function(xsigma_add_scalars_product)
    if(TARGET Scalars::Scalars)
        return()
    endif()
    set(_scalars_src "${VECTORIZATION_THIRDPARTY_DIR}/Scalars")
    if(NOT EXISTS "${_scalars_src}/CMakeLists.txt")
        message(FATAL_ERROR "ThirdParty/Scalars is missing. Initialize the submodule:\n"
                            "  git submodule update --init --recursive ThirdParty/Scalars")
    endif()
    # SCALARS_ENABLE_SVML was seeded from VECTORIZATION_ENABLE_SVML. Do not
    # shadow it with a normal variable: on x86, when the compiler has no built-in
    # SVML intrinsics and SLEEF is off, Scalars forces it ON so transcendentals
    # use the vendored SVML runtime instead of a per-lane scalar fallback.
    set(SCALARS_CPU_BACKEND "${VECTORIZATION_CPU_BACKEND}" CACHE STRING "Scalars CPU backend" FORCE)
    set(SCALARS_GPU_BACKEND "${VECTORIZATION_GPU_BACKEND}" CACHE STRING "Scalars GPU backend" FORCE)
    add_subdirectory("${_scalars_src}" "${VECTORIZATION_THIRDPARTY_BINARY_DIR}/Scalars" EXCLUDE_FROM_ALL)
    if(TARGET sleef)
        # The pinned SLEEF generates sleef.h in the top-level binary directory.
        target_include_directories(sleef INTERFACE "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
        # SLEEF links TLFloat by name but keeps its search directories private.
        # Forward the actual directories on the target: these capture the prefix
        # used inside SLEEF (or a system TLFloat), unlike our parent-scope prefix.
        get_target_property(_vec_sleef_libraries sleef INTERFACE_LINK_LIBRARIES)
        get_target_property(_vec_sleef_link_dirs sleef LINK_DIRECTORIES)
        if("tlfloat" IN_LIST _vec_sleef_libraries AND _vec_sleef_link_dirs)
            foreach(_vec_sleef_link_dir IN LISTS _vec_sleef_link_dirs)
                target_link_directories(sleef INTERFACE "$<BUILD_INTERFACE:${_vec_sleef_link_dir}>")
            endforeach()
        endif()
    endif()
    # Propagate the resolved CPU backend before configuring other consumers.
    # For example, AVX2 on arm64 must also disable LinearAlgebra SIMD.
    if(NOT VECTORIZATION_CPU_BACKEND STREQUAL SCALARS_CPU_BACKEND)
        set(VECTORIZATION_CPU_BACKEND "${SCALARS_CPU_BACKEND}"
            CACHE STRING "CPU SIMD backend resolved by Scalars" FORCE)
    endif()
endfunction()
