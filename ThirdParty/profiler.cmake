# Host overlay: build Profiler product sources only (ThirdParty/Profiler).
# Do not add_subdirectory Profiler's CMakeLists — that tree also compiles
# Profiler's private instrumentation backends. XSigma links Profiler::Profiler.

function(xsigma_add_profiler_product)
    if(TARGET Profiler::Profiler)
        return()
    endif()

    set(_profiler_src "${XSIGMA_THIRDPARTY_DIR}/Profiler")
    if(NOT EXISTS "${_profiler_src}/Profiler/profiler.h")
        message(FATAL_ERROR "ThirdParty/Profiler is missing. Initialize the submodule:\n"
                            "  git submodule update --init ThirdParty/Profiler")
    endif()

    if(NOT TARGET Fmt::fmt)
        set(FMT_TEST OFF CACHE BOOL "Disable fmt tests" FORCE)
        set(FMT_DOC OFF CACHE BOOL "Disable fmt documentation" FORCE)
        set(FMT_INSTALL OFF CACHE BOOL "Disable fmt install" FORCE)
        set(FMT_HEADER_ONLY ON CACHE BOOL "Use header-only fmt library" FORCE)
        add_third_party_library(
            fmt
            LIBRARY_TYPE STATIC
            SOURCE_DIR "${XSIGMA_THIRDPARTY_DIR}/fmt"
            ALIASES "fmt::fmt-header-only"
            XSIGMA_TARGETS "Fmt::fmt=fmt::fmt-header-only|fmt::fmt|fmt"
            INCLUDE_DIR "${XSIGMA_THIRDPARTY_DIR}/fmt/include"
            AUTO_ADD_TO_DEPENDENCIES OFF
        )
    endif()

    set(_profiler_gpu "none")
    if(DEFINED MEMORY_GPU_BACKEND AND NOT MEMORY_GPU_BACKEND STREQUAL "")
        set(_profiler_gpu "${MEMORY_GPU_BACKEND}")
    elseif(DEFINED VECTORIZATION_GPU_BACKEND AND NOT VECTORIZATION_GPU_BACKEND STREQUAL "")
        set(_profiler_gpu "${VECTORIZATION_GPU_BACKEND}")
    endif()

    set(_profiler_has_cuda 0)
    set(_profiler_has_nvtx 0)
    set(_profiler_has_hip 0)
    set(_profiler_has_metal 0)
    set(_profiler_gpu_libs "")

    if(_profiler_gpu STREQUAL "cuda")
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            set(_profiler_has_cuda 1)
            list(APPEND _profiler_gpu_libs CUDA::cudart)
            if(TARGET CUDA::nvToolsExt)
                list(APPEND _profiler_gpu_libs CUDA::nvToolsExt)
                set(_profiler_has_nvtx 1)
            elseif(TARGET CUDA::nvtx3)
                list(APPEND _profiler_gpu_libs CUDA::nvtx3)
                set(_profiler_has_nvtx 1)
            endif()
        endif()
    elseif(_profiler_gpu STREQUAL "hip")
        find_package(hip QUIET)
        if(hip_FOUND)
            set(_profiler_has_hip 1)
            list(APPEND _profiler_gpu_libs hip::host)
        endif()
    elseif(APPLE AND _profiler_gpu STREQUAL "metal")
        set(_profiler_has_metal 1)
    endif()

    set(_profiler_root "${_profiler_src}/Profiler")
    file(GLOB _profiler_common_sources LIST_DIRECTORIES false "${_profiler_root}/common/*.cpp")
    file(GLOB_RECURSE _profiler_native_sources LIST_DIRECTORIES false
         "${_profiler_root}/native/*.cpp")
    set(_profiler_sources ${_profiler_common_sources} ${_profiler_native_sources})

    file(GLOB _profiler_common_headers LIST_DIRECTORIES false
         "${_profiler_root}/*.h" "${_profiler_root}/common/*.h" "${_profiler_root}/util/*.h")
    file(GLOB_RECURSE _profiler_native_headers LIST_DIRECTORIES false
         "${_profiler_root}/native/*.h" "${_profiler_root}/native/*.hxx")
    set(_profiler_headers ${_profiler_common_headers} ${_profiler_native_headers})

    if(_profiler_has_metal)
        enable_language(OBJCXX)
        set(CMAKE_OBJCXX_STANDARD 20)
        set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
        find_library(XSIGMA_PROFILER_METAL_FRAMEWORK Metal REQUIRED)
        find_library(XSIGMA_PROFILER_FOUNDATION_FRAMEWORK Foundation REQUIRED)
        file(GLOB_RECURSE _profiler_mm LIST_DIRECTORIES false "${_profiler_root}/native/*.mm")
        list(APPEND _profiler_sources ${_profiler_mm})
        set_source_files_properties(
            ${_profiler_mm} PROPERTIES COMPILE_OPTIONS "-fobjc-arc;-Wno-deprecated-declarations"
        )
        list(APPEND _profiler_gpu_libs
             ${XSIGMA_PROFILER_METAL_FRAMEWORK} ${XSIGMA_PROFILER_FOUNDATION_FRAMEWORK})
    endif()

    add_library(Profiler EXCLUDE_FROM_ALL ${_profiler_sources} ${_profiler_headers})
    add_library(Profiler::Profiler ALIAS Profiler)
    set_target_properties(
        Profiler PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        FOLDER "ThirdParty"
    )

    if(BUILD_SHARED_LIBS)
        target_compile_definitions(Profiler PUBLIC PROFILER_SHARED_DEFINE PRIVATE PROFILER_BUILDING_DLL)
    else()
        target_compile_definitions(Profiler PUBLIC PROFILER_STATIC_DEFINE)
    endif()

    target_compile_definitions(
        Profiler
        PUBLIC
            PROFILER_HAS_KINETO=0
            PROFILER_HAS_ITT=0
            PROFILER_HAS_CUDA=${_profiler_has_cuda}
            PROFILER_HAS_NVTX=${_profiler_has_nvtx}
            PROFILER_HAS_HIP=${_profiler_has_hip}
            PROFILER_HAS_METAL=${_profiler_has_metal}
            PROFILER_HAS_ROCTX=0
    )

    target_include_directories(
        Profiler
        PUBLIC $<BUILD_INTERFACE:${_profiler_root}>
    )

    if(TARGET Fmt::fmt)
        target_link_libraries(Profiler PUBLIC Fmt::fmt)
    endif()
    if(_profiler_gpu_libs)
        target_link_libraries(Profiler PUBLIC ${_profiler_gpu_libs})
    endif()
endfunction()
