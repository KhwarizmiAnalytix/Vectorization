# Vectorization owns the public configuration; dependencies own implementation
# and package discovery. Never forward an unrelated feature to every dependency.
function(vectorization_forward_dependency prefix target)
  if(TARGET "${target}")
    return()
  endif()
  foreach(feature IN ITEMS CACHE ICECC SANITIZER COVERAGE VALGRIND)
    set(${prefix}_ENABLE_${feature} "${VECTORIZATION_ENABLE_${feature}}"
        CACHE BOOL "Configured by Vectorization" FORCE
    )
  endforeach()
  # Analyze this project's sources; vendored projects own their lint policies.
  foreach(feature IN ITEMS CLANGTIDY FIX IWYU SPELL)
    set(${prefix}_ENABLE_${feature} OFF CACHE BOOL "Dependency analysis disabled" FORCE)
  endforeach()
  foreach(setting IN ITEMS CACHE_BACKEND SANITIZER_TYPE LINKER_CHOICE LTO_MODE)
    set(${prefix}_${setting} "${VECTORIZATION_${setting}}"
        CACHE STRING "Configured by Vectorization" FORCE
    )
  endforeach()
  # Scalars and LinearAlgebra require C++20 even when an older standard is
  # requested for Vectorization. Forward newer standards consistently.
  set(_dependency_standard "${VECTORIZATION_CXX_STANDARD}")
  if(_dependency_standard LESS 20)
    set(_dependency_standard 20)
  endif()
  set(${prefix}_CXX_STANDARD "${_dependency_standard}" CACHE STRING "Configured by Vectorization"
                                                             FORCE
  )
  # This repository's testing and benchmark switches apply to its own targets.
  foreach(feature IN ITEMS TESTING BENCHMARK EXAMPLES)
    set(${prefix}_ENABLE_${feature} OFF CACHE BOOL "Dependency targets disabled" FORCE)
  endforeach()
endfunction()

vectorization_forward_dependency(LOGGING Logging::Logging)
vectorization_forward_dependency(MEMORY Memory::Memory)
vectorization_forward_dependency(SCALARS Scalars::Scalars)
vectorization_forward_dependency(LINALG LinearAlgebra::LinearAlgebra)
vectorization_forward_dependency(PARALLEL Parallel::Parallel)

if(NOT TARGET Logging::Logging)
  # Logging consumes fmt's headers/library, never its optional C++ module.
  # The pinned fmt enables that extra target for clang-cl even though CMake
  # cannot scan modules with its MSVC frontend, which breaks generation.
  set(FMT_MODULE OFF CACHE BOOL "fmt modules are not required by Vectorization" FORCE)
  foreach(feature IN ITEMS MAGICENUM PORTABLE_FLOAT_FORMAT)
    set(LOGGING_ENABLE_${feature} "${VECTORIZATION_ENABLE_${feature}}"
        CACHE BOOL "Configured by Vectorization" FORCE
    )
  endforeach()
endif()
if(NOT TARGET Memory::Memory)
  foreach(feature IN ITEMS MIMALLOC MIMALLOC_STATS NUMA MEMKIND TBB)
    set(MEMORY_ENABLE_${feature} "${VECTORIZATION_ENABLE_${feature}}"
        CACHE BOOL "Configured by Vectorization" FORCE
    )
  endforeach()
endif()
if(NOT TARGET Scalars::Scalars)
  # The pinned SLEEF uses ENABLE_SSL, not DISABLE_SSL. Its OpenSSL helpers
  # register tests even with SLEEF_BUILD_TESTS=OFF and remain unbuilt in an
  # EXCLUDE_FROM_ALL dependency, leaving failing entries in our CTest suite.
  if(VECTORIZATION_ENABLE_SLEEF)
    set(SLEEF_ENABLE_SSL OFF CACHE BOOL "Disable SLEEF's OpenSSL test helpers" FORCE)
  endif()
  foreach(feature IN ITEMS SVML SLEEF ACCELERATE)
    set(SCALARS_ENABLE_${feature} "${VECTORIZATION_ENABLE_${feature}}"
        CACHE BOOL "Configured by Vectorization" FORCE
    )
  endforeach()
  # MKL is used by comparison tests here; Scalars does not need it.
  set(SCALARS_ENABLE_MKL OFF CACHE BOOL "MKL is not required by Scalars" FORCE)
  set(SCALARS_SVE_VECTOR_BITS "${VECTORIZATION_SVE_VECTOR_BITS}"
      CACHE STRING "Configured by Vectorization" FORCE
  )
endif()

# CMake's MSVC rules give /EHsc to CXX only, so clang-cl would build CUDA TUs
# without C++ exception handling while the DLLs they call throw through their
# frames, and inline code would differ from the CXX TUs it is shared with.
if(WIN32 AND VECTORIZATION_GPU_BACKEND STREQUAL "cuda" AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:/EHsc>")
  endif()
  # The device pass has no DLLs, so Clang ignores every dllexport/dllimport
  # there with a -Wignored-attributes warning, once per exported declaration
  # per GPU architecture. Silence that group for the device pass only; the
  # host pass keeps it.
  if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    add_compile_options(
      "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:/clang:-Xarch_device /clang:-Wno-ignored-attributes>"
    )
  else()
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xarch_device -Wno-ignored-attributes>")
  endif()
endif()
# Clang knows CUDA releases only up to its own version and warns about each
# CUDA TU when the toolkit is newer.
if(VECTORIZATION_GPU_BACKEND STREQUAL "cuda" AND CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
  add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Wno-unknown-cuda-version>")
endif()
if(NOT TARGET Memory::Memory)
  # Tensor allocation and expression evaluation must use the same GPU runtime.
  set(MEMORY_GPU_BACKEND "${VECTORIZATION_GPU_BACKEND}" CACHE STRING "Memory GPU backend" FORCE)
  add_subdirectory(
    "${VECTORIZATION_THIRDPARTY_DIR}/Memory" "${CMAKE_CURRENT_BINARY_DIR}/ThirdParty/Memory"
    EXCLUDE_FROM_ALL
  )
endif()
if(WIN32 AND VECTORIZATION_GPU_BACKEND STREQUAL "cuda" AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC" AND VECTORIZATION_CPU_BACKEND STREQUAL "avx2")
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:/clang:-mavx2>)
  endif()
  set(CMAKE_CUDA_RUNTIME_LIBRARY_LINK_OPTIONS_SHARED
      "${CUDAToolkit_LIBRARY_DIR}/cudadevrt.lib"
      "${CUDAToolkit_LIBRARY_DIR}/cudart.lib"
  )
  set(CMAKE_CUDA_RUNTIME_LIBRARY_LINK_OPTIONS_STATIC
      "${CUDAToolkit_LIBRARY_DIR}/cudadevrt.lib"
      "${CUDAToolkit_LIBRARY_DIR}/cudart_static.lib"
  )
  set(CMAKE_CUDA_RUNTIME_LIBRARY_LINK_OPTIONS_SHARED
      "${CUDAToolkit_LIBRARY_DIR}/cudadevrt.lib"
      "${CUDAToolkit_LIBRARY_DIR}/cudart.lib"
      CACHE INTERNAL "CUDA shared runtime link options" FORCE
  )
  set(CMAKE_CUDA_RUNTIME_LIBRARY_LINK_OPTIONS_STATIC
      "${CUDAToolkit_LIBRARY_DIR}/cudadevrt.lib"
      "${CUDAToolkit_LIBRARY_DIR}/cudart_static.lib"
      CACHE INTERNAL "CUDA static runtime link options" FORCE
  )
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL" CACHE STRING "MSVC runtime library" FORCE)
  # Bare .lib names are clang-cl's spelling; the GNU clang++ driver treats them
  # as input files ("no such file or directory: 'kernel32.lib'").
  if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    set(CMAKE_CXX_STANDARD_LIBRARIES
        "kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib"
        CACHE STRING "C++ standard libraries" FORCE
    )
  endif()
endif()
get_target_property(_vec_memory_definitions Memory::Memory INTERFACE_COMPILE_DEFINITIONS)
foreach(_backend IN ITEMS cuda hip metal)
  string(TOUPPER "${_backend}" _backend_upper)
  if(VECTORIZATION_GPU_BACKEND STREQUAL _backend AND NOT "MEMORY_HAS_${_backend_upper}=1" IN_LIST
                                                     _vec_memory_definitions
  )
    message(
      FATAL_ERROR "Vectorization's ${_backend} backend requires Memory built with "
                  "MEMORY_GPU_BACKEND=${_backend}. The existing Memory target is incompatible."
    )
  endif()
endforeach()
if(NOT TARGET Scalars::Scalars)
  # SCALARS_ENABLE_SVML was seeded from VECTORIZATION_ENABLE_SVML above. Do not
  # shadow it with a normal variable: on x86, when the compiler has no built-in
  # SVML intrinsics and SLEEF is off, Scalars forces it ON so transcendentals
  # use the vendored SVML runtime instead of a per-lane scalar fallback.
  set(SCALARS_CPU_BACKEND "${VECTORIZATION_CPU_BACKEND}" CACHE STRING "Scalars CPU backend" FORCE)
  set(SCALARS_GPU_BACKEND "${VECTORIZATION_GPU_BACKEND}" CACHE STRING "Scalars GPU backend" FORCE)
  add_subdirectory(
    "${VECTORIZATION_THIRDPARTY_DIR}/Scalars" "${CMAKE_CURRENT_BINARY_DIR}/ThirdParty/Scalars"
    EXCLUDE_FROM_ALL
  )
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
  # An explicit request must be satisfied; an automatic enable that finds no
  # runtime leaves SCALARS_HAS_SVML=0 (Scalars warns) and keeps the fallback.
  if(VECTORIZATION_ENABLE_SVML AND NOT TARGET SVML::SVML)
    message(
      FATAL_ERROR "VECTORIZATION_ENABLE_SVML=ON requires the SVML runtime in Scalars. "
                  "Provide ThirdParty/Scalars/ThirdParty/svml or disable VECTORIZATION_ENABLE_SVML."
    )
  endif()
  if(SCALARS_ENABLE_SVML AND TARGET SVML::SVML)
    if(VECTORIZATION_ENABLE_SVML)
      set(VECTORIZATION_SVML_STATUS "ON")
    else()
      set(VECTORIZATION_SVML_STATUS "ON (auto)")
    endif()
  else()
    set(VECTORIZATION_SVML_STATUS "OFF")
  endif()
  # Propagate the resolved CPU backend before configuring other consumers.
  # For example, AVX2 on arm64 must also disable LinearAlgebra SIMD.
  if(NOT VECTORIZATION_CPU_BACKEND STREQUAL SCALARS_CPU_BACKEND)
    set(VECTORIZATION_CPU_BACKEND "${SCALARS_CPU_BACKEND}"
        CACHE STRING "CPU SIMD backend resolved by Scalars" FORCE
    )
  endif()
endif()
# Dependencies pass host-compiler flags to CUDA sources as -Xcompiler=<flag>,
# which is right for nvcc but ignored by clang-cl ("unknown argument"), which
# takes the bare flags. Drop the prefix on <target> when clang-cl compiles CUDA.
# /Zc:preprocessor is dropped entirely: it is there for cl.exe behind nvcc, and
# clang-cl, whose preprocessor always conforms, reports it as unused.
function(vectorization_unwrap_cuda_xcompiler_options target)
  if(NOT (WIN32 AND VECTORIZATION_GPU_BACKEND STREQUAL "cuda"
          AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
          AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC" AND TARGET ${target})
  )
    return()
  endif()
  foreach(_property IN ITEMS COMPILE_OPTIONS INTERFACE_COMPILE_OPTIONS)
    get_target_property(_options ${target} ${_property})
    if(_options)
      string(REPLACE "<COMPILE_LANGUAGE:CUDA>:-Xcompiler=" "<COMPILE_LANGUAGE:CUDA>:" _options
                     "${_options}"
      )
      list(REMOVE_ITEM _options "$<$<COMPILE_LANGUAGE:CUDA>:/Zc:preprocessor>")
      set_property(TARGET ${target} PROPERTY ${_property} "${_options}")
    endif()
  endforeach()
endfunction()
# For Scalars this is more than noise: its host SIMD flags (/arch:AVX2
# /D__FMA__ ...) would otherwise be lost, and CUDA TUs would see a different
# Scalars SIMD ABI than ordinary C++ TUs, an ODR violation that miscompiles
# code shared between them.
vectorization_unwrap_cuda_xcompiler_options(Scalars)

# Memory links CUPTI's nvperf_host.lib, whose import library exports DllMain.
# LLD skips that symbol, as it must, and warns on every Memory.dll link.
if(WIN32 AND VECTORIZATION_GPU_BACKEND STREQUAL "cuda" AND TARGET Memory
   AND CMAKE_CXX_COMPILER_LINKER_ID STREQUAL "LLD"
)
  target_link_options(Memory PRIVATE "LINKER:/ignore:importeddllmain")
endif()
