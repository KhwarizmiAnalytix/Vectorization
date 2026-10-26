# Vectorization owns the public configuration; dependencies own implementation
# and package discovery. Never forward an unrelated feature to every dependency.
function(vectorization_forward_dependency prefix target)
  if(TARGET "${target}")
    return()
  endif()
  foreach(
    feature IN
    ITEMS CACHE
          ICECC
          CLANGTIDY
          FIX
          IWYU
          SANITIZER
          COVERAGE
          SPELL
          VALGRIND
  )
    set(${prefix}_ENABLE_${feature} "${VECTORIZATION_ENABLE_${feature}}"
        CACHE BOOL "Configured by Vectorization" FORCE
    )
  endforeach()
  foreach(setting IN ITEMS CACHE_BACKEND SANITIZER_TYPE LINKER_CHOICE LTO_MODE)
    set(${prefix}_${setting} "${VECTORIZATION_${setting}}"
        CACHE STRING "Configured by Vectorization" FORCE
    )
  endforeach()
  # This repository's testing and benchmark switches apply to its own targets.
  foreach(feature IN ITEMS TESTING BENCHMARK EXAMPLES)
    set(${prefix}_ENABLE_${feature} OFF CACHE BOOL "Dependency targets disabled" FORCE)
  endforeach()
endfunction()

vectorization_forward_dependency(LOGGING Logging::Logging)
vectorization_forward_dependency(MEMORY Memory::Memory)
vectorization_forward_dependency(SCALARS Scalars::Scalars)

if(NOT TARGET Logging::Logging)
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

if(NOT TARGET Memory::Memory)
  # Tensor allocation and expression evaluation must use the same GPU runtime.
  set(MEMORY_GPU_BACKEND "${VECTORIZATION_GPU_BACKEND}" CACHE STRING "Memory GPU backend" FORCE)
  add_subdirectory(
    "${XSIGMA_THIRDPARTY_DIR}/Memory" "${CMAKE_CURRENT_BINARY_DIR}/ThirdParty/Memory"
    EXCLUDE_FROM_ALL
  )
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
  # The pinned Scalars probes force its SVML cache option ON for GCC/Clang.
  # Preserve the Scalars option under CMP0126 until that probe is updated.
  set(SCALARS_ENABLE_SVML "${VECTORIZATION_ENABLE_SVML}")
  set(SCALARS_CPU_BACKEND "${VECTORIZATION_CPU_BACKEND}" CACHE STRING "Scalars CPU backend" FORCE)
  set(SCALARS_GPU_BACKEND "${VECTORIZATION_GPU_BACKEND}" CACHE STRING "Scalars GPU backend" FORCE)
  add_subdirectory(
    "${XSIGMA_THIRDPARTY_DIR}/Scalars" "${CMAKE_CURRENT_BINARY_DIR}/ThirdParty/Scalars"
    EXCLUDE_FROM_ALL
  )
  set(SCALARS_ENABLE_SVML "${SCALARS_ENABLE_SVML}"
      CACHE BOOL "Enable Intel SVML short vector math library" FORCE
  )
  if(SCALARS_ENABLE_SVML AND NOT TARGET SVML::SVML)
    message(
      FATAL_ERROR "VECTORIZATION_ENABLE_SVML=ON requires the SVML runtime in Scalars. "
                  "Provide ThirdParty/Scalars/ThirdParty/svml or disable VECTORIZATION_ENABLE_SVML."
    )
  endif()
endif()
