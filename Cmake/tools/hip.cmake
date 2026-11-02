include_guard(GLOBAL)

# =============================================================================
# HIP GPU backend for Vectorization
#
# Mirrors LinearAlgebra's Cmake/hip.cmake and PyTorch cmake/Hip.cmake.
# Called from CMakeLists.txt when VECTORIZATION_GPU_BACKEND=hip; see
# vectorization_apply_hip_flags() for per-target usage.
# HIP/ROCm is Unix-only in this project (ROCm-on-Windows is unvalidated here;
# fail fast rather than attempt an untested find_package(hip) path).
# =============================================================================

if(WIN32)
  message(FATAL_ERROR
    "VECTORIZATION_GPU_BACKEND=hip is not supported on Windows in this project "
    "(HIP/ROCm is Unix-only here). Use VECTORIZATION_GPU_BACKEND=cuda on Windows, "
    "or build on Linux for HIP."
  )
endif()

# Default architecture list — matches Scalars' gpu_backend.cmake and
# LinearAlgebra's Cmake/hip.cmake (PyTorch TORCH_ROCM_ARCH_LIST equivalent).
set(VECTORIZATION_GPU_ARCHITECTURES "gfx900;gfx906;gfx908;gfx90a;gfx1030;gfx1100"
    CACHE STRING "HIP GPU architectures (gfxXXX for AMD, smXX for CUDA offload via hipcc)"
)
# CI passes explicit CMAKE_HIP_ARCHITECTURES (no device at configure time);
# honour it instead of compiling every gfx* default.
if(CMAKE_HIP_ARCHITECTURES AND NOT CMAKE_HIP_ARCHITECTURES STREQUAL "native")
  set(VECTORIZATION_GPU_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
      CACHE STRING "HIP GPU architectures (gfxXXX for AMD, smXX for CUDA offload via hipcc)"
            FORCE
  )
endif()

if(NOT CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES "${VECTORIZATION_GPU_ARCHITECTURES}")
endif()
include(rocm_prefix)
xsigma_setup_rocm_prefix()

# CMake requires Clang directly. Keep explicit CMAKE_HIP_COMPILER / HIPCXX.
if(NOT CMAKE_HIP_COMPILER AND NOT DEFINED ENV{HIPCXX} AND VECTORIZATION_ROCM_PREFIX)
  find_program(_vectorization_hip_clang NAMES clang++
      PATHS "${VECTORIZATION_ROCM_PREFIX}/llvm/bin" "${VECTORIZATION_ROCM_PREFIX}/bin"
      NO_DEFAULT_PATH NO_CACHE)
  if(_vectorization_hip_clang)
    set(CMAKE_HIP_COMPILER "${_vectorization_hip_clang}" CACHE FILEPATH "HIP compiler")
  endif()
  unset(_vectorization_hip_clang)
endif()

# Locate hip::host / hip::device imported targets so the Vectorization target
# can link HIP even when Memory's HIP finder has not run (e.g. MEMORY_GPU_BACKEND=none).
find_package(hip QUIET)
if(NOT hip_FOUND)
  message(WARNING
    "Vectorization: VECTORIZATION_GPU_BACKEND=hip but the HIP package was not found "
    "via find_package(hip). Set ROCM_PATH or CMAKE_PREFIX_PATH to your ROCm install. "
    "Falling back to HIP language detection; the runtime must be reachable via the toolchain."
  )
endif()

# Strip device-only flags from imported host interfaces, including split -x hip.
function(_vectorization_drop_hip_cxx_device_flags target)
    if(NOT TARGET "${target}")
        return()
    endif()
    foreach(_prop IN ITEMS INTERFACE_COMPILE_OPTIONS INTERFACE_LINK_LIBRARIES INTERFACE_LINK_OPTIONS)
        get_property(_prop_set TARGET "${target}" PROPERTY "${_prop}" SET)
        if(NOT _prop_set)
            continue()
        endif()
        get_property(_opts TARGET "${target}" PROPERTY "${_prop}")
        set(_kept "")
        set(_dropped 0)
        set(_i 0)
        list(LENGTH _opts _n)
        while(_i LESS _n)
            list(GET _opts ${_i} _opt)
            set(_consume 1)
            set(_poison 0)
            if(_opt STREQUAL "-x" OR _opt MATCHES "(^|SHELL:)-x$")
                math(EXPR _j "${_i} + 1")
                if(_j LESS _n)
                    list(GET _opts ${_j} _next)
                    if(_next STREQUAL "hip" OR _next MATCHES "^hip>")
                        set(_poison 1)
                        set(_consume 2)
                    endif()
                endif()
            endif()
            if(NOT _poison AND _opt MATCHES "--offload-arch|--hip-link|-xhip|-x[ ;]hip|SHELL:-x;hip"
            )
                set(_poison 1)
            endif()
            if(_poison)
                math(EXPR _dropped "${_dropped} + ${_consume}")
            else()
                list(APPEND _kept "${_opt}")
            endif()
            math(EXPR _i "${_i} + ${_consume}")
        endwhile()
        if(_dropped GREATER 0)
            list(LENGTH _kept _kept_len)
            if(_kept_len EQUAL 0)
                set_property(TARGET "${target}" PROPERTY "${_prop}" "")
            else()
                set_property(TARGET "${target}" PROPERTY "${_prop}" ${_kept})
            endif()
            message(
                STATUS
                    "Vectorization: dropped ${_dropped} host-incompatible ${_prop} from ${target}"
            )
        endif()
    endforeach()
endfunction()

_vectorization_drop_hip_cxx_device_flags(hip::device)
_vectorization_drop_hip_cxx_device_flags(hip::host)

enable_language(HIP)

# ---------------------------------------------------------------------------
# vectorization_apply_hip_flags(<target>)
# Apply HIP language properties and per-target compile/link options.
# Mirrors linalg_apply_hip_flags() and PyTorch's torch_hip_compile_options.
# ---------------------------------------------------------------------------
function(vectorization_apply_hip_flags target)
  _vectorization_drop_hip_cxx_device_flags(hip::host)
  _vectorization_drop_hip_cxx_device_flags(hip::device)
  set_target_properties(${target} PROPERTIES
      HIP_STANDARD ${VECTORIZATION_CXX_STANDARD}
      HIP_STANDARD_REQUIRED ON
      HIP_ARCHITECTURES "${VECTORIZATION_GPU_ARCHITECTURES}")
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>)
  if(TARGET hip::host)
    target_link_libraries(${target} PUBLIC hip::host)
  endif()
  # CMake's HIP language supplies device compilation and linking; hip::device
  # would also pass Clang-only -x hip/--offload-arch flags to ordinary GCC TUs.
endfunction()
