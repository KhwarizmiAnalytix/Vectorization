include_guard()

if(UNIX)
  include(CheckFunctionExists)
endif()

include(CheckIncludeFile)
include(CheckCSourceCompiles)
include(CheckCSourceRuns)
include(CheckCCompilerFlag)
include(CheckCXXSourceCompiles)
include(CheckCXXCompilerFlag)
include(CMakePushCheckState)

# ---[ If running on Ubuntu, check system version and compiler version.
if(EXISTS "/etc/os-release")
  execute_process(
    COMMAND "sed" "-ne" "s/^ID=\\([a-z]\\+\\)$/\\1/p" "/etc/os-release"
    OUTPUT_VARIABLE OS_RELEASE_ID OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  execute_process(
    COMMAND "sed" "-ne" "s/^VERSION_ID=\"\\([0-9\\.]\\+\\)\"$/\\1/p" "/etc/os-release"
    OUTPUT_VARIABLE OS_RELEASE_VERSION_ID OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(OS_RELEASE_ID STREQUAL "ubuntu")
    if(OS_RELEASE_VERSION_ID VERSION_GREATER "17.04")
      if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "6.0.0")
          message(FATAL_ERROR "Please use GCC 6 or higher on Ubuntu 17.04 and higher.")
        endif()
      endif()
    endif()
  endif()
endif()

# ---[ Apply platform-specific optimization flags after compiler validation
if(COMMAND xsigma_apply_platform_flags)
  xsigma_apply_platform_flags()
endif()

# ---[ Check if std::exception_ptr is supported.
cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_FLAGS "-std=c++20")
check_cxx_source_compiles(
  "#include <string>
    #include <exception>
    int main(int argc, char** argv) {
      std::exception_ptr eptr;
      try {
          std::string().at(1);
      } catch(...) {
          eptr = std::current_exception();
      }
    }"
  TMP_EXCEPTION_PTR_SUPPORTED
)

if(TMP_EXCEPTION_PTR_SUPPORTED)
  message("--std::exception_ptr is supported.")
  set(HAS_EXCEPTION_PTR 1)
else()
  message("--std::exception_ptr is NOT supported.")
endif()
cmake_pop_check_state()

if(NOT TMP_NEED_TO_TURN_OFF_DEPRECATION_WARNING AND NOT MSVC)
  message("--Turning off deprecation warning.")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")
endif()

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _xsigma_vec_proc)
if(_xsigma_vec_proc MATCHES "^(x86_64|amd64|i686|i386)$")
  set(_xsigma_vec_is_x86 TRUE)
else()
  set(_xsigma_vec_is_x86 FALSE)
endif()
if(_xsigma_vec_proc MATCHES "^(aarch64|arm64)$")
  set(_xsigma_vec_is_aarch64 TRUE)
else()
  set(_xsigma_vec_is_aarch64 FALSE)
endif()

if(NOT INTERN_BUILD_MOBILE)

  set(VECTORIZATION OFF)
  set(HAS_NEON 0)
  set(HAS_SVE 0)

  if(_xsigma_vec_is_x86)
    # ---[ Check if the compiler has SSE support.
    cmake_push_check_state(RESET)
    if(MSVC)
      set(CMAKE_REQUIRED_FLAGS "/arch:SSE2")
    else()
      set(CMAKE_REQUIRED_FLAGS "-msse -msse2")
    endif()
    check_cxx_source_compiles(
      "#include <immintrin.h>
      int main() {
        __m128 a, b;
        a = _mm_set1_ps (1);
        _mm_add_ps(a, a);
        return 0;
      }"
      TMP_COMPILER_SUPPORTS_SSE_EXTENSIONS
    )
    if(TMP_COMPILER_SUPPORTS_SSE_EXTENSIONS)
      message("--Current compiler supports sse extension.")
      if(VECTORIZATION_CPU_BACKEND STREQUAL "sse")
        set(HAS_SSE 1)
        set(VECTORIZATION ON)
        set(VECTORIZATION_COMPILER_FLAGS "${CMAKE_REQUIRED_FLAGS}")
      endif()
    endif()
    cmake_pop_check_state()

    # ---[ Check if the compiler has AVX support.
    cmake_push_check_state(RESET)
    if(MSVC)
      set(CMAKE_REQUIRED_FLAGS "/arch:AVX /D__F16C__")
    else()
      set(CMAKE_REQUIRED_FLAGS "-mavx -mf16c")
    endif()
    check_cxx_source_compiles(
      "#include <immintrin.h>
      int main() {
        __m256 a, b;
        a = _mm256_set1_ps (1);
        b = a;
        _mm256_add_ps (a,a);
        return 0;
      }"
      TMP_COMPILER_SUPPORTS_AVX_EXTENSIONS
    )
    if(TMP_COMPILER_SUPPORTS_AVX_EXTENSIONS)
      message("--Current compiler supports avx extension.")
      if(VECTORIZATION_CPU_BACKEND STREQUAL "avx")
        set(HAS_AVX 1)
        set(VECTORIZATION ON)
        set(VECTORIZATION_COMPILER_FLAGS "${CMAKE_REQUIRED_FLAGS}")
      endif()
    endif()
    cmake_pop_check_state()

    # ---[ Check if the compiler has AVX2 support.
    cmake_push_check_state(RESET)
    if(MSVC)
      set(CMAKE_REQUIRED_FLAGS "/arch:AVX2 /D__F16C__")
    else()
      set(CMAKE_REQUIRED_FLAGS "-mavx2 -mf16c")
    endif()
    check_cxx_source_compiles(
      "#include <immintrin.h>
      int main() {
        __m256i a, b;
        a = _mm256_set1_epi8 (1);
        b = a;
        _mm256_add_epi8 (a,a);
        __m256i x;
        _mm256_extract_epi64(x, 0); // we rely on this in our AVX2 code
        return 0;
      }"
      TMP_COMPILER_SUPPORTS_AVX2_EXTENSIONS
    )
    if(TMP_COMPILER_SUPPORTS_AVX2_EXTENSIONS)
      message("--Current compiler supports avx2 extension.")
      if(VECTORIZATION_CPU_BACKEND STREQUAL "avx2")
        set(HAS_AVX2 1)
        set(VECTORIZATION ON)
        set(VECTORIZATION_COMPILER_FLAGS "${CMAKE_REQUIRED_FLAGS}")
      endif()
    endif()
    cmake_pop_check_state()

    # ---[ Check if the compiler has AVX512 support.
    # Compile-only: do not change this to check_cxx_source_runs(). Hosted CI
    # (ubuntu-latest) can emit AVX-512 but typically has no avx512f silicon; a
    # run-time compile probe would disable the backend. Test *execution* is
    # gated separately by Scripts/helpers/cpu_isa.py.
    cmake_push_check_state(RESET)
    if(MSVC)
      # /arch:AVX512 implies __AVX512F__, __AVX512CD__, __AVX512BW__, __AVX512DQ__, __AVX512VL__
      set(CMAKE_REQUIRED_FLAGS "/arch:AVX512 /D__F16C__")
    else()
      set(CMAKE_REQUIRED_FLAGS "-mavx512f -mavx512dq -mavx512vl -mavx512bw -mavx512cd -mf16c")
    endif()
    check_cxx_source_compiles(
      "#if defined(_MSC_VER)
     #include <intrin.h>
     #else
     #include <immintrin.h>
     #endif
     // check avx512f
     __m512 addConstant(__m512 arg) {
       return _mm512_add_ps(arg, _mm512_set1_ps(1.f));
     }
     // check avx512dq
     __m512 andConstant(__m512 arg) {
       return _mm512_and_ps(arg, _mm512_set1_ps(1.f));
     }
     int main() {
       __m512i a = _mm512_set1_epi32(1);
       __m256i ymm = _mm512_extracti64x4_epi64(a, 0);
       ymm = _mm256_abs_epi64(ymm); // check avx512vl
       __mmask16 m = _mm512_cmp_epi32_mask(a, a, _MM_CMPINT_EQ);
       __m512i r = _mm512_andnot_si512(a, a);
     }"
      TMP_COMPILER_SUPPORTS_AVX512_EXTENSIONS
    )
    if(TMP_COMPILER_SUPPORTS_AVX512_EXTENSIONS)
      message("--Current compiler supports avx512f extension.")
      if(VECTORIZATION_CPU_BACKEND STREQUAL "avx512")
        set(HAS_AVX512 1)
        set(VECTORIZATION ON)
        set(VECTORIZATION_COMPILER_FLAGS "${CMAKE_REQUIRED_FLAGS}")
      endif()
    endif()
    cmake_pop_check_state()

    # ---[ Check if the compiler has FMA support.
    cmake_push_check_state(RESET)
    if(MSVC)
      set(CMAKE_REQUIRED_FLAGS "${VECTORIZATION_COMPILER_FLAGS} /D__FMA__")
    else()
      set(CMAKE_REQUIRED_FLAGS "${VECTORIZATION_COMPILER_FLAGS} -mfma")
    endif()
    check_cxx_source_compiles(
      "#if defined(_MSC_VER)
     #include <intrin.h>
     #else
     #include <immintrin.h>
     #endif

      int main() {
        __m128 a, b;
        a = _mm_set1_ps (1);
        b = _mm_set1_ps (1);
        a = _mm_fmadd_ps(a,b,b);
        return 0;
      }"
      TMP_COMPILER_SUPPORTS_FMA_EXTENSIONS
    )
    if(TMP_COMPILER_SUPPORTS_FMA_EXTENSIONS)
      message("--Current compiler supports fma extension.")
      # Only apply -mfma when an x86 CPU SIMD backend is actually selected. Unlike the
      # SSE/AVX/AVX2/AVX512 probes above (each gated on its own VECTORIZATION_CPU_BACKEND
      # match), this check used to run unconditionally and would set VECTORIZATION_COMPILER_FLAGS
      # even for VECTORIZATION_CPU_BACKEND=no, leaking -mfma into every consumer (including
      # CUDA/HIP device-compiler host passes, which don't use CPU SIMD at all).
      if(NOT VECTORIZATION_CPU_BACKEND STREQUAL "no")
        set(VECTORIZATION_COMPILER_FLAGS "${CMAKE_REQUIRED_FLAGS}")
      endif()
    endif()
    cmake_pop_check_state()

  endif() # _xsigma_vec_is_x86

  if(_xsigma_vec_is_x86)
    # ---[ Check if the compiler has SVML support.
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_FLAGS "${VECTORIZATION_COMPILER_FLAGS}")
    check_cxx_source_compiles(
      "#if defined(_MSC_VER)
     #include <intrin.h>
     #else
     #include <immintrin.h>
     #endif

      int main() {
        __m256 a, b;
        a = _mm256_setzero_ps();
        b = _mm256_exp_ps(a);
        b = _mm256_cos_ps(a);
        b = _mm256_tanh_ps(a);
        return 0;
      }"
      TMP_COMPILER_SUPPORTS_SVML_EXTENSIONS
    )

    if(NOT TMP_COMPILER_SUPPORTS_SVML_EXTENSIONS AND VECTORIZATION AND NOT
                                                                       VECTORIZATION_ENABLE_SLEEF
    )
      message(
        WARNING
          "--Current compiler does not supports SVML functoins. Turn ON VECTORIZATION_ENABLE_SVML"
      )
      set(VECTORIZATION_ENABLE_SVML ON CACHE BOOL "Enable Intel SVML short vector math library"
                                             FORCE
      )
    endif()
    cmake_pop_check_state()

    # Locate ThirdParty SVML binaries when the flag is ON.
    if(VECTORIZATION_ENABLE_SVML)
      find_package(SVML QUIET)
      if(SVML_FOUND)
        message("--SVML: ThirdParty package found.")
      else()
        message(
          WARNING "--VECTORIZATION_ENABLE_SVML=ON but ThirdParty SVML binaries were not found. "
                  "Ensure ThirdParty/svml/ contains the required libraries."
        )
      endif()
    endif()

  endif() # _xsigma_vec_is_x86 (SVML)

  if(_xsigma_vec_is_aarch64)
    # ---[ AArch64 NEON — probe armv8.2-a first (Cortex-A75+, Neoverse N1, Apple A12+),
    # fall back to baseline armv8-a so the binary runs on older targets.
    cmake_push_check_state(RESET)
    if(NOT MSVC)
      check_cxx_compiler_flag("-march=armv8.2-a" _xsigma_vec_neon_armv82)
      if(_xsigma_vec_neon_armv82)
        set(CMAKE_REQUIRED_FLAGS "-march=armv8.2-a")
      else()
        set(CMAKE_REQUIRED_FLAGS "-march=armv8-a")
      endif()
    endif()
    check_cxx_source_compiles(
      "#include <arm_neon.h>
       int main() {
         float32x4_t a = vdupq_n_f32(1.f);
         float32x4_t b = vfmaq_f32(a, a, a);
         (void)b;
         return 0;
       }"
      TMP_COMPILER_SUPPORTS_ARM_NEON
    )
    if(TMP_COMPILER_SUPPORTS_ARM_NEON AND VECTORIZATION_CPU_BACKEND STREQUAL "neon")
      message("--Current compiler supports AArch64 NEON.")
      set(HAS_NEON 1)
      set(VECTORIZATION ON)
      if(NOT MSVC)
        set(VECTORIZATION_COMPILER_FLAGS "${CMAKE_REQUIRED_FLAGS}")
      endif()
    else()
      set(VECTORIZATION_ENABLE_SLEEF OFF
          CACHE BOOL "Enable SLEEF (auto-enabled: Accelerate unavailable on non-Apple)" FORCE
      )

    endif()
    cmake_pop_check_state()

    # ---[ AArch64 SVE — vector width is controlled by VECTORIZATION_SVE_VECTOR_BITS.
    # Default is 128 (matches backend/sve/*/simd.h); set to 256/512 once backends support it.
    cmake_push_check_state(RESET)
    if(NOT MSVC)
      set(CMAKE_REQUIRED_FLAGS
          "-march=armv8-a+sve -msve-vector-bits=${VECTORIZATION_SVE_VECTOR_BITS}"
      )
    endif()
    check_cxx_source_compiles(
      "#include <arm_sve.h>
       int main() {
         svbool_t pg = svptrue_b32();
         (void)pg;
         return 0;
       }"
      TMP_COMPILER_SUPPORTS_ARM_SVE128
    )
    if(TMP_COMPILER_SUPPORTS_ARM_SVE128 AND VECTORIZATION_CPU_BACKEND STREQUAL "sve")
      message("--Current compiler supports AArch64 SVE "
              "(${VECTORIZATION_SVE_VECTOR_BITS}-bit vector length)."
      )
      set(HAS_SVE 1)
      set(VECTORIZATION ON)
      if(NOT MSVC)
        set(VECTORIZATION_COMPILER_FLAGS "${CMAKE_REQUIRED_FLAGS}")
      endif()
    endif()
    cmake_pop_check_state()

    # ---[ Probe Apple Accelerate vForce (vvexpf/vvsinf/vvcosf/vvlogf).
    # These symbols live in <Accelerate/Accelerate.h> and require -framework Accelerate.
    # If they are absent (Linux/Android AArch64, old SDK, non-Apple toolchain) we
    # automatically switch on SLEEF so the caller never has to set the flag manually.
    if(APPLE AND HAS_NEON AND NOT VECTORIZATION_ENABLE_SLEEF)
      cmake_push_check_state(RESET)
      set(CMAKE_REQUIRED_LIBRARIES "-framework Accelerate")
      check_cxx_source_compiles(
        "#include <Accelerate/Accelerate.h>
         int main() {
           const int n = 4;
           float x[4] = {1.f, 2.f, 3.f, 4.f};
           float y[4];
           vvexpf(y, x, &n);
           vvsinf(y, x, &n);
           vvcosf(y, x, &n);
           vvlogf(y, x, &n);
           return 0;
         }"
        TMP_ACCELERATE_VFORCE_SUPPORTED
      )
      cmake_pop_check_state()
      if(TMP_ACCELERATE_VFORCE_SUPPORTED)
        message(STATUS "Accelerate vForce (vvexpf/vvsinf/vvcosf/vvlogf): supported -- "
                       "auto-enabling VECTORIZATION_ENABLE_ACCELERATE."
        )
        set(VECTORIZATION_ENABLE_ACCELERATE ON
            CACHE BOOL "Enable Apple Accelerate vForce (auto-enabled)" FORCE
        )
      else()
        message(STATUS "Accelerate vForce (vvexpf/vvsinf/vvcosf/vvlogf): NOT supported -- "
                       "auto-enabling VECTORIZATION_ENABLE_SLEEF."
        )
        set(VECTORIZATION_ENABLE_SLEEF ON
            CACHE BOOL "Enable SLEEF (auto-enabled: Accelerate vForce unavailable)" FORCE
        )
      endif()
    elseif(NOT APPLE AND HAS_NEON AND NOT VECTORIZATION_ENABLE_SLEEF)
      # Non-Apple AArch64 never has Accelerate; enable SLEEF automatically.
      message(STATUS "Non-Apple AArch64 detected: Accelerate vForce unavailable -- "
                     "auto-enabling VECTORIZATION_ENABLE_SLEEF."
      )
      set(VECTORIZATION_ENABLE_SLEEF ON
          CACHE BOOL "Enable SLEEF (auto-enabled: Accelerate unavailable on non-Apple)" FORCE
      )
    endif()

  endif()

endif()

if(USE_NATIVE_ARCH)
  check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
  if(COMPILER_SUPPORTS_MARCH_NATIVE)
    string(APPEND CMAKE_C_FLAGS " -march=native")
    string(APPEND CMAKE_CXX_FLAGS " -march=native")
  else()
    message(WARNING "Your compiler does not support -march=native. Turn off this warning "
                    "by setting -DUSE_NATIVE_ARCH=OFF."
    )
  endif()
endif()
