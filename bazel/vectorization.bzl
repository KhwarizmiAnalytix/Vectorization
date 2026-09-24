load("//bazel:xsigma.bzl", "xsigma_copts", "xsigma_defines", "xsigma_linkopts")

# C++ standard for Vectorization — mirrors CMake VECTORIZATION_CXX_STANDARD (default: 20)
VECTORIZATION_CXX_STD = "c++20"

def vectorization_copts():
    return xsigma_copts(cxx_std = VECTORIZATION_CXX_STD)

def vectorization_defines():
    """Returns compile definitions for the Vectorization library.

    Mirrors this repo's include/CMakeLists.txt VECTORIZATION_HAS_* flags.
    Values not wired here (VECTORIZATION_HAS_SVML/SLEEF/ACCELERATE/MKL/
    LIBTORCH) are CMake-only in this repo's Bazel build (see
    Scripts/setup_bazel.py's --help) and are always compiled out (=0).
    """
    defines = xsigma_defines()

    # --- CPU backend -> VECTORIZATION_HAS_* (one-hot, driven by cpu_* config_settings) ---

    # SSE
    defines += select({
        "//bazel:cpu_sse": ["VECTORIZATION_HAS_SSE=1"],
        "//conditions:default": ["VECTORIZATION_HAS_SSE=0"],
    })

    # AVX2 implies AVX; set once to avoid a duplicate -DVECTORIZATION_HAS_AVX define.
    defines += select({
        "//bazel:cpu_avx": ["VECTORIZATION_HAS_AVX=1"],
        "//bazel:cpu_avx2": ["VECTORIZATION_HAS_AVX=1"],
        "//conditions:default": ["VECTORIZATION_HAS_AVX=0"],
    })

    defines += select({
        "//bazel:cpu_avx2": ["VECTORIZATION_HAS_AVX2=1"],
        "//conditions:default": ["VECTORIZATION_HAS_AVX2=0"],
    })

    defines += select({
        "//bazel:cpu_avx512": ["VECTORIZATION_HAS_AVX512=1"],
        "//conditions:default": ["VECTORIZATION_HAS_AVX512=0"],
    })

    defines += select({
        "//bazel:cpu_neon": ["VECTORIZATION_HAS_NEON=1"],
        "//conditions:default": ["VECTORIZATION_HAS_NEON=0"],
    })

    defines += select({
        "//bazel:cpu_sve": ["VECTORIZATION_HAS_SVE=1"],
        "//conditions:default": ["VECTORIZATION_HAS_SVE=0"],
    })

    # VECTORIZATION_VECTORIZED: 1 when any CPU SIMD backend is active.
    defines += select({
        "//bazel:cpu_sse": ["VECTORIZATION_VECTORIZED=1"],
        "//bazel:cpu_avx": ["VECTORIZATION_VECTORIZED=1"],
        "//bazel:cpu_avx2": ["VECTORIZATION_VECTORIZED=1"],
        "//bazel:cpu_avx512": ["VECTORIZATION_VECTORIZED=1"],
        "//bazel:cpu_neon": ["VECTORIZATION_VECTORIZED=1"],
        "//bazel:cpu_sve": ["VECTORIZATION_VECTORIZED=1"],
        "//conditions:default": ["VECTORIZATION_VECTORIZED=0"],
    })

    # --- GPU backend -> VECTORIZATION_HAS_* (driven by enable_{cuda,hip,metal}) ---
    defines += select({
        "//bazel:enable_cuda": ["VECTORIZATION_HAS_CUDA=1"],
        "//conditions:default": ["VECTORIZATION_HAS_CUDA=0"],
    })
    defines += select({
        "//bazel:enable_hip": ["VECTORIZATION_HAS_HIP=1"],
        "//conditions:default": ["VECTORIZATION_HAS_HIP=0"],
    })
    defines += select({
        "//bazel:enable_metal": ["VECTORIZATION_HAS_METAL=1"],
        "//conditions:default": ["VECTORIZATION_HAS_METAL=0"],
    })

    # CMake-only backends: never wired to a --define in this repo's Bazel build.
    defines += [
        "VECTORIZATION_HAS_SVML=0",
        "VECTORIZATION_HAS_SLEEF=0",
        "VECTORIZATION_HAS_ACCELERATE=0",
        "VECTORIZATION_HAS_MKL=0",
        "VECTORIZATION_HAS_LIBTORCH=0",
    ]

    # Profiler presence (mirrors _vec_has_profiler in include/CMakeLists.txt).
    defines += select({
        "//bazel:disable_profiler": ["VECTORIZATION_HAS_PROFILER=0"],
        "//conditions:default": ["VECTORIZATION_HAS_PROFILER=1"],
    })

    # Test builds define VECTORIZATION_HAS_GTEST=1 directly (see Testing/Cxx/BUILD.bazel);
    # the library itself does not depend on GoogleTest.
    defines += ["VECTORIZATION_HAS_GTEST=0", "VECTORIZATION_PACKET_SIZE=1"]

    return defines

def vectorization_linkopts():
    return xsigma_linkopts()

def vectorization_arch_copts():
    """Returns the -m.../-march=... ISA compiler flags for the selected CPU
    backend. Mirrors include/Cmake/utils.cmake's CMAKE_REQUIRED_FLAGS probes
    (VECTORIZATION_COMPILER_FLAGS), which include/CMakeLists.txt applies as
    PUBLIC target_compile_options so every consumer inherits the correct ISA.
    Bazel copts are not transitive the same way — targets that include
    Vectorization headers using simd<T> directly (rather than only calling
    into pre-built .cpp/.o) must apply the same select() themselves, the way
    Testing/Cxx/BUILD.bazel does.
    """
    # A single flat select(): Bazel resolves "cpu_sse_windows" over "cpu_sse" on
    # Windows because the former is a strict specialization of the latter
    # (same define_values, plus a constraint_values match) -- see
    # bazel/BUILD.bazel. Neon/SVE have no Windows target here, so they need no
    # such override.
    return select({
        "//bazel:cpu_sse": ["-msse", "-msse2"],
        "//bazel:cpu_avx": ["-mavx", "-mf16c", "-mfma"],
        "//bazel:cpu_avx2": ["-mavx2", "-mf16c", "-mfma"],
        "//bazel:cpu_avx512": [
            "-mavx512f",
            "-mavx512dq",
            "-mavx512vl",
            "-mavx512bw",
            "-mavx512cd",
            "-mf16c",
            "-mfma",
        ],
        "//bazel:cpu_neon": ["-march=armv8-a"],
        "//bazel:cpu_sve": ["-march=armv8-a+sve", "-msve-vector-bits=128"],
        "//bazel:cpu_sse_windows": ["/arch:SSE2"],
        "//bazel:cpu_avx_windows": ["/arch:AVX", "/D__F16C__"],
        "//bazel:cpu_avx2_windows": ["/arch:AVX2", "/D__F16C__"],
        "//bazel:cpu_avx512_windows": ["/arch:AVX512", "/D__F16C__"],
        "//conditions:default": [],
    })
