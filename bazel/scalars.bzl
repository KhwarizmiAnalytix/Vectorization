"""Scalars backend settings follow Vectorization's CPU/GPU selection."""

load("//bazel:vectorization.bzl", "vectorization_arch_copts", "vectorization_copts")

def scalars_copts():
    return vectorization_copts() + vectorization_arch_copts()

def scalars_defines():
    defines = [
        "SCALARS_HAS_SVML=0",
        "SCALARS_COMPILER_SUPPORTS_SVML_EXTENSIONS=0",
        "SCALARS_HAS_SLEEF=0",
        "SCALARS_HAS_MKL=0",
        "SCALARS_HAS_ACCELERATE=0",
        "SCALARS_PACKET_SIZE=1",
    ]
    for backend in ["sse", "avx", "avx2", "avx512", "neon", "sve"]:
        values = {
            Label("//bazel:cpu_" + backend): ["SCALARS_HAS_" + backend.upper() + "=1"],
            "//conditions:default": ["SCALARS_HAS_" + backend.upper() + "=0"],
        }
        if backend == "avx":
            values[Label("//bazel:cpu_avx2")] = ["SCALARS_HAS_AVX=1"]
        defines += select(values)
    defines += select({
        Label("//bazel:cpu_" + backend): ["SCALARS_VECTORIZED=1"]
        for backend in ["sse", "avx", "avx2", "avx512", "neon", "sve"]
    } | {"//conditions:default": ["SCALARS_VECTORIZED=0"]})
    for backend in ["avx2", "avx512"]:
        defines += select({
            Label("//bazel:cpu_" + backend): ["SCALARS_CPU_CAPABILITY_" + backend.upper()],
            "//conditions:default": [],
        })
    for backend in ["cuda", "hip", "metal"]:
        defines += select({
            Label("//bazel:enable_" + backend): ["SCALARS_HAS_" + backend.upper() + "=1"],
            "//conditions:default": ["SCALARS_HAS_" + backend.upper() + "=0"],
        })
    return defines
