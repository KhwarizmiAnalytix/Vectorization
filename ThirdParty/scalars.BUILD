load("@rules_cc//cc:defs.bzl", "cc_library")
# Scalars owns scalar types and the CPU/GPU SIMD implementations.
load("@bazel_skylib//rules:copy_file.bzl", "copy_file")
load("@xsigma_vectorization//bazel:scalars.bzl", "scalars_copts", "scalars_defines")

package(default_visibility = ["//visibility:public"])

genrule(
    name = "metal_kernels_source_gen",
    srcs = [
        "include/backend/gpu/metal/kernels.metal",
        "include/backend/gpu/metal/metal_kernels_source.h.in",
    ],
    outs = ["include/backend/gpu/metal/metal_kernels_source.h"],
    cmd = "python3 $(location @xsigma_vectorization//bazel:gen_metal_kernels_source.py) $(location include/backend/gpu/metal/kernels.metal) $(location include/backend/gpu/metal/metal_kernels_source.h.in) $@",
    tools = ["@xsigma_vectorization//bazel:gen_metal_kernels_source.py"],
)

copy_file(
    name = "metal_dispatch_objcxx",
    src = "include/backend/gpu/metal/metal_dispatch.mm",
    out = "include/backend/gpu/metal/metal_dispatch_mm.cc",
)

cc_library(
    name = "scalars_metal_objcxx",
    srcs = [":metal_dispatch_objcxx"],
    hdrs = glob(["include/**/*.h"]) + [":metal_kernels_source_gen"],
    copts = scalars_copts() + ["-x", "objective-c++", "-fobjc-arc"],
    defines = scalars_defines(),
    includes = ["include"],
    linkopts = ["-framework", "Metal", "-framework", "Foundation"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "Scalars",
    srcs = [
        "src/core/scalar_type.cpp",
        "src/math/complex_math.cpp",
        "src/math/normal_cdf.cpp",
    ],
    hdrs = glob(["include/**/*.h"], exclude = ["include/backend/cpu/mkl/**"]),
    copts = scalars_copts(),
    defines = scalars_defines() + select({
        "@xsigma_vectorization//bazel:shared_libs": ["SCALARS_SHARED_DEFINE"],
        "//conditions:default": ["SCALARS_STATIC_DEFINE"],
    }),
    local_defines = select({
        "@xsigma_vectorization//bazel:shared_libs": ["SCALARS_BUILDING_DLL"],
        "//conditions:default": [],
    }),
    includes = ["include"],
    deps = select({
        "@xsigma_vectorization//bazel:enable_metal": [":scalars_metal_objcxx"],
        "//conditions:default": [],
    }),
)
