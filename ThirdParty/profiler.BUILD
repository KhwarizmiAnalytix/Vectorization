# =============================================================================
# Overlay BUILD for the KhwarizmiAnalytix/Profiler submodule (ThirdParty/Profiler).
# Do not edit files inside that submodule — this overlay lives in XSigma.
# XSigma depends on Profiler only (public <profiler.h> / profiler::session).
# =============================================================================
load("@bazel_skylib//rules:copy_file.bzl", "copy_file")
load("//bazel:profiler.bzl", "profiler_copts", "profiler_defines", "profiler_linkopts")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "profiler_hdrs",
    srcs = glob(
        [
            "Profiler/*.h",
            "Profiler/**/*.h",
        ],
        exclude = [
            "Testing/**",
            "Profiler/native/**",
            "Profiler/bespoke/**",
        ],
        allow_empty = True,
    ),
)

filegroup(
    name = "profiler_srcs",
    srcs = glob(
        [
            "Profiler/*.cpp",
            "Profiler/**/*.cpp",
        ],
        exclude = [
            "Profiler/native/**/*.cpp",
            "Profiler/bespoke/**",
            "Testing/**",
            "examples/**",
            "consumer/**",
            "third_party/**",
        ],
        allow_empty = True,
    ),
)

# cc_library srcs don't accept .mm without rules_apple. Copy to a .cc suffix and compile
# as Objective-C++ in a dedicated target so ARC copts don't leak onto plain C++ TUs
# (same split as //Library/Memory:memory_metal_objcxx).
copy_file(
    name = "metal_gpu_probe_objcxx",
    src = "Profiler/native/gpu/metal_gpu_probe.mm",
    out = "Profiler/native/gpu/metal_gpu_probe_mm.cc",
)

cc_library(
    name = "profiler_metal_objcxx",
    srcs = select({
        "@//bazel:enable_metal": [":metal_gpu_probe_objcxx"],
        "//conditions:default": [],
    }),
    hdrs = [
        ":profiler_hdrs",
    ] + glob(["Profiler/native/**/*.h", "Profiler/native/**/*.hxx"], exclude = ["Testing/**"], allow_empty = True),
    copts = profiler_copts() + select({
        "@//bazel:enable_metal": [
            "-x",
            "objective-c++",
            "-fobjc-arc",
            "-Wno-deprecated-declarations",
        ],
        "//conditions:default": [],
    }),
    defines = profiler_defines() + select({
        "//bazel:shared_libs": ["PROFILER_SHARED_DEFINE", "PROFILER_BUILDING_DLL"],
        "//conditions:default": ["PROFILER_STATIC_DEFINE"],
    }),
    includes = ["Profiler"],
    linkopts = profiler_linkopts(),
    visibility = ["//visibility:public"],
    alwayslink = True,
)

cc_library(
    name = "Profiler",
    srcs = [
        ":profiler_srcs",
    ] + glob(["Profiler/native/**/*.cpp"], exclude = ["Testing/**"], allow_empty = True),
    hdrs = [
        ":profiler_hdrs",
    ] + glob(["Profiler/native/**/*.h", "Profiler/native/**/*.hxx"], exclude = ["Testing/**"], allow_empty = True),
    copts = profiler_copts(),
    defines = profiler_defines() + select({
        "//bazel:shared_libs": ["PROFILER_SHARED_DEFINE", "PROFILER_BUILDING_DLL"],
        "//conditions:default": ["PROFILER_STATIC_DEFINE"],
    }),
    includes = [
        "Profiler",
    ],
    linkopts = profiler_linkopts(),
    linkstatic = select({
        "//bazel:shared_libs": False,
        "//conditions:default": True,
    }),
    deps = [
        # No //Library/Core dependency: nothing under native/ actually includes a
        # Core header; CMake never linked Core either. A prior Core dep here would
        # create a Memory -> Profiler -> Core -> Memory cycle.
        "@fmt//:fmt",
    ] + select({
        "//bazel:enable_cuda": ["@local_config_cuda//:cuda"],
        "//conditions:default": [],
    }) + select({
        "//bazel:enable_hip": ["@local_config_hip//:hip"],
        "//conditions:default": [],
    }) + select({
        "@//bazel:enable_metal": [":profiler_metal_objcxx"],
        "//conditions:default": [],
    }),
    # native/core/profiler_factory.h's registered profilers (host_tracer_factory,
    # python_tracer_factory, gpu_tracer_factory) self-register via static-init --
    # no other code references a symbol in those translation units. Bazel links this
    # target as a static archive (see linkstatic above); without alwayslink, the linker
    # drops those objects and host/GPU tracing never starts. CMake's shared-library
    # build doesn't hit this since every object file in a .so is linked.
    alwayslink = True,
)

# Profiler is consumed as a pure third-party dependency: its test suite
# (Testing/Cxx, ProfilerCxxTests) builds only from the standalone Profiler
# repo, never from XSigma — same as the CMake side (xsigma_add_profiler
# forces PROFILER_ENABLE_TESTING/EXAMPLES OFF).
