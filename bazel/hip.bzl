"""HIP compilation through Bazel's selected ROCm Clang C++ toolchain."""

load("@bazel_skylib//rules:copy_file.bzl", "copy_file")
load("@rules_cc//cc:defs.bzl", "cc_library")

def hip_library(name, srcs, enabled, copts = [], **kwargs):
    """Compile device sources in HIP mode; leave CPU builds SDK-independent.

    Bazel's cc_library accepts C++ suffixes, so stage .hip files as .cc.
    ROCm Clang detects its SDK relative to the compiler installation. Select
    it via --repo_env=CC=<rocm>/llvm/bin/clang and --repo_env=CXX=<rocm>/llvm/bin/clang++.
    Headless builds can pass --copt=--offload-arch=gfx90a (or another GPU).

    Args:
        name: Name of the device library.
        srcs: Device translation units.
        enabled: Backend config_setting label.
        copts: Additional compiler options.
        **kwargs: Additional cc_library attributes.
    """
    staged = []
    for src in srcs:
        if src.endswith(".hip"):
            staged_src = name + "/" + src.rsplit("/", 1)[-1] + ".cc"
            copy_file(name = name + "_" + src.rsplit("/", 1)[-1], src = src, out = staged_src)
            staged.append(staged_src)
        else:
            staged.append(src)
    cc_library(
        name = name,
        srcs = select({enabled: staged, "//conditions:default": []}),
        copts = copts + ["-x", "hip"],
        target_compatible_with = select({enabled: [], "//conditions:default": ["@platforms//:incompatible"]}),
        **kwargs
    )
