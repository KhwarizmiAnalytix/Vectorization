"""Expose an installed ROCm SDK without modifying vendor sources.

Only fetched when a HIP target is selected. Device compilation uses the
ROCm Clang C++ toolchain selected by --repo_env=CC / --repo_env=CXX.
"""

def _rocm_repository_impl(ctx):
    if ctx.os.name != "linux":
        fail("HIP builds require Linux and a ROCm Clang toolchain")
    prefix = ctx.os.environ.get("ROCM_PATH") or ctx.os.environ.get("HIP_PATH") or "/opt/rocm"
    sdk = ctx.path(prefix)
    if not sdk.get_child("include/hip/hip_runtime.h").exists:
        fail("ROCm SDK not found at %s; set --repo_env=ROCM_PATH=/path/to/rocm" % prefix)
    ctx.symlink(sdk.get_child("include"), "include")
    lib = sdk.get_child("lib")
    if not lib.get_child("libamdhip64.so").exists:
        lib = sdk.get_child("lib64")
    if not lib.get_child("libamdhip64.so").exists:
        fail("ROCm runtime libamdhip64.so missing under %s/lib or lib64" % prefix)
    ctx.symlink(lib, "lib")
    ctx.file("BUILD.bazel", '''
load("@rules_cc//cc:defs.bzl", "cc_library")
package(default_visibility = ["//visibility:public"])
cc_library(
    name = "hip",
    hdrs = glob(["include/**"]),
    srcs = glob(["lib/libamdhip64.so*"]),
    includes = ["include"],
    defines = ["__HIP_PLATFORM_AMD__=1"],
)
cc_library(
    name = "hipblas",
    srcs = ["lib/libhipblas.so"] + glob(["lib/libhipblas.so.*"], allow_empty = True),
    deps = [":hip"],
)
cc_library(
    name = "hipsolver",
    srcs = ["lib/libhipsolver.so"] + glob(["lib/libhipsolver.so.*"], allow_empty = True),
    deps = [":hip"],
)
''')

rocm_repository = repository_rule(
    implementation = _rocm_repository_impl,
    environ = ["ROCM_PATH", "HIP_PATH"],
    local = True,
)
