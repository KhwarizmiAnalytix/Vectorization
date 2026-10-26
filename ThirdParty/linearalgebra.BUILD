# Standalone LinearAlgebra library staged into @linear_algebra by
# bazel/linear_algebra_repository.bzl. Include root is the repository root;
# public headers live under include/, implementation files under src/.
#
# Deps are remapped from LinearAlgebra's own workspace names to
# Vectorization's: @xsigma_logging → @logging, @scalars//:scalars → @scalars//:Scalars.
load("@rules_cc//cc:defs.bzl", "cc_library")
load("//bazel:linear_algebra.bzl", "linear_algebra_copts", "linear_algebra_defines", "linear_algebra_linkopts")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "linear_algebra_hdrs",
    srcs = glob(
        [
            "include/common/*.h",
            "include/memory/*.h",
            "include/util/*.h",
            "include/matrix_operation/*.h",
            "src/common/*.h",
            "src/matrix_operation/*.h",
            "src/matrix_operation/cpu/blas/*.h",
        ],
        allow_empty = True,
    ),
)

filegroup(
    name = "linear_algebra_srcs",
    srcs = glob(
        [
            "src/common/*.cpp",
            "src/matrix_operation/cpu/*.cpp",
            "src/matrix_operation/cpu/native/*.cpp",
        ],
        allow_empty = True,
    ) + select({
        "//bazel:linalg_enable_mkl": glob(["src/matrix_operation/cpu/mkl/*.cpp"], allow_empty = True),
        "//conditions:default": [],
    }) + select({
        "//bazel:linalg_enable_blas": glob(["src/matrix_operation/cpu/blas/*.cpp"], allow_empty = True),
        "//conditions:default": [],
    }),
)

cc_library(
    name = "linear_algebra_lib",
    srcs = [":linear_algebra_srcs"],
    hdrs = [":linear_algebra_hdrs"],
    copts = linear_algebra_copts(),
    defines = linear_algebra_defines() + select({
        "//bazel:shared_libs": ["LINALG_SHARED_DEFINE"],
        "//conditions:default": ["LINALG_STATIC_DEFINE"],
    }),
    local_defines = select({
        "//bazel:shared_libs": ["LINALG_BUILDING_DLL"],
        "//conditions:default": [],
    }),
    linkopts = linear_algebra_linkopts(),
    linkstatic = select({
        "//bazel:shared_libs": False,
        "//conditions:default": True,
    }),
    deps = ["@logging//:Logging", "@scalars//:Scalars"],
    alwayslink = True,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "LinearAlgebra",
    deps = [":linear_algebra_lib"],
    visibility = ["//visibility:public"],
)
