# Overlay BUILD for the KhwarizmiAnalytix/Logging submodule (ThirdParty/Logging).
# Do not edit files inside that submodule — this overlay lives in this repo, mirroring
# the same technique XSigma itself uses for its own ThirdParty/logging.BUILD.
# This repo depends on @logging//:Logging only. Tracks the submodule's own
# BUILD.bazel: include root is the repo root, public headers live under logging/
# (e.g. "logging/logging.h", "logging/logger/logger.h") — the submodule was
# restructured from a flat include/ layout to this logging/ layout.
load("//bazel:logging.bzl", "logging_copts", "logging_defines", "logging_linkopts")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "logging_hdrs",
    srcs = glob(
        [
            "logging/*.h",
            "logging/common/*.h",
            "logging/util/*.h",
            "logging/logger/*.h",
        ],
        allow_empty = True,
    ),
)

filegroup(
    name = "logging_srcs",
    srcs = glob(
        [
            "logging/util/*.cpp",
        ],
        allow_empty = True,
    ),
)

cc_library(
    name = "logging_lib",
    srcs = [
        ":logging_srcs",
        "logging/logger/back_trace.cpp",
        "logging/logger/logger.cpp",
    ],
    hdrs = [":logging_hdrs"],
    copts = logging_copts(),
    defines = logging_defines() + select({
        "//bazel:shared_libs": ["LOGGING_SHARED_DEFINE"],
        "//conditions:default": ["LOGGING_STATIC_DEFINE"],
    }),
    local_defines = select({
        "//bazel:shared_libs": ["LOGGING_BUILDING_DLL"],
        "//conditions:default": [],
    }),
    # Logging's own headers include each other as "logging/util/..." from the
    # repo root, and Memory includes them as <logging/util/...>, <logging/logger/logger.h>
    # (angle-bracket form, which needs a -isystem/-I path, not just the -iquote path
    # that "includes" would add without "."). "logging/logger" is also needed:
    # sources include bare "logger_verbosity_enum.h".
    includes = [
        ".",
        "logging/logger",
    ],
    linkopts = logging_linkopts() + select({
        "@platforms//os:windows": ["dbghelp.lib"],
        "//conditions:default": [],
    }),
    linkstatic = select({
        "//bazel:shared_libs": False,
        "//conditions:default": True,
    }),
    deps = [
        "@fmt//:fmt",
    ] + select({
        "//bazel:disable_magic_enum": [],
        "//conditions:default": ["@magic_enum//:magic_enum"],
    }) + select({
        "//bazel:logging_glog": ["@glog//:glog"],
        "//bazel:logging_loguru": ["@loguru//:loguru"],
        "//bazel:logging_native": [],
        "//bazel:logging_spdlog": ["@spdlog//:spdlog"],
        "//conditions:default": ["@loguru//:loguru"],
    }),
    alwayslink = False,
)

cc_library(
    name = "Logging",
    deps = [":logging_lib"],
    visibility = ["//visibility:public"],
)
