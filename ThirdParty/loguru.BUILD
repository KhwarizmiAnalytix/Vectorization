# BUILD file for Loguru logging library
# Copied from ThirdParty/Logging/ThirdParty/loguru.BUILD: provides build rules for the
# third-party Loguru library without modifying the submodule itself.

cc_library(
    name = "loguru",
    srcs = ["loguru.cpp"],
    hdrs = ["loguru.hpp"],
    copts = select({
        "@platforms//os:windows": [
            "/W0",  # Disable warnings
            "/std:c++17",
        ],
        "@platforms//os:macos": [
            "-w",  # Disable warnings
            "-std=c++17",
            "-fPIC",
        ],
        "//conditions:default": [
            "-w",  # Disable warnings
            "-std=c++17",
            "-fPIC",
        ],
    }),
    defines = [
        "LOGURU_WITH_STREAMS=1",
    ],
    includes = ["."],
    linkopts = select({
        "@platforms//os:windows": [],
        "@platforms//os:macos": ["-lpthread"],
        "//conditions:default": ["-lpthread", "-ldl"],
    }),
    linkstatic = True,
    visibility = ["//visibility:public"],
)

