# =============================================================================
# mimalloc Library BUILD Configuration
# =============================================================================
# Microsoft's high-performance memory allocator
# Equivalent to ThirdParty/mimalloc
# =============================================================================

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mimalloc",
    srcs = [
        "src/static.c",
    ],
    hdrs = glob([
        "include/**/*.h",
    ]),
    textual_hdrs = glob([
        "src/**/*.c",
        "src/**/*.h",
    ]),
    copts = select({
        "@platforms//os:windows": [
            "/W0",  # Disable warnings
        ],
        "//conditions:default": [
            "-w",  # Disable warnings
        ],
    }),
    # Mirrors CMake MI_OVERRIDE=OFF: do not define MI_MALLOC_OVERRIDE at all.
    defines = [],
    includes = [
        "include",
        "src",  # Include src directory for internal includes
    ],
    linkopts = select({
        # Mirrors mimalloc's Windows system library dependencies (see CMake logic).
        "@platforms//os:windows": [
            "psapi.lib",
            "shell32.lib",
            "user32.lib",
            "advapi32.lib",
            "bcrypt.lib",
        ],
        "@platforms//os:macos": [],
        "//conditions:default": ["-lpthread"],
    }),
    linkstatic = True,
)
