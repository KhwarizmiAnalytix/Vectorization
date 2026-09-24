# Standalone Memory library (KhwarizmiAnalytix/Memory). Public headers live under
# include/, which is also the include root (e.g. "allocator.h", "common/data_ptr.h").
load("@bazel_skylib//rules:copy_file.bzl", "copy_file")
load("//bazel:memory.bzl", "memory_copts", "memory_defines", "memory_linkopts")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "memory_hdrs",
    srcs = glob(
        [
            "include/*.h",
            "include/**/*.h",
        ],
        allow_empty = True,
    ),
)

filegroup(
    name = "memory_srcs",
    srcs = glob(
        [
            "include/*.cpp",
            "include/common/*.cpp",
            "include/helper/*.cpp",
            "include/profiler/*.cpp",
        ],
        allow_empty = True,
    ),
)

# cc_library's srcs don't accept .mm (Objective-C++) without rules_apple's objc_library,
# which this repo doesn't otherwise depend on. Work around it the standard way: copy each
# .mm to a .cc-suffixed generated file (copy_file, not glob) and force Objective-C++ mode
# back on via -x objective-c++ in copts -- scoped to a small dedicated cc_library (below)
# rather than applied to the whole memory_lib target, matching CMake's per-file
# set_source_files_properties(${memory_objcxx_sources} PROPERTIES COMPILE_OPTIONS "-fobjc-arc").
copy_file(
    name = "metal_buffer_allocator_objcxx",
    src = "include/gpu/metal/metal_buffer_allocator.mm",
    out = "include/gpu/metal/metal_buffer_allocator_mm.cc",
)

copy_file(
    name = "metal_caching_allocator_objcxx",
    src = "include/gpu/metal/metal_caching_allocator.mm",
    out = "include/gpu/metal/metal_caching_allocator_mm.cc",
)

cc_library(
    name = "memory_metal_objcxx",
    srcs = select({
        "//bazel:enable_metal": [
            ":metal_buffer_allocator_objcxx",
            ":metal_caching_allocator_objcxx",
        ],
        "//conditions:default": [],
    }),
    hdrs = [":memory_hdrs"] + select({
        "//bazel:enable_metal": glob(["include/gpu/*.h", "include/gpu/metal/*.h"], allow_empty = True),
        "//conditions:default": [],
    }),
    copts = memory_copts() + select({
        "//bazel:enable_metal": ["-x", "objective-c++", "-fobjc-arc"],
        "//conditions:default": [],
    }),
    defines = memory_defines() + select({
        "//bazel:shared_libs": ["MEMORY_SHARED_DEFINE", "MEMORY_BUILDING_DLL"],
        "//conditions:default": ["MEMORY_STATIC_DEFINE"],
    }),
    includes = ["include"],
    deps = ["@logging//:Logging"] + select({
        "//bazel:disable_profiler": [],
        "//conditions:default": ["@profiler//:Profiler"],
    }),
    visibility = ["//visibility:private"],
)

cc_library(
    name = "memory_lib",
    srcs = [
        ":memory_srcs",
    ] + select({
        "//bazel:enable_cuda": glob(["include/gpu/*.cpp"], allow_empty = True),
        "//bazel:enable_hip": glob(["include/gpu/*.cpp"], allow_empty = True),
        # gpu/*.cpp (cuda_caching_allocator.cpp) is included for Metal too, matching CMake
        # (CMakeLists.txt only strips it for GPU backend "none") -- the file's own
        # #if MEMORY_HAS_CUDA || MEMORY_HAS_HIP guard makes it compile to an empty TU here.
        "//bazel:enable_metal": glob(["include/gpu/*.cpp"], allow_empty = True),
        "//conditions:default": [],
    }),
    hdrs = [
        ":memory_hdrs",
    ] + select({
        "//bazel:enable_cuda": glob(["include/gpu/*.h"], allow_empty = True),
        "//bazel:enable_hip": glob(["include/gpu/*.h"], allow_empty = True),
        "//bazel:enable_metal": glob(["include/gpu/*.h", "include/gpu/metal/*.h"], allow_empty = True),
        "//conditions:default": [],
    }),
    copts = memory_copts(),
    defines = memory_defines() + select({
        "//bazel:shared_libs": ["MEMORY_SHARED_DEFINE", "MEMORY_BUILDING_DLL"],
        "//conditions:default": ["MEMORY_STATIC_DEFINE"],
    }),
    includes = ["include"],
    linkopts = memory_linkopts() + select({
        "@platforms//os:windows": [],
        "@platforms//os:macos": [],
        "//conditions:default": ["-lpthread", "-ldl", "-lrt"],
    }) + select({
        # Mirrors Cmake/metal.cmake's find_library(Metal)/find_library(Foundation).
        "//bazel:enable_metal": ["-framework", "Metal", "-framework", "Foundation"],
        "//conditions:default": [],
    }) + select({
        # Mirrors Cmake/numa.cmake's find_package(Numa) -> Numa::numa (a plain system
        # library, -lnuma on Linux; Unix/Linux-only, matching numa.cmake's own guard).
        "//bazel:enable_numa": ["-lnuma"],
        "//conditions:default": [],
    }),
    linkstatic = select({
        "//bazel:shared_libs": False,
        "//conditions:default": True,
    }),
    deps = [
        "@logging//:Logging",
    ] + select({
        "//bazel:disable_profiler": [],
        "//conditions:default": ["@profiler//:Profiler"],
    }) + select({
        "//bazel:disable_mimalloc": [],
        "//conditions:default": ["@mimalloc//:mimalloc"],
    }) + select({
        "//bazel:enable_cuda": ["@local_config_cuda//:cuda"],
        "//conditions:default": [],
    }) + select({
        "//bazel:enable_hip": ["@local_config_hip//:hip"],
        "//conditions:default": [],
    }) + select({
        "//bazel:enable_metal": [":memory_metal_objcxx"],
        "//conditions:default": [],
    }) + select({
        # Mirrors CMakeLists.txt (if(MEMORY_ENABLE_TBB AND TARGET Tbb::tbbmalloc)
        # list(APPEND MEMORY_DEPENDENCY_LIBS Tbb::tbbmalloc)).
        "//bazel:memory_enable_tbb": ["@tbb//:tbbmalloc"],
        "//conditions:default": [],
    }),
    alwayslink = False,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "Memory",
    deps = [":memory_lib"],
    visibility = ["//visibility:public"],
)
