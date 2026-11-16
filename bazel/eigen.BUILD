# Eigen (header-only), bundled by LinearAlgebra at ThirdParty/LinearAlgebra/ThirdParty/eigen.
# Used only as a benchmark baseline (Testing/Cxx/BenchmarkTensorBaselines.cpp).
load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "eigen",
    hdrs = glob(["Eigen/**"]),
    defines = ["EIGEN_DONT_PARALLELIZE"],
    includes = ["."],
)
