/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * Single-source CPU/GPU check for the fused update
 *
 *     X += Y + alpha * Z + exp(beta * U)
 *
 * with X, Y, Z, U double vectors and alpha, beta double scalars. The statement
 * lives once in fused_update() and is instantiated unchanged for CPU tensors
 * and GPU tensors; the destination's device selects the evaluator at run time.
 *
 * Because this file references tensor<, CMake compiles it as a CUDA/HIP
 * translation unit when a GPU backend is active (see Testing/Cxx/CMakeLists.txt).
 */

#include "VectorizationTest.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "terminals/tensor.h"

#if VECTORIZATION_HAS_CUDA || VECTORIZATION_HAS_HIP
#include <backend/gpu/gpu_stream_compat.h>
#endif

namespace
{

using vectorization::device_enum;
using vectorization::tensor;

constexpr std::size_t kN     = 1027;  // not a multiple of any SIMD width: exercises the tail
constexpr double      kAlpha = 0.75;
constexpr double      kBeta  = -1.25;
constexpr double      kTol   = 1e-13;

// The one statement under test. Identical source for every backend.
template <typename Vec>
void fused_update(Vec& X, Vec const& Y, Vec const& Z, Vec const& U, double alpha, double beta)
{
    X += Y + alpha * Z + ::exp(beta * U);
}

struct inputs
{
    std::vector<double> x, y, z, u;
};

inputs make_inputs()
{
    inputs in;
    in.x.resize(kN);
    in.y.resize(kN);
    in.z.resize(kN);
    in.u.resize(kN);
    for (std::size_t i = 0; i < kN; ++i)
    {
        double const t = static_cast<double>(i) / static_cast<double>(kN);
        in.x[i]        = 1.0 - t;        // (0, 1]
        in.y[i]        = 2.0 * t - 1.0;  // [-1, 1)
        in.z[i]        = 0.5 + t;        // [0.5, 1.5)
        in.u[i]        = 1.6 * t - 0.8;  // [-0.8, 0.8)
    }
    return in;
}

std::vector<double> reference(inputs const& in)
{
    std::vector<double> ref(kN);
    for (std::size_t i = 0; i < kN; ++i)
    {
        ref[i] = in.x[i] + (in.y[i] + kAlpha * in.z[i] + std::exp(kBeta * in.u[i]));
    }
    return ref;
}

// Upload, run fused_update() on `device`, download.
std::vector<double> run_on(device_enum device, inputs const& in)
{
    tensor<double> X(kN, device), Y(kN, device), Z(kN, device), U(kN, device);
    X.copy_from_host(in.x);
    Y.copy_from_host(in.y);
    Z.copy_from_host(in.z);
    U.copy_from_host(in.u);

    fused_update(X, Y, Z, U, kAlpha, kBeta);

    EXPECT_EQ(X.device(), device);
    return X.to_host_vector();
}

void expect_near_rel(std::vector<double> const& got, std::vector<double> const& ref)
{
    ASSERT_EQ(got.size(), ref.size());
    for (std::size_t i = 0; i < ref.size(); ++i)
    {
        double const allowed = kTol * std::max(1.0, std::fabs(ref[i]));
        EXPECT_NEAR(got[i], ref[i], allowed) << " at i=" << i;
    }
}

}  // namespace

VECTORIZATIONTEST(TensorFusedUpdate, CpuDouble)
{
    inputs const in = make_inputs();
    expect_near_rel(run_on(device_enum::CPU, in), reference(in));
    END_TEST();
}

VECTORIZATIONTEST(TensorFusedUpdate, GpuDouble)
{
#if VECTORIZATION_HAS_CUDA || VECTORIZATION_HAS_HIP
#if VECTORIZATION_HAS_CUDA
    constexpr device_enum kGpu = device_enum::CUDA;
#else
    constexpr device_enum kGpu = device_enum::HIP;
#endif
    int ndev = 0;
    gpuGetDeviceCount(&ndev);
    if (ndev == 0)
    {
        GTEST_SKIP() << "No GPU device";
    }
    inputs const              in  = make_inputs();
    std::vector<double> const gpu = run_on(kGpu, in);
    expect_near_rel(gpu, reference(in));
    // Same source, both backends: results must agree with each other too.
    expect_near_rel(gpu, run_on(device_enum::CPU, in));
#elif VECTORIZATION_HAS_METAL
    GTEST_SKIP() << "Metal backend is float-only (no fp64 on Apple GPUs)";
#else
    GTEST_SKIP() << "No GPU backend (CUDA/HIP) is enabled";
#endif
    END_TEST();
}
