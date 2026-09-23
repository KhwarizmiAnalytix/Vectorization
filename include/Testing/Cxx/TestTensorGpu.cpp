/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * Integration tests for tensor expression evaluation on the GPU backend
 * (CUDA, HIP, or Metal).
 *
 * Tensors are allocated with kActiveGpuDevice (device_enum::CUDA, ::HIP, or
 * ::METAL depending on which backend is active) so that
 * expressions_evaluator::run dispatches to the GPU path. CUDA and HIP are
 * treated identically by the evaluator dispatch (expressions_evaluator.h);
 * Metal is a separate dispatch condition (expressions_evaluator_metal.h) but
 * shares this same test file since the *tensor<T>*-level behavior it verifies
 * is backend-agnostic. Results are copied back to the host via
 * tensor::to_host_vector() and compared against std:: math.
 *
 * Tests are skipped automatically when no GPU device is present at runtime.
 * Metal is float-only (MSL has no double type on any Apple GPU) — the
 * *Double tests skip themselves under Metal rather than attempting a
 * construction that would throw (see kMetalOnlyBackend below).
 *
 * This file is compiled as a CUDA or HIP translation unit (CMake sets
 * LANGUAGE CUDA/HIP on it) so that GPU expression-template kernels are
 * instantiated correctly, while remaining a plain .cpp source that needs no
 * __CUDACC__/__HIPCC__ guards. Metal needs no such CMake language — it's
 * ordinary host C++, with device-count queries routed through the tiny
 * Objective-C++ shim in metal_device_probe.mm (xsigma_metal_device_count)
 * since this file itself must stay plain .cpp.
 */

// VectorizationTest.h pulls in the nvcc fmt/int128 ADL shim itself (guarded
// by VECTORIZATION_HAS_CUDA) -- see Testing/VectorizationTest.h.in and
// cuda_fmt_int128_fix.h for the full explanation.
#include "VectorizationTest.h"

#if VECTORIZATION_HAS_CUDA || VECTORIZATION_HAS_HIP || VECTORIZATION_HAS_METAL

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "backend/gpu/gpu_stream_compat.h"
#include "common/scalar_helper_functions.h"
#include "terminals/tensor.h"

namespace
{

using namespace vectorization;

#if VECTORIZATION_HAS_CUDA
constexpr device_enum kActiveGpuDevice  = device_enum::CUDA;
constexpr bool        kMetalOnlyBackend = false;
#elif VECTORIZATION_HAS_HIP
constexpr device_enum kActiveGpuDevice  = device_enum::HIP;
constexpr bool        kMetalOnlyBackend = false;
#elif VECTORIZATION_HAS_METAL
constexpr device_enum kActiveGpuDevice  = device_enum::METAL;
constexpr bool        kMetalOnlyBackend = true;
#endif

// ---- Comparison helper ---------------------------------------------------

// Check element-wise: |got[i] - ref[i]| <= tol * max(1, |ref[i]|) + tol
template <typename T>
void expect_near_rel(const std::vector<T>& got, const std::vector<T>& ref, double tol)
{
    ASSERT_EQ(got.size(), ref.size());
    for (size_t i = 0; i < ref.size(); ++i)
    {
        double err     = std::fabs(static_cast<double>(got[i]) - static_cast<double>(ref[i]));
        double allowed = tol * std::max(1.0, std::fabs(static_cast<double>(ref[i]))) + tol;
        EXPECT_LE(err, allowed) << " at i=" << i << "  got=" << got[i] << "  ref=" << ref[i];
    }
}

// ---- Input generation ---------------------------------------------------

template <typename T>
void make_inputs(std::vector<T>& ha, std::vector<T>& hb, size_t N)
{
    ha.resize(N);
    hb.resize(N);
    for (size_t i = 0; i < N; ++i)
    {
        double t = static_cast<double>(i) / static_cast<double>(N);
        ha[i]    = static_cast<T>(t * 4.0 - 2.0);  // [-2, 2)
        hb[i]    = static_cast<T>(t + 0.5);        // [0.5, 1.5)
    }
}

// ---- Test bodies --------------------------------------------------------

// Fill a GPU tensor with a scalar and verify the values come back correctly.
template <typename T>
void test_fill()
{
    constexpr size_t N    = 1024;
    constexpr T      kVal = static_cast<T>(2.71828);
    constexpr double tol  = std::is_same_v<T, float> ? 5e-6 : 1e-13;

    tensor<T> ga(N, kActiveGpuDevice);
    ga = kVal;  // Dispatches to fill_gpu kernel

    auto result = ga.to_host_vector();
    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(result[i], kVal, tol) << " at i=" << i;
}

// Element-wise add, sub, and mul on GPU tensors.
template <typename T>
void test_binary_ops()
{
    constexpr size_t N   = 1024;
    constexpr double tol = std::is_same_v<T, float> ? 5e-5 : 1e-11;

    std::vector<T> ha, hb;
    make_inputs(ha, hb, N);

    tensor<T> ga(N, kActiveGpuDevice), gb(N, kActiveGpuDevice);
    ga.copy_from_host(ha);
    gb.copy_from_host(hb);

    // Add
    {
        tensor<T> gc(N, kActiveGpuDevice);
        gc                    = ga + gb;  // → run_gpu(add_expr, gc.data(), N)
        auto           result = gc.to_host_vector();
        std::vector<T> ref(N);
        for (size_t i = 0; i < N; ++i)
            ref[i] = ha[i] + hb[i];
        expect_near_rel(result, ref, tol);
    }

    // Subtract
    {
        tensor<T> gc(N, kActiveGpuDevice);
        gc                    = ga - gb;
        auto           result = gc.to_host_vector();
        std::vector<T> ref(N);
        for (size_t i = 0; i < N; ++i)
            ref[i] = ha[i] - hb[i];
        expect_near_rel(result, ref, tol);
    }

    // Element-wise multiply (operator* is element-wise for 1-D tensors)
    {
        tensor<T> gc(N, kActiveGpuDevice);
        gc                    = ga * gb;
        auto           result = gc.to_host_vector();
        std::vector<T> ref(N);
        for (size_t i = 0; i < N; ++i)
            ref[i] = ha[i] * hb[i];
        expect_near_rel(result, ref, tol);
    }

    // Scalar + tensor
    {
        tensor<T> gc(N, kActiveGpuDevice);
        gc                    = ga + static_cast<T>(1);
        auto           result = gc.to_host_vector();
        std::vector<T> ref(N);
        for (size_t i = 0; i < N; ++i)
            ref[i] = ha[i] + static_cast<T>(1);
        expect_near_rel(result, ref, tol);
    }

    // Destination aliases a leaf (`a = a + b`): must not overwrite `a` before `b` is read.
    {
        tensor<T> acc(N, kActiveGpuDevice);
        acc.copy_from_host(ha);
        acc                   = acc + gb;
        auto           result = acc.to_host_vector();
        std::vector<T> ref(N);
        for (size_t i = 0; i < N; ++i)
            ref[i] = ha[i] + hb[i];
        expect_near_rel(result, ref, tol);
    }
}

// Unary math ops: exp, sqrt, log on GPU tensors.
template <typename T>
void test_unary_math()
{
    constexpr size_t N   = 1024;
    constexpr double tol = std::is_same_v<T, float> ? 5e-5 : 1e-11;

    // Use strictly positive values for sqrt/log
    std::vector<T> ha(N);
    for (size_t i = 0; i < N; ++i)
        ha[i] = static_cast<T>(static_cast<double>(i + 1) / static_cast<double>(N) * 2.0);

    tensor<T> ga(N, kActiveGpuDevice);
    ga.copy_from_host(ha);

    // exp
    {
        tensor<T> gc(N, kActiveGpuDevice);
        gc                    = ::exp(ga);
        auto           result = gc.to_host_vector();
        std::vector<T> ref(N);
        for (size_t i = 0; i < N; ++i)
            ref[i] = std::exp(ha[i]);
        expect_near_rel(result, ref, tol);
    }

    // sqrt
    {
        tensor<T> gc(N, kActiveGpuDevice);
        gc                    = ::sqrt(ga);
        auto           result = gc.to_host_vector();
        std::vector<T> ref(N);
        for (size_t i = 0; i < N; ++i)
            ref[i] = std::sqrt(ha[i]);
        expect_near_rel(result, ref, tol);
    }

    // log
    {
        tensor<T> gc(N, kActiveGpuDevice);
        gc                    = ::log(ga);
        auto           result = gc.to_host_vector();
        std::vector<T> ref(N);
        for (size_t i = 0; i < N; ++i)
            ref[i] = std::log(ha[i]);
        expect_near_rel(result, ref, tol);
    }

    // fabs
    {
        std::vector<T> ha_neg(N);
        for (size_t i = 0; i < N; ++i)
            ha_neg[i] = static_cast<T>(-static_cast<double>(ha[i]));
        tensor<T> ga_neg(N, kActiveGpuDevice);
        ga_neg.copy_from_host(ha_neg);

        tensor<T> gc(N, kActiveGpuDevice);
        gc          = ::fabs(ga_neg);
        auto result = gc.to_host_vector();
        for (size_t i = 0; i < N; ++i)
            EXPECT_NEAR(result[i], ha[i], static_cast<T>(tol)) << " at i=" << i;
    }
}

// Compound expression: the full expression tree is fused into a single kernel.
template <typename T>
void test_compound()
{
    constexpr size_t N = 1024;
    // exp amplifies float errors; use a looser tolerance
    constexpr double tol = std::is_same_v<T, float> ? 5e-4 : 1e-10;

    std::vector<T> ha(N), hb(N);
    for (size_t i = 0; i < N; ++i)
    {
        double t = static_cast<double>(i) / static_cast<double>(N);
        ha[i]    = static_cast<T>(t * 2.0 - 1.0);  // [-1, 1)
        hb[i]    = static_cast<T>(t + 0.5);        // [0.5, 1.5)
    }

    tensor<T> ga(N, kActiveGpuDevice), gb(N, kActiveGpuDevice);
    ga.copy_from_host(ha);
    gb.copy_from_host(hb);

    // exp(a) + sqrt(b): fused into one run_gpu call
    {
        tensor<T> gc(N, kActiveGpuDevice);
        gc                    = ::exp(ga) + ::sqrt(gb);
        auto           result = gc.to_host_vector();
        std::vector<T> ref(N);
        for (size_t i = 0; i < N; ++i)
            ref[i] = std::exp(ha[i]) + std::sqrt(hb[i]);
        expect_near_rel(result, ref, tol);
    }

    // (a + b) * scalar: mixed tensor and scalar operands
    {
        constexpr T kAlpha = static_cast<T>(1.5);
        tensor<T>   gc(N, kActiveGpuDevice);
        gc                    = (ga + gb) * kAlpha;
        auto           result = gc.to_host_vector();
        std::vector<T> ref(N);
        for (size_t i = 0; i < N; ++i)
            ref[i] = (ha[i] + hb[i]) * kAlpha;
        expect_near_rel(result, ref, tol);
    }

    // y + a + 5*d — one fused kernel (Metal JIT / CUDA tree eval), no per-node temps
    {
        constexpr T kFive = static_cast<T>(5);
        tensor<T>   gd(N, kActiveGpuDevice);
        gd.copy_from_host(hb);
        tensor<T> gx(N, kActiveGpuDevice);
        gx                    = ga + gb + kFive * gd;
        auto           result = gx.to_host_vector();
        std::vector<T> ref(N);
        for (size_t i = 0; i < N; ++i)
            ref[i] = ha[i] + hb[i] + kFive * hb[i];
        expect_near_rel(result, ref, tol);

        tensor<T> acc(N, kActiveGpuDevice);
        acc.copy_from_host(ha);
        acc        = acc + gb + kFive * gd;
        auto acc_h = acc.to_host_vector();
        expect_near_rel(acc_h, ref, tol);
    }
}

// Ops that used to throw on Metal (min/max/pow/floor/if_else/...) now fuse like CPU/CUDA.
template <typename T>
void test_fused_catalog()
{
    constexpr size_t N   = 1024;
    constexpr double tol = std::is_same_v<T, float> ? 5e-4 : 1e-10;

    std::vector<T> ha(N), hb(N);
    for (size_t i = 0; i < N; ++i)
    {
        double t = static_cast<double>(i) / static_cast<double>(N);
        ha[i]    = static_cast<T>(t * 2.0 - 1.0);
        hb[i]    = static_cast<T>(t + 0.5);
    }

    tensor<T> ga(N, kActiveGpuDevice), gb(N, kActiveGpuDevice);
    ga.copy_from_host(ha);
    gb.copy_from_host(hb);

    {
        tensor<T> gc(N, kActiveGpuDevice);
        gc                    = min(ga, gb);
        auto           result = gc.to_host_vector();
        std::vector<T> ref(N);
        for (size_t i = 0; i < N; ++i)
            ref[i] = std::min(ha[i], hb[i]);
        expect_near_rel(result, ref, tol);
    }
    {
        tensor<T> gc(N, kActiveGpuDevice);
        gc                    = max(ga, gb) + ::floor(ga);
        auto           result = gc.to_host_vector();
        std::vector<T> ref(N);
        for (size_t i = 0; i < N; ++i)
            ref[i] = std::max(ha[i], hb[i]) + std::floor(ha[i]);
        expect_near_rel(result, ref, tol);
    }
    {
        tensor<T> gc(N, kActiveGpuDevice);
        gc                    = ::pow(gb, static_cast<T>(2)) + ::hypot(ga, gb);
        auto           result = gc.to_host_vector();
        std::vector<T> ref(N);
        for (size_t i = 0; i < N; ++i)
            ref[i] = std::pow(hb[i], static_cast<T>(2)) + std::hypot(ha[i], hb[i]);
        expect_near_rel(result, ref, tol);
    }
    {
        tensor<T> gc(N, kActiveGpuDevice);
        gc                    = ::if_else(ga > static_cast<T>(0), ga, gb);
        auto           result = gc.to_host_vector();
        std::vector<T> ref(N);
        for (size_t i = 0; i < N; ++i)
            ref[i] = ha[i] > static_cast<T>(0) ? ha[i] : hb[i];
        expect_near_rel(result, ref, tol);
    }
    {
        tensor<T> gc(N, kActiveGpuDevice);
        gc                    = ::cbrt(ga) + ::sqr(gb) * ::invsqrt(gb);
        auto           result = gc.to_host_vector();
        std::vector<T> ref(N);
        for (size_t i = 0; i < N; ++i)
            ref[i] = std::cbrt(ha[i]) + std::sqr(hb[i]) * std::invsqrt(hb[i]);
        expect_near_rel(result, ref, tol);
    }
    {
        tensor<T> acc(N, kActiveGpuDevice);
        acc.copy_from_host(ha);
        acc += gb;
        auto           result = acc.to_host_vector();
        std::vector<T> ref(N);
        for (size_t i = 0; i < N; ++i)
            ref[i] = ha[i] + hb[i];
        expect_near_rel(result, ref, tol);
    }
}

template <typename T>
void test_reductions()
{
    constexpr size_t N       = 4096;
    constexpr double sum_tol = std::is_same_v<T, float> ? 2e-2 : 1e-10;
    constexpr double mm_tol  = std::is_same_v<T, float> ? 5e-6 : 1e-13;

    tensor<T> ones(N, kActiveGpuDevice);
    ones = static_cast<T>(1);
    EXPECT_NEAR(static_cast<double>(accumulate(ones)), static_cast<double>(N), sum_tol);

    std::vector<T> ha;
    std::vector<T> hb;
    make_inputs(ha, hb, N);
    tensor<T> ga(N, kActiveGpuDevice);
    tensor<T> gb(N, kActiveGpuDevice);
    ga.copy_from_host(ha);
    gb.copy_from_host(hb);

    T min_ref = std::numeric_limits<T>::max();
    T max_ref = -std::numeric_limits<T>::max();
    T sum_ref = static_cast<T>(0);
    for (size_t i = 0; i < N; ++i)
    {
        T const v = ha[i] + hb[i] * static_cast<T>(2);
        sum_ref += v;
        min_ref = std::min(min_ref, v);
        max_ref = std::max(max_ref, v);
    }

    EXPECT_NEAR(
        static_cast<double>(accumulate(ga + gb * static_cast<T>(2))),
        static_cast<double>(sum_ref),
        sum_tol);
    EXPECT_NEAR(
        static_cast<double>(hmin(ga + gb * static_cast<T>(2))),
        static_cast<double>(min_ref),
        mm_tol);
    EXPECT_NEAR(
        static_cast<double>(hmax(ga + gb * static_cast<T>(2))),
        static_cast<double>(max_ref),
        mm_tol);

    tensor<T> empty(static_cast<size_t>(0), kActiveGpuDevice);
    EXPECT_EQ(accumulate(empty), static_cast<T>(0));
    EXPECT_EQ(hmin(empty), std::numeric_limits<T>::max());
    EXPECT_EQ(hmax(empty), -std::numeric_limits<T>::max());

    tensor<T> one(static_cast<size_t>(1), kActiveGpuDevice);
    one = static_cast<T>(-3.5);
    EXPECT_NEAR(static_cast<double>(accumulate(one)), -3.5, mm_tol);
    EXPECT_NEAR(static_cast<double>(hmin(one)), -3.5, mm_tol);
    EXPECT_NEAR(static_cast<double>(hmax(one)), -3.5, mm_tol);
}

}  // namespace

// --------------------------------------------------------------------------
// Scalar fill
// --------------------------------------------------------------------------
VECTORIZATIONTEST(TensorGpu, FillFloat)
{
    int ndev = 0;
    gpuGetDeviceCount(&ndev);
    if (ndev == 0)
        GTEST_SKIP() << "No GPU device";
    test_fill<float>();
    END_TEST();
}

VECTORIZATIONTEST(TensorGpu, StoresDeviceIndex)
{
    int ndev = 0;
    gpuGetDeviceCount(&ndev);
    if (ndev == 0)
        GTEST_SKIP() << "No GPU device";
    tensor<float> t(64, kActiveGpuDevice, 0);
    EXPECT_EQ(kActiveGpuDevice, t.device());
    EXPECT_EQ(0, t.device_index());
    END_TEST();
}

VECTORIZATIONTEST(TensorGpu, CopyToHostWritesCallerBuffer)
{
    int ndev = 0;
    gpuGetDeviceCount(&ndev);
    if (ndev == 0)
        GTEST_SKIP() << "No GPU device";

    constexpr size_t N    = 256;
    constexpr float  kVal = 1.25f;
    tensor<float>    ga(N, kActiveGpuDevice);
    ga = kVal;
    std::vector<float> host(N, 0.f);
    ga.copy_to_host(host.data());
    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(host[i], kVal, 5e-6f) << " at i=" << i;
    ga.copy_to_host(host.data(), N);
    EXPECT_NEAR(host[0], kVal, 5e-6f);
    END_TEST();
}
VECTORIZATIONTEST(TensorGpu, CopyClonesIndependentStorage)
{
    int ndev = 0;
    gpuGetDeviceCount(&ndev);
    if (ndev == 0)
        GTEST_SKIP() << "No GPU device";

    constexpr size_t N    = 64;
    constexpr float  kVal = 1.5f;
    tensor<float>    src(N, kActiveGpuDevice);
    src = kVal;
    tensor<float> dst(src);
    EXPECT_EQ(src.device(), dst.device());
    EXPECT_EQ(src.device_index(), dst.device_index());
    EXPECT_NE(src.data(), dst.data());
    expect_near_rel(dst.to_host_vector(), src.to_host_vector(), 5e-6);
    src        = static_cast<float>(9);
    auto src_h = src.to_host_vector();
    auto dst_h = dst.to_host_vector();
    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(dst_h[i], kVal, 5e-6f);
    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(src_h[i], 9.0f, 5e-6f);
    END_TEST();
}
VECTORIZATIONTEST(TensorGpu, ExpressionCtorKeepsDevice)
{
    int ndev = 0;
    gpuGetDeviceCount(&ndev);
    if (ndev == 0)
        GTEST_SKIP() << "No GPU device";

    constexpr size_t   N   = 64;
    constexpr double   tol = 5e-5;
    std::vector<float> ha, hb;
    make_inputs(ha, hb, N);
    tensor<float> ga(N, kActiveGpuDevice), gb(N, kActiveGpuDevice);
    ga.copy_from_host(ha);
    gb.copy_from_host(hb);
    tensor<float> gc = ga + gb;
    EXPECT_EQ(kActiveGpuDevice, gc.device());
    EXPECT_EQ(0, gc.device_index());
    std::vector<float> ref(N);
    for (size_t i = 0; i < N; ++i)
        ref[i] = ha[i] + hb[i];
    expect_near_rel(gc.to_host_vector(), ref, tol);
    auto cloned = gc.clone();
    EXPECT_EQ(gc.device(), cloned.device());
    EXPECT_NE(gc.data(), cloned.data());
    expect_near_rel(cloned.to_host_vector(), gc.to_host_vector(), 5e-6);
    END_TEST();
}
VECTORIZATIONTEST(TensorGpu, ExpressionLeavesAliasStorage)
{
    int ndev = 0;
    gpuGetDeviceCount(&ndev);
    if (ndev == 0)
        GTEST_SKIP() << "No GPU device";

    constexpr size_t N = 64;
    tensor<float>    ga(N, kActiveGpuDevice), gb(N, kActiveGpuDevice);
    ga     = 1.5f;
    gb     = 2.5f;
    auto e = ga + gb;
    EXPECT_EQ(e.lhs().data(), ga.data());
    EXPECT_EQ(e.rhs().data(), gb.data());
    tensor<float> gc = e;
    EXPECT_EQ(kActiveGpuDevice, gc.device());
    std::vector<float> got = gc.to_host_vector();
    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(got[i], 4.0f, 5e-6f);
    END_TEST();
}
VECTORIZATIONTEST(TensorGpu, LinspaceAndToCpu)
{
    int ndev = 0;
    gpuGetDeviceCount(&ndev);
    if (ndev == 0)
        GTEST_SKIP() << "No GPU device";

    tensor<float> g(0.0f, 4.0f, 5u, kActiveGpuDevice);
    EXPECT_EQ(g.device(), kActiveGpuDevice);
    tensor<float> h = g.to_cpu();
    EXPECT_EQ(h.device(), device_enum::CPU);
    EXPECT_NE(h.data(), g.data());
    EXPECT_EQ(h.size(), 5u);
    EXPECT_NEAR(h[0], 0.0f, 5e-6f);
    EXPECT_NEAR(h[2], 2.0f, 5e-6f);
    EXPECT_NEAR(h[4], 4.0f, 5e-6f);
    END_TEST();
}
VECTORIZATIONTEST(TensorGpu, FillDouble)
{
    if (kMetalOnlyBackend)
        GTEST_SKIP() << "Metal backend is float-only (no fp64 on Apple GPUs)";
    int ndev = 0;
    gpuGetDeviceCount(&ndev);
    if (ndev == 0)
        GTEST_SKIP() << "No GPU device";
    test_fill<double>();
    END_TEST();
}

// --------------------------------------------------------------------------
// Binary ops
// --------------------------------------------------------------------------
VECTORIZATIONTEST(TensorGpu, BinaryOpsFloat)
{
    int ndev = 0;
    gpuGetDeviceCount(&ndev);
    if (ndev == 0)
        GTEST_SKIP() << "No GPU device";
    test_binary_ops<float>();
    END_TEST();
}
VECTORIZATIONTEST(TensorGpu, BinaryOpsDouble)
{
    if (kMetalOnlyBackend)
        GTEST_SKIP() << "Metal backend is float-only (no fp64 on Apple GPUs)";
    int ndev = 0;
    gpuGetDeviceCount(&ndev);
    if (ndev == 0)
        GTEST_SKIP() << "No GPU device";
    test_binary_ops<double>();
    END_TEST();
}

// --------------------------------------------------------------------------
// Unary math
// --------------------------------------------------------------------------
VECTORIZATIONTEST(TensorGpu, UnaryMathFloat)
{
    int ndev = 0;
    gpuGetDeviceCount(&ndev);
    if (ndev == 0)
        GTEST_SKIP() << "No GPU device";
    test_unary_math<float>();
    END_TEST();
}
VECTORIZATIONTEST(TensorGpu, UnaryMathDouble)
{
    if (kMetalOnlyBackend)
        GTEST_SKIP() << "Metal backend is float-only (no fp64 on Apple GPUs)";
    int ndev = 0;
    gpuGetDeviceCount(&ndev);
    if (ndev == 0)
        GTEST_SKIP() << "No GPU device";
    test_unary_math<double>();
    END_TEST();
}

// --------------------------------------------------------------------------
// Compound (fused) expressions
// --------------------------------------------------------------------------
VECTORIZATIONTEST(TensorGpu, CompoundFloat)
{
    int ndev = 0;
    gpuGetDeviceCount(&ndev);
    if (ndev == 0)
        GTEST_SKIP() << "No GPU device";
    test_compound<float>();
    END_TEST();
}
VECTORIZATIONTEST(TensorGpu, FusedCatalogFloat)
{
    int ndev = 0;
    gpuGetDeviceCount(&ndev);
    if (ndev == 0)
        GTEST_SKIP() << "No GPU device";
    test_fused_catalog<float>();
    END_TEST();
}
VECTORIZATIONTEST(TensorGpu, CompoundDouble)
{
    if (kMetalOnlyBackend)
        GTEST_SKIP() << "Metal backend is float-only (no fp64 on Apple GPUs)";
    int ndev = 0;
    gpuGetDeviceCount(&ndev);
    if (ndev == 0)
        GTEST_SKIP() << "No GPU device";
    test_compound<double>();
    END_TEST();
}
VECTORIZATIONTEST(TensorGpu, FusedCatalogDouble)
{
    if (kMetalOnlyBackend)
        GTEST_SKIP() << "Metal backend is float-only (no fp64 on Apple GPUs)";
    int ndev = 0;
    gpuGetDeviceCount(&ndev);
    if (ndev == 0)
        GTEST_SKIP() << "No GPU device";
    test_fused_catalog<double>();
    END_TEST();
}

VECTORIZATIONTEST(TensorGpu, ReductionsFloat)
{
    int ndev = 0;
    gpuGetDeviceCount(&ndev);
    if (ndev == 0)
        GTEST_SKIP() << "No GPU device";
    test_reductions<float>();
    END_TEST();
}

VECTORIZATIONTEST(TensorGpu, ReductionsDouble)
{
    if (kMetalOnlyBackend)
        GTEST_SKIP() << "Metal backend is float-only (no fp64 on Apple GPUs)";
    int ndev = 0;
    gpuGetDeviceCount(&ndev);
    if (ndev == 0)
        GTEST_SKIP() << "No GPU device";
    test_reductions<double>();
    END_TEST();
}

// --------------------------------------------------------------------------
// stream_guard — ambient current stream (see terminals/stream_guard.h)
// --------------------------------------------------------------------------
VECTORIZATIONTEST(TensorGpu, StreamGuardRedirectsExpressionConstruction)
{
#if VECTORIZATION_HAS_CUDA || VECTORIZATION_HAS_HIP
    int ndev = 0;
    gpuGetDeviceCount(&ndev);
    if (ndev == 0)
        GTEST_SKIP() << "No GPU device";

    gpu_stream_t stream = nullptr;
    ASSERT_EQ(gpuSuccess, gpuStreamCreate(&stream));

    constexpr size_t N = 64;
    {
        // a, b, and c (below) must be destroyed -- and any cross-stream events they
        // recorded against `stream` via record_expression_streams retired -- before
        // `stream` itself is destroyed at the end of this test; a live tensor can still
        // reference the stream via the caching allocator's deferred-reuse bookkeeping
        // (see cuda_caching_allocator.cpp's record_stream/deallocate) even after the
        // guard that redirected it goes out of scope.
        tensor<float> a(N, kActiveGpuDevice), b(N, kActiveGpuDevice);
        a = 1.5f;
        b = 2.5f;
        {
            stream_guard guard(stream, 0);
            // init_from_expression must allocate and evaluate `c` on the guard's stream,
            // not the default stream.
            tensor<float> c = a + b;
            EXPECT_EQ(stream, c.stream());
            ASSERT_EQ(gpuSuccess, gpuStreamSynchronize(stream));
            std::vector<float> got = c.to_host_vector();
            for (size_t i = 0; i < N; ++i)
                EXPECT_NEAR(got[i], 4.0f, 5e-6f);
        }
        EXPECT_EQ(nullptr, current_stream(0));
    }
    ASSERT_EQ(gpuSuccess, gpuStreamDestroy(stream));
#else
    GTEST_SKIP() << "stream_guard requires CUDA or HIP (Metal has no stream concept)";
#endif
    END_TEST();
}

VECTORIZATIONTEST(TensorGpu, StreamGuardRedirectsAssignmentIntoExistingTensor)
{
#if VECTORIZATION_HAS_CUDA || VECTORIZATION_HAS_HIP
    int ndev = 0;
    gpuGetDeviceCount(&ndev);
    if (ndev == 0)
        GTEST_SKIP() << "No GPU device";

    gpu_stream_t stream = nullptr;
    ASSERT_EQ(gpuSuccess, gpuStreamCreate(&stream));

    constexpr size_t N = 64;
    {
        // a, b, c must be destroyed before `stream` is (see the comment in
        // StreamGuardRedirectsExpressionConstruction above for why).
        tensor<float> a(N, kActiveGpuDevice), b(N, kActiveGpuDevice), c(N, kActiveGpuDevice);
        a = 1.5f;
        b = 2.5f;
        {
            // operator=(E const&) into a pre-existing tensor should also pick up the
            // ambient stream (and record it with the caching allocator) rather than
            // always falling back to the default stream.
            stream_guard guard(stream, 0);
            c = a + b;
        }
        ASSERT_EQ(gpuSuccess, gpuStreamSynchronize(stream));
        std::vector<float> got = c.to_host_vector();
        for (size_t i = 0; i < N; ++i)
            EXPECT_NEAR(got[i], 4.0f, 5e-6f);
    }
    ASSERT_EQ(gpuSuccess, gpuStreamDestroy(stream));
#else
    GTEST_SKIP() << "stream_guard requires CUDA or HIP (Metal has no stream concept)";
#endif
    END_TEST();
}

VECTORIZATIONTEST(TensorGpu, StreamGuardNestedRestoresPreviousStream)
{
#if VECTORIZATION_HAS_CUDA || VECTORIZATION_HAS_HIP
    int ndev = 0;
    gpuGetDeviceCount(&ndev);
    if (ndev == 0)
        GTEST_SKIP() << "No GPU device";

    gpu_stream_t outer = nullptr;
    gpu_stream_t inner = nullptr;
    ASSERT_EQ(gpuSuccess, gpuStreamCreate(&outer));
    ASSERT_EQ(gpuSuccess, gpuStreamCreate(&inner));

    EXPECT_EQ(nullptr, current_stream(0));
    {
        stream_guard outer_guard(outer, 0);
        EXPECT_EQ(outer, current_stream(0));
        {
            stream_guard inner_guard(inner, 0);
            EXPECT_EQ(inner, current_stream(0));
        }
        EXPECT_EQ(outer, current_stream(0));
    }
    EXPECT_EQ(nullptr, current_stream(0));

    ASSERT_EQ(gpuSuccess, gpuStreamDestroy(outer));
    ASSERT_EQ(gpuSuccess, gpuStreamDestroy(inner));
#else
    GTEST_SKIP() << "stream_guard requires CUDA or HIP (Metal has no stream concept)";
#endif
    END_TEST();
}

VECTORIZATIONTEST(TensorGpu, StreamGuardPerDeviceIndexIsolated)
{
#if VECTORIZATION_HAS_CUDA || VECTORIZATION_HAS_HIP
    int ndev = 0;
    gpuGetDeviceCount(&ndev);
    if (ndev == 0)
        GTEST_SKIP() << "No GPU device";

    gpu_stream_t stream = nullptr;
    ASSERT_EQ(gpuSuccess, gpuStreamCreate(&stream));
    {
        stream_guard guard(stream, 0);
        EXPECT_EQ(stream, current_stream(0));
        // A guard scoped to device_index 0 must not leak into another index's
        // ambient stream.
        EXPECT_EQ(nullptr, current_stream(1));
    }
    ASSERT_EQ(gpuSuccess, gpuStreamDestroy(stream));
#else
    GTEST_SKIP() << "stream_guard requires CUDA or HIP (Metal has no stream concept)";
#endif
    END_TEST();
}

#else  // !(VECTORIZATION_HAS_CUDA || VECTORIZATION_HAS_HIP || VECTORIZATION_HAS_METAL)

VECTORIZATIONTEST(TensorGpu, FillFloat)
{
    GTEST_SKIP() << "Test disabled: no GPU backend (CUDA/HIP/Metal) is enabled";
    END_TEST();
}
VECTORIZATIONTEST(TensorGpu, StoresDeviceIndex)
{
    GTEST_SKIP() << "Test disabled: no GPU backend (CUDA/HIP/Metal) is enabled";
    END_TEST();
}
VECTORIZATIONTEST(TensorGpu, CopyToHostWritesCallerBuffer)
{
    GTEST_SKIP() << "Test disabled: no GPU backend (CUDA/HIP/Metal) is enabled";
    END_TEST();
}
VECTORIZATIONTEST(TensorGpu, CopyClonesIndependentStorage)
{
    GTEST_SKIP() << "Test disabled: no GPU backend (CUDA/HIP/Metal) is enabled";
    END_TEST();
}
VECTORIZATIONTEST(TensorGpu, ExpressionCtorKeepsDevice)
{
    GTEST_SKIP() << "Test disabled: no GPU backend (CUDA/HIP/Metal) is enabled";
    END_TEST();
}
VECTORIZATIONTEST(TensorGpu, ExpressionLeavesAliasStorage)
{
    GTEST_SKIP() << "Test disabled: no GPU backend (CUDA/HIP/Metal) is enabled";
    END_TEST();
}
VECTORIZATIONTEST(TensorGpu, LinspaceAndToCpu)
{
    GTEST_SKIP() << "Test disabled: no GPU backend (CUDA/HIP/Metal) is enabled";
    END_TEST();
}
VECTORIZATIONTEST(TensorGpu, FillDouble)
{
    GTEST_SKIP() << "Test disabled: no GPU backend (CUDA/HIP/Metal) is enabled";
    END_TEST();
}
VECTORIZATIONTEST(TensorGpu, BinaryOpsFloat)
{
    GTEST_SKIP() << "Test disabled: no GPU backend (CUDA/HIP/Metal) is enabled";
    END_TEST();
}
VECTORIZATIONTEST(TensorGpu, BinaryOpsDouble)
{
    GTEST_SKIP() << "Test disabled: no GPU backend (CUDA/HIP/Metal) is enabled";
    END_TEST();
}
VECTORIZATIONTEST(TensorGpu, UnaryMathFloat)
{
    GTEST_SKIP() << "Test disabled: no GPU backend (CUDA/HIP/Metal) is enabled";
    END_TEST();
}
VECTORIZATIONTEST(TensorGpu, UnaryMathDouble)
{
    GTEST_SKIP() << "Test disabled: no GPU backend (CUDA/HIP/Metal) is enabled";
    END_TEST();
}
VECTORIZATIONTEST(TensorGpu, CompoundFloat)
{
    GTEST_SKIP() << "Test disabled: no GPU backend (CUDA/HIP/Metal) is enabled";
    END_TEST();
}
VECTORIZATIONTEST(TensorGpu, FusedCatalogFloat)
{
    GTEST_SKIP() << "Test disabled: no GPU backend (CUDA/HIP/Metal) is enabled";
    END_TEST();
}
VECTORIZATIONTEST(TensorGpu, CompoundDouble)
{
    GTEST_SKIP() << "Test disabled: no GPU backend (CUDA/HIP/Metal) is enabled";
    END_TEST();
}
VECTORIZATIONTEST(TensorGpu, FusedCatalogDouble)
{
    GTEST_SKIP() << "Test disabled: no GPU backend (CUDA/HIP/Metal) is enabled";
    END_TEST();
}
VECTORIZATIONTEST(TensorGpu, ReductionsFloat)
{
    GTEST_SKIP() << "Test disabled: no GPU backend (CUDA/HIP/Metal) is enabled";
    END_TEST();
}
VECTORIZATIONTEST(TensorGpu, ReductionsDouble)
{
    GTEST_SKIP() << "Test disabled: no GPU backend (CUDA/HIP/Metal) is enabled";
    END_TEST();
}
VECTORIZATIONTEST(TensorGpu, StreamGuardRedirectsExpressionConstruction)
{
    GTEST_SKIP() << "Test disabled: no GPU backend (CUDA/HIP/Metal) is enabled";
    END_TEST();
}
VECTORIZATIONTEST(TensorGpu, StreamGuardRedirectsAssignmentIntoExistingTensor)
{
    GTEST_SKIP() << "Test disabled: no GPU backend (CUDA/HIP/Metal) is enabled";
    END_TEST();
}
VECTORIZATIONTEST(TensorGpu, StreamGuardNestedRestoresPreviousStream)
{
    GTEST_SKIP() << "Test disabled: no GPU backend (CUDA/HIP/Metal) is enabled";
    END_TEST();
}
VECTORIZATIONTEST(TensorGpu, StreamGuardPerDeviceIndexIsolated)
{
    GTEST_SKIP() << "Test disabled: no GPU backend (CUDA/HIP/Metal) is enabled";
    END_TEST();
}

#endif  // VECTORIZATION_HAS_CUDA || VECTORIZATION_HAS_HIP || VECTORIZATION_HAS_METAL
