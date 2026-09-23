/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * CPU vs CUDA correctness tests.
 * Runs XSigma tensor operations on CPU and the equivalent torch operations on
 * a CUDA device, then compares results.  Skipped automatically when CUDA is
 * not available at runtime.
 */

#include "VectorizationTest.h"

#if VECTORIZATION_HAS_LIBTORCH

#include "TestTensorLibTorchHelpers.h"

namespace
{
using namespace libtorch_test;

template <typename T>
void test_libtorch_gpu()
{
    if (!torch::cuda::is_available())
    {
        GTEST_SKIP() << "CUDA not available — skipping GPU comparison test";
        return;
    }

    constexpr std::size_t n   = 1024;
    constexpr double      tol = std::is_same_v<T, float> ? 1e-5 : 1e-12;

    auto ax = rand_vec<T>(n, T(-2), T(2), 10);
    auto ay = rand_vec<T>(n, T(-2), T(2), 11);

    vectorization::tensor<T> xa(ax.data(), n);
    vectorization::tensor<T> xb(ay.data(), n);

    // addition: XSigma CPU vs torch CUDA
    vectorization::tensor<T> out_add(n);
    out_add = xa + xb;

    auto ta_gpu  = to_torch(ax.data(), n).to(torch::kCUDA);
    auto tb_gpu  = to_torch(ay.data(), n).to(torch::kCUDA);
    auto add_gpu = (ta_gpu + tb_gpu).cpu();

    EXPECT_LT(max_diff(out_add, add_gpu), tol) << "CPU+CPU vs GPU+GPU add";

    // exp: XSigma CPU vs torch CUDA
    vectorization::tensor<T> out_exp(n);
    out_exp      = ::exp(xa);
    auto exp_gpu = torch::exp(ta_gpu).cpu();

    EXPECT_LT(max_diff(out_exp, exp_gpu), tol * 10) << "CPU exp vs GPU exp";
}

}  // namespace

VECTORIZATIONTEST(LibTorch, GpuVsCpuFloat)
{
    test_libtorch_gpu<float>();
    END_TEST();
}

VECTORIZATIONTEST(LibTorch, GpuVsCpuDouble)
{
    test_libtorch_gpu<double>();
    END_TEST();
}

#else

VECTORIZATIONTEST(LibTorch, GpuVsCpuFloat)
{
    END_TEST();
}
VECTORIZATIONTEST(LibTorch, GpuVsCpuDouble)
{
    END_TEST();
}

#endif  // VECTORIZATION_HAS_LIBTORCH
