/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * 2-D tensor correctness tests against LibTorch.
 * Verifies that elementwise kernels produce shape-independent results when the
 * backing storage is reinterpreted as a rows×cols matrix.
 * Covers: +, -, *, exp, sqrt, tanh.
 */

#include "VectorizationTest.h"

#if VECTORIZATION_HAS_LIBTORCH

#include "TestTensorLibTorchHelpers.h"

namespace
{
using namespace libtorch_test;

template <typename T>
void test_libtorch_2d()
{
    using tensor_t             = vectorization::tensor<T>;
    constexpr std::size_t rows = 32;
    constexpr std::size_t cols = 33;  // non-multiple of common SIMD widths
    constexpr std::size_t n    = rows * cols;
    constexpr double      tol  = std::is_same_v<T, float> ? 1e-5 : 1e-13;

    auto ax = rand_vec<T>(n, T(-2), T(2), 7);
    auto ay = rand_vec<T>(n, T(-2), T(2), 8);
    auto az = rand_vec<T>(n, T(0.5), T(3), 9);  // positive for log/sqrt

    tensor_t xa(ax.data(), rows, cols);
    tensor_t xb(ay.data(), rows, cols);
    tensor_t xc(az.data(), rows, cols);

    auto ta = to_torch_2d(ax.data(), rows, cols);
    auto tb = to_torch_2d(ay.data(), rows, cols);
    auto tc = to_torch_2d(az.data(), rows, cols);

    {
        tensor_t r(rows, cols);
        r = xa + xb;
        EXPECT_LT(max_diff(r, ta + tb), tol) << "2D add";
    }
    {
        tensor_t r(rows, cols);
        r = xa - xb;
        EXPECT_LT(max_diff(r, ta - tb), tol) << "2D sub";
    }
    {
        tensor_t r(rows, cols);
        r = xa * xb;
        EXPECT_LT(max_diff(r, ta * tb), tol) << "2D mul";
    }
    {
        tensor_t r(rows, cols);
        r = ::exp(xc);
        EXPECT_LT(max_diff(r, torch::exp(tc)), tol * 10) << "2D exp";
    }
    {
        tensor_t r(rows, cols);
        r = ::sqrt(xc);
        EXPECT_LT(max_diff(r, torch::sqrt(tc)), tol) << "2D sqrt";
    }
    {
        tensor_t r(rows, cols);
        r = ::tanh(xa);
        EXPECT_LT(max_diff(r, torch::tanh(ta)), tol * 10) << "2D tanh";
    }
}

}  // namespace

VECTORIZATIONTEST(LibTorch, TwoDimFloat)
{
    test_libtorch_2d<float>();
    END_TEST();
}

VECTORIZATIONTEST(LibTorch, TwoDimDouble)
{
    test_libtorch_2d<double>();
    END_TEST();
}

#else

VECTORIZATIONTEST(LibTorch, TwoDimFloat)
{
    END_TEST();
}
VECTORIZATIONTEST(LibTorch, TwoDimDouble)
{
    END_TEST();
}

#endif  // VECTORIZATION_HAS_LIBTORCH
