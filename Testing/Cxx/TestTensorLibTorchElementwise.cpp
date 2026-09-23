/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * 1-D elementwise arithmetic and basic transcendental correctness tests.
 * Covers: +, -, *, /, exp, log, log2, log10, sqrt, sin, cos, tanh, fabs, neg.
 */

#include "VectorizationTest.h"

#if VECTORIZATION_HAS_LIBTORCH

#include "TestTensorLibTorchHelpers.h"

namespace
{
using namespace libtorch_test;

template <typename T>
void test_libtorch_elementwise()
{
    using tensor_t            = vectorization::tensor<T>;
    constexpr std::size_t n   = 512 + 7;  // deliberate non-multiple of SIMD width
    constexpr double      tol = std::is_same_v<T, float> ? 1e-5 : 1e-13;

    auto ax = rand_vec<T>(n, T(-4), T(4), 1);
    auto ay = rand_vec<T>(n, T(-4), T(4), 2);
    auto az = rand_vec<T>(n, T(1), T(4), 3);  // positive for log/sqrt/div

    tensor_t xa(ax.data(), n);
    tensor_t xb(ay.data(), n);
    tensor_t xc(az.data(), n);

    auto ta = to_torch(ax.data(), n);
    auto tb = to_torch(ay.data(), n);
    auto tc = to_torch(az.data(), n);

    {
        tensor_t r(n);
        r = xa + xb;
        EXPECT_LT(max_diff(r, ta + tb), tol) << "add";
    }
    {
        tensor_t r(n);
        r = xa - xb;
        EXPECT_LT(max_diff(r, ta - tb), tol) << "sub";
    }
    {
        tensor_t r(n);
        r = xa * xb;
        EXPECT_LT(max_diff(r, ta * tb), tol) << "mul";
    }
    {
        tensor_t r(n);
        r = xa / xc;
        EXPECT_LT(max_diff(r, ta / tc), tol) << "div";
    }
    {
        tensor_t r(n);
        r = ::exp(xa);
        EXPECT_LT(max_diff(r, torch::exp(ta)), tol * 10) << "exp";
    }
    {
        tensor_t r(n);
        r = ::log(xc);
        EXPECT_LT(max_diff(r, torch::log(tc)), tol * 10) << "log";
    }
    {
        tensor_t r(n);
        r = ::log2(xc);
        EXPECT_LT(max_diff(r, torch::log2(tc)), tol * 10) << "log2";
    }
    {
        tensor_t r(n);
        r = ::log10(xc);
        EXPECT_LT(max_diff(r, torch::log10(tc)), tol * 10) << "log10";
    }
    {
        tensor_t r(n);
        r = ::sqrt(xc);
        EXPECT_LT(max_diff(r, torch::sqrt(tc)), tol) << "sqrt";
    }
    {
        tensor_t r(n);
        r = ::sin(xa);
        EXPECT_LT(max_diff(r, torch::sin(ta)), tol * 10) << "sin";
    }
    {
        tensor_t r(n);
        r = ::cos(xa);
        EXPECT_LT(max_diff(r, torch::cos(ta)), tol * 10) << "cos";
    }
    {
        tensor_t r(n);
        r = ::tanh(xa);
        EXPECT_LT(max_diff(r, torch::tanh(ta)), tol * 10) << "tanh";
    }
    {
        tensor_t r(n);
        r = ::fabs(xa);
        EXPECT_LT(max_diff(r, torch::abs(ta)), tol) << "fabs";
    }
    {
        tensor_t r(n);
        r = -xa;
        EXPECT_LT(max_diff(r, -ta), tol) << "neg";
    }
}

}  // namespace

VECTORIZATIONTEST(LibTorch, ElementwiseFloat)
{
    test_libtorch_elementwise<float>();
    END_TEST();
}

VECTORIZATIONTEST(LibTorch, ElementwiseDouble)
{
    test_libtorch_elementwise<double>();
    END_TEST();
}

#else

VECTORIZATIONTEST(LibTorch, ElementwiseFloat)
{
    END_TEST();
}
VECTORIZATIONTEST(LibTorch, ElementwiseDouble)
{
    END_TEST();
}

#endif  // VECTORIZATION_HAS_LIBTORCH
