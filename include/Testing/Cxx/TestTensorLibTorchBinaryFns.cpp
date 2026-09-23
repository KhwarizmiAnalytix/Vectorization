/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * 1-D binary math function correctness tests.
 * Covers: pow, element-wise min, element-wise max, fma (a*b+c).
 */

#include "VectorizationTest.h"

#if VECTORIZATION_HAS_LIBTORCH

#include "TestTensorLibTorchHelpers.h"

namespace
{
using namespace libtorch_test;

template <typename T>
void test_libtorch_binary_fns()
{
    using tensor_t                = vectorization::tensor<T>;
    constexpr std::size_t n       = 512 + 7;
    constexpr double      tol     = std::is_same_v<T, float> ? 1e-4 : 1e-12;
    constexpr double      tol_pow = std::is_same_v<T, float> ? 2e-3 : 1e-11;

    auto ax = rand_vec<T>(n, T(-3), T(3), 4);
    auto ay = rand_vec<T>(n, T(-3), T(3), 5);
    auto az = rand_vec<T>(n, T(0.5), T(4), 6);  // positive base for pow

    tensor_t xa(ax.data(), n);
    tensor_t xb(ay.data(), n);
    tensor_t xc(az.data(), n);

    auto ta = to_torch(ax.data(), n);
    auto tb = to_torch(ay.data(), n);
    auto tc = to_torch(az.data(), n);

    {
        tensor_t r(n);
        r = ::pow(xc, xb);
        EXPECT_LT(max_diff(r, torch::pow(tc, tb)), tol_pow) << "pow";
    }
    {
        tensor_t r(n);
        r = ::min(xa, xb);
        EXPECT_LT(max_diff(r, torch::minimum(ta, tb)), tol) << "min";
    }
    {
        tensor_t r(n);
        r = ::max(xa, xb);
        EXPECT_LT(max_diff(r, torch::maximum(ta, tb)), tol) << "max";
    }
    {
        // fma(a, b, c) = a*b + c; torch::addcmul(c, a, b) = c + 1*a*b
        tensor_t r(n);
        r = xa * xb + xc;
        EXPECT_LT(max_diff(r, torch::addcmul(tc, ta, tb)), tol) << "fma (a*b+c)";
    }
}

}  // namespace

VECTORIZATIONTEST(LibTorch, BinaryFnsFloat)
{
    test_libtorch_binary_fns<float>();
    END_TEST();
}

VECTORIZATIONTEST(LibTorch, BinaryFnsDouble)
{
    test_libtorch_binary_fns<double>();
    END_TEST();
}

#else

VECTORIZATIONTEST(LibTorch, BinaryFnsFloat)
{
    END_TEST();
}
VECTORIZATIONTEST(LibTorch, BinaryFnsDouble)
{
    END_TEST();
}

#endif  // VECTORIZATION_HAS_LIBTORCH
