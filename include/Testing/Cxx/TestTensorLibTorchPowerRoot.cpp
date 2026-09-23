/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * LibTorch comparison tests: power / root (sqrt, sqr, invsqrt, cbrt).
 */

#include "VectorizationTest.h"

#if VECTORIZATION_HAS_LIBTORCH

#include "TestTensorLibTorchHelpers.h"

namespace
{
using namespace libtorch_test;

template <typename T>
void test_libtorch_power_root()
{
    using tensor_t = vectorization::tensor<T>;

    constexpr std::size_t n         = 512 + 7;
    constexpr double      tol       = std::is_same_v<T, float> ? 1e-5 : 1e-13;
    constexpr double      tol_loose = std::is_same_v<T, float> ? 1e-4 : 1e-11;

    auto a_gen = rand_vec<T>(n, T(-3), T(3), 30);
    auto a_pos = rand_vec<T>(n, T(0.5), T(4), 32);

    tensor_t xa(a_gen.data(), n);
    tensor_t xpos(a_pos.data(), n);

    auto ta   = to_torch(a_gen.data(), n);
    auto tpos = to_torch(a_pos.data(), n);

    {
        tensor_t r(n);
        r = ::sqrt(xpos);
        EXPECT_LT(max_diff(r, torch::sqrt(tpos)), tol) << "sqrt";
    }
    {
        tensor_t r(n);
        r = ::sqr(xa);
        EXPECT_LT(max_diff(r, ta * ta), tol) << "sqr";
    }
    {
        tensor_t r(n);
        r = ::invsqrt(xpos);
        EXPECT_LT(max_diff(r, T(1) / torch::sqrt(tpos)), 20 * tol_loose) << "invsqrt";
    }
    {
        tensor_t r(n);
        r = ::cbrt(xpos);
        EXPECT_LT(max_diff(r, torch::pow(tpos, T(1) / T(3))), tol_loose) << "cbrt";
    }
}

}  // namespace

VECTORIZATIONTEST(LibTorch, PowerRootFloat)
{
    test_libtorch_power_root<float>();
    END_TEST();
}

VECTORIZATIONTEST(LibTorch, PowerRootDouble)
{
    test_libtorch_power_root<double>();
    END_TEST();
}

#else

VECTORIZATIONTEST(LibTorch, PowerRootFloat)
{
    END_TEST();
}
VECTORIZATIONTEST(LibTorch, PowerRootDouble)
{
    END_TEST();
}

#endif  // VECTORIZATION_HAS_LIBTORCH
