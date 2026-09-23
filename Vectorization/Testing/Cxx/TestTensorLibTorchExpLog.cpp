/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * LibTorch comparison tests: exponential and logarithm
 * (exp, expm1, exp2, log, log1p, log2, log10).
 */

#include "VectorizationTest.h"

#if VECTORIZATION_HAS_LIBTORCH

#include "TestTensorLibTorchHelpers.h"

namespace
{
using namespace libtorch_test;

template <typename T>
void test_libtorch_exp_log()
{
    using tensor_t = vectorization::tensor<T>;

    constexpr std::size_t n         = 512 + 7;
    constexpr double      tol_loose = std::is_same_v<T, float> ? 1e-4 : 1e-11;

    auto a_gen = rand_vec<T>(n, T(-3), T(3), 30);
    auto a_pos = rand_vec<T>(n, T(0.5), T(4), 32);

    tensor_t xa(a_gen.data(), n);
    tensor_t xpos(a_pos.data(), n);

    auto ta   = to_torch(a_gen.data(), n);
    auto tpos = to_torch(a_pos.data(), n);

    // exponential
    {
        tensor_t r(n);
        r = ::exp(xa);
        EXPECT_LT(max_diff(r, torch::exp(ta)), tol_loose) << "exp";
    }
    {
        tensor_t r(n);
        r = ::expm1(xa);
        EXPECT_LT(max_diff(r, torch::expm1(ta)), tol_loose) << "expm1";
    }
    {
        tensor_t r(n);
        r = ::exp2(xa);
        EXPECT_LT(max_diff(r, torch::exp2(ta)), tol_loose) << "exp2";
    }

    // logarithm
    {
        tensor_t r(n);
        r = ::log(xpos);
        EXPECT_LT(max_diff(r, torch::log(tpos)), tol_loose) << "log";
    }
    {
        tensor_t r(n);
        r = ::log1p(xpos);
        EXPECT_LT(max_diff(r, torch::log1p(tpos)), tol_loose) << "log1p";
    }
    {
        tensor_t r(n);
        r = ::log2(xpos);
        EXPECT_LT(max_diff(r, torch::log2(tpos)), tol_loose) << "log2";
    }
    {
        tensor_t r(n);
        r = ::log10(xpos);
        EXPECT_LT(max_diff(r, torch::log10(tpos)), tol_loose) << "log10";
    }
}

}  // namespace

VECTORIZATIONTEST(LibTorch, ExpLogFloat)
{
    test_libtorch_exp_log<float>();
    END_TEST();
}

VECTORIZATIONTEST(LibTorch, ExpLogDouble)
{
    test_libtorch_exp_log<double>();
    END_TEST();
}

#else

VECTORIZATIONTEST(LibTorch, ExpLogFloat)
{
    END_TEST();
}
VECTORIZATIONTEST(LibTorch, ExpLogDouble)
{
    END_TEST();
}

#endif  // VECTORIZATION_HAS_LIBTORCH
