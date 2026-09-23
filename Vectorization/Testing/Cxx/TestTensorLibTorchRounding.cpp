/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * LibTorch comparison tests: abs / neg / rounding (fabs, neg, floor, ceil, trunc).
 */

#include "VectorizationTest.h"

#if VECTORIZATION_HAS_LIBTORCH

#include "TestTensorLibTorchHelpers.h"

namespace
{
using namespace libtorch_test;

template <typename T>
void test_libtorch_rounding()
{
    using tensor_t = vectorization::tensor<T>;

    constexpr std::size_t n   = 512 + 7;
    constexpr double      tol = std::is_same_v<T, float> ? 1e-5 : 1e-13;

    auto a_gen = rand_vec<T>(n, T(-3), T(3), 30);

    tensor_t xa(a_gen.data(), n);
    auto     ta = to_torch(a_gen.data(), n);

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
    {
        tensor_t r(n);
        r = ::floor(xa);
        EXPECT_LT(max_diff(r, torch::floor(ta)), tol) << "floor";
    }
    {
        tensor_t r(n);
        r = ::ceil(xa);
        EXPECT_LT(max_diff(r, torch::ceil(ta)), tol) << "ceil";
    }
    {
        tensor_t r(n);
        r = ::trunc(xa);
        EXPECT_LT(max_diff(r, torch::trunc(ta)), tol) << "trunc";
    }
}

}  // namespace

VECTORIZATIONTEST(LibTorch, RoundingFloat)
{
    std::cout << "\n=== Build Config ===" << std::endl;
    std::cout << torch::show_config() << std::endl;
    test_libtorch_rounding<float>();
    END_TEST();
}

VECTORIZATIONTEST(LibTorch, RoundingDouble)
{
    test_libtorch_rounding<double>();
    END_TEST();
}

#else

VECTORIZATIONTEST(LibTorch, RoundingFloat)
{
    END_TEST();
}
VECTORIZATIONTEST(LibTorch, RoundingDouble)
{
    END_TEST();
}

#endif  // VECTORIZATION_HAS_LIBTORCH
