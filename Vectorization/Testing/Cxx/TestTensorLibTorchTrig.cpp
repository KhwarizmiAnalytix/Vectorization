/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * LibTorch comparison tests: trigonometry (sin, cos, tan, asin, acos, atan).
 */

#include "VectorizationTest.h"

#if VECTORIZATION_HAS_LIBTORCH

#include "TestTensorLibTorchHelpers.h"

namespace
{
using namespace libtorch_test;

template <typename T>
void test_libtorch_trig()
{
    using tensor_t = vectorization::tensor<T>;

    constexpr std::size_t n         = 512 + 7;
    constexpr double      tol_loose = std::is_same_v<T, float> ? 1e-4 : 1e-11;

    auto a_gen  = rand_vec<T>(n, T(-3), T(3), 30);
    auto a_unit = rand_vec<T>(n, T(-0.9), T(0.9), 33);
    auto a_sm   = rand_vec<T>(n, T(-0.5), T(0.5), 35);

    tensor_t xa(a_gen.data(), n);
    tensor_t xunit(a_unit.data(), n);
    tensor_t xsm(a_sm.data(), n);

    auto ta    = to_torch(a_gen.data(), n);
    auto tunit = to_torch(a_unit.data(), n);
    auto tsm   = to_torch(a_sm.data(), n);

    {
        tensor_t r(n);
        r = ::sin(xa);
        EXPECT_LT(max_diff(r, torch::sin(ta)), tol_loose) << "sin";
    }
    {
        tensor_t r(n);
        r = ::cos(xa);
        EXPECT_LT(max_diff(r, torch::cos(ta)), tol_loose) << "cos";
    }
    {
        tensor_t r(n);
        r = ::tan(xsm);
        EXPECT_LT(max_diff(r, torch::tan(tsm)), tol_loose) << "tan";
    }
    {
        tensor_t r(n);
        r = ::asin(xunit);
        EXPECT_LT(max_diff(r, torch::asin(tunit)), tol_loose) << "asin";
    }
    {
        tensor_t r(n);
        r = ::acos(xunit);
        EXPECT_LT(max_diff(r, torch::acos(tunit)), tol_loose) << "acos";
    }
    {
        tensor_t r(n);
        r = ::atan(xa);
        EXPECT_LT(max_diff(r, torch::atan(ta)), tol_loose) << "atan";
    }
}

}  // namespace

VECTORIZATIONTEST(LibTorch, TrigFloat)
{
    test_libtorch_trig<float>();
    END_TEST();
}

VECTORIZATIONTEST(LibTorch, TrigDouble)
{
    test_libtorch_trig<double>();
    END_TEST();
}

#else

VECTORIZATIONTEST(LibTorch, TrigFloat)
{
    END_TEST();
}
VECTORIZATIONTEST(LibTorch, TrigDouble)
{
    END_TEST();
}

#endif  // VECTORIZATION_HAS_LIBTORCH
