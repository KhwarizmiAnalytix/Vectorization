/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * LibTorch comparison tests: hyperbolic functions
 * (sinh, cosh, tanh, asinh, acosh, atanh).
 */

#include "VectorizationTest.h"

#if VECTORIZATION_HAS_LIBTORCH

#include "TestTensorLibTorchHelpers.h"

namespace
{
using namespace libtorch_test;

template <typename T>
void test_libtorch_hyperbolic()
{
    using tensor_t = vectorization::tensor<T>;

    constexpr std::size_t n         = 512 + 7;
    constexpr double      tol_loose = std::is_same_v<T, float> ? 1e-4 : 1e-11;

    auto a_gen  = rand_vec<T>(n, T(-3), T(3), 30);
    auto a_unit = rand_vec<T>(n, T(-0.9), T(0.9), 33);
    auto a_ge1  = rand_vec<T>(n, T(1.0), T(4), 34);
    auto a_sm   = rand_vec<T>(n, T(-0.5), T(0.5), 35);

    tensor_t xa(a_gen.data(), n);
    tensor_t xunit(a_unit.data(), n);
    tensor_t xge1(a_ge1.data(), n);
    tensor_t xsm(a_sm.data(), n);

    auto ta    = to_torch(a_gen.data(), n);
    auto tunit = to_torch(a_unit.data(), n);
    auto tge1  = to_torch(a_ge1.data(), n);
    auto tsm   = to_torch(a_sm.data(), n);

    {
        tensor_t r(n);
        r = ::sinh(xsm);
        EXPECT_LT(max_diff(r, torch::sinh(tsm)), tol_loose) << "sinh";
    }
    {
        tensor_t r(n);
        r = ::cosh(xsm);
        EXPECT_LT(max_diff(r, torch::cosh(tsm)), tol_loose) << "cosh";
    }
    {
        tensor_t r(n);
        r = ::tanh(xa);
        EXPECT_LT(max_diff(r, torch::tanh(ta)), tol_loose) << "tanh";
    }
    {
        tensor_t r(n);
        r = ::asinh(xa);
        EXPECT_LT(max_diff(r, torch::asinh(ta)), tol_loose) << "asinh";
    }
    {
        tensor_t r(n);
        r = ::acosh(xge1);
        EXPECT_LT(max_diff(r, torch::acosh(tge1)), tol_loose) << "acosh";
    }
    {
        tensor_t r(n);
        r = ::atanh(xunit);
        EXPECT_LT(max_diff(r, torch::atanh(tunit)), tol_loose) << "atanh";
    }
}

}  // namespace

VECTORIZATIONTEST(LibTorch, HyperbolicFloat)
{
    test_libtorch_hyperbolic<float>();
    END_TEST();
}

VECTORIZATIONTEST(LibTorch, HyperbolicDouble)
{
    test_libtorch_hyperbolic<double>();
    END_TEST();
}

#else

VECTORIZATIONTEST(LibTorch, HyperbolicFloat)
{
    END_TEST();
}
VECTORIZATIONTEST(LibTorch, HyperbolicDouble)
{
    END_TEST();
}

#endif  // VECTORIZATION_HAS_LIBTORCH
