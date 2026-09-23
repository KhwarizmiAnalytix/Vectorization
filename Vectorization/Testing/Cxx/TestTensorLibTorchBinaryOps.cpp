/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * LibTorch comparison tests: binary arithmetic, binary math functions,
 * compound assignment, and fma.
 * Covers: +, -, *, /, +=, -=, *=, /=, pow, hypot, min, max, copysign, fma.
 */

#include "VectorizationTest.h"

#if VECTORIZATION_HAS_LIBTORCH

#include "TestTensorLibTorchHelpers.h"

namespace
{
using namespace libtorch_test;

template <typename T>
void test_libtorch_binary_ops()
{
    using tensor_t = vectorization::tensor<T>;

    constexpr std::size_t n         = 512 + 7;
    constexpr double      tol       = std::is_same_v<T, float> ? 1e-5 : 1e-13;
    constexpr double      tol_loose = std::is_same_v<T, float> ? 1e-4 : 1e-11;
    constexpr double      tol_pow   = std::is_same_v<T, float> ? 2e-3 : 1e-11;

    auto a_gen  = rand_vec<T>(n, T(-3), T(3), 30);
    auto a_gen2 = rand_vec<T>(n, T(-3), T(3), 31);
    auto a_pos  = rand_vec<T>(n, T(0.5), T(4), 32);

    tensor_t xa(a_gen.data(), n);
    tensor_t xb(a_gen2.data(), n);
    tensor_t xpos(a_pos.data(), n);

    auto ta   = to_torch(a_gen.data(), n);
    auto tb   = to_torch(a_gen2.data(), n);
    auto tpos = to_torch(a_pos.data(), n);

    // binary arithmetic
    {
        tensor_t r(n);
        r = xa + xb;
        EXPECT_LT(max_diff(r, ta + tb), tol) << "+";
    }
    {
        tensor_t r(n);
        r = xa - xb;
        EXPECT_LT(max_diff(r, ta - tb), tol) << "-";
    }
    {
        tensor_t r(n);
        r = xa * xb;
        EXPECT_LT(max_diff(r, ta * tb), tol) << "*";
    }
    {
        tensor_t r(n);
        r = xa / xpos;
        EXPECT_LT(max_diff(r, ta / tpos), tol) << "/";
    }

    // binary math functions
    {
        tensor_t r(n);
        r = ::pow(xpos, xb);
        EXPECT_LT(max_diff(r, torch::pow(tpos, tb)), tol_pow) << "pow";
    }
    {
        tensor_t r(n);
        r = ::hypot(xa, xb);
        EXPECT_LT(max_diff(r, torch::hypot(ta, tb)), tol) << "hypot";
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
        tensor_t r(n);
        r = ::copysign(xpos, xb);
        EXPECT_LT(max_diff(r, torch::copysign(tpos, tb)), tol) << "copysign";
    }

    // compound assignment (each uses an independent copy of a_gen)
    {
        auto     t = a_gen;
        tensor_t r(t.data(), n);
        r += xb;
        EXPECT_LT(max_diff(r, ta + tb), tol) << "+=";
    }
    {
        auto     t = a_gen;
        tensor_t r(t.data(), n);
        r -= xb;
        EXPECT_LT(max_diff(r, ta - tb), tol) << "-=";
    }
    {
        auto     t = a_gen;
        tensor_t r(t.data(), n);
        r *= xb;
        EXPECT_LT(max_diff(r, ta * tb), tol) << "*=";
    }
    {
        auto     t = a_gen;
        tensor_t r(t.data(), n);
        r /= xpos;
        EXPECT_LT(max_diff(r, ta / tpos), tol) << "/=";
    }

    // fma(a, b, c) = a*b + c; torch::addcmul(c, a, b) = c + 1*a*b
    {
        tensor_t r(n);
        r = ::fma(xa, xb, xpos);
        EXPECT_LT(max_diff(r, torch::addcmul(tpos, ta, tb)), tol_loose) << "fma";
    }
}

}  // namespace

VECTORIZATIONTEST(LibTorch, BinaryOpsFloat)
{
    test_libtorch_binary_ops<float>();
    END_TEST();
}

VECTORIZATIONTEST(LibTorch, BinaryOpsDouble)
{
    test_libtorch_binary_ops<double>();
    END_TEST();
}

#else

VECTORIZATIONTEST(LibTorch, BinaryOpsFloat)
{
    END_TEST();
}
VECTORIZATIONTEST(LibTorch, BinaryOpsDouble)
{
    END_TEST();
}

#endif  // VECTORIZATION_HAS_LIBTORCH
