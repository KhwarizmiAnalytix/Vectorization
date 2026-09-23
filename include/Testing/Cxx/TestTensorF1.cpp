/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * Tensor tests: unary element-wise ops vs std:: (one tensor argument).
 * Also tests XSigma tensor ops against Intel MKL VML when available.
 */

#include "VectorizationTest.h"
#if VECTORIZATION_VECTORIZED

#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>

#include "common/vectorization_macros.h"
#include "terminals/tensor.h"

#if VECTORIZATION_HAS_MKL
#include "backend/cpu/mkl/mkl_vml.h"
#endif

namespace vectorization
{
namespace tensor_tests
{

template <typename T>
struct tensor_abs_tol_unary
{
};

template <>
struct tensor_abs_tol_unary<float>
{
    static constexpr float transcendental = 2e-3f;
};

template <>
struct tensor_abs_tol_unary<double>
{
    static constexpr double transcendental = 8e-12;
};

template <typename T>
void expect_tensor_near_elementwise(tensor<T> const& got, std::vector<T> const& ref, T abs_tol)
{
    ASSERT_EQ(got.size(), ref.size());
    double max_d = 0.;
    for (std::size_t i = 0; i < ref.size(); ++i)
    {
        T const d = std::fabs(got.data()[i] - ref[i]);
        max_d     = std::max<double>(max_d, static_cast<double>(d));
    }
    EXPECT_LE(max_d, abs_tol);
}

template <typename T>
void fill_uniform(std::vector<T>& v, std::mt19937& gen, T lo, T hi)
{
    std::uniform_real_distribution<T> dist(lo, hi);
    for (auto& x : v)
        x = dist(gen);
}

template <typename T>
void test_tensor_unary_vs_std()
{
    using tensor_t      = tensor<T>;
    std::size_t const n = (2U << 8U) + 3U;
    std::mt19937      gen(9001u);
    T const           tol = tensor_abs_tol_unary<T>::transcendental;

    std::vector<T> ax(n);
    fill_uniform(ax, gen, static_cast<T>(-1), static_cast<T>(1));
    std::vector<T> asqrt(n);
    fill_uniform(asqrt, gen, static_cast<T>(0), static_cast<T>(4));

    tensor_t ta(ax.data(), n);
    tensor_t tsq(asqrt.data(), n);

    {
        tensor_t out(n);
        out = ::exp(ta);
        std::vector<T> ref(n);
        for (std::size_t i = 0; i < n; ++i)
            ref[i] = std::exp(ax[i]);
        expect_tensor_near_elementwise(out, ref, tol);
    }
    {
        tensor_t out(n);
        out = ::sin(ta);
        std::vector<T> ref(n);
        for (std::size_t i = 0; i < n; ++i)
            ref[i] = std::sin(ax[i]);
        expect_tensor_near_elementwise(out, ref, tol);
    }
    {
        tensor_t out(n);
        out = ::sqrt(tsq);
        std::vector<T> ref(n);
        for (std::size_t i = 0; i < n; ++i)
            ref[i] = std::sqrt(asqrt[i]);
        expect_tensor_near_elementwise(out, ref, tol);
    }
    {
        tensor_t out(n);
        out = ::fabs(ta);
        std::vector<T> ref(n);
        for (std::size_t i = 0; i < n; ++i)
            ref[i] = std::fabs(ax[i]);
        expect_tensor_near_elementwise(out, ref, std::numeric_limits<T>::epsilon() * T(64));
    }

    std::size_t const r = 12;
    std::size_t const c = 24;
    std::vector<T>    a2(r * c);
    fill_uniform(a2, gen, static_cast<T>(-0.5), static_cast<T>(0.5));
    tensor_t t2d(a2.data(), r, c);
    tensor_t out2d(r, c);
    out2d = ::cos(t2d);
    std::vector<T> ref2(r * c);
    for (std::size_t i = 0; i < r * c; ++i)
        ref2[i] = std::cos(a2[i]);
    expect_tensor_near_elementwise(out2d, ref2, tol);
}

#if VECTORIZATION_HAS_MKL

template <typename T>
void test_tensor_unary_vs_mkl_vml()
{
    using tensor_t      = tensor<T>;
    std::size_t const n = (2U << 8U) + 3U;
    std::mt19937      gen(9001u);
    T const           tol = tensor_abs_tol_unary<T>::transcendental;

    std::vector<T> ax(n);
    fill_uniform(ax, gen, static_cast<T>(-1), static_cast<T>(1));
    std::vector<T> asqrt(n);
    fill_uniform(asqrt, gen, static_cast<T>(0.1), static_cast<T>(4));

    tensor_t ta(ax.data(), n);
    tensor_t tsq(asqrt.data(), n);

    // exp
    {
        tensor_t out(n);
        out = ::exp(ta);
        std::vector<T> ref(n);
        mkl_vml::exp(static_cast<MKL_INT>(n), ax.data(), ref.data());
        expect_tensor_near_elementwise(out, ref, tol);
    }
    // log
    {
        tensor_t out(n);
        out = ::log(tsq);
        std::vector<T> ref(n);
        mkl_vml::log(static_cast<MKL_INT>(n), asqrt.data(), ref.data());
        expect_tensor_near_elementwise(out, ref, tol);
    }
    // sqrt
    {
        tensor_t out(n);
        out = ::sqrt(tsq);
        std::vector<T> ref(n);
        mkl_vml::sqrt(static_cast<MKL_INT>(n), asqrt.data(), ref.data());
        expect_tensor_near_elementwise(out, ref, tol);
    }
    // sin
    {
        tensor_t out(n);
        out = ::sin(ta);
        std::vector<T> ref(n);
        mkl_vml::sin(static_cast<MKL_INT>(n), ax.data(), ref.data());
        expect_tensor_near_elementwise(out, ref, tol);
    }
    // cos
    {
        tensor_t out(n);
        out = ::cos(ta);
        std::vector<T> ref(n);
        mkl_vml::cos(static_cast<MKL_INT>(n), ax.data(), ref.data());
        expect_tensor_near_elementwise(out, ref, tol);
    }
    // tanh
    {
        tensor_t out(n);
        out = ::tanh(ta);
        std::vector<T> ref(n);
        mkl_vml::tanh(static_cast<MKL_INT>(n), ax.data(), ref.data());
        expect_tensor_near_elementwise(out, ref, tol);
    }
    // fabs
    {
        tensor_t out(n);
        out = ::fabs(ta);
        std::vector<T> ref(n);
        mkl_vml::fabs(static_cast<MKL_INT>(n), ax.data(), ref.data());
        expect_tensor_near_elementwise(out, ref, std::numeric_limits<T>::epsilon() * T(64));
    }
}

#endif  // VECTORIZATION_HAS_MKL

}  // namespace tensor_tests
}  // namespace vectorization

VECTORIZATIONTEST(Math, TensorUnary)
{
    using namespace vectorization::tensor_tests;
    test_tensor_unary_vs_std<float>();
    test_tensor_unary_vs_std<double>();
    END_TEST();
}

#if VECTORIZATION_HAS_MKL
VECTORIZATIONTEST(Math, TensorUnaryVsMklVml)
{
    using namespace vectorization::tensor_tests;
    test_tensor_unary_vs_mkl_vml<float>();
    test_tensor_unary_vs_mkl_vml<double>();
    END_TEST();
}
#else
VECTORIZATIONTEST(Math, TensorUnaryVsMklVml)
{
    END_TEST();
}
#endif

#else
VECTORIZATIONTEST(Math, TensorUnary)
{
    END_TEST();
}
VECTORIZATIONTEST(Math, TensorUnaryVsMklVml)
{
    END_TEST();
}

#endif  // VECTORIZATION_VECTORIZED
