/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * Tensor tests: binary element-wise ops vs std:: (two tensor arguments).
 */

#include "VectorizationTest.h"
#if VECTORIZATION_VECTORIZED

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "common/vectorization_macros.h"
#include "terminals/tensor.h"

namespace vectorization
{
namespace tensor_tests
{

template <typename T>
struct tensor_abs_tol_binary
{
};

template <>
struct tensor_abs_tol_binary<float>
{
    static constexpr float general = 2e-4f;
    static constexpr float pow_    = 2e-3f;
};

template <>
struct tensor_abs_tol_binary<double>
{
    static constexpr double general = 5e-13;
    static constexpr double pow_    = 8e-12;
};

template <typename T>
T expect_tensor_near_elementwise(tensor<T> const& got, std::vector<T> const& ref)
{
    T max_d = 0;
    for (std::size_t i = 0; i < ref.size(); ++i)
    {
        T const d = std::fabs(got.data()[i] - ref[i]);
        max_d     = max_d > d ? max_d : d;
    }
    return max_d;
}

template <typename T>
void fill_uniform(std::vector<T>& v, std::mt19937& gen, T lo, T hi)
{
    std::uniform_real_distribution<T> dist(lo, hi);
    for (auto& x : v)
        x = dist(gen);
}

template <typename value_t>
double scalar_as_double(value_t x)
{
    return static_cast<double>(x);
}

template <typename T>
void test_tensor_binary_vs_std()
{
    using tensor_t      = tensor<T>;
    std::size_t const n = (2U << 8U) + 5U;
    std::mt19937      gen(12011u);
    T const           tol_gen = tensor_abs_tol_binary<T>::general;
    T const           tol_pow = tensor_abs_tol_binary<T>::pow_;

    std::vector<T> ax(n);
    std::vector<T> ay(n);
    fill_uniform(ax, gen, static_cast<T>(-8), static_cast<T>(8));
    fill_uniform(ay, gen, static_cast<T>(-8), static_cast<T>(8));

    tensor_t a(ax.data(), n);
    tensor_t b(ay.data(), n);

    {
        tensor_t out(n);
        out = a + b;
        std::vector<T> ref(n);
        for (std::size_t i = 0; i < n; ++i)
            ref[i] = ax[i] + ay[i];
        EXPECT_LE(expect_tensor_near_elementwise(out, ref), tol_gen);
    }
    {
        tensor_t out(n);
        out = a * b;
        std::vector<T> ref(n);
        for (std::size_t i = 0; i < n; ++i)
            ref[i] = ax[i] * ay[i];
        EXPECT_LE(expect_tensor_near_elementwise(out, ref), tol_gen);
    }
    {
        tensor_t out(n);
        out = min(a, b);
        std::vector<T> ref(n);
        for (std::size_t i = 0; i < n; ++i)
            ref[i] = std::min(ax[i], ay[i]);
        EXPECT_LE(
            expect_tensor_near_elementwise(out, ref), std::numeric_limits<T>::epsilon() * T(64));
    }
    {
        tensor_t out(n);
        out = max(a, b);
        std::vector<T> ref(n);
        for (std::size_t i = 0; i < n; ++i)
            ref[i] = std::max(ax[i], ay[i]);
        EXPECT_LE(
            expect_tensor_near_elementwise(out, ref), std::numeric_limits<T>::epsilon() * T(64));
    }
    {
        std::vector<T> px(n);
        std::vector<T> py(n);
        fill_uniform(px, gen, static_cast<T>(0.1), static_cast<T>(3.0));
        fill_uniform(py, gen, static_cast<T>(-4.0), static_cast<T>(4.0));
        tensor_t pa(px.data(), n);
        tensor_t pb(py.data(), n);
        tensor_t out(n);
        out = pow(pa, pb);

        std::vector<T> ref(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            ref[i] = std::pow(px[i], py[i]);
        }
        EXPECT_LE(expect_tensor_near_elementwise(out, ref), tol_pow);
    }
}

}  // namespace tensor_tests
}  // namespace vectorization

VECTORIZATIONTEST(Math, TensorBinary)
{
    using namespace vectorization::tensor_tests;
    test_tensor_binary_vs_std<float>();
    test_tensor_binary_vs_std<double>();
    END_TEST();
}
#else
VECTORIZATIONTEST(Math, TensorBinary)
{
    END_TEST();
}

#endif  // VECTORIZATION_VECTORIZED
