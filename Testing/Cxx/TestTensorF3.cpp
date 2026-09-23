/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * Tensor tests: ternary element-wise ops vs std:: (three tensor arguments; fma).
 */

#include "VectorizationTest.h"
#if VECTORIZATION_VECTORIZED

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include "common/vectorization_macros.h"
#include "terminals/tensor.h"

namespace vectorization
{
namespace tensor_tests
{

template <typename T>
struct tensor_abs_tol_ternary
{
};

template <>
struct tensor_abs_tol_ternary<float>
{
    static constexpr float fma_tol = 1.2e-3f;
};

template <>
struct tensor_abs_tol_ternary<double>
{
    static constexpr double fma_tol = 8e-12;
};

template <typename T>
void expect_tensor_near_elementwise(tensor<T> const& got, std::vector<T> const& ref, T abs_tol)
{
    ASSERT_EQ(got.size(), ref.size());
    for (std::size_t i = 0; i < ref.size(); ++i)
    {
        T const d = std::fabs(got.data()[i] - ref[i]);
        EXPECT_LE(d, abs_tol) << "i=" << i << " got=" << got.data()[i] << " ref=" << ref[i];
    }
}

template <typename T>
void fill_uniform(std::vector<T>& v, std::mt19937& gen, T lo, T hi)
{
    std::uniform_real_distribution<T> dist(lo, hi);
    for (auto& x : v)
        x = dist(gen);
}

template <typename T>
void test_tensor_ternary_vs_std()
{
    using tensor_t      = tensor<T>;
    std::size_t const n = (2U << 8U) + 7U;
    std::mt19937      gen(13021u);
    T const           tol = tensor_abs_tol_ternary<T>::fma_tol;

    std::vector<T> ax(n);
    std::vector<T> ay(n);
    std::vector<T> az(n);
    fill_uniform(ax, gen, static_cast<T>(-2), static_cast<T>(2));
    fill_uniform(ay, gen, static_cast<T>(-2), static_cast<T>(2));
    fill_uniform(az, gen, static_cast<T>(-2), static_cast<T>(2));

    tensor_t a(ax.data(), n);
    tensor_t b(ay.data(), n);
    tensor_t c(az.data(), n);

    tensor_t out(n);
    out = ::fma(a, b, c);

    std::vector<T> ref(n);
    for (std::size_t i = 0; i < n; ++i)
        ref[i] = std::fma(ax[i], ay[i], az[i]);

    expect_tensor_near_elementwise(out, ref, tol);
}

}  // namespace tensor_tests
}  // namespace vectorization

VECTORIZATIONTEST(Math, TensorTernary)
{
    using namespace vectorization::tensor_tests;
    test_tensor_ternary_vs_std<float>();
    test_tensor_ternary_vs_std<double>();
    END_TEST();
}
#else
VECTORIZATIONTEST(Math, TensorTernary)
{
    END_TEST();
}

#endif  // VECTORIZATION_VECTORIZED
