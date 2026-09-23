/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * SIMD tests: horizontal reductions (accumulate, hmin, hmax).
 */

#include "VectorizationTest.h"
#if VECTORIZATION_VECTORIZED

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <utility>

#include "backend/simd.h"
#include "common/scalar_helper_functions.h"
#include "common/vectorization_macros.h"

namespace vectorization
{
namespace simd_tests
{

inline constexpr int kRandomTrials = 256;

template <typename value_t, std::size_t N>
void fill_uniform(std::array<value_t, N>& xs, std::mt19937& gen, value_t lo, value_t hi)
{
    std::uniform_real_distribution<value_t> dist(lo, hi);
    for (auto& x : xs)
        x = dist(gen);
}

template <typename value_t>
void test_accumulate(value_t tolerance)
{
    constexpr std::size_t n = simd<value_t>::size;
    using simd_t            = typename simd<value_t>::simd_t;

    std::array<value_t, n> xs{};
    std::mt19937           gen(5489u);
    value_t const scaled_tol = tolerance * static_cast<value_t>(std::max<std::size_t>(4, n));

    for (int trial = 0; trial < kRandomTrials; ++trial)
    {
        fill_uniform(xs, gen, static_cast<value_t>(-6), static_cast<value_t>(6));
        simd_t     a   = simd<value_t>::loadu(xs.data());
        auto const e   = simd<value_t>::accumulate(a);
        value_t    sum = 0;
        for (std::size_t i = 0; i < n; ++i)
            sum += xs[i];
        EXPECT_LE(std::fabs(e - sum), scaled_tol);
    }
}

template <typename value_t>
void test_hmin(value_t tolerance)
{
    constexpr std::size_t n = simd<value_t>::size;
    using simd_t            = typename simd<value_t>::simd_t;

    std::array<value_t, n> xs{};
    std::mt19937           gen(5489u);

    for (int trial = 0; trial < kRandomTrials; ++trial)
    {
        fill_uniform(xs, gen, static_cast<value_t>(-40), static_cast<value_t>(40));
        simd_t     a   = simd<value_t>::loadu(xs.data());
        auto const e   = simd<value_t>::hmin(a);
        value_t    ref = xs[0];
        for (std::size_t i = 1; i < n; ++i)
            ref = std::min(ref, xs[i]);
        EXPECT_LE(std::fabs(e - ref), tolerance);
    }
}

template <typename value_t>
void test_hmax(value_t tolerance)
{
    constexpr std::size_t n = simd<value_t>::size;
    using simd_t            = typename simd<value_t>::simd_t;

    std::array<value_t, n> xs{};
    std::mt19937           gen(5489u);

    for (int trial = 0; trial < kRandomTrials; ++trial)
    {
        fill_uniform(xs, gen, static_cast<value_t>(-40), static_cast<value_t>(40));
        simd_t     a   = simd<value_t>::loadu(xs.data());
        auto const e   = simd<value_t>::hmax(a);
        value_t    ref = xs[0];
        for (std::size_t i = 1; i < n; ++i)
            ref = std::max(ref, xs[i]);
        EXPECT_LE(std::fabs(e - ref), tolerance);
    }
}

template <typename value_t>
void test_all_simd_horizontal(value_t tolerance)
{
    test_accumulate(tolerance);
    test_hmax(tolerance);
    test_hmin(tolerance);
}
}  // namespace simd_tests
}  // namespace vectorization

VECTORIZATIONTEST(Math, SimdH)
{
    using namespace vectorization::simd_tests;
    constexpr float  kSimdTolF = 0.0007f;
    constexpr double kSimdTolD = 5.e-12;
    test_all_simd_horizontal<float>(kSimdTolF);
    test_all_simd_horizontal<double>(kSimdTolD);
    END_TEST();
}
#else
VECTORIZATIONTEST(Math, SimdH)
{
    END_TEST();
}

#endif  // VECTORIZATION_VECTORIZED
