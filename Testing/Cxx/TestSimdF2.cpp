/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * SIMD tests: two-argument (binary) packet operations vs scalar std/math.
 */

#include "VectorizationTest.h"
#if VECTORIZATION_VECTORIZED

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <random>
#include <type_traits>
#include <utility>

#include "backend/simd.h"
#include "common/scalar_helper_functions.h"
#include "common/vectorization_macros.h"

namespace vectorization
{
namespace simd_tests
{

#define XSIGMA_SIMD_BINARY_TAG(Scalar, name) \
    ((std::is_same_v<Scalar, float>) ? (#name ".f32") : (#name ".f64"))

inline constexpr int kRandomTrials = 256;

template <typename value_t>
double scalar_as_double(value_t x)
{
    return static_cast<double>(x);
}

template <typename value_t, std::size_t N>
void fill_uniform(std::array<value_t, N>& xs, std::mt19937& gen, value_t lo, value_t hi)
{
    std::uniform_real_distribution<value_t> dist(lo, hi);
    for (auto& x : xs)
        x = dist(gen);
}

template <typename value_t, std::size_t N>
void fill_binary_div_safe(std::array<value_t, N>& xs, std::array<value_t, N>& ys, std::mt19937& gen)
{
    std::uniform_real_distribution<value_t> wide(
        static_cast<value_t>(-50), static_cast<value_t>(50));
    // Lower bound on |y| for stable division. Must be well below half the width of `wide`
    // so rejection sampling terminates (epsilon * 1e9 would exceed 50 for float/double).
    value_t const guard = std::max(
        static_cast<value_t>(1e-3),
        std::numeric_limits<value_t>::epsilon() * static_cast<value_t>(1024));
    for (std::size_t i = 0; i < N; ++i)
    {
        xs[i]     = wide(gen);
        value_t y = wide(gen);
        while (std::fabs(y) < guard)
            y = wide(gen);
        ys[i] = y;
    }
}

template <typename value_t, typename FillXY, typename SimdOp, typename RefOp>
void binary_random_vs_std(
    char const* case_tag, value_t tolerance, FillXY&& fill_xy, SimdOp&& simd_op, RefOp&& ref_op)
{
    constexpr std::size_t n = simd<value_t>::size;
    using simd_t            = typename simd<value_t>::simd_t;

    std::array<value_t, n> xs{};
    std::array<value_t, n> ys{};
    std::array<value_t, n> out{};
    std::mt19937           gen(5489u);

    value_t err_max = 0;
    for (int trial = 0; trial < kRandomTrials; ++trial)
    {
        SCOPED_TRACE(::testing::Message() << case_tag << " trial=" << trial << " (binary vs std)");
        fill_xy(xs, ys, gen);
        simd_t a = simd<value_t>::loadu(xs.data());
        simd_t b = simd<value_t>::loadu(ys.data());
        simd_t c = simd<value_t>::setzero();
        simd_op(a, b, c);
        simd<value_t>::storeu(c, out.data());
        for (std::size_t i = 0; i < n; ++i)
        {
            value_t const ref = ref_op(xs[i], ys[i]);
            value_t const err = std::fabs(out[i] - ref);
            err_max           = err_max > err ? err_max : err;
#if 0
            EXPECT_LE(err, tolerance)
                << "\n  case: " << case_tag << "  lane=" << i << "  trial=" << trial
                << "\n  x     = " << std::setprecision(21) << scalar_as_double(xs[i])
                << "\n  y     = " << std::setprecision(21) << scalar_as_double(ys[i])
                << "\n  simd  = " << std::setprecision(21) << scalar_as_double(out[i])
                << "\n  ref   = " << std::setprecision(21) << scalar_as_double(ref)
                << "\n  |err| = " << std::setprecision(21) << scalar_as_double(err)
                << "   tol = " << std::setprecision(21) << scalar_as_double(tolerance);
#endif
        }
    }
    EXPECT_LE(err_max, tolerance);
}

template <typename value_t, typename SimdBinary, typename RefBinary>
void binary_uniform_boxes_vs_std(
    char const*  case_tag,
    value_t      tolerance,
    value_t      lo1,
    value_t      hi1,
    value_t      lo2,
    value_t      hi2,
    SimdBinary&& simd_op,
    RefBinary&&  ref_op)
{
    binary_random_vs_std<value_t>(
        case_tag,
        tolerance,
        [=](std::array<value_t, simd<value_t>::size>& xs,
            std::array<value_t, simd<value_t>::size>& ys,
            std::mt19937&                             gen)
        {
            fill_uniform(xs, gen, lo1, hi1);
            fill_uniform(ys, gen, lo2, hi2);
        },
        std::forward<SimdBinary>(simd_op),
        std::forward<RefBinary>(ref_op));
}

template <typename value_t>
void test_add(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    binary_uniform_boxes_vs_std<value_t>(
        XSIGMA_SIMD_BINARY_TAG(value_t, add),
        tolerance,
        static_cast<value_t>(-40),
        static_cast<value_t>(40),
        static_cast<value_t>(-40),
        static_cast<value_t>(40),
        [](simd_t a, simd_t b, simd_t& c) { c = simd<value_t>::add(a, b); },
        [](value_t x, value_t y) { return std::add(x, y); });
}

template <typename value_t>
void test_sub(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    binary_uniform_boxes_vs_std<value_t>(
        XSIGMA_SIMD_BINARY_TAG(value_t, sub),
        tolerance,
        static_cast<value_t>(-40),
        static_cast<value_t>(40),
        static_cast<value_t>(-40),
        static_cast<value_t>(40),
        [](simd_t a, simd_t b, simd_t& c) { c = simd<value_t>::sub(a, b); },
        [](value_t x, value_t y) { return std::sub(x, y); });
}

template <typename value_t>
void test_mul(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    binary_uniform_boxes_vs_std<value_t>(
        XSIGMA_SIMD_BINARY_TAG(value_t, mul),
        tolerance,
        static_cast<value_t>(-25),
        static_cast<value_t>(25),
        static_cast<value_t>(-25),
        static_cast<value_t>(25),
        [](simd_t a, simd_t b, simd_t& c) { c = simd<value_t>::mul(a, b); },
        [](value_t x, value_t y) { return std::mul(x, y); });
}

template <typename value_t>
void test_min(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    binary_uniform_boxes_vs_std<value_t>(
        XSIGMA_SIMD_BINARY_TAG(value_t, min),
        tolerance,
        static_cast<value_t>(-40),
        static_cast<value_t>(40),
        static_cast<value_t>(-40),
        static_cast<value_t>(40),
        [](simd_t a, simd_t b, simd_t& c) { c = simd<value_t>::min(a, b); },
        [](value_t x, value_t y) { return std::min(x, y); });
}

template <typename value_t>
void test_max(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    binary_uniform_boxes_vs_std<value_t>(
        XSIGMA_SIMD_BINARY_TAG(value_t, max),
        tolerance,
        static_cast<value_t>(-40),
        static_cast<value_t>(40),
        static_cast<value_t>(-40),
        static_cast<value_t>(40),
        [](simd_t a, simd_t b, simd_t& c) { c = simd<value_t>::max(a, b); },
        [](value_t x, value_t y) { return std::max(x, y); });
}

template <typename value_t>
void test_hypot(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    binary_uniform_boxes_vs_std<value_t>(
        XSIGMA_SIMD_BINARY_TAG(value_t, hypot),
        tolerance,
        static_cast<value_t>(-80),
        static_cast<value_t>(80),
        static_cast<value_t>(-80),
        static_cast<value_t>(80),
        [](simd_t a, simd_t b, simd_t& c) { c = simd<value_t>::hypot(a, b); },
        [](value_t x, value_t y) { return std::hypot(x, y); });
}

template <typename value_t>
void test_signcopy(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    binary_uniform_boxes_vs_std<value_t>(
        XSIGMA_SIMD_BINARY_TAG(value_t, signcopy),
        tolerance,
        static_cast<value_t>(-80),
        static_cast<value_t>(80),
        static_cast<value_t>(-80),
        static_cast<value_t>(80),
        [](simd_t a, simd_t b, simd_t& c) { c = simd<value_t>::signcopy(a, b); },
        [](value_t x, value_t y) { return std::signcopy(x, y); });
}

template <typename value_t>
void test_div(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    // Division stacks ULP error vs scalar; use slightly looser bound than add/mul.
    value_t const div_tol = tolerance * static_cast<value_t>(2);
    binary_random_vs_std<value_t>(
        XSIGMA_SIMD_BINARY_TAG(value_t, div),
        div_tol,
        [](std::array<value_t, simd<value_t>::size>& xs,
           std::array<value_t, simd<value_t>::size>& ys,
           std::mt19937&                             gen) { fill_binary_div_safe(xs, ys, gen); },
        [](simd_t a, simd_t b, simd_t& c) { c = simd<value_t>::div(a, b); },
        [](value_t x, value_t y) { return std::div(x, y); });
}

template <typename value_t>
void test_pow(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    // Packet pow often differs from std::pow by ~1 ULP in the last mantissa bits.
    value_t const pow_tol = tolerance * static_cast<value_t>(2);
    binary_random_vs_std<value_t>(
        XSIGMA_SIMD_BINARY_TAG(value_t, pow),
        pow_tol,
        [](std::array<value_t, simd<value_t>::size>& xs,
           std::array<value_t, simd<value_t>::size>& ys,
           std::mt19937&                             gen)
        {
            fill_uniform(xs, gen, static_cast<value_t>(0.05), static_cast<value_t>(12));
            fill_uniform(ys, gen, static_cast<value_t>(-4), static_cast<value_t>(4));
        },
        [](simd_t a, simd_t b, simd_t& c) { c = simd<value_t>::pow(a, b); },
        [](value_t x, value_t y) { return std::pow(x, y); });
}

template <typename value_t>
void test_all_simd_binary(value_t tolerance)
{
    test_add(tolerance);
    test_sub(tolerance);
    test_mul(tolerance);
    test_div(tolerance);
    test_min(tolerance);
    test_max(tolerance);
    test_pow(tolerance);
    test_hypot(tolerance);
    test_signcopy(tolerance);
}

#undef XSIGMA_SIMD_BINARY_TAG

}  // namespace simd_tests
}  // namespace vectorization

VECTORIZATIONTEST(Math, SimdF2)
{
    using namespace vectorization::simd_tests;
    constexpr float  kSimdTolF = 0.0007f;
    constexpr double kSimdTolD = 5.e-12;
    test_all_simd_binary<float>(kSimdTolF);
    test_all_simd_binary<double>(kSimdTolD);
    END_TEST();
}
#else
VECTORIZATIONTEST(Math, SimdF2)
{
    END_TEST();
}

#endif  // VECTORIZATION_VECTORIZED
