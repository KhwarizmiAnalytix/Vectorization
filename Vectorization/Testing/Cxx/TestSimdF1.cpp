/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * SIMD tests: unary ops that do not rely on Intel SVML (sqrt, rounding, invsqrt NR path, etc.).
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

#define XSIGMA_SIMD_UNARY_TAG(Scalar, name) \
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

template <typename value_t, typename FillXs, typename SimdOp, typename RefOp>
void unary_random_vs_std(
    char const* case_tag, value_t tolerance, FillXs&& fill_xs, SimdOp&& simd_op, RefOp&& ref_op)
{
    constexpr std::size_t n = simd<value_t>::size;
    using simd_t            = typename simd<value_t>::simd_t;

    std::array<value_t, n> xs{};
    std::array<value_t, n> out{};
    std::mt19937           gen(5489u);
    double                 max_d = 0.;
    for (int trial = 0; trial < kRandomTrials; ++trial)
    {
        SCOPED_TRACE(
            ::testing::Message() << "failing_function=" << case_tag << " trial=" << trial
                                 << " (unary vs std)");
        fill_xs(xs, gen);
        simd_t a = simd<value_t>::loadu(xs.data());
        simd_t c = simd<value_t>::setzero();
        simd_op(a, c);
        simd<value_t>::storeu(c, out.data());
        for (std::size_t i = 0; i < n; ++i)
        {
            value_t const ref = ref_op(xs[i]);
            value_t const err = std::fabs(out[i] - ref);
            max_d             = std::max<double>(max_d, static_cast<double>(err));
#if 0
            EXPECT_LE(err, tolerance)
                << "\n  failing_function: " << case_tag << "  lane=" << i << "  trial=" << trial
                << "\n  x     = " << std::setprecision(21) << scalar_as_double(xs[i])
                << "\n  simd  = " << std::setprecision(21) << scalar_as_double(out[i])
                << "\n  ref   = " << std::setprecision(21) << scalar_as_double(ref)
                << "\n  |err| = " << std::setprecision(21) << scalar_as_double(err)
                << "   tol = " << std::setprecision(21) << scalar_as_double(tolerance);
#endif
        }
    }
    EXPECT_LE(max_d, tolerance);
}

template <typename value_t, typename SimdUnary, typename RefUnary>
void unary_uniform_vs_std(
    char const* case_tag,
    value_t     tolerance,
    value_t     lo,
    value_t     hi,
    SimdUnary&& simd_op,
    RefUnary&&  ref_op)
{
    unary_random_vs_std<value_t>(
        case_tag,
        tolerance,
        [=](std::array<value_t, simd<value_t>::size>& xs, std::mt19937& gen)
        { fill_uniform(xs, gen, lo, hi); },
        std::forward<SimdUnary>(simd_op),
        std::forward<RefUnary>(ref_op));
}

template <typename value_t>
void test_sqrt(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    unary_uniform_vs_std<value_t>(
        XSIGMA_SIMD_UNARY_TAG(value_t, sqrt),
        tolerance,
        static_cast<value_t>(1e-18),
        static_cast<value_t>(500),
        [](simd_t a, simd_t& c) { c = simd<value_t>::sqrt(a); },
        [](value_t x) { return std::sqrt(x); });
}

template <typename value_t>
void test_sqr(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    unary_uniform_vs_std<value_t>(
        XSIGMA_SIMD_UNARY_TAG(value_t, sqr),
        tolerance,
        static_cast<value_t>(-25),
        static_cast<value_t>(25),
        [](simd_t a, simd_t& c) { c = simd<value_t>::sqr(a); },
        [](value_t x) { return std::sqr(x); });
}

template <typename value_t>
void test_ceil(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    unary_uniform_vs_std<value_t>(
        XSIGMA_SIMD_UNARY_TAG(value_t, ceil),
        tolerance,
        static_cast<value_t>(-75),
        static_cast<value_t>(75),
        [](simd_t a, simd_t& c) { c = simd<value_t>::ceil(a); },
        [](value_t x) { return std::ceil(x); });
}

template <typename value_t>
void test_floor(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    unary_uniform_vs_std<value_t>(
        XSIGMA_SIMD_UNARY_TAG(value_t, floor),
        tolerance,
        static_cast<value_t>(-75),
        static_cast<value_t>(75),
        [](simd_t a, simd_t& c) { c = simd<value_t>::floor(a); },
        [](value_t x) { return std::floor(x); });
}

template <typename value_t>
void test_trunc(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    unary_uniform_vs_std<value_t>(
        XSIGMA_SIMD_UNARY_TAG(value_t, trunc),
        tolerance,
        static_cast<value_t>(-75),
        static_cast<value_t>(75),
        [](simd_t a, simd_t& c) { c = simd<value_t>::trunc(a); },
        [](value_t x) { return std::trunc(x); });
}

template <typename value_t>
void test_invsqrt(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    unary_uniform_vs_std<value_t>(
        XSIGMA_SIMD_UNARY_TAG(value_t, invsqrt),
        15. * tolerance,
        static_cast<value_t>(1e-18),
        static_cast<value_t>(500),
        [](simd_t a, simd_t& c) { c = simd<value_t>::invsqrt(a); },
        [](value_t x) { return std::invsqrt(x); });
}

template <typename value_t>
void test_fabs(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    unary_uniform_vs_std<value_t>(
        XSIGMA_SIMD_UNARY_TAG(value_t, fabs),
        tolerance,
        static_cast<value_t>(-120),
        static_cast<value_t>(120),
        [](simd_t a, simd_t& c) { c = simd<value_t>::fabs(a); },
        [](value_t x) { return std::fabs(x); });
}

template <typename value_t>
void test_neg(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    unary_uniform_vs_std<value_t>(
        XSIGMA_SIMD_UNARY_TAG(value_t, neg),
        tolerance,
        static_cast<value_t>(-120),
        static_cast<value_t>(120),
        [](simd_t a, simd_t& c) { c = simd<value_t>::neg(a); },
        [](value_t x) { return std::neg(x); });
}

template <typename value_t>
void test_all_simd_f1_native(value_t tolerance)
{
    test_sqrt(tolerance);
    test_sqr(tolerance);
    test_ceil(tolerance);
    test_floor(tolerance);
    test_trunc(tolerance);
    test_invsqrt(tolerance);
    test_fabs(tolerance);
    test_neg(tolerance);
}

#undef XSIGMA_SIMD_UNARY_TAG

}  // namespace simd_tests
}  // namespace vectorization

VECTORIZATIONTEST(Math, SimdF1)
{
    using namespace vectorization::simd_tests;
    constexpr float  kSimdTolF = 0.0007f;
    constexpr double kSimdTolD = 5.e-12;
    test_all_simd_f1_native<float>(kSimdTolF);
    test_all_simd_f1_native<double>(kSimdTolD);
    END_TEST();
}

#else

VECTORIZATIONTEST(Math, SimdF1)
{
    END_TEST();
}

#endif  // VECTORIZATION_VECTORIZED
