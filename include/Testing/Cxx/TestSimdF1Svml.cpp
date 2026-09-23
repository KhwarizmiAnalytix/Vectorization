/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * SIMD tests: unary transcendentals and related intrinsics (Intel SVML / compiler vector math).
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

template <typename value_t, std::size_t N>
void fill_open_unit_interval(std::array<value_t, N>& xs, std::mt19937& gen)
{
    std::uniform_real_distribution<value_t> dist(0, 1);
    value_t const                           margin =
        std::max(std::numeric_limits<value_t>::epsilon() * value_t(1000), value_t(1e-7));
    for (auto& x : xs)
    {
        x = dist(gen);
        x = margin + (value_t(1) - value_t(2) * margin) * x;
    }
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
            EXPECT_LE(err, tolerance)
                << "\n  failing_function: " << case_tag << "  lane=" << i << "  trial=" << trial
                << "\n  x     = " << std::setprecision(21) << scalar_as_double(xs[i])
                << "\n  simd  = " << std::setprecision(21) << scalar_as_double(out[i])
                << "\n  ref   = " << std::setprecision(21) << scalar_as_double(ref)
                << "\n  |err| = " << std::setprecision(21) << scalar_as_double(err)
                << "   tol = " << std::setprecision(21) << scalar_as_double(tolerance);
        }
    }
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
void test_exp(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    unary_uniform_vs_std<value_t>(
        XSIGMA_SIMD_UNARY_TAG(value_t, exp),
        tolerance,
        static_cast<value_t>(-8),
        static_cast<value_t>(8),
        [](simd_t a, simd_t& c) { c = simd<value_t>::exp(a); },
        [](value_t x) { return std::exp(x); });
}

template <typename value_t>
void test_expm1(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    unary_uniform_vs_std<value_t>(
        XSIGMA_SIMD_UNARY_TAG(value_t, expm1),
        tolerance,
        static_cast<value_t>(-8),
        static_cast<value_t>(8),
        [](simd_t a, simd_t& c) { c = simd<value_t>::expm1(a); },
        [](value_t x) { return std::expm1(x); });
}

template <typename value_t>
void test_exp2(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    unary_uniform_vs_std<value_t>(
        XSIGMA_SIMD_UNARY_TAG(value_t, exp2),
        tolerance,
        static_cast<value_t>(-12),
        static_cast<value_t>(12),
        [](simd_t a, simd_t& c) { c = simd<value_t>::exp2(a); },
        [](value_t x) { return std::exp2(x); });
}

template <typename value_t>
void test_log(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    unary_uniform_vs_std<value_t>(
        XSIGMA_SIMD_UNARY_TAG(value_t, log),
        tolerance,
        static_cast<value_t>(1e-18),
        static_cast<value_t>(500),
        [](simd_t a, simd_t& c) { c = simd<value_t>::log(a); },
        [](value_t x) { return std::log(x); });
}

template <typename value_t>
void test_log1p(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    unary_uniform_vs_std<value_t>(
        XSIGMA_SIMD_UNARY_TAG(value_t, log1p),
        tolerance,
        static_cast<value_t>(-0.95),
        static_cast<value_t>(500),
        [](simd_t a, simd_t& c) { c = simd<value_t>::log1p(a); },
        [](value_t x) { return std::log1p(x); });
}

template <typename value_t>
void test_log2(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    unary_uniform_vs_std<value_t>(
        XSIGMA_SIMD_UNARY_TAG(value_t, log2),
        tolerance,
        static_cast<value_t>(1e-18),
        static_cast<value_t>(500),
        [](simd_t a, simd_t& c) { c = simd<value_t>::log2(a); },
        [](value_t x) { return std::log2(x); });
}

template <typename value_t>
void test_sin(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    unary_uniform_vs_std<value_t>(
        XSIGMA_SIMD_UNARY_TAG(value_t, sin),
        tolerance,
        static_cast<value_t>(-25),
        static_cast<value_t>(25),
        [](simd_t a, simd_t& c) { c = simd<value_t>::sin(a); },
        [](value_t x) { return std::sin(x); });
}

template <typename value_t>
void test_cos(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    unary_uniform_vs_std<value_t>(
        XSIGMA_SIMD_UNARY_TAG(value_t, cos),
        tolerance,
        static_cast<value_t>(-25),
        static_cast<value_t>(25),
        [](simd_t a, simd_t& c) { c = simd<value_t>::cos(a); },
        [](value_t x) { return std::cos(x); });
}

template <typename value_t>
void test_tan(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    unary_uniform_vs_std<value_t>(
        XSIGMA_SIMD_UNARY_TAG(value_t, tan),
        tolerance,
        static_cast<value_t>(-1.2),
        static_cast<value_t>(1.2),
        [](simd_t a, simd_t& c) { c = simd<value_t>::tan(a); },
        [](value_t x) { return std::tan(x); });
}

template <typename value_t>
void test_asin(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    unary_uniform_vs_std<value_t>(
        XSIGMA_SIMD_UNARY_TAG(value_t, asin),
        tolerance,
        static_cast<value_t>(-0.999),
        static_cast<value_t>(0.999),
        [](simd_t a, simd_t& c) { c = simd<value_t>::asin(a); },
        [](value_t x) { return std::asin(x); });
}

template <typename value_t>
void test_acos(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    unary_uniform_vs_std<value_t>(
        XSIGMA_SIMD_UNARY_TAG(value_t, acos),
        tolerance,
        static_cast<value_t>(-0.999),
        static_cast<value_t>(0.999),
        [](simd_t a, simd_t& c) { c = simd<value_t>::acos(a); },
        [](value_t x) { return std::acos(x); });
}

template <typename value_t>
void test_atan(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    unary_uniform_vs_std<value_t>(
        XSIGMA_SIMD_UNARY_TAG(value_t, atan),
        tolerance,
        static_cast<value_t>(-25),
        static_cast<value_t>(25),
        [](simd_t a, simd_t& c) { c = simd<value_t>::atan(a); },
        [](value_t x) { return std::atan(x); });
}

template <typename value_t>
void test_sinh(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    unary_uniform_vs_std<value_t>(
        XSIGMA_SIMD_UNARY_TAG(value_t, sinh),
        tolerance,
        static_cast<value_t>(-8),
        static_cast<value_t>(8),
        [](simd_t a, simd_t& c) { c = simd<value_t>::sinh(a); },
        [](value_t x) { return std::sinh(x); });
}

template <typename value_t>
void test_cosh(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    unary_uniform_vs_std<value_t>(
        XSIGMA_SIMD_UNARY_TAG(value_t, cosh),
        tolerance,
        static_cast<value_t>(-8),
        static_cast<value_t>(8),
        [](simd_t a, simd_t& c) { c = simd<value_t>::cosh(a); },
        [](value_t x) { return std::cosh(x); });
}

template <typename value_t>
void test_tanh(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    unary_uniform_vs_std<value_t>(
        XSIGMA_SIMD_UNARY_TAG(value_t, tanh),
        tolerance,
        static_cast<value_t>(-8),
        static_cast<value_t>(8),
        [](simd_t a, simd_t& c) { c = simd<value_t>::tanh(a); },
        [](value_t x) { return std::tanh(x); });
}

template <typename value_t>
void test_asinh(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    unary_uniform_vs_std<value_t>(
        XSIGMA_SIMD_UNARY_TAG(value_t, asinh),
        tolerance,
        static_cast<value_t>(-80),
        static_cast<value_t>(80),
        [](simd_t a, simd_t& c) { c = simd<value_t>::asinh(a); },
        [](value_t x) { return std::asinh(x); });
}

template <typename value_t>
void test_acosh(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    unary_uniform_vs_std<value_t>(
        XSIGMA_SIMD_UNARY_TAG(value_t, acosh),
        tolerance,
        static_cast<value_t>(1),
        static_cast<value_t>(80),
        [](simd_t a, simd_t& c) { c = simd<value_t>::acosh(a); },
        [](value_t x) { return std::acosh(x); });
}

template <typename value_t>
void test_atanh(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    unary_uniform_vs_std<value_t>(
        XSIGMA_SIMD_UNARY_TAG(value_t, atanh),
        tolerance,
        static_cast<value_t>(-0.99),
        static_cast<value_t>(0.99),
        [](simd_t a, simd_t& c) { c = simd<value_t>::atanh(a); },
        [](value_t x) { return std::atanh(x); });
}

template <typename value_t>
void test_cbrt(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    unary_uniform_vs_std<value_t>(
        XSIGMA_SIMD_UNARY_TAG(value_t, cbrt),
        tolerance,
        static_cast<value_t>(-125),
        static_cast<value_t>(125),
        [](simd_t a, simd_t& c) { c = simd<value_t>::cbrt(a); },
        [](value_t x) { return std::cbrt(x); });
}

template <typename value_t>
void test_cdf(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    unary_uniform_vs_std<value_t>(
        XSIGMA_SIMD_UNARY_TAG(value_t, cdf),
        tolerance,
        static_cast<value_t>(-8),
        static_cast<value_t>(8),
        [](simd_t a, simd_t& c) { c = simd<value_t>::cdf(a); },
        [](value_t x) { return std::cdf(x); });
}

template <typename value_t>
void test_inv_cdf(value_t tolerance)
{
    using simd_t = typename simd<value_t>::simd_t;
    unary_random_vs_std<value_t>(
        XSIGMA_SIMD_UNARY_TAG(value_t, inv_cdf),
        tolerance,
        [](std::array<value_t, simd<value_t>::size>& xs, std::mt19937& gen)
        { fill_open_unit_interval(xs, gen); },
        [](simd_t a, simd_t& c) { c = simd<value_t>::inv_cdf(a); },
        [](value_t x) { return std::inv_cdf(x); });
}

template <typename value_t>
void test_all_simd_f1_svml(value_t tolerance)
{
    test_exp(tolerance);
    test_expm1(tolerance);
    test_exp2(tolerance);
    test_log(tolerance);
    test_log1p(tolerance);
    test_log2(tolerance);
    test_sin(tolerance);
    test_cos(tolerance);
    test_tan(tolerance);
    test_asin(tolerance);
    test_acos(tolerance);
    test_atan(tolerance);
    test_sinh(tolerance);
    test_cosh(tolerance);
    test_tanh(tolerance);
    test_asinh(tolerance);
    test_acosh(tolerance);
    test_atanh(tolerance);
    test_cbrt(tolerance);
    test_cdf(tolerance);
    test_inv_cdf(tolerance);
}

#undef XSIGMA_SIMD_UNARY_TAG

}  // namespace simd_tests
}  // namespace vectorization

VECTORIZATIONTEST(Math, SimdF1Svml)
{
    using namespace vectorization::simd_tests;
    constexpr float  kSimdTolF = 0.0007f;
    constexpr double kSimdTolD = 5.e-12;
    test_all_simd_f1_svml<float>(kSimdTolF);
    test_all_simd_f1_svml<double>(kSimdTolD);
    END_TEST();
}

#else

VECTORIZATIONTEST(Math, SimdF1Svml)
{
    END_TEST();
}

#endif  // VECTORIZATION_VECTORIZED
