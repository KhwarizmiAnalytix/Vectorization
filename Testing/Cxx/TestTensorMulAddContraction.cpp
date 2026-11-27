/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * Multiply-add contraction (VECTORIZATION_FUSE_MUL_ADD) and the CPU flat
 * evaluation path.
 *
 * When contraction is enabled, `x * y + z`, `z + x * y` and `z += x * y` build
 * fma(x, y, z), so every element must equal std::fma bit for bit -- in the SIMD
 * body as well as in the scalar peel and tail. When it is disabled the result
 * must equal separate multiply-then-add rounding instead.
 *
 * Inputs are chosen so that fused and unfused rounding differ, and lengths and
 * storage offsets are odd so the unaligned peel, the aligned body and the tail
 * of the flat kernel all run.
 */

#include "VectorizationTest.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "terminals/tensor.h"

using vectorization::tensor;

namespace
{
// Values in [1, 2) with full mantissas (an irrational step), so products are
// rarely exact in T.
template <typename T> std::vector<T> mantissas(std::size_t n, double step)
{
    std::vector<T> v(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        const double x = static_cast<double>(i + 1) * step;
        v[i]           = static_cast<T>(1.0 + (x - std::floor(x)));
    }
    return v;
}

// -(x * y) rounded to T: fma(x, y, z) then yields the product's rounding error
// (usually nonzero) while multiply-then-add yields exactly 0.
template <typename T>
std::vector<T> negated_products(std::vector<T> const& x, std::vector<T> const& y)
{
    std::vector<T> v(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        volatile T product = x[i] * y[i];
        v[i]               = -product;
    }
    return v;
}

template <typename T> T reference(T x, T y, T z)
{
#if VECTORIZATION_FUSE_MUL_ADD
    return std::fma(x, y, z);
#else
    // Volatile keeps a compiler from contracting the reference itself.
    volatile T product = x * y;
    return product + z;
#endif
}

template <typename T> void fill(tensor<T>& t, std::vector<T> const& v)
{
    for (std::size_t i = 0; i < v.size(); ++i)
    {
        t[i] = v[i];
    }
}

template <typename T, typename View> bool matches(View const& got, std::vector<T> const& want)
{
    for (std::size_t i = 0; i < want.size(); ++i)
    {
        if (got[i] != want[i])
        {
            return false;
        }
    }
    return true;
}

template <typename T> void check_contraction()
{
    constexpr std::size_t n  = 1027;  // odd: SIMD body plus scalar tail
    const auto            av = mantissas<T>(n, 0.6180339887498949);
    const auto            bv = mantissas<T>(n, 0.4142135623730950);
    const auto            cv = negated_products(av, bv);

    tensor<T> a(n), b(n), c(n), out(n);
    fill(a, av);
    fill(b, bv);
    fill(c, cv);

    std::vector<T> want(n);
    std::vector<T> want_scaled(n);
    std::size_t    differs = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
        want[i]             = reference(av[i], bv[i], cv[i]);
        want_scaled[i]      = reference(av[i], T(2.5), bv[i]);
        volatile T separate = av[i] * bv[i];
        differs += (separate + cv[i]) != std::fma(av[i], bv[i], cv[i]) ? 1 : 0;
    }
    // Guard the test itself: the inputs must distinguish the two roundings.
    EXPECT_GT(differs, std::size_t{0});

    out = a * b + c;
    EXPECT_TRUE(matches(out, want));

    out = c + a * b;
    EXPECT_TRUE(matches(out, want));

    out = a * T(2.5) + b;
    EXPECT_TRUE(matches(out, want_scaled));

    out = T(2.5) * a + b;
    EXPECT_TRUE(matches(out, want_scaled));

    // An lvalue `x * y` node is copied into the fma node, not moved from.
    const auto product = a * b;
    out                = product + c;
    EXPECT_TRUE(matches(out, want));
    out = product + c;
    EXPECT_TRUE(matches(out, want));

    // `z += x * y` contracts like `z = z + x * y`.
    out = c;
    out += a * b;
    EXPECT_TRUE(matches(out, want));

    // A scalar addend.
    std::vector<T> want_scalar(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        want_scalar[i] = reference(av[i], bv[i], T(3));
    }
    out = a * b + T(3);
    EXPECT_TRUE(matches(out, want_scalar));

    // Reductions see the same contraction.
    T expected_sum = T(0);
    for (std::size_t i = 0; i < n; ++i)
    {
        expected_sum += want[i];
    }
    const T got_sum = vectorization::sum(a * b + c);
    EXPECT_NEAR(static_cast<double>(got_sum),
        static_cast<double>(expected_sum),
        1e-3 * static_cast<double>(n));
}

template <typename E> struct is_fma_node : std::false_type
{
};

template <typename L, typename M, typename R>
struct is_fma_node<vectorization::trinary_expression<L, M, R, vectorization::fma_evaluator>>
    : std::true_type
{
};

template <typename T> void check_contracted_type()
{
    tensor<T>      a(3), b(3), c(3);
    constexpr bool fused = VECTORIZATION_FUSE_MUL_ADD != 0;
    static_assert(is_fma_node<decltype(a * b + c)>::value == fused);
    static_assert(is_fma_node<decltype(c + a * b)>::value == fused);
    static_assert(is_fma_node<decltype(a * T(2) + c)>::value == fused);
    static_assert(is_fma_node<decltype(a * b + T(1))>::value == fused);
    // Only addition contracts.
    static_assert(!is_fma_node<decltype(a * b - c)>::value);
    static_assert(!is_fma_node<decltype(a + b)>::value);
    static_assert(!is_fma_node<decltype((a + b) * c)>::value);
}

template <typename T> void check_contraction_retains()
{
    tensor<T> a(8), b(8), c(8);
    // Each owner holds one reference; each captured leaf adds exactly one. Moving the
    // rvalue `a * b` node's leaves into the fma node must not add or drop any.
    const auto expression = a * b + c;
#if VECTORIZATION_FUSE_MUL_ADD
    EXPECT_EQ(expression.lhs().storage.use_count(), 2);
    EXPECT_EQ(expression.mhs().storage.use_count(), 2);
    EXPECT_EQ(expression.rhs().storage.use_count(), 2);
#else
    EXPECT_EQ(expression.lhs().lhs().storage.use_count(), 2);
    EXPECT_EQ(expression.lhs().rhs().storage.use_count(), 2);
    EXPECT_EQ(expression.rhs().storage.use_count(), 2);
#endif
}

template <typename T> void check_flat_offsets()
{
    // Contiguous views starting 0-9 elements into their allocations. Equal source and
    // destination offsets take the aligned-load body after a scalar peel; unequal ones
    // are not co-aligned and fall back to unaligned loads. The odd length leaves a tail.
    constexpr std::size_t n     = 517;
    constexpr int64_t     slack = 10;
    const auto            av    = mantissas<T>(n + slack, 0.6180339887498949);
    const auto            bv    = mantissas<T>(n + slack, 0.4142135623730950);
    tensor<T>             a(n + slack), b(n + slack), out(n + slack);
    fill(a, av);
    fill(b, bv);
    out = T(0);

    const int64_t offsets[] = {0, 1, 3, 9};
    for (const int64_t d : offsets)
    {
        for (const int64_t sx : offsets)
        {
            for (const int64_t sy : offsets)
            {
                auto       dst = out.slice(0, d, d + static_cast<int64_t>(n));
                const auto x   = a.slice(0, sx, sx + static_cast<int64_t>(n));
                const auto y   = b.slice(0, sy, sy + static_cast<int64_t>(n));
                dst            = x * T(2) - y;
                bool ok        = true;
                for (std::size_t i = 0; i < n; ++i)
                {
                    const T want = av[i + static_cast<std::size_t>(sx)] * T(2) -
                                   bv[i + static_cast<std::size_t>(sy)];
                    ok = ok && dst[i] == want;
                }
                EXPECT_TRUE(ok) << "dst offset " << d << ", x offset " << sx << ", y offset " << sy;
            }
        }
    }
}

template <typename T> void check_reduction_offsets()
{
    // Reductions peel a scalar head up to the first aligned element of the first
    // operand, then run whole SIMD registers (aligned loads when the operands are
    // co-aligned) and a scalar tail.
    constexpr std::size_t n     = 1029;
    constexpr int64_t     slack = 10;
    const auto            av    = mantissas<T>(n + slack, 0.6180339887498949);
    const auto            bv    = mantissas<T>(n + slack, 0.4142135623730950);
    tensor<T>             a(n + slack), b(n + slack);
    fill(a, av);
    fill(b, bv);

    const int64_t offsets[] = {0, 1, 3, 9};
    for (const int64_t sx : offsets)
    {
        for (const int64_t sy : offsets)
        {
            const auto x     = a.slice(0, sx, sx + static_cast<int64_t>(n));
            const auto y     = b.slice(0, sy, sy + static_cast<int64_t>(n));
            double     sum_x = 0, dot = 0;
            uint64_t   above = 0;
            for (std::size_t i = 0; i < n; ++i)
            {
                const double xi = av[i + static_cast<std::size_t>(sx)];
                const double yi = bv[i + static_cast<std::size_t>(sy)];
                sum_x += xi;
                dot += xi * yi;
                above += xi > yi ? 1 : 0;
            }
            const double tol = std::is_same_v<T, float> ? 1e-4 : 1e-12;
            EXPECT_NEAR(static_cast<double>(vectorization::sum(x)), sum_x, tol * sum_x)
                << "x offset " << sx;
            EXPECT_NEAR(static_cast<double>(vectorization::sum(x * y)), dot, tol * dot)
                << "x offset " << sx << ", y offset " << sy;
            EXPECT_EQ(vectorization::count_nonzero(x > y), above)
                << "x offset " << sx << ", y offset " << sy;
        }
    }
}
}  // namespace

VECTORIZATIONTEST(TensorMulAddContraction, Float)
{
    check_contraction<float>();
    check_contracted_type<float>();
    check_contraction_retains<float>();
    END_TEST();
}

VECTORIZATIONTEST(TensorMulAddContraction, Double)
{
    check_contraction<double>();
    check_contracted_type<double>();
    check_contraction_retains<double>();
    END_TEST();
}

VECTORIZATIONTEST(TensorFlatPath, OffsetViewsFloat)
{
    check_flat_offsets<float>();
    check_reduction_offsets<float>();
    END_TEST();
}

VECTORIZATIONTEST(TensorFlatPath, OffsetViewsDouble)
{
    check_flat_offsets<double>();
    check_reduction_offsets<double>();
    END_TEST();
}
