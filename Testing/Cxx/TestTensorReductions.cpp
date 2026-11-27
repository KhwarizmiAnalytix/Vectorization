/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * This file is part of XSigma and is licensed under a dual-license model:
 *
 *   - Open-source License (GPLv3):
 *       Free for personal, academic, and research use under the terms of
 *       the GNU General Public License v3.0 or later.
 *
 *   - Commercial License:
 *       A commercial license is required for proprietary, closed-source,
 *       or SaaS usage. Contact us to obtain a commercial agreement.
 *
 * Contact: licensing@xsigma.co.uk
 * Website: https://www.xsigma.co.uk
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include "VectorizationTest.h"
#include "terminals/tensor.h"

namespace
{
template <typename T> void check_reductions()
{
    using namespace vectorization;
    using tensor_t = tensor<T>;
    static_assert(std::is_same_v<decltype(mean(std::declval<tensor_t const&>())), T>);
    static_assert(std::is_same_v<decltype(all(std::declval<tensor_t const&>())), bool>);
    static_assert(
        std::is_same_v<decltype(count_nonzero(std::declval<tensor_t const&>())), std::uint64_t>);
    for (size_t n : {size_t(0),
             size_t(1),
             size_t(3),
             size_t(17),
             size_t(255),
             size_t(257),
             size_t(1025),
             size_t(66001)})
    {
        SCOPED_TRACE(n);
        tensor_t x(n);
        x = T(1);
        EXPECT_EQ(prod(x), T(1));
        EXPECT_EQ(sum(x), T(n));
        EXPECT_EQ(count_nonzero(x), n);
        EXPECT_TRUE(all(x));
        EXPECT_EQ(any(x), n != 0);
        EXPECT_EQ(nansum(x), T(n));
        if (n == 0)
        {
            EXPECT_TRUE(std::isnan(mean(x)));
            EXPECT_TRUE(std::isnan(nanmean(x)));
        }
        else
        {
            EXPECT_EQ(mean(x), T(1));
            EXPECT_EQ(nanmean(x), T(1));
            x[n - 1] = T(0);
            EXPECT_FALSE(all(x));
            EXPECT_EQ(count_nonzero(x), n - 1);
            EXPECT_EQ(prod(x), T(0));
            x = T(0);
            EXPECT_FALSE(any(x));
            EXPECT_EQ(count_nonzero(x), 0u);
        }
    }
    tensor_t x(67);
    x     = T(1);
    x[3]  = T(-2);
    x[66] = T(3);
    EXPECT_EQ(prod(x), T(-6));
    EXPECT_EQ(mean(x + T(2)), T(200) / T(67));
    EXPECT_TRUE(all(x != T(0)));
    EXPECT_TRUE(any(x < T(0)));
    EXPECT_EQ(count_nonzero(x < T(0)), 1u);
    EXPECT_EQ(sum(x < T(0)), T(1));
    EXPECT_EQ(mean(x < T(0)), T(1) / T(67));
    EXPECT_EQ(count_nonzero(x * T(0)), 0u);

    const T nan = std::numeric_limits<T>::quiet_NaN();
    x           = nan;
    EXPECT_EQ(nansum(x), T(0));
    EXPECT_TRUE(std::isnan(nanmean(x)));
    EXPECT_EQ(count_nonzero(x), 67u);  // NaN is truthy, as in PyTorch
    EXPECT_TRUE(all(x));
    EXPECT_TRUE(any(x));
    x[0]  = T(2);
    x[66] = T(4);
    EXPECT_EQ(nansum(x), T(6));
    EXPECT_EQ(nanmean(x), T(3));
    EXPECT_EQ(nansum(x * T(2)), T(12));
    EXPECT_TRUE(std::isnan(mean(x)));
    EXPECT_TRUE(std::isnan(prod(x)));
    x[0] = std::numeric_limits<T>::infinity();
    EXPECT_EQ(nansum(x), std::numeric_limits<T>::infinity());
    x[66] = -std::numeric_limits<T>::infinity();
    EXPECT_TRUE(std::isnan(nansum(x)));  // only input NaNs are ignored
}

template <typename T> void check_strides_and_shapes()
{
    using namespace vectorization;
    using tensor_t = tensor<T>;
    using dims_t   = typename tensor_t::dimensions_type;
    tensor_t x(dims_t{3, 5});
    for (size_t i = 0; i < x.size(); ++i)
    {
        x[i] = T(i + 1);
    }
    auto transposed = x.t();
    auto columns    = x.slice(1, 0, 5, 2);
    EXPECT_EQ(mean(transposed), T(8));
    EXPECT_EQ(count_nonzero(columns), 9u);
    EXPECT_EQ(nansum(columns * T(2)), T(144));
    EXPECT_TRUE(all(transposed));
    EXPECT_TRUE(any(columns - T(1)));
    tensor_t row(dims_t{5});
    row = T(1);
    EXPECT_EQ(mean(x + row), T(9));
    EXPECT_EQ(count_nonzero(x + row), 15u);
    EXPECT_EQ(nanmean(x + row), T(9));
    tensor_t bad(dims_t{7});
    EXPECT_ANY_THROW((void)prod(x + bad));
    EXPECT_ANY_THROW((void)mean(x + bad));
    EXPECT_ANY_THROW((void)count_nonzero(x + bad));
    EXPECT_ANY_THROW((void)nanmean(x + bad));
    const auto& readonly = x;
    EXPECT_EQ(count_nonzero(readonly.as_view()), 15u);
}
// Reference reductions over a logical (C-order) element list.
template <typename T> struct reference
{
    double sum = 0, scale = 0, prod = 1;
    T      min = std::numeric_limits<T>::max(), max = -std::numeric_limits<T>::max();
    size_t nonzero = 0;
    explicit reference(std::vector<T> const& v)
    {
        for (T x : v)
        {
            sum += x;
            scale += std::abs(static_cast<double>(x));
            prod *= x;
            min = std::fmin(min, x);
            max = std::fmax(max, x);
            nonzero += x != T(0);
        }
    }
};

// Checks every whole-expression reduction of `e` against its materialized values.
template <typename T, typename E>
void expect_reductions_match(E const& e, std::vector<T> const& values)
{
    using namespace vectorization;
    const reference<T> ref(values);
    const double       tol = (std::is_same_v<T, float> ? 1e-5 : 1e-12) * std::max(1.0, ref.scale);
    EXPECT_NEAR(static_cast<double>(sum(e)), ref.sum, tol);
    EXPECT_NEAR(static_cast<double>(accumulate(e)), ref.sum, tol);
    EXPECT_EQ(hmin(e), ref.min);
    EXPECT_EQ(hmax(e), ref.max);
    EXPECT_EQ(count_nonzero(e), ref.nonzero);
}

// Values in [-2, 2] that are distinct per position, so min / max locate a
// single element and a wrong offset or a skipped element is visible.
template <typename T> void fill_pattern(vectorization::tensor<T>& x)
{
    for (size_t i = 0; i < x.size(); ++i)
    {
        x[i] = static_cast<T>(std::sin(0.37 * static_cast<double>(i) + 0.1) * 2.0);
    }
}

// hmin / hmax / accumulate now share the multi-accumulator reduction core.
template <typename T> void check_extrema()
{
    using namespace vectorization;
    using tensor_t = tensor<T>;
    const T nan    = std::numeric_limits<T>::quiet_NaN();
    EXPECT_EQ(hmin(tensor_t(size_t(0))), std::numeric_limits<T>::max());
    EXPECT_EQ(hmax(tensor_t(size_t(0))), -std::numeric_limits<T>::max());
    EXPECT_EQ(accumulate(tensor_t(size_t(0))), T(0));
    for (size_t n : {size_t(1),
             size_t(3),
             size_t(7),
             size_t(8),
             size_t(9),
             size_t(31),
             size_t(33),
             size_t(257),
             size_t(1031)})
    {
        SCOPED_TRACE(n);
        tensor_t x(n + 1);
        fill_pattern(x);
        // Offset by one element so the head and tail peels are both exercised.
        auto view   = x.slice(0, 1, static_cast<int64_t>(n) + 1);
        auto values = view.to_host_vector();
        expect_reductions_match<T>(view, values);
        expect_reductions_match<T>(x, x.to_host_vector());

#if !(VECTORIZATION_HAS_NEON || VECTORIZATION_HAS_SVE)
        // NaN elements are skipped, wherever they fall (head, SIMD body, tail).
        // NEON / SVE min / max propagate NaN in the vector body instead.
        for (size_t at : {size_t(0), n / 2, n - 1})
        {
            tensor_t y(n);
            fill_pattern(y);
            const T kept_min = hmin(y), kept_max = hmax(y);
            y[at]         = nan;
            const auto v  = y.to_host_vector();
            T          lo = std::numeric_limits<T>::max(), hi = -std::numeric_limits<T>::max();
            for (T e : v)
            {
                lo = std::fmin(lo, e);
                hi = std::fmax(hi, e);
            }
            EXPECT_EQ(hmin(y), lo);
            EXPECT_EQ(hmax(y), hi);
            EXPECT_LE(kept_min, hmin(y));
            EXPECT_GE(kept_max, hmax(y));
        }
#endif
    }
#if !(VECTORIZATION_HAS_NEON || VECTORIZATION_HAS_SVE)
    tensor_t all_nan(size_t(40));
    all_nan = nan;
    EXPECT_EQ(hmin(all_nan), std::numeric_limits<T>::max());
    EXPECT_EQ(hmax(all_nan), -std::numeric_limits<T>::max());
#else
    (void)nan;
#endif
}

// Strided and broadcast reductions run through the iteration planner: rows
// with unit inner stride (packed SIMD body), rows with a broadcast operand
// (splatted loads), scalar rows, and plans with more than one outer dimension.
template <typename T> void check_planned_reductions()
{
    using namespace vectorization;
    using tensor_t = tensor<T>;
    using dims_t   = typename tensor_t::dimensions_type;

    // Row lengths below, at and above the SIMD width, with and without a tail.
    for (int64_t cols : {int64_t(3), int64_t(8), int64_t(13), int64_t(37), int64_t(509)})
    {
        SCOPED_TRACE(cols);
        tensor_t m(dims_t{9, static_cast<size_t>(cols + 5)});
        fill_pattern(m);
        auto window = m.slice(1, 2, cols + 2);  // unit inner stride, padded rows
        expect_reductions_match<T>(window, window.to_host_vector());
        expect_reductions_match<T>(
            window * T(2) + T(1), tensor_t(window * T(2) + T(1)).to_host_vector());
        auto stepped = m.slice(1, 0, cols + 5, 2);  // inner stride 2: scalar rows
        expect_reductions_match<T>(stepped, stepped.to_host_vector());
        auto transposed = window.t();  // rows walk the original columns
        expect_reductions_match<T>(transposed, transposed.to_host_vector());

        // Broadcast operands: a row vector (stride 0 along rows) and a column
        // vector (stride 0 along the row itself, splatted in the SIMD body).
        tensor_t row(dims_t{static_cast<size_t>(cols)});
        tensor_t col(dims_t{9, 1});
        fill_pattern(row);
        fill_pattern(col);
        expect_reductions_match<T>(window + row, tensor_t(window + row).to_host_vector());
        expect_reductions_match<T>(window * col, tensor_t(window * col).to_host_vector());
        expect_reductions_match<T>(row + col, tensor_t(row + col).to_host_vector());
        // Product through the broadcast path; every element is exactly 1.
        EXPECT_EQ(prod(col * T(0) + window * T(0) + T(1)), T(1));
        EXPECT_EQ(prod(window * T(0) + T(-1)), (9 * cols) % 2 == 0 ? T(1) : T(-1));
    }

    // Three outer dimensions that do not merge: rows are stepped by the counter.
    tensor_t cube(dims_t{4, 6, 21});
    fill_pattern(cube);
    auto inner = cube.slice(1, 1, 5).slice(2, 2, 19);
    expect_reductions_match<T>(inner, inner.to_host_vector());
    expect_reductions_match<T>(inner - T(3), tensor_t(inner - T(3)).to_host_vector());

    // Empty and single-element strided views.
    tensor_t grid(dims_t{4, 6});
    fill_pattern(grid);
    auto none = grid.slice(1, 2, 2);
    EXPECT_EQ(sum(none), T(0));
    EXPECT_EQ(count_nonzero(none), 0u);
    EXPECT_EQ(hmax(none), -std::numeric_limits<T>::max());
    auto one = grid.slice(0, 1, 2).slice(1, 3, 4);
    EXPECT_EQ(sum(one), grid.at(1, 3));
    EXPECT_EQ(hmin(one), grid.at(1, 3));

    // Rank above a strided_ref's limit: reduced element by element.
    tensor_t deep(dims_t{2, 2, 2, 2, 2, 2, 2, 2, 3});
    fill_pattern(deep);
    auto deep_view = deep.slice(8, 0, 2);
    ASSERT_FALSE(deep_view.is_contiguous());
    expect_reductions_match<T>(deep_view, deep_view.to_host_vector());

    // Counts over a strided view large enough to flush the SIMD count lanes.
    tensor_t big(dims_t{300, 260});
    big             = T(1);
    auto big_window = big.slice(1, 1, 258);
    EXPECT_EQ(count_nonzero(big_window), size_t(300) * 257);
    EXPECT_EQ(count_nonzero(big_window - T(1)), 0u);
}

// The packed assignment fast path: exact aliases are fine, partial overlap
// still reaches the general checker and is rejected.
template <typename T> void check_packed_assignment()
{
    using namespace vectorization;
    using tensor_t = tensor<T>;
    for (size_t n : {size_t(1), size_t(9), size_t(1027)})
    {
        SCOPED_TRACE(n);
        tensor_t a(n), b(n), out(n);
        fill_pattern(a);
        b       = T(2);
        out     = a * b + T(1);
        auto av = a.to_host_vector();
        for (size_t i = 0; i < n; ++i)
        {
            EXPECT_EQ(out[i], av[i] * T(2) + T(1));
        }
        a = a + b;  // exact alias
        for (size_t i = 0; i < n; ++i)
        {
            EXPECT_EQ(a[i], av[i] + T(2));
        }
        tensor_t buffer(n + 1);
        buffer  = T(1);
        auto lo = buffer.slice(0, 0, static_cast<int64_t>(n));
        auto hi = buffer.slice(0, 1, static_cast<int64_t>(n) + 1);
        if (n > 1)
        {
            EXPECT_ANY_THROW(lo = hi + T(1));  // views share n - 1 elements
        }
        else
        {
            lo = hi + T(1);  // disjoint single elements
            EXPECT_EQ(buffer[0], T(2));
        }
    }
}
}  // namespace

VECTORIZATIONTEST(TensorReductions, Float)
{
    check_reductions<float>();
    check_strides_and_shapes<float>();
    END_TEST();
}
VECTORIZATIONTEST(TensorReductions, Double)
{
    check_reductions<double>();
    check_strides_and_shapes<double>();
    END_TEST();
}
VECTORIZATIONTEST(TensorReductions, Extrema)
{
    check_extrema<float>();
    check_extrema<double>();
    END_TEST();
}
VECTORIZATIONTEST(TensorReductions, Planned)
{
    check_planned_reductions<float>();
    check_planned_reductions<double>();
    END_TEST();
}
VECTORIZATIONTEST(TensorReductions, PackedAssignment)
{
    check_packed_assignment<float>();
    check_packed_assignment<double>();
    END_TEST();
}
VECTORIZATIONTEST(TensorReductions, LargeExactCount)
{
    constexpr size_t             n = (size_t(1) << 24) + 3;
    vectorization::tensor<float> x(n);
    x        = 1.0f;
    x[n - 1] = 0.0f;
    EXPECT_EQ(vectorization::count_nonzero(x), n - 1);
    EXPECT_EQ(vectorization::count_nonzero(x + 1.0f), n);
    END_TEST();
}
