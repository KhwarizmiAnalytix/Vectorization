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

// Shapes and strides of elementwise expressions (design §5, §7, §8): result
// shape inference, broadcasting, shape validation, strided sources and
// destinations through the iteration planner, strided reductions, and the
// metadata-only view operations.

#include <cstddef>
#include <vector>

#include "VectorizationTest.h"
#include "common/vectorization_macros.h"
#include "terminals/tensor.h"

namespace
{

template <typename T> using dims_of = typename vectorization::tensor<T>::dimensions_type;

// Fills `t` with distinct values in its logical C order.
template <typename T> void iota(vectorization::tensor<T>& t, T start = T(1))
{
    for (size_t i = 0; i < t.size(); ++i)
    {
        t[i] = start + static_cast<T>(i);
    }
}

template <typename T> std::vector<int64_t> shape_of(vectorization::tensor<T> const& t)
{
    auto const d = t.dimensions();
    return std::vector<int64_t>(d.begin(), d.end());
}

/// @brief Result shapes and broadcasting rules.
template <typename T> VECTORIZATION_NOINLINE void test_shapes_broadcasting()
{
    using tensor_t = vectorization::tensor<T>;
    using dims_t   = dims_of<T>;

    // Construction keeps the operands' shape instead of flattening to 1-D.
    {
        tensor_t a(dims_t{2, 3});
        tensor_t b(dims_t{2, 3});
        iota(a);
        iota(b, T(10));
        tensor_t c = a + b;
        EXPECT_EQ(shape_of(c), (std::vector<int64_t>{2, 3}));
        for (size_t i = 0; i < c.size(); ++i)
        {
            EXPECT_EQ(c[i], a[i] + b[i]);
        }
    }

    // Row broadcast [2, 3] + [3], with a scalar.
    {
        tensor_t x(dims_t{2, 3});
        tensor_t bias(dims_t{3});
        iota(x);
        iota(bias, T(100));
        tensor_t y = x + bias * T(2);
        EXPECT_EQ(shape_of(y), (std::vector<int64_t>{2, 3}));
        for (size_t i = 0; i < 2; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                EXPECT_EQ(y.at(i, j), x.at(i, j) + bias[j] * T(2));
            }
        }
    }

    // Outer product shape [4, 1] * [1, 5] -> [4, 5].
    {
        tensor_t col(dims_t{4, 1});
        tensor_t row(dims_t{1, 5});
        iota(col);
        iota(row, T(10));
        tensor_t p = col * row;
        EXPECT_EQ(shape_of(p), (std::vector<int64_t>{4, 5}));
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j < 5; ++j)
            {
                EXPECT_EQ(p.at(i, j), col[i] * row[j]);
            }
        }
    }

    // Column broadcast with rows long enough for the SIMD body and a tail:
    // [m, n] + [m, 1] puts a stride-0 leaf in the inner dimension (tier 2).
    {
        const size_t m = 5;
        const size_t n = 37;
        tensor_t     x(dims_t{m, n});
        tensor_t     c(dims_t{m, 1});
        iota(x);
        iota(c, T(1000));
        tensor_t y(dims_t{m, n});
        y = x - c;
        for (size_t i = 0; i < m; ++i)
        {
            for (size_t j = 0; j < n; ++j)
            {
                EXPECT_EQ(y.at(i, j), x.at(i, j) - c[i]);
            }
        }
    }

    // Leading dimensions broadcast: [2, 3, 4] + [3, 4] + [4].
    {
        tensor_t a(dims_t{2, 3, 4});
        tensor_t b(dims_t{3, 4});
        tensor_t c(dims_t{4});
        iota(a);
        iota(b, T(50));
        iota(c, T(500));
        tensor_t r = a + b + c;
        EXPECT_EQ(shape_of(r), (std::vector<int64_t>{2, 3, 4}));
        for (size_t i = 0; i < 2; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                for (size_t k = 0; k < 4; ++k)
                {
                    EXPECT_EQ(
                        r.at(dims_t{i, j, k}), a.at(dims_t{i, j, k}) + b.at(dims_t{j, k}) + c[k]);
                }
            }
        }
    }

    logging::set_exception_mode(logging::exception_mode::THROW);

    // Incompatible shapes are rejected in every build type.
    {
        tensor_t a(dims_t{2, 3});
        tensor_t b(dims_t{2});
        tensor_t c(dims_t{3, 2});
        tensor_t d(4u);
        tensor_t e(8u);
        EXPECT_THROW({ tensor_t r = a + b; }, logging::exception);
        EXPECT_THROW({ tensor_t r = d + e; }, logging::exception);
        // Same element count, different shape.
        EXPECT_THROW({ c = a + a; }, logging::exception);
        EXPECT_THROW({ (void)vectorization::accumulate(d + e); }, logging::exception);
    }

    // Assignment never broadcasts the destination: the result must equal its shape.
    {
        tensor_t a(dims_t{3});
        tensor_t dst(dims_t{2, 3});
        tensor_t small(dims_t{3});
        EXPECT_THROW({ small = dst + a; }, logging::exception);
        EXPECT_NO_THROW({ dst = dst * T(0) + a; });
    }
}

/// @brief Strided sources and destinations through the planner.
template <typename T> VECTORIZATION_NOINLINE void test_shapes_strided_evaluation()
{
    using tensor_t = vectorization::tensor<T>;
    using dims_t   = dims_of<T>;

    // Transposed source plus a packed source.
    {
        tensor_t m(dims_t{3, 5});
        tensor_t b(dims_t{5, 3});
        iota(m);
        iota(b, T(100));
        tensor_t r = m.t() + b;
        EXPECT_EQ(shape_of(r), (std::vector<int64_t>{5, 3}));
        for (size_t i = 0; i < 5; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                EXPECT_EQ(r.at(i, j), m.at(j, i) + b.at(i, j));
            }
        }
    }

    // Stepped slice source and stepped slice destination.
    {
        tensor_t m(dims_t{4, 10});
        iota(m);
        auto     even = m.slice(1, 0, 10, 2);  // [4, 5], strides [10, 2]
        auto     odd  = m.slice(1, 1, 10, 2);
        tensor_t keep = m.clone();
        EXPECT_FALSE(even.is_contiguous());
        tensor_t src(dims_t{4, 5});
        iota(src, T(1000));
        even = src * T(2);
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j < 5; ++j)
            {
                EXPECT_EQ(m.at(i, 2 * j), src.at(i, j) * T(2));
                EXPECT_EQ(odd.at(i, j), keep.at(i, 2 * j + 1));  // untouched
            }
        }
    }

    // Rank-3 permuted source, broadcast leaf and a transposed destination
    // window, compared against a naive loop.
    {
        tensor_t a(dims_t{2, 3, 4});
        iota(a);
        auto     p = a.permute(dims_t{2, 0, 1});  // [4, 2, 3]
        tensor_t b(dims_t{3});
        iota(b, T(-7));
        tensor_t r(dims_t{4, 2, 3});
        r = p * b + T(1);
        for (size_t i = 0; i < 4; ++i)
        {
            for (size_t j = 0; j < 2; ++j)
            {
                for (size_t k = 0; k < 3; ++k)
                {
                    EXPECT_EQ(r.at(dims_t{i, j, k}), a.at(dims_t{j, k, i}) * b[k] + T(1));
                }
            }
        }
    }

    // In-place update through a view (exact aliasing is allowed).
    {
        tensor_t m(dims_t{3, 4});
        iota(m);
        tensor_t keep = m.clone();
        auto     tv   = m.t();
        tv            = tv + T(1);
        for (size_t i = 0; i < m.size(); ++i)
        {
            EXPECT_EQ(m[i], keep[i] + T(1));
        }
    }

    // Large enough to split rows across threads.
    {
        const size_t rows = 300;
        const size_t cols = 211;
        tensor_t     m(dims_t{cols, rows});
        tensor_t     bias(dims_t{cols});
        iota(m);
        iota(bias);
        tensor_t r  = m.t() + bias;  // [rows, cols]
        bool     ok = true;
        for (size_t i = 0; i < rows && ok; ++i)
        {
            for (size_t j = 0; j < cols && ok; ++j)
            {
                ok = r.at(i, j) == m.at(j, i) + bias[j];
            }
        }
        EXPECT_TRUE(ok);
    }
}

/// @brief Reductions over strided tensors and broadcast expressions.
template <typename T> VECTORIZATION_NOINLINE void test_shapes_reductions()
{
    using tensor_t = vectorization::tensor<T>;
    using dims_t   = dims_of<T>;

    tensor_t m(dims_t{4, 6});
    iota(m);
    auto cols = m.slice(1, 1, 6, 2);  // columns 1, 3, 5

    T expect_sum = 0;
    T expect_min = cols.at(0, 0);
    T expect_max = cols.at(0, 0);
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            const T v = cols.at(i, j);
            expect_sum += v;
            expect_min = v < expect_min ? v : expect_min;
            expect_max = v > expect_max ? v : expect_max;
        }
    }
    EXPECT_EQ(vectorization::accumulate(cols), expect_sum);
    EXPECT_EQ(vectorization::hmin(cols), expect_min);
    EXPECT_EQ(vectorization::hmax(cols), expect_max);
    EXPECT_EQ(vectorization::accumulate(cols * T(2)), T(2) * expect_sum);

    // Broadcast expression: sum over [4, 6] of m + row.
    tensor_t row(dims_t{6});
    iota(row);
    T expect = 0;
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 6; ++j)
        {
            expect += m.at(i, j) + row[j];
        }
    }
    EXPECT_EQ(vectorization::accumulate(m + row), expect);
}

/// @brief view / reshape / expand / squeeze / unsqueeze / flatten / slice.
template <typename T> VECTORIZATION_NOINLINE void test_shapes_view_operations()
{
    using tensor_t = vectorization::tensor<T>;
    using dims_t   = dims_of<T>;

    logging::set_exception_mode(logging::exception_mode::THROW);

    tensor_t m(dims_t{4, 6});
    iota(m);

    // view of a row slice merges dimensions that are adjacent in memory.
    {
        auto rows = m.slice(0, 1, 3);  // [2, 6], packed
        auto flat = rows.view(dims_t{12});
        EXPECT_EQ(flat.data(), rows.data());
        EXPECT_EQ(flat[0], m.at(1, 0));
        // A column slice can still split its row dimension without copying.
        auto cols  = m.slice(1, 0, 4);  // [4, 4], strides [6, 1]
        auto split = cols.view(dims_t{2, 2, 4});
        EXPECT_EQ(split.data(), cols.data());
        EXPECT_EQ(split.stride(0), 12);
        EXPECT_EQ(split.stride(1), 6);
        EXPECT_EQ(split.at(dims_t{1, 1, 3}), m.at(3, 3));
        // ...but cannot merge rows that are not adjacent.
        EXPECT_THROW({ (void)cols.view(dims_t{16}); }, logging::exception);
        EXPECT_THROW({ (void)m.view(dims_t{5, 5}); }, logging::exception);
    }

    // reshape falls back to a packed copy when no view exists.
    {
        auto tr = m.t();
        auto r  = tr.reshape(dims_t{24});
        EXPECT_TRUE(r.is_contiguous());
        EXPECT_NE(r.data(), m.data());
        for (size_t i = 0; i < 24; ++i)
        {
            EXPECT_EQ(r[i], tr[i]);
        }
        auto v = m.reshape(dims_t{2, 12});
        EXPECT_EQ(v.data(), m.data());
        EXPECT_EQ(m.t().flatten().size(), 24u);
    }

    // expand / broadcast_to: stride-0 views usable as operands.
    {
        tensor_t b(dims_t{3, 1});
        iota(b);
        auto e = b.expand(dims_t{2, 3, 4});
        EXPECT_EQ(shape_of(e), (std::vector<int64_t>{2, 3, 4}));
        EXPECT_EQ(e.stride(0), 0);
        EXPECT_EQ(e.stride(2), 0);
        EXPECT_EQ(e.at(dims_t{1, 2, 3}), b[2]);
        tensor_t c = e + T(0);
        EXPECT_EQ(c.at(dims_t{0, 1, 2}), b[1]);
        EXPECT_THROW({ (void)b.expand(dims_t{2, 4}); }, logging::exception);
        EXPECT_EQ(b.broadcast_to(dims_t{3, 5}).size(), 15u);
    }

    // squeeze / unsqueeze.
    {
        tensor_t s(dims_t{1, 3, 1, 2});
        auto     q = s.squeeze();
        EXPECT_EQ(shape_of(q), (std::vector<int64_t>{3, 2}));
        EXPECT_EQ(shape_of(s.squeeze(0)), (std::vector<int64_t>{3, 1, 2}));
        EXPECT_EQ(shape_of(s.squeeze(1)), (std::vector<int64_t>{1, 3, 1, 2}));
        auto u = m.unsqueeze(1);
        EXPECT_EQ(shape_of(u), (std::vector<int64_t>{4, 1, 6}));
        EXPECT_EQ(u.stride(1), 6);
        EXPECT_TRUE(u.is_contiguous());
        EXPECT_EQ(shape_of(m.unsqueeze(2)), (std::vector<int64_t>{4, 6, 1}));
        EXPECT_THROW({ (void)m.unsqueeze(3); }, logging::exception);

        // Squeezing every dimension gives a 0-d tensor with one element.
        tensor_t one(dims_t{1, 1});
        one[0] = T(7);
        auto z = one.squeeze();
        EXPECT_EQ(z.rank(), 0u);
        EXPECT_EQ(z.size(), 1u);
        EXPECT_EQ(z[0], T(7));
    }

    // slice: Python bounds, step >= 1.
    {
        tensor_t v(10u);
        iota(v, T(0));
        auto tail = v.slice(0, -3);
        EXPECT_EQ(tail.size(), 3u);
        EXPECT_EQ(tail[0], T(7));
        auto mid = v.slice(0, 2, -2);  // [2, 8)
        EXPECT_EQ(mid.size(), 6u);
        EXPECT_EQ(mid[5], T(7));
        auto all = v.slice(0, std::nullopt, std::nullopt, 3);
        EXPECT_EQ(all.size(), 4u);
        EXPECT_EQ(all[3], T(9));
        EXPECT_TRUE(v.slice(0, 8, 2).empty());
        EXPECT_THROW({ (void)v.slice(0, 0, 10, 0); }, logging::exception);
        EXPECT_THROW({ (void)v.slice(0, 0, 10, -1); }, logging::exception);
    }
}

}  // namespace

VECTORIZATIONTEST(Math, TensorShapes)
{
    test_shapes_broadcasting<float>();
    test_shapes_broadcasting<double>();
    test_shapes_strided_evaluation<float>();
    test_shapes_strided_evaluation<double>();
    test_shapes_reductions<float>();
    test_shapes_reductions<double>();
    test_shapes_view_operations<float>();
    test_shapes_view_operations<double>();

    END_TEST();
}
