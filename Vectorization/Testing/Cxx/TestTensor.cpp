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

#include <cmath>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "VectorizationTest.h"
#include "common/vectorization_macros.h"
#include "terminals/tensor.h"

namespace
{

// The tensor test body is split into several noinline helpers: as one function
// it is large enough to trip an internal compiler error in GCC 13's basic-block
// reordering pass, and keeping the helpers out-of-line stops the compiler from
// merging them back together.

/// @brief Construction, copy/move semantics, and external-data borrowing.
template <typename T>
VECTORIZATION_NOINLINE void test_tensor_construction()
{
    using tensor_t = vectorization::tensor<T>;
    using dims_t   = typename tensor_t::dimensions_type;
    const T eps    = std::is_same<T, float>::value ? T(1e-5) : T(1e-10);

    // -----------------------------------------------------------------------
    // SIMD metadata
    // -----------------------------------------------------------------------
#if VECTORIZATION_VECTORIZED
    EXPECT_EQ(tensor_t::length(), static_cast<size_t>(simd<T>::size));
#else
    EXPECT_EQ(tensor_t::length(), 1u);
#endif

    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    // 1-D
    {
        tensor_t v(6u);
        EXPECT_EQ(v.rank(), 1u);
        EXPECT_EQ(v.size(), 6u);
        EXPECT_EQ(v.size(0), 6);
        EXPECT_EQ(v.stride(0), 1);
        EXPECT_FALSE(v.empty());
        EXPECT_TRUE(v.is_contiguous());
        EXPECT_EQ(v.contiguous().data(), v.data());
    }

    // 1-D linspace
    {
        tensor_t ls(T(0), T(4), 5u);
        EXPECT_EQ(ls.size(), 5u);
        EXPECT_NEAR(static_cast<T>(ls[0]), T(0), eps);
        EXPECT_NEAR(static_cast<T>(ls[2]), T(2), eps);
        EXPECT_NEAR(static_cast<T>(ls[4]), T(4), eps);
    }

    // 1-D initializer_list
    {
        tensor_t il{T(1), T(2), T(3)};
        EXPECT_EQ(il.size(), 3u);
        EXPECT_EQ(static_cast<T>(il[0]), T(1));
        EXPECT_EQ(static_cast<T>(il[1]), T(2));
        EXPECT_EQ(static_cast<T>(il[2]), T(3));
    }

    // 2-D
    {
        tensor_t m(3u, 4u);
        EXPECT_EQ(m.rank(), 2u);
        EXPECT_EQ(m.dimension(0), 3u);
        EXPECT_EQ(m.dimension(1), 4u);
        EXPECT_EQ(m.size(), 12u);
        EXPECT_EQ(m.stride(0), 4);
        EXPECT_EQ(m.stride(1), 1);
        EXPECT_TRUE(m.is_contiguous());
    }

    // 2-D initializer_list
    {
        tensor_t il2{{T(1), T(2)}, {T(3), T(4)}};
        EXPECT_EQ(il2.dimension(0), 2u);
        EXPECT_EQ(il2.dimension(1), 2u);
        EXPECT_EQ(static_cast<T>(il2.at(0u, 0u)), T(1));
        EXPECT_EQ(static_cast<T>(il2.at(0u, 1u)), T(2));
        EXPECT_EQ(static_cast<T>(il2.at(1u, 0u)), T(3));
        EXPECT_EQ(static_cast<T>(il2.at(1u, 1u)), T(4));
    }

    // N-D const ref dims
    {
        tensor_t volume(dims_t{2, 3, 4});
        EXPECT_EQ(volume.rank(), 3u);
        EXPECT_EQ(volume.size(), 24u);
        EXPECT_EQ(volume.dimension(0), 2u);
        EXPECT_EQ(volume.dimension(1), 3u);
        EXPECT_EQ(volume.dimension(2), 4u);
        EXPECT_EQ(volume.stride(0), 12);
        EXPECT_EQ(volume.stride(1), 4);
        EXPECT_EQ(volume.stride(2), 1);
    }

    // Rank > MAX_INLINE_SIZE (5) uses the outline sizes_and_strides path.
    {
        tensor_t t6(dims_t{2, 2, 2, 2, 2, 2});
        EXPECT_EQ(t6.rank(), 6u);
        EXPECT_EQ(t6.size(), 64u);
        EXPECT_TRUE(t6.is_contiguous());
        EXPECT_EQ(t6.stride(5), 1);
        EXPECT_EQ(t6.stride(0), 32);
        t6.at(dims_t{1, 1, 1, 1, 1, 1}) = T(9);
        const T last                    = static_cast<T>(t6.at(dims_t{1, 1, 1, 1, 1, 1}));
        EXPECT_EQ(last, T(9));
        EXPECT_EQ(static_cast<T>(t6[63]), T(9));
    }

    // N-D move dims
    {
        dims_t   shape{2, 5};
        tensor_t from_moved_shape(std::move(shape));
        EXPECT_EQ(from_moved_shape.rank(), 2u);
        EXPECT_EQ(from_moved_shape.size(), 10u);
    }

    // External data wrap (non-owning borrow)
    {
        std::vector<T> buf(6, T(9));
        tensor_t       ext(buf.data(), dims_t{2, 3});
        EXPECT_EQ(ext.dimension(0), 2u);
        EXPECT_EQ(ext.dimension(1), 3u);
        EXPECT_EQ(ext.size(), 6u);
        EXPECT_EQ(ext.data(), buf.data());
        EXPECT_EQ(static_cast<T>(ext.at(0u, 0u)), T(9));
        EXPECT_EQ(static_cast<T>(ext.at(1u, 2u)), T(9));
        tensor_t cloned(ext);
        EXPECT_NE(cloned.data(), ext.data());
        EXPECT_TRUE(cloned == ext);
        ext.at(0u, 0u) = T(1);
        EXPECT_EQ(static_cast<T>(cloned.at(0u, 0u)), T(9));
    }

    // Copy constructor: always deep-clones (independent storage)
    {
        tensor_t orig(4u);
        orig = T(7);
        tensor_t cp(orig);
        EXPECT_EQ(cp.size(), orig.size());
        EXPECT_EQ(cp.rank(), orig.rank());
        EXPECT_NE(cp.data(), orig.data());
        EXPECT_TRUE(cp == orig);
        orig[0] = T(99);
        EXPECT_NE(static_cast<T>(cp[0]), T(99));
    }

    // Move constructor
    {
        tensor_t src(5u);
        src              = T(3);
        T*       raw_ptr = src.data();
        tensor_t moved(std::move(src));
        EXPECT_EQ(moved.size(), 5u);
        EXPECT_EQ(moved.data(), raw_ptr);
    }

    // Copy assignment: always deep-clones (independent storage)
    {
        tensor_t a(3u), b(3u);
        a = T(1);
        b = T(2);
        a = b;
        EXPECT_NE(a.data(), b.data());
        EXPECT_TRUE(a == b);
        b[0] = T(9);
        EXPECT_NE(static_cast<T>(a[0]), T(9));
    }

    // Move assignment
    {
        tensor_t a(3u), b(4u);
        b          = T(5);
        T* raw_ptr = b.data();
        a          = std::move(b);
        EXPECT_EQ(a.size(), 4u);
        EXPECT_EQ(a.data(), raw_ptr);
    }
}

/// @brief Shape metadata, element access, cloning, and view creation.
template <typename T>
VECTORIZATION_NOINLINE void test_tensor_shape_access()
{
    using tensor_t = vectorization::tensor<T>;
    using dims_t   = typename tensor_t::dimensions_type;
    const T eps    = std::is_same<T, float>::value ? T(1e-5) : T(1e-10);

    // -----------------------------------------------------------------------
    // Shape metadata
    // -----------------------------------------------------------------------
    {
        tensor_t m(2u, 5u);

        // dimensions() span
        auto dims = m.dimensions();
        EXPECT_EQ(dims.size(), 2u);
        EXPECT_EQ(dims[0], 2);
        EXPECT_EQ(dims[1], 5);

        // strides() span
        auto strs = m.strides();
        EXPECT_EQ(strs.size(), 2u);
        EXPECT_EQ(strs[0], 5);
        EXPECT_EQ(strs[1], 1);

        // size(dim) / stride(dim) / dimension(n)
        EXPECT_EQ(m.size(0), 2);
        EXPECT_EQ(m.size(1), 5);
        EXPECT_EQ(m.stride(0), 5);
        EXPECT_EQ(m.stride(1), 1);
        EXPECT_EQ(m.dimension(0), 2u);
        EXPECT_EQ(m.dimension(1), 5u);
    }

    // empty()
    {
        tensor_t empty_t;
        EXPECT_TRUE(empty_t.empty());
        tensor_t non_empty(1u);
        EXPECT_FALSE(non_empty.empty());
    }

    // -----------------------------------------------------------------------
    // Element access
    // -----------------------------------------------------------------------
    {
        // operator[] and at(i)
        tensor_t v(5u);
        v    = T(0);
        v[0] = T(10);
        v[4] = T(40);
        EXPECT_EQ(static_cast<T>(v[0]), T(10));
        EXPECT_EQ(static_cast<T>(v.at(size_t(0))), T(10));
        EXPECT_EQ(static_cast<T>(v[4]), T(40));

        // begin / end iteration
        T sum = T(0);
        for (auto it = v.begin(); it != v.end(); ++it)
            sum += *it;
        EXPECT_NEAR(sum, T(50), eps);

        // at(i, j)
        tensor_t m(2u, 3u);
        m            = T(0);
        m.at(0u, 1u) = T(5);
        m.at(1u, 2u) = T(7);
        EXPECT_EQ(static_cast<T>(m.at(0u, 1u)), T(5));
        EXPECT_EQ(static_cast<T>(m.at(1u, 2u)), T(7));

        // at(dimensions_type) — N-D
        tensor_t t3(dims_t{2, 3, 4});
        t3                     = T(0);
        t3.at(dims_t{1, 2, 3}) = T(99);
        EXPECT_EQ(static_cast<T>(t3.at(dims_t{1, 2, 3})), T(99));
    }

    // -----------------------------------------------------------------------
    // Clone (deep copy)
    // -----------------------------------------------------------------------
    {
        tensor_t orig(3u);
        orig     = T(7);
        auto dst = orig.clone();
        EXPECT_EQ(dst.size(), orig.size());
        EXPECT_TRUE(dst == orig);
        // Verify independence: modifying orig doesn't affect dst
        orig[0] = T(99);
        EXPECT_NE(static_cast<T>(dst[0]), T(99));

        tensor_t m(2u, 3u);
        m            = T(0);
        m.at(0u, 2u) = T(6);
        auto packed  = m.t().clone();
        EXPECT_TRUE(packed.is_contiguous());
        EXPECT_EQ(packed.dimension(0), 3u);
        EXPECT_EQ(packed.dimension(1), 2u);
        EXPECT_EQ(static_cast<T>(packed.at(2u, 0u)), T(6));

        auto cpu = m.t().to_cpu();
        EXPECT_EQ(cpu.device(), vectorization::device_enum::CPU);
        EXPECT_TRUE(cpu.is_contiguous());
        EXPECT_EQ(static_cast<T>(cpu.at(2u, 0u)), T(6));
        EXPECT_NE(cpu.data(), m.data());

        tensor_t packed_cpu(2u, 3u);
        packed_cpu = T(4);
        auto alias = packed_cpu.to_cpu();
        EXPECT_EQ(alias.data(), packed_cpu.data());
        EXPECT_TRUE(alias.is_contiguous());
    }

    // -----------------------------------------------------------------------
    // Views
    // -----------------------------------------------------------------------

    // t() — transpose: swaps shape and strides, shares data, non-contiguous
    {
        tensor_t m(2u, 3u);
        m            = T(0);
        m.at(0u, 2u) = T(6);
        auto tv      = m.t();
        EXPECT_EQ(tv.dimension(0), 3u);
        EXPECT_EQ(tv.dimension(1), 2u);
        EXPECT_EQ(tv.stride(0), m.stride(1));
        EXPECT_EQ(tv.stride(1), m.stride(0));
        EXPECT_FALSE(tv.is_contiguous());
        EXPECT_EQ(tv.data(), m.data());
        EXPECT_EQ(tv.at(2u, 0u), T(6));
        auto packed = tv.contiguous();
        EXPECT_TRUE(packed.is_contiguous());
        EXPECT_EQ(packed.at(2u, 0u), T(6));
        EXPECT_NE(packed.data(), tv.data());
    }

    // permute(): reorders axes
    {
        tensor_t volume(dims_t{2, 3, 4});
        volume        = T(0);
        auto permuted = volume.permute(dims_t{2, 0, 1});
        EXPECT_EQ(permuted.dimension(0), 4u);
        EXPECT_EQ(permuted.dimension(1), 2u);
        EXPECT_EQ(permuted.dimension(2), 3u);
        EXPECT_EQ(permuted.stride(0), volume.stride(2));
        EXPECT_EQ(permuted.stride(1), volume.stride(0));
        EXPECT_EQ(permuted.stride(2), volume.stride(1));
        EXPECT_EQ(permuted.data(), volume.data());
    }
}

/// @brief Slicing, permutation, and expression assignment into views.
template <typename T>
VECTORIZATION_NOINLINE void test_tensor_views_slices()
{
    using tensor_t = vectorization::tensor<T>;
    using dims_t   = typename tensor_t::dimensions_type;
    const T eps    = std::is_same<T, float>::value ? T(1e-5) : T(1e-10);

    // permute() moving a size-1 axis must not spuriously break contiguity: a singleton
    // dimension's stride is never read by any valid index, so relocating it can't change
    // whether the tensor is genuinely flat-indexable (data[i] for i in [0, numel)).
    // Regression for recompute_cpu_simd_alignment_state() requiring every dimension's
    // stride -- including size-1 ones -- to satisfy the packed-stride recurrence, instead
    // of skipping size-1 dimensions like PyTorch's compute_contiguous() does.
    {
        tensor_t vol(dims_t{4, 1, 5});
        EXPECT_TRUE(vol.is_contiguous());
        auto moved = vol.permute(dims_t{1, 0, 2});
        EXPECT_EQ(moved.dimension(0), 1u);
        EXPECT_EQ(moved.dimension(1), 4u);
        EXPECT_EQ(moved.dimension(2), 5u);
        EXPECT_TRUE(moved.is_contiguous());
        EXPECT_EQ(moved.data(), vol.data());
    }

    // view(): reinterprets shape with new contiguous strides, shares data
    {
        tensor_t v(12u);
        for (size_t i = 0; i < 12; ++i)
            v[i] = static_cast<T>(i);
        auto vw = v.view(dims_t{3, 4});
        EXPECT_EQ(vw.dimension(0), 3u);
        EXPECT_EQ(vw.dimension(1), 4u);
        EXPECT_EQ(vw.size(), 12u);
        EXPECT_TRUE(vw.is_contiguous());
        EXPECT_EQ(vw.data(), v.data());
        // Modification through view is reflected in source
        vw.at(0u, 0u) = T(99);
        EXPECT_EQ(static_cast<T>(v[0]), T(99));
    }

    // reshape(): same semantics as view for contiguous source
    {
        tensor_t v(6u);
        for (size_t i = 0; i < 6; ++i)
            v[i] = static_cast<T>(i);
        auto rs = v.reshape(dims_t{2, 3});
        EXPECT_EQ(rs.dimension(0), 2u);
        EXPECT_EQ(rs.dimension(1), 3u);
        EXPECT_EQ(rs.size(), 6u);
        EXPECT_EQ(rs.data(), v.data());
        EXPECT_TRUE(rs.is_contiguous());
    }

    // slice(dim, start, stop, step): shrinks one dimension, multiplies its stride
    {
        tensor_t v(10u);
        for (size_t i = 0; i < 10; ++i)
            v[i] = static_cast<T>(i);

        // indices 1, 4, 7 (step=3 through [1,8))
        auto sl = v.slice(0, 1, 8, 3);
        EXPECT_EQ(sl.size(), 3u);
        EXPECT_EQ(sl.stride(0), 3);
        EXPECT_EQ(sl.data(), v.data() + 1);
        EXPECT_EQ(*(sl.data() + 0 * sl.stride(0)), T(1));
        EXPECT_EQ(*(sl.data() + 1 * sl.stride(0)), T(4));
        EXPECT_EQ(*(sl.data() + 2 * sl.stride(0)), T(7));
        EXPECT_EQ(static_cast<T>(sl[0]), T(1));
        EXPECT_EQ(static_cast<T>(sl[1]), T(4));
        EXPECT_EQ(static_cast<T>(sl[2]), T(7));

        // stop==-1 means to end
        auto sl2 = v.slice(0, 5);
        EXPECT_EQ(sl2.size(), 5u);
        EXPECT_EQ(sl2.data(), v.data() + 5);

        // negative start wraps like Python
        auto sl3 = v.slice(0, -3);
        EXPECT_EQ(sl3.size(), 3u);
        EXPECT_EQ(static_cast<T>(sl3[0]), T(7));

        // empty slice
        auto sl_empty = v.slice(0, 3, 3);
        EXPECT_EQ(sl_empty.size(), 0u);
        EXPECT_TRUE(sl_empty.empty());
    }

    // 2-D slice: size() is numel, at() uses strides
    {
        tensor_t m(4u, 5u);
        for (size_t i = 0; i < 4; ++i)
            for (size_t j = 0; j < 5; ++j)
                m.at(i, j) = static_cast<T>(i * 10 + j);
        auto sl = m.slice(0, 1, 3);
        EXPECT_EQ(sl.dimension(0), 2u);
        EXPECT_EQ(sl.dimension(1), 5u);
        EXPECT_EQ(sl.size(), 10u);
        EXPECT_EQ(static_cast<T>(sl.at(0u, 0u)), T(10));
        EXPECT_EQ(static_cast<T>(sl.at(1u, 4u)), T(24));
    }

    // -----------------------------------------------------------------------
    // sizes_and_strides: scalar at()/[] use strides; expressions do not
    // -----------------------------------------------------------------------

    // t(): every at(i, j) of the view is original at(j, i); storage is shared.
    {
        tensor_t m(2u, 3u);
        for (size_t i = 0; i < 2; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                m.at(i, j) = static_cast<T>(10 * i + j);
            }
        }
        auto tv = m.t();
        EXPECT_EQ(tv.rank(), 2u);
        EXPECT_EQ(tv.size(), m.size());
        EXPECT_EQ(tv.data(), m.data());
        EXPECT_FALSE(tv.is_contiguous());
        EXPECT_EQ(tv.stride(0), 1);
        EXPECT_EQ(tv.stride(1), 3);
        auto const tv_strides = tv.strides();
        EXPECT_EQ(tv_strides.size(), 2u);
        EXPECT_EQ(tv_strides[0], 1);
        EXPECT_EQ(tv_strides[1], 3);
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 2; ++j)
            {
                EXPECT_EQ(static_cast<T>(tv.at(i, j)), static_cast<T>(m.at(j, i)))
                    << "t() at(" << i << "," << j << ")";
            }
        }
        // operator[] walks C-order of the *view* shape through logical_offset.
        EXPECT_EQ(static_cast<T>(tv[0]), T(0));
        EXPECT_EQ(static_cast<T>(tv[1]), T(10));
        EXPECT_EQ(static_cast<T>(tv[2]), T(1));
        EXPECT_EQ(static_cast<T>(tv[5]), T(12));
        tv.at(1u, 0u) = T(77);
        EXPECT_EQ(static_cast<T>(m.at(0u, 1u)), T(77));
    }

    // permute(): N-D at(dims) follows the remapped axes and strides.
    {
        tensor_t volume(dims_t{2, 3, 4});
        for (size_t i = 0; i < 2; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                for (size_t k = 0; k < 4; ++k)
                {
                    volume.at(dims_t{i, j, k}) = static_cast<T>(100 * i + 10 * j + k);
                }
            }
        }
        auto permuted = volume.permute(dims_t{2, 0, 1});
        EXPECT_EQ(permuted.data(), volume.data());
        EXPECT_FALSE(permuted.is_contiguous());
        EXPECT_EQ(permuted.stride(0), volume.stride(2));
        EXPECT_EQ(permuted.stride(1), volume.stride(0));
        EXPECT_EQ(permuted.stride(2), volume.stride(1));
        for (size_t i = 0; i < 2; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                for (size_t k = 0; k < 4; ++k)
                {
                    const T got      = static_cast<T>(permuted.at(dims_t{k, i, j}));
                    const T expected = static_cast<T>(volume.at(dims_t{i, j, k}));
                    EXPECT_EQ(got, expected);
                }
            }
        }
    }

    // Column slice: dim-1 size shrinks, its stride is multiplied by step.
    {
        tensor_t m(3u, 5u);
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 5; ++j)
            {
                m.at(i, j) = static_cast<T>(10 * i + j);
            }
        }
        auto cols = m.slice(1, 1, 5, 2);
        EXPECT_EQ(cols.dimension(0), 3u);
        EXPECT_EQ(cols.dimension(1), 2u);
        EXPECT_EQ(cols.size(), 6u);
        EXPECT_EQ(cols.stride(0), m.stride(0));
        EXPECT_EQ(cols.stride(1), m.stride(1) * 2);
        EXPECT_FALSE(cols.is_contiguous());
        EXPECT_EQ(static_cast<T>(cols.at(0u, 0u)), T(1));
        EXPECT_EQ(static_cast<T>(cols.at(0u, 1u)), T(3));
        EXPECT_EQ(static_cast<T>(cols.at(2u, 0u)), T(21));
        EXPECT_EQ(static_cast<T>(cols.at(2u, 1u)), T(23));
        EXPECT_EQ(static_cast<T>(cols[0]), T(1));
        EXPECT_EQ(static_cast<T>(cols[1]), T(3));
        EXPECT_EQ(static_cast<T>(cols[2]), T(11));
    }

    // Expressions load packed contiguous memory; a strided view must be packed first.
    {
        tensor_t m(2u, 3u);
        for (size_t i = 0; i < 2; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                m.at(i, j) = static_cast<T>(i + j);
            }
        }
        auto packed = m.t().contiguous();
        EXPECT_TRUE(packed.is_contiguous());
        EXPECT_EQ(packed.stride(0), 2);
        EXPECT_EQ(packed.stride(1), 1);
        EXPECT_EQ(static_cast<T>(packed.at(2u, 1u)), static_cast<T>(m.at(1u, 2u)));

        tensor_t ones = packed.clone();
        ones          = T(1);
        // Expression ctor is 1-D (expr.size() only); assign into a shaped clone.
        tensor_t sum = packed.clone();
        sum          = packed + ones;
        EXPECT_TRUE(sum.is_contiguous());
        EXPECT_EQ(sum.rank(), packed.rank());
        EXPECT_EQ(sum.size(), packed.size());
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 2; ++j)
            {
                EXPECT_NEAR(
                    static_cast<T>(sum.at(i, j)), static_cast<T>(packed.at(i, j)) + T(1), eps);
            }
        }
    }

    // Assigning an expression into a non-contiguous destination must fail loudly
    // rather than silently write through the raw pointer as if it were packed
    // (expressions_evaluator::run/fill hard-check the destination).
    {
        tensor_t m(2u, 3u);
        for (size_t i = 0; i < 2; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                m.at(i, j) = static_cast<T>(i + j);
            }
        }
        auto tv = m.t();
        EXPECT_FALSE(tv.is_contiguous());

        // Contiguous sources, non-contiguous destination: isolates the destination
        // check in expressions_evaluator::run() from the (pre-existing) source check
        // in store_operand().
        tensor_t a = m.contiguous();
        tensor_t b = m.contiguous();
        EXPECT_TRUE(a.is_contiguous());
        EXPECT_TRUE(b.is_contiguous());

        logging::set_exception_mode(logging::exception_mode::THROW);
        EXPECT_THROW({ tv = a + b; }, logging::exception);
        EXPECT_THROW({ tv = T(1); }, logging::exception);
    }

    // linspace n==1; empty 2-D initializer
    {
        tensor_t one(T(4), T(9), 1u);
        EXPECT_EQ(one.size(), 1u);
        EXPECT_NEAR(static_cast<T>(one[0]), T(4), eps);

        tensor_t z(std::initializer_list<std::initializer_list<T>>{});
        EXPECT_TRUE(z.empty());
    }
}

/// @brief Shape/state predicates and expression evaluation.
template <typename T>
VECTORIZATION_NOINLINE void test_tensor_predicates()
{
    using tensor_t = vectorization::tensor<T>;
    const T eps    = std::is_same<T, float>::value ? T(1e-5) : T(1e-10);

    // -----------------------------------------------------------------------
    // Predicates
    // -----------------------------------------------------------------------
    {
        // is_zero
        tensor_t z(4u);
        z = T(0);
        EXPECT_TRUE(z.is_zero());
        z[2] = T(1);
        EXPECT_FALSE(z.is_zero());

        // non_negative
        tensor_t nn(3u);
        nn = T(1);
        EXPECT_TRUE(nn.non_negative());
        nn[0] = T(-1);
        EXPECT_FALSE(nn.non_negative());

        // positive
        tensor_t pos(3u);
        pos = T(1);
        EXPECT_TRUE(pos.positive());
        pos[0] = T(0);
        EXPECT_FALSE(pos.positive());

        // symmetric
        tensor_t sym{{T(1), T(2)}, {T(2), T(1)}};
        EXPECT_TRUE(sym.symmetric());
        sym.at(0u, 1u) = T(3);
        EXPECT_FALSE(sym.symmetric());

        // identity
        tensor_t id(2u, 2u);
        id            = T(0);
        id.at(0u, 0u) = T(1);
        id.at(1u, 1u) = T(1);
        EXPECT_TRUE(id.identity());
        id.at(0u, 1u) = T(1);
        EXPECT_FALSE(id.identity());

        // trace
        tensor_t tr{{T(3), T(0)}, {T(0), T(5)}};
        EXPECT_NEAR(static_cast<T>(tr.trace()), T(8), eps);

        // is_correlation: unit diagonal, symmetric, all |off-diag| <= 1
        tensor_t corr{{T(1), T(0.5)}, {T(0.5), T(1)}};
        EXPECT_TRUE(corr.is_correlation());
        tensor_t ncorr{{T(1), T(2)}, {T(2), T(1)}};
        EXPECT_FALSE(ncorr.is_correlation());

        // operator== and !=
        tensor_t a(3u), b(3u);
        a = T(2);
        b = T(2);
        EXPECT_TRUE(a == b);
        b[1] = T(3);
        EXPECT_FALSE(a == b);
        EXPECT_TRUE(a != b);
    }

    // -----------------------------------------------------------------------
    // Expressions
    // -----------------------------------------------------------------------
    {
        tensor_t a(4u), b(4u);
        a = T(3);
        b = T(2);

        auto e = a + b;
        EXPECT_EQ(e.lhs().data(), a.data());
        EXPECT_EQ(e.rhs().data(), b.data());
        EXPECT_EQ(e.size(), a.size());

        // Expression construction
        tensor_t c = a + b;
        EXPECT_EQ(c.size(), 4u);
        EXPECT_NEAR(static_cast<T>(c[0]), T(5), eps);

        // Scalar assignment via expression
        tensor_t sum(4u);
        sum = a + b;
        EXPECT_NEAR(static_cast<T>(sum[0]), T(5), eps);

        tensor_t acc = a.clone();
        acc          = acc + b;
        EXPECT_NEAR(static_cast<T>(acc[0]), T(5), eps);

        // Unary expression
        tensor_t ex = exp(a);
        EXPECT_EQ(ex.size(), 4u);
        EXPECT_NEAR(static_cast<T>(ex[0]), std::exp(T(3)), T(1e-4));
    }
}

/// @brief Contiguity semantics and stream formatting.
template <typename T>
VECTORIZATION_NOINLINE void test_tensor_contiguity()
{
    using tensor_t = vectorization::tensor<T>;
    using dims_t   = typename tensor_t::dimensions_type;

    // -----------------------------------------------------------------------
    // Contiguity semantics -- see Docs/vectorization_tensor_contiguity.md.
    // A tensor is "contiguous" here iff data()[i] for i in [0,size()) matches
    // the row-major element i (the flat-indexing invariant CPU SIMD/scalar
    // loops rely on). A size-1 dimension's stride is never read by any valid
    // index, so it must not gate contiguity (matches PyTorch's
    // TensorImpl::compute_contiguous()).
    // -----------------------------------------------------------------------

    // Fresh construction is always contiguous, at any rank.
    {
        tensor_t v(5u);
        EXPECT_TRUE(v.is_contiguous());
        tensor_t m(dims_t{3, 4});
        EXPECT_TRUE(m.is_contiguous());
        tensor_t vol(dims_t{2, 3, 4});
        EXPECT_TRUE(vol.is_contiguous());
    }

    // A size-0 dimension makes the tensor unconditionally contiguous (same
    // convention as rank-0): there is no element that could be out of place.
    {
        tensor_t empty_dim(dims_t{0, 5});
        EXPECT_EQ(empty_dim.size(), 0u);
        EXPECT_TRUE(empty_dim.is_contiguous());
    }

    // Copy construction/assignment recomputes contiguity from the copied
    // shape/strides -- it mirrors the source, it is not always true.
    {
        tensor_t m(dims_t{3, 4});
        tensor_t transposed = m.t();
        EXPECT_FALSE(transposed.is_contiguous());

        tensor_t copy_ctor(transposed);
        EXPECT_FALSE(copy_ctor.is_contiguous());

        tensor_t copy_assign;
        copy_assign = transposed;
        EXPECT_FALSE(copy_assign.is_contiguous());

        tensor_t contiguous_src(dims_t{3, 4});
        tensor_t copy_of_contiguous(contiguous_src);
        EXPECT_TRUE(copy_of_contiguous.is_contiguous());
    }

    // Move construction/assignment: the moved-to tensor mirrors the source's
    // (cached, not recomputed) flag; the moved-from tensor resets to its
    // default empty/contiguous state.
    {
        tensor_t m(dims_t{3, 4});
        tensor_t transposed = m.t();
        EXPECT_FALSE(transposed.is_contiguous());

        tensor_t moved(std::move(transposed));
        EXPECT_FALSE(moved.is_contiguous());
        EXPECT_TRUE(transposed.is_contiguous());  // moved-from: reset to empty/contiguous
        EXPECT_EQ(transposed.size(), 0u);

        tensor_t move_target;
        move_target = std::move(moved);
        EXPECT_FALSE(move_target.is_contiguous());
        EXPECT_TRUE(moved.is_contiguous());
    }

    // t(): a genuine rank-2 transpose is non-contiguous; transposing a
    // vector-shaped (N,1) tensor to (1,N) stays contiguous -- the size-1
    // dimension carries no memory-layout information either way. This is the
    // simplest instance of the bug shape fixed this session (see the
    // "permute() moving a size-1 axis" regression test above).
    {
        tensor_t mat(dims_t{2, 3});
        EXPECT_TRUE(mat.is_contiguous());
        EXPECT_FALSE(mat.t().is_contiguous());

        tensor_t col(dims_t{5, 1});
        EXPECT_TRUE(col.is_contiguous());
        tensor_t row = col.t();
        EXPECT_EQ(row.dimension(0), 1u);
        EXPECT_EQ(row.dimension(1), 5u);
        EXPECT_TRUE(row.is_contiguous());
        EXPECT_EQ(row.data(), col.data());
    }

    // permute(): the identity permutation is always contiguous; reordering
    // real (size>1) axes generally is not.
    {
        tensor_t vol(dims_t{2, 3, 4});
        EXPECT_TRUE(vol.permute(dims_t{0, 1, 2}).is_contiguous());
        EXPECT_FALSE(vol.permute(dims_t{2, 0, 1}).is_contiguous());
        EXPECT_FALSE(vol.permute(dims_t{1, 0, 2}).is_contiguous());
    }

    // view()/reshape(): always produce freshly-built canonical strides, so
    // the result is always contiguous (both require a contiguous source).
    {
        tensor_t v(12u);
        EXPECT_TRUE(v.view(dims_t{3, 4}).is_contiguous());
        EXPECT_TRUE(v.reshape(dims_t{2, 2, 3}).is_contiguous());
    }

    // slice(): whether the result stays contiguous depends on which
    // dimension is sliced and how -- not simply "slicing implies
    // non-contiguous". See Docs/vectorization_tensor_contiguity.md #4 for the
    // full case table this mirrors.
    {
        tensor_t m(dims_t{4, 5});
        EXPECT_TRUE(m.is_contiguous());

        // Full no-op slice: identical to the source.
        EXPECT_TRUE(m.slice(0, 0, 4, 1).is_contiguous());

        // Contiguous sub-range of whole outer rows: still a packed block.
        EXPECT_TRUE(m.slice(0, 1, 3, 1).is_contiguous());

        // Outer dim sliced down to a single row: that size-1 result
        // dimension is skipped, leaving one packed inner row.
        EXPECT_TRUE(m.slice(0, 1, 2, 1).is_contiguous());

        // Sub-range of the inner dimension: each row now leaves a real gap.
        EXPECT_FALSE(m.slice(1, 0, 3, 1).is_contiguous());

        // Inner dim sliced down to a single column: the outer dimension's
        // stride (5) no longer equals the expected packed stride (1) --
        // genuine stride-5 gaps between the selected elements. Contrast with
        // the single-row case above: collapsing to size 1 is not itself
        // sufficient for contiguity, only for exempting that one dimension.
        EXPECT_FALSE(m.slice(1, 1, 2, 1).is_contiguous());

        // Strided slice (step > 1): introduces gaps.
        EXPECT_FALSE(m.slice(1, 0, 5, 2).is_contiguous());

        // Slicing to an empty result: unconditionally contiguous (numel==0).
        auto empty_slice = m.slice(0, 2, 2, 1);
        EXPECT_EQ(empty_slice.size(), 0u);
        EXPECT_TRUE(empty_slice.is_contiguous());
    }

    // clone(): always a freshly packed, contiguous copy, regardless of
    // whether the source was contiguous.
    {
        tensor_t m(dims_t{3, 4});
        tensor_t transposed = m.t();
        EXPECT_FALSE(transposed.is_contiguous());
        tensor_t cloned = transposed.clone();
        EXPECT_TRUE(cloned.is_contiguous());
        EXPECT_NE(cloned.data(), transposed.data());
    }

    // contiguous(): borrows (same pointer) when already contiguous, clones
    // (new pointer) otherwise; the result is always contiguous either way.
    {
        tensor_t m(dims_t{3, 4});
        tensor_t already = m.contiguous();
        EXPECT_TRUE(already.is_contiguous());
        EXPECT_EQ(already.data(), m.data());

        tensor_t transposed = m.t();
        tensor_t packed     = transposed.contiguous();
        EXPECT_TRUE(packed.is_contiguous());
        EXPECT_NE(packed.data(), transposed.data());
    }

    // -----------------------------------------------------------------------
    // Formatting
    // -----------------------------------------------------------------------
    {
        tensor_t    v{T(1), T(2)};
        std::string s = v.to_string();
        EXPECT_FALSE(s.empty());
        EXPECT_NE(s, std::string("[]"));

        std::ostringstream oss;
        oss << v;
        EXPECT_EQ(oss.str(), s);
    }
}

/// @brief Runs the full tensor test suite for element type @p T.
template <typename T>
void test_tensor()
{
    test_tensor_construction<T>();
    test_tensor_shape_access<T>();
    test_tensor_views_slices<T>();
    test_tensor_predicates<T>();
    test_tensor_contiguity<T>();
}
}  // namespace

VECTORIZATIONTEST(Math, Tensor)
{
    test_tensor<float>();
    test_tensor<double>();

    END_TEST();
}
