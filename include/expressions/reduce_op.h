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

#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

#include "common/vectorization_macros.h"

namespace vectorization
{

// Horizontal reduction over an expression (or a device buffer of partials).
// Matches CPU accumulate / hmin / hmax identities: empty sum is 0, empty min
// is max(), empty max is -max(). Preserve these legacy extrema identities.
enum class reduce_op : int
{
    sum = 0,
    min = 1,
    max = 2,
    product,
    all,
    any,
    nansum,
    count_nonzero,
    count_valid
};

template <reduce_op Op>
inline constexpr bool reduce_is_count =
    Op == reduce_op::count_nonzero || Op == reduce_op::count_valid;

// Counts must not round at float's 2^24 consecutive-integer limit.
template <typename T, reduce_op Op>
using reduce_accumulator_t = std::conditional_t<reduce_is_count<Op>, std::uint64_t, T>;

// Transform input only on the first pass; subsequent passes combine partials.
template <typename T, reduce_op Op>
VECTORIZATION_FUNCTION_ATTRIBUTE reduce_accumulator_t<T, Op> reduce_transform(T x)
{
    if constexpr (Op == reduce_op::all || Op == reduce_op::any || Op == reduce_op::count_nonzero)
    {
        return x != T(0);
    }
    else if constexpr (Op == reduce_op::count_valid)
    {
        return x == x;
    }
    else if constexpr (Op == reduce_op::nansum)
    {
        return x == x ? x : T(0);
    }
    else
    {
        return x;
    }
}

template <typename T> VECTORIZATION_FUNCTION_ATTRIBUTE T reduce_finite_max()
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
        "reduce_identity supports float and double");
    return std::numeric_limits<T>::max();
}

template <typename T, reduce_op Op> VECTORIZATION_FUNCTION_ATTRIBUTE T reduce_identity()
{
    if constexpr (Op == reduce_op::product || Op == reduce_op::all)
    {
        return static_cast<T>(1);
    }
    else if constexpr (Op != reduce_op::min && Op != reduce_op::max)
    {
        return static_cast<T>(0);
    }
    else if constexpr (Op == reduce_op::min)
    {
        return reduce_finite_max<T>();
    }
    else
    {
        return -reduce_finite_max<T>();
    }
}

template <typename T, reduce_op Op> VECTORIZATION_FUNCTION_ATTRIBUTE T reduce_combine(T a, T b)
{
    if constexpr (Op == reduce_op::product)
    {
        return a * b;
    }
    else if constexpr (Op == reduce_op::all)
    {
        return static_cast<T>(a != T(0) && b != T(0));
    }
    else if constexpr (Op == reduce_op::any)
    {
        return static_cast<T>(a != T(0) || b != T(0));
    }
    else if constexpr (Op != reduce_op::min && Op != reduce_op::max)
    {
        return a + b;
    }
    else if constexpr (Op == reduce_op::min)
    {
        return a < b ? a : b;
    }
    else
    {
        return a > b ? a : b;
    }
}

}  // namespace vectorization
