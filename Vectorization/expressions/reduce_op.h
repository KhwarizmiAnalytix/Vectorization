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

#include <cfloat>
#include <type_traits>

#include "common/vectorization_macros.h"

namespace vectorization
{

// Horizontal reduction over an expression (or a device buffer of partials).
// Matches CPU accumulate / hmin / hmax identities: empty sum is 0, empty min
// is max(), empty max is -max() (not lowest(), which is the smallest *positive*
// finite float).
enum class reduce_op : int
{
    sum = 0,
    min = 1,
    max = 2
};

// MSVC+nvcc treats std::numeric_limits<T>::max() as a host-only function, so
// a __host__ __device__ identity must use the C macros (literal constants).
template <typename T>
VECTORIZATION_FUNCTION_ATTRIBUTE T reduce_finite_max()
{
    if constexpr (std::is_same_v<T, float>)
    {
        return FLT_MAX;
    }
    else
    {
        static_assert(std::is_same_v<T, double>, "reduce_identity supports float and double");
        return DBL_MAX;
    }
}

template <typename T, reduce_op Op>
VECTORIZATION_FUNCTION_ATTRIBUTE T reduce_identity()
{
    if constexpr (Op == reduce_op::sum)
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

template <typename T, reduce_op Op>
VECTORIZATION_FUNCTION_ATTRIBUTE T reduce_combine(T a, T b)
{
    if constexpr (Op == reduce_op::sum)
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
