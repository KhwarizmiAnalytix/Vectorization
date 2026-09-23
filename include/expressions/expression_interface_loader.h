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

#include "backend/simd.h"
#include "common/scalar_helper_functions.h"
#include "common/vectorization_macros.h"
#include "common/vectorization_type_traits.h"

namespace vectorization
{
template <typename value_t>
class tensor;

template <typename LHS, bool vectorize, bool aligned>
class expression_loader final
{
public:
    using rmv_lhs = vectorization::remove_cvref_t<LHS>;

    VECTORIZATION_FUNCTION_ATTRIBUTE static auto evaluate(
        rmv_lhs const& expr, size_t index) noexcept
    {
        if constexpr (vectorization::is_base_expression<rmv_lhs>::value)
        {
            if constexpr (vectorize)
            {
                using value_t = typename scalar_type<rmv_lhs, rmv_lhs>::value;

                auto const* ptr = expr.data() + index;

#if VECTORIZATION_VECTORIZED
                if constexpr (aligned)
                    return simd<value_t>::load(ptr);
                else
                    return simd<value_t>::loadu(ptr);
#else
                return simd<value_t>::loadu(ptr);
#endif
            }
            else
            {
                return expr.data()[index];
            }
        }
        else if constexpr (vectorization::is_pure_expression<rmv_lhs>::value)
        {
            return rmv_lhs::template evaluate<vectorize, aligned>(expr, index);
        }
        else if constexpr (std::is_fundamental<rmv_lhs>::value)
        {
            return expr;
        }
    }

    // Prefetch is a CPU-only hint; issued by the caller one SIMD register ahead of the
    // matching evaluate(expr, index) call instead of being buried inside it, so hot loops
    // control prefetch distance explicitly (see expressions_evaluator.h). Mirrors evaluate()'s
    // dispatch so every leaf of a composite expression tree still gets prefetched.
    VECTORIZATION_FUNCTION_ATTRIBUTE static void prefetch(
        rmv_lhs const& expr, size_t index) noexcept
    {
        if constexpr (vectorization::is_base_expression<rmv_lhs>::value)
        {
            if constexpr (vectorize)
            {
#if !VECTORIZATION_ON_GPU_DEVICE
                using value_t = typename scalar_type<rmv_lhs, rmv_lhs>::value;
                simd<value_t>::prefetch(expr.data() + index + simd<value_t>::size);
#endif
            }
        }
        else if constexpr (vectorization::is_pure_expression<rmv_lhs>::value)
        {
            rmv_lhs::template prefetch<vectorize, aligned>(expr, index);
        }
    }
};
}  // namespace vectorization
