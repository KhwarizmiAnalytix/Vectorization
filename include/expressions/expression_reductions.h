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

#ifndef VECTORIZATION_EXPRESSIONS_INCLUDES_INSIDE
Do_not_include_expression_reductions_directly_use_expressions_h;
#endif

// Included after expressions_evaluator.h: shares device placement, shape
// validation and strided lowering with the original horizontal reductions.

namespace vectorization
{
namespace detail
{
#if VECTORIZATION_VECTORIZED
template <typename T, reduce_op Op>
VECTORIZATION_HOST_FUNCTION_ATTRIBUTE
    typename simd<T>::simd_t reduce_packet_transform(typename simd<T>::simd_t x)
{
    using S         = simd<T>;
    const auto zero = S::set(T(0));
    const auto one  = S::set(T(1));
    if constexpr (Op == reduce_op::all || Op == reduce_op::any || Op == reduce_op::count_nonzero)
    {
        return S::if_else(S::neq(x, zero), one, zero);
    }
    else if constexpr (Op == reduce_op::count_valid)
    {
        return S::if_else(S::eq(x, x), one, zero);
    }
    else if constexpr (Op == reduce_op::nansum)
    {
        return S::if_else(S::eq(x, x), x, zero);
    }
    else
    {
        return x;
    }
}

template <typename T, reduce_op Op>
VECTORIZATION_HOST_FUNCTION_ATTRIBUTE typename simd<T>::simd_t reduce_packet_combine(
    typename simd<T>::simd_t a, typename simd<T>::simd_t b)
{
    if constexpr (Op == reduce_op::product)
    {
        return simd<T>::mul(a, b);
    }
    else if constexpr (Op == reduce_op::all)
    {
        return simd<T>::min(a, b);
    }
    else if constexpr (Op == reduce_op::any)
    {
        return simd<T>::max(a, b);
    }
    else
    {
        return simd<T>::add(a, b);
    }
}
#endif

template <reduce_op Op, typename E>
VECTORIZATION_HOST_FUNCTION_ATTRIBUTE auto reduce_expression(E const& expression)
{
    using T     = typename scalar_type<E, E>::value;
    using acc_t = reduce_accumulator_t<T, Op>;
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
        "reductions support float and double expressions");
    acc_t result = reduce_identity<acc_t, Op>();
    if (try_reduce_gpu<E, T, Op>(expression, result))
    {
        return result;
    }
    if (reduce_strided(expression,
            result,
            [](acc_t a, T b) { return reduce_combine<acc_t, Op>(a, reduce_transform<T, Op>(b)); }))
    {
        return result;
    }

    size_t       i = 0;
    const size_t n = expression.size();
#if VECTORIZATION_VECTORIZED
    using S                     = simd<T>;
    constexpr size_t length     = E::length();
    constexpr size_t packet     = VECTORIZATION_PACKET_SIZE < 4 ? 4 : VECTORIZATION_PACKET_SIZE;
    constexpr size_t block      = length * packet;
    const size_t     vector_end = n - n % length;
    while (i < vector_end)
    {
        // Bounded count batches keep each float lane exact. Convert each lane
        // to uint64_t before merging; never sum a large count in float.
        const size_t remaining = vector_end - i;
        const size_t batch =
            reduce_is_count<Op> && remaining > 4096 * length ? 4096 * length : remaining;
        const size_t       end = i + batch;
        typename S::simd_t accumulators[packet];
        unroll<packet>(
            [&](auto k)
            {
                accumulators[decltype(k)::value] =
                    S::set(static_cast<T>(reduce_identity<acc_t, Op>()));
            });
        auto update = [&](size_t k, size_t offset)
        {
            auto x = expression_loader<E, true, false>::evaluate(expression, offset);
            // Comparison expressions produce predicates, including on ISAs
            // whose predicate and numeric register types happen to coincide.
            using scalar_result =
                decltype(expression_loader<E, false, false>::evaluate(expression, 0));
            if constexpr (std::is_same_v<scalar_result, bool>)
            {
                accumulators[k] = reduce_packet_combine<T, Op>(accumulators[k],
                    reduce_packet_transform<T, Op>(S::if_else(x, S::set(T(1)), S::set(T(0)))));
            }
            else
            {
                accumulators[k] = reduce_packet_combine<T, Op>(
                    accumulators[k], reduce_packet_transform<T, Op>(x));
            }
        };
        for (; end - i >= block; i += block)
        {
            unroll<packet>(
                [&](auto k) { update(decltype(k)::value, i + decltype(k)::value * length); });
        }
        for (; i < end; i += length)
        {
            update(0, i);
        }
        if constexpr (reduce_is_count<Op> || Op == reduce_op::product)
        {
            T lanes[length];
            for (size_t k = 0; k < packet; ++k)
            {
                S::storeu(accumulators[k], lanes);
                for (size_t lane = 0; lane < length; ++lane)
                {
                    result = reduce_combine<acc_t, Op>(result, static_cast<acc_t>(lanes[lane]));
                }
            }
        }
        else
        {
            auto combined = accumulators[0];
            for (size_t k = 1; k < packet; ++k)
            {
                combined = reduce_packet_combine<T, Op>(combined, accumulators[k]);
            }
            if constexpr (Op == reduce_op::all)
            {
                result = reduce_combine<acc_t, Op>(result, S::hmin(combined));
            }
            else if constexpr (Op == reduce_op::any)
            {
                result = reduce_combine<acc_t, Op>(result, S::hmax(combined));
            }
            else
            {
                result += static_cast<T>(S::accumulate(combined));
            }
        }
    }
#endif
    for (; i < n; ++i)
    {
        const T x = expression_loader<E, false, false>::evaluate(expression, i);
        result    = reduce_combine<acc_t, Op>(result, reduce_transform<T, Op>(x));
    }
    return result;
}
}  // namespace detail

// Whole-expression reductions; no intermediate tensor is constructed.
template <typename E> VECTORIZATION_HOST_FUNCTION_ATTRIBUTE auto sum(E const& expression)
{
    return detail::reduce_expression<reduce_op::sum>(expression);
}

template <typename E> VECTORIZATION_HOST_FUNCTION_ATTRIBUTE auto prod(E const& expression)
{
    return detail::reduce_expression<reduce_op::product>(expression);
}

template <typename E> VECTORIZATION_HOST_FUNCTION_ATTRIBUTE auto mean(E const& expression)
{
    using T       = typename scalar_type<E, E>::value;
    const T total = sum(expression);  // also validates shapes and device placement for empty inputs
    return expression.size() == 0 ? std::numeric_limits<T>::quiet_NaN()
                                  : total / static_cast<T>(expression.size());
}

template <typename E> VECTORIZATION_HOST_FUNCTION_ATTRIBUTE bool all(E const& expression)
{
    return detail::reduce_expression<reduce_op::all>(expression) != 0;
}

template <typename E> VECTORIZATION_HOST_FUNCTION_ATTRIBUTE bool any(E const& expression)
{
    return detail::reduce_expression<reduce_op::any>(expression) != 0;
}

template <typename E>
VECTORIZATION_HOST_FUNCTION_ATTRIBUTE std::uint64_t count_nonzero(E const& expression)
{
    return detail::reduce_expression<reduce_op::count_nonzero>(expression);
}

template <typename E> VECTORIZATION_HOST_FUNCTION_ATTRIBUTE auto nansum(E const& expression)
{
    return detail::reduce_expression<reduce_op::nansum>(expression);
}

template <typename E> VECTORIZATION_HOST_FUNCTION_ATTRIBUTE auto nanmean(E const& expression)
{
    using T          = typename scalar_type<E, E>::value;
    const T    total = nansum(expression);
    const auto count = detail::reduce_expression<reduce_op::count_valid>(expression);
    return count == 0 ? std::numeric_limits<T>::quiet_NaN() : total / static_cast<T>(count);
}
}  // namespace vectorization
