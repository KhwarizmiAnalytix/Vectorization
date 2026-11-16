/*
 * Vectorization: High-Performance Quantitative Library
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "linalg/solve.h"

namespace vectorization
{
namespace detail
{
template <typename A> auto inv_impl(A const& a, bool check_errors, char const* op)
{
    using T                                     = operand_value_t<A>;
    auto                                factors = lu_factor_impl(a, check_errors, op);
    const auto                          n       = static_cast<size_t>(factor_matrix_order(a));
    const auto                          batch   = factor_batch_count(a);
    typename tensor<T>::dimensions_type dims(a.dimensions().begin(), a.dimensions().end());
    tensor<T>                           inverse(dims, device_enum::CPU, a.device_index());
    if (inverse.size() != 0)
    {
        std::vector<T> identity(n * n, T(0));
        for (size_t i = 0; i < n; ++i)
        {
            identity[i * n + i] = T(1);
        }
        for (size_t b = 0; b < batch; ++b)
        {
            lu_solve_standard_block(factors.LU.data() + b * n * n,
                factors.pivots.data() + b * n,
                identity.data(),
                inverse.data() + b * n * n,
                static_cast<int64_t>(n),
                static_cast<int64_t>(n));
        }
    }
    return inv_ex_result<T>{std::move(inverse), std::move(factors.info)};
}
}  // namespace detail

template <typename A, std::enable_if_t<are_tensor_operands_v<A>, bool> = true>
auto inv_ex(A const& a, bool check_errors = false)
{
    return detail::inv_impl(a, check_errors, "inv_ex");
}

template <typename A, std::enable_if_t<are_tensor_operands_v<A>, bool> = true> auto inv(A const& a)
{
    return detail::inv_impl(a, true, "inv").inverse;
}

template <typename A, std::enable_if_t<are_tensor_operands_v<A>, bool> = true>
auto slogdet(A const& a)
{
    using T            = operand_value_t<A>;
    auto       factors = detail::lu_factor_impl(a, false, "slogdet");
    auto       dims    = detail::factor_batch_dims<T>(a);
    tensor<T>  sign(dims, device_enum::CPU, a.device_index());
    tensor<T>  logabsdet(dims, device_enum::CPU, a.device_index());
    const auto n = static_cast<size_t>(detail::factor_matrix_order(a));
    for (size_t b = 0; b < sign.size(); ++b)
    {
        T s      = 1;
        T logabs = 0;
        for (size_t i = 0; i < n; ++i)
        {
            const T diagonal = factors.LU[b * n * n + i * n + i];
            if (factors.pivots[b * n + i] != static_cast<T>(i + 1))
            {
                s = -s;
            }
            s *= diagonal > T(0)    ? T(1)
                 : diagonal < T(0)  ? T(-1)
                 : diagonal == T(0) ? T(0)
                                    : std::numeric_limits<T>::quiet_NaN();
            logabs += std::log(std::abs(diagonal));
        }
        sign[b]      = s;
        logabsdet[b] = logabs;
    }
    return slogdet_result<T>{std::move(sign), std::move(logabsdet)};
}

template <typename A, std::enable_if_t<are_tensor_operands_v<A>, bool> = true> auto det(A const& a)
{
    using T            = operand_value_t<A>;
    auto       factors = detail::lu_factor_impl(a, false, "det");
    tensor<T>  result(detail::factor_batch_dims<T>(a), device_enum::CPU, a.device_index());
    const auto n = static_cast<size_t>(detail::factor_matrix_order(a));
    for (size_t b = 0; b < result.size(); ++b)
    {
        T value = 1;
        for (size_t i = 0; i < n; ++i)
        {
            value *= factors.LU[b * n * n + i * n + i];
            if (factors.pivots[b * n + i] != static_cast<T>(i + 1))
            {
                value = -value;
            }
        }
        result[b] = value;
    }
    return result;
}

namespace linalg
{
using vectorization::det;
using vectorization::inv;
using vectorization::inv_ex;
using vectorization::slogdet;
}  // namespace linalg
}  // namespace vectorization
