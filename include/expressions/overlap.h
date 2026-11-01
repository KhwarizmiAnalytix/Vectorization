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

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <numeric>
#include <type_traits>

#include "expressions/expression_shape.h"
#include "expressions/strided_ref.h"

namespace vectorization
{
enum class overlap_status : uint8_t
{
    none,
    full,
    partial,
    too_hard
};

namespace detail
{
struct overlap_axis
{
    int64_t size;
    int64_t stride;
};

template <typename T, int N>
int overlap_axes(strided_ref<T, N> const& ref, std::array<overlap_axis, N>& axes)
{
    int count = 0;
    for (int d = 0; d < ref.rank; ++d)
    {
        if (ref.sizes[d] > 1)
        {
            axes[count++] = {ref.sizes[d], ref.strides[d]};
        }
    }
    std::sort(axes.begin(),
        axes.begin() + count,
        [](auto const& a, auto const& b) { return a.stride < b.stride; });
    return count;
}

template <typename T, int N> uint64_t overlap_extent(strided_ref<T, N> const& ref)
{
    uint64_t extent = 0;
    for (int d = 0; d < ref.rank; ++d)
    {
        extent += static_cast<uint64_t>(ref.sizes[d] - 1) * static_cast<uint64_t>(ref.strides[d]);
    }
    return extent;
}

inline int64_t floor_div(int64_t n, int64_t d)
{
    return n / d - (n % d < 0 ? 1 : 0);
}
inline int64_t ceil_div(int64_t n, int64_t d)
{
    return n / d + (n % d > 0 ? 1 : 0);
}
}  // namespace detail

template <typename T, int N> bool is_non_overlapping(strided_ref<T, N> const& ref)
{
    if (ref.numel == 0)
    {
        return true;
    }
    std::array<detail::overlap_axis, N> axes{};
    const int                           count  = detail::overlap_axes<T, N>(ref, axes);
    uint64_t                            extent = 0;
    for (int d = 0; d < count; ++d)
    {
        if (axes[d].stride <= 0 || static_cast<uint64_t>(axes[d].stride) <= extent)
        {
            return false;
        }
        extent += static_cast<uint64_t>(axes[d].size - 1) * static_cast<uint64_t>(axes[d].stride);
    }
    return true;
}

template <typename A, typename B, int N, int M>
overlap_status classify_overlap(strided_ref<A, N> const& a, strided_ref<B, M> const& b)
{
    if (a.numel == 0 || b.numel == 0 || !a.data() || !b.data() || a.device() != b.device() ||
        a.device_index() != b.device_index())
    {
        return overlap_status::none;
    }
    const auto ap = reinterpret_cast<uintptr_t>(a.data());
    const auto bp = reinterpret_cast<uintptr_t>(b.data());
    const auto ae = ap + (detail::overlap_extent(a) + 1) * sizeof(A);
    const auto be = bp + (detail::overlap_extent(b) + 1) * sizeof(B);
    if (ae <= bp || be <= ap)
    {
        return overlap_status::none;
    }
    if constexpr (!std::is_same_v<std::remove_const_t<A>, std::remove_const_t<B>>)
    {
        return overlap_status::too_hard;
    }
    bool same = ap == bp && a.rank == b.rank;
    for (int d = 0; same && d < a.rank; ++d)
    {
        same = a.sizes[d] == b.sizes[d] && a.strides[d] == b.strides[d];
    }
    if (same && is_non_overlapping(a))
    {
        return overlap_status::full;
    }
    if (!is_non_overlapping(a) || !is_non_overlapping(b))
    {
        return overlap_status::too_hard;
    }
    if (detail::overlap_extent(a) + 1 == a.numel && detail::overlap_extent(b) + 1 == b.numel)
    {
        return overlap_status::partial;
    }

    const auto distance = ap <= bp ? bp - ap : ap - bp;
    if (distance % sizeof(A) != 0 || distance / sizeof(A) > static_cast<uint64_t>(INT64_MAX))
    {
        return overlap_status::too_hard;
    }
    const int64_t delta = ap <= bp ? static_cast<int64_t>(distance / sizeof(A))
                                   : -static_cast<int64_t>(distance / sizeof(A));
    std::array<detail::overlap_axis, N> ax{};
    std::array<detail::overlap_axis, M> bx{};
    const int                           na  = detail::overlap_axes<A, N>(a, ax);
    const int                           nb  = detail::overlap_axes<B, M>(b, bx);
    int64_t                             gcd = 0;
    for (int d = 0; d < na; ++d)
    {
        gcd = std::gcd(gcd, ax[d].stride);
    }
    for (int d = 0; d < nb; ++d)
    {
        gcd = std::gcd(gcd, bx[d].stride);
    }
    if (gcd && delta % gcd != 0)
    {
        return overlap_status::none;
    }
    if (na != nb || na == 0)
    {
        return overlap_status::too_hard;
    }
    for (int d = 0; d < na; ++d)
    {
        if (ax[d].stride != bx[d].stride)
        {
            return overlap_status::too_hard;
        }
        if (d && ax[d].stride % ax[d - 1].stride != 0)
        {
            return overlap_status::too_hard;
        }
    }
    if (delta % ax[0].stride != 0)
    {
        return overlap_status::none;
    }
    // All possible carries form an integer interval. Signed bounds are essential:
    // offsets can intersect through a positive outer and negative inner coefficient.
    int64_t lo = delta / ax[0].stride;
    int64_t hi = lo;
    for (int d = 0; d < na; ++d)
    {
        const int64_t lower = -(bx[d].size - 1);
        const int64_t upper = ax[d].size - 1;
        if (d + 1 == na)
        {
            return hi >= lower && lo <= upper ? overlap_status::partial : overlap_status::none;
        }
        // Avoid overflowing subtraction on externally supplied huge layouts.
        if (lo < INT64_MIN + upper || hi > INT64_MAX + lower)
        {
            return overlap_status::too_hard;
        }
        const int64_t period = ax[d + 1].stride / ax[d].stride;
        lo                   = detail::ceil_div(lo - upper, period);
        hi                   = detail::floor_div(hi - lower, period);
        if (lo > hi)
        {
            return overlap_status::none;
        }
    }
    return overlap_status::none;
}

template <typename E, typename T, int N>
void check_assignment_overlap(E const& expr, strided_ref<T, N> const& dst, bool noalias)
{
#ifdef NDEBUG
    if (noalias)
    {
        return;
    }
#endif
    using raw = remove_cvref_t<E>;
    if constexpr (is_pure_expression<raw>::value)
    {
        if constexpr (VECTORIZATION_EXPR_HAS_LHS(expr))
        {
            check_assignment_overlap(expr.lhs(), dst, noalias);
        }
        if constexpr (VECTORIZATION_EXPR_HAS_MHS(expr))
        {
            check_assignment_overlap(expr.mhs(), dst, noalias);
        }
        check_assignment_overlap(expr.rhs(), dst, noalias);
    }
    else if constexpr (detail::is_data_view<raw>::value)
    {
        using value_t = std::remove_pointer_t<decltype(expr.data())>;
        strided_ref<value_t> ref;
        ref.view_      = expr;
        ref.rank       = 1;
        ref.sizes[0]   = static_cast<int64_t>(expr.size());
        ref.strides[0] = 1;
        ref.numel      = expr.size();
        ref.contiguous = true;
        check_assignment_overlap(ref, dst, noalias);
    }
    else if constexpr (is_strided_ref<raw>::value)
    {
        const auto status = classify_overlap(dst, expr);
        VECTORIZATION_CHECK(status != overlap_status::partial,
            "assignment has partial overlap; use dst = src.clone()");
        VECTORIZATION_CHECK(noalias || status != overlap_status::too_hard,
            "assignment overlap is too hard to prove; use src.clone() or dst.noalias() = expr");
    }
}
}  // namespace vectorization
