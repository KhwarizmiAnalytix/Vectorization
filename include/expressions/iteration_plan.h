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
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#if VECTORIZATION_ENABLE_PARALLEL
#include "include/tools/parallel_tools.h"
#endif
#include "common/vectorization_macros.h"
#include "common/vectorization_type_traits.h"
#include "expressions/expression_interface.h"
#include "expressions/expression_interface_loader.h"
#include "expressions/expression_shape.h"
#include "expressions/strided_ref.h"

// CPU iteration planner for elementwise expressions whose operands are not all
// packed with the destination's shape (design §5). Modelled on PyTorch's
// TensorIterator: operand strides are expressed in the destination's index
// space, dimensions are sorted and merged, and the merged space is walked one
// row at a time by a loop2d kernel. Rows are advanced with a dimension
// counter, so no index is computed by division inside the loop.

namespace vectorization
{

// Destination plus tensor leaves an expression may have.
static constexpr int kMaxOperands = 32;

// Loop tiers after merging (design §5):
//   0 — one dimension, every operand unit-stride: a single SIMD row
//   1 — innermost stride 1 for every operand: SIMD rows
//   2 — destination unit-stride, sources 1 or 0 (row broadcast): SIMD rows, splatted loads
//   3 — anything else: scalar rows
struct iteration_plan
{
    int         rank{0};        // after dropping size-1 dimensions and merging
    int         n_operands{0};  // destination + every tensor leaf, in tree order
    int         tier{3};
    int64_t     numel{0};
    int64_t     sizes[kMaxRank]{};                  // outermost first; sizes[rank - 1] is the row
    int64_t     strides[kMaxOperands][kMaxRank]{};  // operand 0 is the destination
    void const* bases[kMaxOperands]{};              // first element of each source (index 0 unused)

    VECTORIZATION_FUNCTION_ATTRIBUTE int64_t row_size() const noexcept
    {
        return rank > 0 ? sizes[rank - 1] : 0;
    }
    VECTORIZATION_FUNCTION_ATTRIBUTE int64_t rows() const noexcept
    {
        const int64_t n = row_size();
        return n > 0 ? numel / n : 0;
    }
};

namespace detail
{

// Appends each tensor leaf's base pointer and its strides in the result's index
// space (stride 0 on broadcast dimensions), in the same order lower_rows()
// visits leaves.
template <typename E>
VECTORIZATION_HOST_FUNCTION_ATTRIBUTE void plan_leaves(
    E const& expr, expression_shape const& out, iteration_plan& plan)
{
    using expr_t = vectorization::remove_cvref_t<E>;
    if constexpr (is_pure_expression<expr_t>::value)
    {
        if constexpr (VECTORIZATION_EXPR_HAS_MHS(expr))
        {
            plan_leaves(expr.lhs(), out, plan);
            plan_leaves(expr.mhs(), out, plan);
            plan_leaves(expr.rhs(), out, plan);
        }
        else if constexpr (VECTORIZATION_EXPR_HAS_LHS(expr))
        {
            plan_leaves(expr.lhs(), out, plan);
            plan_leaves(expr.rhs(), out, plan);
        }
        else
        {
            plan_leaves(expr.rhs(), out, plan);
        }
    }
    else if constexpr (is_strided_ref<expr_t>::value || is_data_view<expr_t>::value)
    {
        VECTORIZATION_CHECK(plan.n_operands < kMaxOperands,
            "expression has more than {} tensor operands",
            kMaxOperands - 1);
        const int op   = plan.n_operands++;
        plan.bases[op] = expr.data();
        if constexpr (is_strided_ref<expr_t>::value)
        {
            broadcast_strides(out, expr.rank, expr.sizes, expr.strides, plan.strides[op]);
        }
        else
        {
            const int64_t size   = static_cast<int64_t>(expr.size());
            const int64_t stride = 1;
            broadcast_strides(out, 1, &size, &stride, plan.strides[op]);
        }
    }
}

// Type of an expression whose tensor leaves are replaced by row_ref.
template <typename E, typename = void> struct row_lowered
{
    using type = E;  // scalars
};

template <typename T, int MaxRank> struct row_lowered<strided_ref<T, MaxRank>>
{
    using type = row_ref<std::remove_const_t<T>>;
};

template <typename T> struct row_lowered<memory::data_view<T>>
{
    using type = row_ref<std::remove_const_t<T>>;
};

template <typename E>
using row_lowered_t = typename row_lowered<vectorization::remove_cvref_t<E>>::type;

template <typename L, typename Ev> struct row_lowered<unary_expression<L, Ev>>
{
    using type = unary_expression<row_lowered_t<stored_operand_t<remove_cvref_t<L>>>, Ev>;
};

template <typename L, typename R, typename Ev> struct row_lowered<binary_expression<L, R, Ev>>
{
    using type = binary_expression<row_lowered_t<stored_operand_t<remove_cvref_t<L>>>,
        row_lowered_t<stored_operand_t<remove_cvref_t<R>>>,
        Ev>;
};

template <typename L, typename M, typename R, typename Ev>
struct row_lowered<trinary_expression<L, M, R, Ev>>
{
    using type = trinary_expression<row_lowered_t<stored_operand_t<remove_cvref_t<L>>>,
        row_lowered_t<stored_operand_t<remove_cvref_t<M>>>,
        row_lowered_t<stored_operand_t<remove_cvref_t<R>>>,
        Ev>;
};

// Rebuilds the tree with row_ref leaves; pointers are bound per row by bind_row().
template <typename E>
VECTORIZATION_HOST_FUNCTION_ATTRIBUTE row_lowered_t<E> lower_rows(E const& expr)
{
    using expr_t = vectorization::remove_cvref_t<E>;
    if constexpr (is_pure_expression<expr_t>::value)
    {
        if constexpr (VECTORIZATION_EXPR_HAS_MHS(expr))
        {
            return row_lowered_t<E>(
                lower_rows(expr.lhs()), lower_rows(expr.mhs()), lower_rows(expr.rhs()));
        }
        else if constexpr (VECTORIZATION_EXPR_HAS_LHS(expr))
        {
            return row_lowered_t<E>(lower_rows(expr.lhs()), lower_rows(expr.rhs()));
        }
        else
        {
            return row_lowered_t<E>(lower_rows(expr.rhs()));
        }
    }
    else if constexpr (is_strided_ref<expr_t>::value || is_data_view<expr_t>::value)
    {
        return row_lowered_t<E>{};
    }
    else
    {
        return expr;
    }
}

// Points every row_ref leaf at its element `offsets[op]` and sets its inner stride.
template <typename L>
VECTORIZATION_HOST_FUNCTION_ATTRIBUTE void bind_row(
    L& lowered, iteration_plan const& plan, int64_t const* offsets, int& op) noexcept
{
    using expr_t = vectorization::remove_cvref_t<L>;
    if constexpr (is_pure_expression<expr_t>::value)
    {
        if constexpr (VECTORIZATION_EXPR_HAS_MHS(lowered))
        {
            bind_row(lowered.mutable_lhs(), plan, offsets, op);
            bind_row(lowered.mutable_mhs(), plan, offsets, op);
            bind_row(lowered.mutable_rhs(), plan, offsets, op);
        }
        else if constexpr (VECTORIZATION_EXPR_HAS_LHS(lowered))
        {
            bind_row(lowered.mutable_lhs(), plan, offsets, op);
            bind_row(lowered.mutable_rhs(), plan, offsets, op);
        }
        else
        {
            bind_row(lowered.mutable_rhs(), plan, offsets, op);
        }
    }
    else if constexpr (is_row_ref<expr_t>::value)
    {
        using value_t  = typename expr_t::value_type;
        lowered.ptr    = static_cast<value_t const*>(plan.bases[op]) + offsets[op];
        lowered.stride = plan.strides[op][plan.rank - 1];
        ++op;
    }
}

// Orders dimensions outermost first by destination stride, ties broken by the
// sources' strides (planner step 4).
inline bool dim_outer_than(iteration_plan const& plan, int a, int b) noexcept
{
    for (int op = 0; op < plan.n_operands; ++op)
    {
        if (plan.strides[op][a] != plan.strides[op][b])
        {
            return plan.strides[op][a] > plan.strides[op][b];
        }
    }
    return false;
}

// Drops size-1 dimensions, sorts the rest, and merges neighbours that are
// adjacent in memory for every operand (planner step 4), then classifies the tier.
inline void sort_merge_classify(iteration_plan& plan)
{
    int perm[kMaxRank];
    int m = 0;
    for (int d = 0; d < plan.rank; ++d)
    {
        if (plan.sizes[d] != 1)
        {
            perm[m++] = d;
        }
    }
    if (m == 0)
    {
        // A single element: one row of one element.
        plan.rank     = 1;
        plan.sizes[0] = 1;
        for (int op = 0; op < plan.n_operands; ++op)
        {
            plan.strides[op][0] = 0;
        }
        plan.tier = 3;
        return;
    }

    // Stable insertion sort; rank is at most kMaxRank.
    for (int i = 1; i < m; ++i)
    {
        const int d = perm[i];
        int       j = i;
        while (j > 0 && dim_outer_than(plan, d, perm[j - 1]))
        {
            perm[j] = perm[j - 1];
            --j;
        }
        perm[j] = d;
    }

    int64_t sizes[kMaxRank];
    int64_t strides[kMaxOperands][kMaxRank];
    int     r = 0;
    sizes[0]  = plan.sizes[perm[0]];
    for (int op = 0; op < plan.n_operands; ++op)
    {
        strides[op][0] = plan.strides[op][perm[0]];
    }
    for (int k = 1; k < m; ++k)
    {
        const int d         = perm[k];
        bool      mergeable = true;
        for (int op = 0; op < plan.n_operands && mergeable; ++op)
        {
            mergeable = strides[op][r] == plan.strides[op][d] * plan.sizes[d];
        }
        if (mergeable)
        {
            sizes[r] *= plan.sizes[d];
            for (int op = 0; op < plan.n_operands; ++op)
            {
                strides[op][r] = plan.strides[op][d];
            }
        }
        else
        {
            ++r;
            sizes[r] = plan.sizes[d];
            for (int op = 0; op < plan.n_operands; ++op)
            {
                strides[op][r] = plan.strides[op][d];
            }
        }
    }

    plan.rank = r + 1;
    for (int d = 0; d < plan.rank; ++d)
    {
        plan.sizes[d] = sizes[d];
        for (int op = 0; op < plan.n_operands; ++op)
        {
            plan.strides[op][d] = strides[op][d];
        }
    }

    const int inner        = plan.rank - 1;
    bool      all_unit     = true;
    bool      sources_simd = true;
    for (int op = 0; op < plan.n_operands; ++op)
    {
        const int64_t s = plan.strides[op][inner];
        all_unit        = all_unit && s == 1;
        if (op > 0)
        {
            sources_simd = sources_simd && (s == 0 || s == 1);
        }
    }
    if (all_unit)
    {
        plan.tier = plan.rank == 1 ? 0 : 1;
    }
    else if (plan.strides[0][inner] == 1 && sources_simd)
    {
        plan.tier = 2;
    }
    else
    {
        plan.tier = 3;
    }
}

// Evaluates one row of `inner_n` elements into dst_row.
template <typename L, typename value_t>
VECTORIZATION_HOST_FUNCTION_ATTRIBUTE void loop_row(
    L const& lowered, value_t* dst_row, int64_t dst_stride, int64_t inner_n, int tier) noexcept
{
    int64_t i = 0;
    if (tier <= 2)
    {
#if VECTORIZATION_VECTORIZED
        constexpr int64_t length = static_cast<int64_t>(simd<value_t>::size);
        for (; i + length <= inner_n; i += length)
        {
            const auto temp =
                expression_loader<L, true, false>::evaluate(lowered, static_cast<size_t>(i));
            simd<value_t>::storeu(temp, dst_row + i);
        }
#endif
        for (; i < inner_n; ++i)
        {
            dst_row[i] =
                expression_loader<L, false, false>::evaluate(lowered, static_cast<size_t>(i));
        }
        return;
    }
    for (; i < inner_n; ++i)
    {
        dst_row[i * dst_stride] =
            expression_loader<L, false, false>::evaluate(lowered, static_cast<size_t>(i));
    }
}

// Runs rows [row_begin, row_end) of the plan. The outer coordinates are
// derived once per call; each following row is reached by a counter step.
template <typename L, typename value_t>
VECTORIZATION_HOST_FUNCTION_ATTRIBUTE void loop2d(L lowered,
    value_t*                                        dst,
    iteration_plan const&                           plan,
    int64_t                                         row_begin,
    int64_t                                         row_end) noexcept
{
    const int     outer   = plan.rank - 1;
    const int64_t inner_n = plan.row_size();
    int64_t       coord[kMaxRank]{};
    int64_t       offsets[kMaxOperands]{};

    int64_t rem = row_begin;
    for (int d = outer - 1; d >= 0; --d)
    {
        coord[d] = rem % plan.sizes[d];
        rem /= plan.sizes[d];
        for (int op = 0; op < plan.n_operands; ++op)
        {
            offsets[op] += coord[d] * plan.strides[op][d];
        }
    }

    const int64_t dst_stride = plan.strides[0][outer];
    for (int64_t row = row_begin; row < row_end; ++row)
    {
        int op = 1;
        bind_row(lowered, plan, offsets, op);
        loop_row(lowered, dst + offsets[0], dst_stride, inner_n, plan.tier);

        for (int d = outer - 1; d >= 0; --d)
        {
            ++coord[d];
            for (int k = 0; k < plan.n_operands; ++k)
            {
                offsets[k] += plan.strides[k][d];
            }
            if (coord[d] < plan.sizes[d])
            {
                break;
            }
            for (int k = 0; k < plan.n_operands; ++k)
            {
                offsets[k] -= coord[d] * plan.strides[k][d];
            }
            coord[d] = 0;
        }
    }
}
}  // namespace detail

// Builds the plan for `dst = expr`, where `shape` is the validated result shape
// and equals the destination's shape.
template <typename E, typename T, int MaxRank>
VECTORIZATION_HOST_FUNCTION_ATTRIBUTE iteration_plan make_iteration_plan(
    E const& expr, strided_ref<T, MaxRank> const& dst, expression_shape const& shape)
{
    iteration_plan plan;
    plan.rank       = shape.rank;
    plan.numel      = static_cast<int64_t>(shape.numel());
    plan.n_operands = 1;
    for (int d = 0; d < shape.rank; ++d)
    {
        plan.sizes[d] = shape.sizes[d];
        // A destination element written twice would make the result depend on
        // loop order (design §6, internal overlap).
        VECTORIZATION_CHECK(dst.strides[d] != 0 || dst.sizes[d] <= 1 || plan.numel == 0,
            "destination has stride 0 in dimension {} and cannot be written; "
            "call contiguous() first",
            d);
        plan.strides[0][d] = dst.sizes[d] == 1 ? 0 : dst.strides[d];
    }
    detail::plan_leaves(expr, shape, plan);
    if (plan.numel == 0)
    {
        plan.rank = 0;
        plan.tier = 0;
        return plan;
    }
    detail::sort_merge_classify(plan);
    return plan;
}

// Evaluates `dst = expr` through the planner (any strides, broadcasting).
// Rows are split across threads above the grain size; chunks never share a
// destination row.
template <typename E, typename T, int MaxRank>
VECTORIZATION_HOST_FUNCTION_ATTRIBUTE void run_planned(
    E const& expr, strided_ref<T, MaxRank> const& dst, expression_shape const& shape)
{
    const iteration_plan plan = make_iteration_plan(expr, dst, shape);
    if (plan.numel == 0)
    {
        return;
    }
    const auto    lowered = detail::lower_rows(expr);
    T*            out     = dst.data();
    const int64_t rows    = plan.rows();

    static constexpr int64_t kGrainSize = 32768;  // elements, as in PyTorch
#if VECTORIZATION_ENABLE_PARALLEL
    if (plan.numel < kGrainSize || rows < 2)
    {
        detail::loop2d(lowered, out, plan, 0, rows);
        return;
    }
    const int64_t row_grain = std::max<int64_t>(1, kGrainSize / plan.row_size());
    auto _loop2d_fn = [&](size_t begin, size_t end) {
        detail::loop2d(lowered, out, plan, static_cast<int64_t>(begin), static_cast<int64_t>(end));
    };
    parallel_tools::parallel_for(
        size_t{0}, static_cast<size_t>(rows), static_cast<size_t>(row_grain), _loop2d_fn);
#else
    detail::loop2d(lowered, out, plan, 0, rows);
#endif
}

}  // namespace vectorization
