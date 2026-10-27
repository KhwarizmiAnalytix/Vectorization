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

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

#include "common/data_view.h"
#include "common/vectorization_macros.h"
#include "common/vectorization_type_traits.h"
#include "expressions/strided_ref.h"

namespace vectorization
{

// Result shape of an elementwise expression: every tensor leaf's shape
// broadcast against the others, right-aligned as in NumPy and PyTorch
// (design §5, planner step 2). Scalars do not contribute.
struct expression_shape
{
    int     rank{0};
    int64_t sizes[kMaxRank]{};

    VECTORIZATION_FUNCTION_ATTRIBUTE size_t numel() const noexcept
    {
        size_t n = 1;
        for (int d = 0; d < rank; ++d)
        {
            n *= static_cast<size_t>(sizes[d]);
        }
        return n;
    }
};

namespace detail
{
template <typename T> struct is_data_view : std::false_type
{
};

template <typename T> struct is_data_view<memory::data_view<T>> : std::true_type
{
};

// Broadcast `acc` with one leaf shape. Dimensions are matched from the right;
// a pair is compatible when the sizes are equal or one of them is 1.
VECTORIZATION_CUDA_FUNCTION_TYPE inline bool broadcast_into(
    expression_shape& acc, int rank, int64_t const* sizes) noexcept
{
    if (rank > kMaxRank)
    {
        return false;
    }
    const int out_rank = acc.rank > rank ? acc.rank : rank;
    int64_t   out[kMaxRank]{};
    for (int i = 0; i < out_rank; ++i)
    {
        const int64_t a = i < acc.rank ? acc.sizes[acc.rank - 1 - i] : 1;
        const int64_t b = i < rank ? sizes[rank - 1 - i] : 1;
        if (a != b && a != 1 && b != 1)
        {
            return false;
        }
        out[out_rank - 1 - i] = a == 1 ? b : a;
    }
    acc.rank = out_rank;
    for (int d = 0; d < out_rank; ++d)
    {
        acc.sizes[d] = out[d];
    }
    return true;
}

template <typename E>
VECTORIZATION_FUNCTION_ATTRIBUTE bool accumulate_shape(
    E const& expr, expression_shape& acc) noexcept
{
    using expr_t = vectorization::remove_cvref_t<E>;
    if constexpr (is_pure_expression<expr_t>::value)
    {
        if constexpr (VECTORIZATION_EXPR_HAS_MHS(expr))
        {
            return accumulate_shape(expr.lhs(), acc) && accumulate_shape(expr.mhs(), acc) &&
                   accumulate_shape(expr.rhs(), acc);
        }
        else if constexpr (VECTORIZATION_EXPR_HAS_LHS(expr))
        {
            return accumulate_shape(expr.lhs(), acc) && accumulate_shape(expr.rhs(), acc);
        }
        else
        {
            return accumulate_shape(expr.rhs(), acc);
        }
    }
    else if constexpr (is_strided_ref<expr_t>::value)
    {
        return broadcast_into(acc, expr.rank, expr.sizes);
    }
    else if constexpr (is_data_view<expr_t>::value)
    {
        const auto n = static_cast<int64_t>(expr.size());
        return broadcast_into(acc, 1, &n);
    }
    else
    {
        return true;  // scalars have no shape
    }
}

// Strides that read a leaf of shape (rank, sizes, strides) in the index space
// of `out`: leading dimensions the leaf lacks and size-1 dimensions that `out`
// broadcasts get stride 0 (design §5, planner step 3).
VECTORIZATION_CUDA_FUNCTION_TYPE inline void broadcast_strides(expression_shape const& out,
    int                                                                                rank,
    int64_t const*                                                                     sizes,
    int64_t const*                                                                     strides,
    int64_t* result) noexcept
{
    const int lead = out.rank - rank;
    for (int d = 0; d < out.rank; ++d)
    {
        const int j = d - lead;
        result[d]   = (j < 0 || sizes[j] == 1) ? 0 : strides[j];
    }
}

// C-order packing test that ignores size-1 dimensions (same rule as
// tensor::recompute_cpu_simd_alignment_state and PyTorch's compute_contiguous).
VECTORIZATION_CUDA_FUNCTION_TYPE inline bool is_packed(
    int rank, int64_t const* sizes, int64_t const* strides) noexcept
{
    int64_t expected = 1;
    for (int d = rank - 1; d >= 0; --d)
    {
        if (sizes[d] == 0)
        {
            return true;
        }
        if (sizes[d] == 1)
        {
            continue;
        }
        if (strides[d] != expected)
        {
            return false;
        }
        expected *= sizes[d];
    }
    return true;
}

template <typename E>
VECTORIZATION_HOST_FUNCTION_ATTRIBUTE void describe_leaf_shapes(E const& expr, std::string& out)
{
    using expr_t = vectorization::remove_cvref_t<E>;
    if constexpr (is_pure_expression<expr_t>::value)
    {
        if constexpr (VECTORIZATION_EXPR_HAS_MHS(expr))
        {
            describe_leaf_shapes(expr.lhs(), out);
            describe_leaf_shapes(expr.mhs(), out);
            describe_leaf_shapes(expr.rhs(), out);
        }
        else if constexpr (VECTORIZATION_EXPR_HAS_LHS(expr))
        {
            describe_leaf_shapes(expr.lhs(), out);
            describe_leaf_shapes(expr.rhs(), out);
        }
        else
        {
            describe_leaf_shapes(expr.rhs(), out);
        }
    }
    else if constexpr (is_strided_ref<expr_t>::value || is_data_view<expr_t>::value)
    {
        if (!out.empty())
        {
            out += ", ";
        }
        if constexpr (is_strided_ref<expr_t>::value)
        {
            out += "[";
            for (int d = 0; d < expr.rank; ++d)
            {
                out += (d ? ", " : "") + std::to_string(expr.sizes[d]);
            }
            out += "]";
        }
        else
        {
            out += "[" + std::to_string(expr.size()) + "]";
        }
    }
}

// True when every leaf reads its i-th element at flat index i in the result's
// C order: each is packed and holds exactly `numel` elements, so nothing is
// broadcast. Expressions that pass need no planning (design §5, contiguous
// fast path); the others go through the planner or rebase_leaves().
template <typename E>
VECTORIZATION_FUNCTION_ATTRIBUTE bool leaves_flat(E const& expr, size_t numel) noexcept
{
    using expr_t = vectorization::remove_cvref_t<E>;
    if constexpr (is_pure_expression<expr_t>::value)
    {
        if constexpr (VECTORIZATION_EXPR_HAS_MHS(expr))
        {
            return leaves_flat(expr.lhs(), numel) && leaves_flat(expr.mhs(), numel) &&
                   leaves_flat(expr.rhs(), numel);
        }
        else if constexpr (VECTORIZATION_EXPR_HAS_LHS(expr))
        {
            return leaves_flat(expr.lhs(), numel) && leaves_flat(expr.rhs(), numel);
        }
        else
        {
            return leaves_flat(expr.rhs(), numel);
        }
    }
    else if constexpr (is_strided_ref<expr_t>::value)
    {
        return expr.contiguous && expr.numel == numel;
    }
    else if constexpr (is_data_view<expr_t>::value)
    {
        return expr.size() == numel;
    }
    else
    {
        return true;
    }
}

// Rewrites every strided_ref leaf of a (copied) expression into the index
// space of `out`: rank, sizes and numel become the result's, strides become
// broadcast strides. Flat-index evaluation of the rewritten tree over
// [0, out.numel()) is then correct for any strides and broadcasting; the
// GPU kernels and reductions use it, the CPU elementwise path uses the planner.
template <typename E>
VECTORIZATION_HOST_FUNCTION_ATTRIBUTE void rebase_leaves(E& expr, expression_shape const& out)
{
    using expr_t = vectorization::remove_cvref_t<E>;
    if constexpr (is_pure_expression<expr_t>::value)
    {
        if constexpr (VECTORIZATION_EXPR_HAS_MHS(expr))
        {
            rebase_leaves(expr.mutable_lhs(), out);
            rebase_leaves(expr.mutable_mhs(), out);
            rebase_leaves(expr.mutable_rhs(), out);
        }
        else if constexpr (VECTORIZATION_EXPR_HAS_LHS(expr))
        {
            rebase_leaves(expr.mutable_lhs(), out);
            rebase_leaves(expr.mutable_rhs(), out);
        }
        else
        {
            rebase_leaves(expr.mutable_rhs(), out);
        }
    }
    else if constexpr (is_strided_ref<expr_t>::value)
    {
        int64_t strides[kMaxRank]{};
        broadcast_strides(out, expr.rank, expr.sizes, expr.strides, strides);
        expr.rank = out.rank;
        for (int d = 0; d < out.rank; ++d)
        {
            expr.sizes[d]   = out.sizes[d];
            expr.strides[d] = strides[d];
        }
        expr.numel      = out.numel();
        expr.contiguous = is_packed(expr.rank, expr.sizes, expr.strides);
    }
    else if constexpr (is_data_view<expr_t>::value)
    {
        // data_view carries no strides, so it can only be read flat.
        VECTORIZATION_CHECK(expr.size() == out.numel(),
            "a data_view operand of {} elements cannot be broadcast to {} elements",
            expr.size(),
            out.numel());
    }
}
}  // namespace detail

// Broadcast shape of `expr`, or false when two leaves are incompatible.
// Device-safe; the evaluator calls expression_result_shape() instead.
template <typename E>
VECTORIZATION_FUNCTION_ATTRIBUTE bool try_expression_shape(
    E const& expr, expression_shape& out) noexcept
{
    out = expression_shape{};
    return detail::accumulate_shape(expr, out);
}

// Element count of the broadcast result; 0 when the leaf shapes are
// incompatible (evaluation then fails in expression_result_shape()).
template <typename E>
VECTORIZATION_FUNCTION_ATTRIBUTE size_t expression_numel(E const& expr) noexcept
{
    expression_shape shape;
    return try_expression_shape(expr, shape) ? shape.numel() : 0;
}

inline std::string shape_to_string(int rank, int64_t const* sizes)
{
    std::string s = "[";
    for (int d = 0; d < rank; ++d)
    {
        s += (d ? ", " : "") + std::to_string(sizes[d]);
    }
    return s + "]";
}

// Validated broadcast shape: raises in every build type when two leaves are
// incompatible or a leaf exceeds kMaxRank dimensions.
template <typename E>
VECTORIZATION_HOST_FUNCTION_ATTRIBUTE expression_shape expression_result_shape(E const& expr)
{
    expression_shape shape;
    if (!try_expression_shape(expr, shape))
    {
        std::string leaves;
        detail::describe_leaf_shapes(expr, leaves);
        VECTORIZATION_CHECK(
            false, "expression operands have incompatible shapes for broadcasting: {}", leaves);
    }
    return shape;
}

// Copy of `expr` whose leaves read the broadcast result in flat C order.
template <typename E>
VECTORIZATION_HOST_FUNCTION_ATTRIBUTE vectorization::remove_cvref_t<E> flat_lowered(
    E const& expr, expression_shape const& shape)
{
    vectorization::remove_cvref_t<E> lowered(expr);
    detail::rebase_leaves(lowered, shape);
    return lowered;
}

}  // namespace vectorization
