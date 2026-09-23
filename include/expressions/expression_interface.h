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
#include <type_traits>
#include <utility>

#include "backend/simd.h"
#include "common/data_view.h"
#include "common/scalar_helper_functions.h"
#include "common/vectorization_macros.h"
#include "common/vectorization_type_traits.h"
#include "expressions/expression_interface_loader.h"

namespace vectorization
{
// std::conditional_t instantiates both branches, so a helper trait is required:
// tensor leaves become a data_view (pointer alias, GPU-memcpy-safe); nested
// expression nodes stay by value.
template <typename T, bool = is_tensor<vectorization::remove_cvref_t<T>>::value>
struct stored_operand
{
    using type = vectorization::remove_cvref_t<T>;
};

template <typename T>
struct stored_operand<T, true>
{
    using type = memory::data_view<typename vectorization::remove_cvref_t<T>::value_type>;
};

template <typename T>
using stored_operand_t = typename stored_operand<T>::type;

// Tensor leaves are borrowed, not cloned. The source tensor must outlive the
// expression (the usual ET contract: `c = a + b` is fine; `auto e = tensor(n) + a`
// dangles). Nested expression nodes are moved when they are rvalues.
template <typename T>
VECTORIZATION_FUNCTION_ATTRIBUTE stored_operand_t<T> store_operand(T&& x)
{
    using raw = vectorization::remove_cvref_t<T>;
    if constexpr (is_tensor<raw>::value)
    {
#if !VECTORIZATION_ON_GPU_DEVICE
        VECTORIZATION_CHECK(
            x.is_contiguous(), "expression operands must be contiguous; call contiguous() first");
#endif
        using view_t = memory::data_view<typename raw::value_type>;
        return view_t::borrow(
            x.data(),
            x.size(),
            x.device(),
            x.device_index(),
            static_cast<typename view_t::stream_t>(x.stream()));
    }
    else
    {
        return std::forward<T>(x);
    }
}

/*!
 * \brief An unary expression
 *
 * This expression applies an unary operator on each element of a sub expression
 */
template <typename LHS, typename EVALUATOR>
class unary_expression final
{
    using rmv_lhs    = vectorization::remove_cvref_t<LHS>;
    using stored_lhs = stored_operand_t<rmv_lhs>;
    stored_lhs lhs_;

public:
    VECTORIZATION_FUNCTION_ATTRIBUTE size_t size() const noexcept { return lhs_.size(); }

    static constexpr size_t length() { return rmv_lhs::length(); }

    VECTORIZATION_FUNCTION_ATTRIBUTE explicit unary_expression(rmv_lhs const& lhs)
        : lhs_(store_operand(lhs))
    {
    }

    VECTORIZATION_FUNCTION_ATTRIBUTE explicit unary_expression(rmv_lhs&& lhs)
        : lhs_(store_operand(std::move(lhs)))
    {
    }

    // Defaulted copy/move must carry __host__ __device__ for CUDA/HIP device
    // code to be able to copy expression nodes (e.g. when capturing in a kernel).
    VECTORIZATION_FUNCTION_ATTRIBUTE unary_expression(unary_expression const& e)     = default;
    VECTORIZATION_FUNCTION_ATTRIBUTE unary_expression(unary_expression&& e) noexcept = default;

    // Expressions are invariant
    VECTORIZATION_FUNCTION_ATTRIBUTE unary_expression& operator=(unary_expression const& e) =
        delete;
    VECTORIZATION_FUNCTION_ATTRIBUTE unary_expression& operator=(unary_expression&& e) = delete;

    VECTORIZATION_FUNCTION_ATTRIBUTE const auto& rhs() const { return lhs_; }

    template <bool vectorize, bool aligned>
    VECTORIZATION_FUNCTION_ATTRIBUTE static auto evaluate(
        unary_expression const& expr, size_t index) noexcept
    {
        const auto& rhs =
            expression_loader<stored_lhs, vectorize, aligned>::evaluate(expr.rhs(), index);
        return EVALUATOR::functor(rhs);
    }

    template <bool vectorize, bool aligned>
    VECTORIZATION_FUNCTION_ATTRIBUTE static void prefetch(
        unary_expression const& expr, size_t index) noexcept
    {
        expression_loader<stored_lhs, vectorize, aligned>::prefetch(expr.rhs(), index);
    }
};

/*!
 * \brief A binary expression
 *
 * A binary expression has a left hand side expression and a right hand side expression and for each
 * element applies a binary operator to both expressions.
 */
template <typename LHS, typename RHS, typename EVALUATOR>
class binary_expression final
{
    using rmv_lhs    = vectorization::remove_cvref_t<LHS>;
    using rmv_rhs    = vectorization::remove_cvref_t<RHS>;
    using stored_lhs = stored_operand_t<rmv_lhs>;
    using stored_rhs = stored_operand_t<rmv_rhs>;

    stored_lhs lhs_;
    stored_rhs rhs_;

public:
    VECTORIZATION_FUNCTION_ATTRIBUTE size_t size() const noexcept
    {
        if constexpr (vectorization::is_expression<rmv_rhs>::value)  // NOLINT
            return rhs_.size();
        else if constexpr (vectorization::is_expression<rmv_lhs>::value)  // NOLINT
            return lhs_.size();
        else
            return 0;
    }

    VECTORIZATION_FUNCTION_ATTRIBUTE static constexpr size_t length()
    {
        if constexpr (
            vectorization::is_expression<rmv_rhs>::value &&
            vectorization::is_expression<rmv_lhs>::value)
        {
            static_assert(
                rmv_rhs::length() == rmv_lhs::length(), "expressions have different strides!");
            return rmv_rhs::length();
        }
        else if constexpr (vectorization::is_expression<rmv_rhs>::value)
        {
            return rmv_rhs::length();
        }
        else if constexpr (vectorization::is_expression<rmv_lhs>::value)
        {
            return rmv_lhs::length();
        }
        else
        {
            static_assert(
                vectorization::is_expression<rmv_lhs>::value ||
                    vectorization::is_expression<rmv_rhs>::value,
                "binary_expression: neither operand is an expression");
            return 0;  // unreachable; satisfies the compiler
        }
    }

    VECTORIZATION_FUNCTION_ATTRIBUTE binary_expression(rmv_lhs const& lhs, rmv_rhs const& rhs)
        : lhs_(store_operand(lhs)), rhs_(store_operand(rhs))
    {
    }

    VECTORIZATION_FUNCTION_ATTRIBUTE binary_expression(rmv_lhs&& lhs, rmv_rhs&& rhs)
        : lhs_(store_operand(std::move(lhs))), rhs_(store_operand(std::move(rhs)))
    {
    }

    VECTORIZATION_FUNCTION_ATTRIBUTE binary_expression(binary_expression const& e)     = default;
    VECTORIZATION_FUNCTION_ATTRIBUTE binary_expression(binary_expression&& e) noexcept = default;

    // Expressions are invariant
    VECTORIZATION_FUNCTION_ATTRIBUTE binary_expression& operator=(binary_expression const& e) =
        delete;
    VECTORIZATION_FUNCTION_ATTRIBUTE binary_expression& operator=(binary_expression&& e) = delete;

    VECTORIZATION_FUNCTION_ATTRIBUTE const auto& lhs() const { return lhs_; }
    VECTORIZATION_FUNCTION_ATTRIBUTE const auto& rhs() const { return rhs_; }

    template <bool vectorize, bool aligned>
    VECTORIZATION_FUNCTION_ATTRIBUTE static auto evaluate(
        binary_expression const& expr, size_t index) noexcept
    {
        const auto& lhs =
            expression_loader<stored_lhs, vectorize, aligned>::evaluate(expr.lhs(), index);
        const auto& rhs =
            expression_loader<stored_rhs, vectorize, aligned>::evaluate(expr.rhs(), index);
        return EVALUATOR::functor(lhs, rhs);
    }

    template <bool vectorize, bool aligned>
    VECTORIZATION_FUNCTION_ATTRIBUTE static void prefetch(
        binary_expression const& expr, size_t index) noexcept
    {
        expression_loader<stored_lhs, vectorize, aligned>::prefetch(expr.lhs(), index);
        expression_loader<stored_rhs, vectorize, aligned>::prefetch(expr.rhs(), index);
    }
};

/*!
 * \brief A trinary expression
 *
 * A trinary expression has a left hand side expression, middle hand side and a right hand side
 * expression and for each element applies a trinary operator to all expressions.
 */
template <typename LHS, typename MHS, typename RHS, typename EVALUATOR>
class trinary_expression final
{
    using rmv_lhs    = vectorization::remove_cvref_t<LHS>;
    using rmv_mhs    = vectorization::remove_cvref_t<MHS>;
    using rmv_rhs    = vectorization::remove_cvref_t<RHS>;
    using stored_lhs = stored_operand_t<rmv_lhs>;
    using stored_mhs = stored_operand_t<rmv_mhs>;
    using stored_rhs = stored_operand_t<rmv_rhs>;

    stored_lhs lhs_;
    stored_mhs mhs_;
    stored_rhs rhs_;

public:
    VECTORIZATION_FUNCTION_ATTRIBUTE size_t size() const noexcept
    {
        if constexpr (vectorization::is_expression<rmv_lhs>::value)
            return lhs_.size();
        else if constexpr (vectorization::is_expression<rmv_mhs>::value)
            return mhs_.size();
        else if constexpr (vectorization::is_expression<rmv_rhs>::value)
            return rhs_.size();
        else
        {
            static_assert(
                vectorization::is_expression<rmv_lhs>::value ||
                    vectorization::is_expression<rmv_mhs>::value ||
                    vectorization::is_expression<rmv_rhs>::value,
                "trinary_expression: no operand is an expression");
            return 0;  // unreachable; satisfies the compiler
        }
    }

    VECTORIZATION_FUNCTION_ATTRIBUTE static constexpr size_t length()
    {
        if constexpr (vectorization::is_expression<rmv_lhs>::value)
            return rmv_lhs::length();
        else if constexpr (vectorization::is_expression<rmv_mhs>::value)
            return rmv_mhs::length();
        else if constexpr (vectorization::is_expression<rmv_rhs>::value)
            return rmv_rhs::length();
        else
        {
            static_assert(
                vectorization::is_expression<rmv_lhs>::value ||
                    vectorization::is_expression<rmv_mhs>::value ||
                    vectorization::is_expression<rmv_rhs>::value,
                "trinary_expression: no operand is an expression");
            return 0;  // unreachable; satisfies the compiler
        }
    }

    VECTORIZATION_FUNCTION_ATTRIBUTE trinary_expression(
        rmv_lhs const& lhs, rmv_mhs const& e1, rmv_rhs const& e2)
        : lhs_(store_operand(lhs)), mhs_(store_operand(e1)), rhs_(store_operand(e2))
    {
    }

    VECTORIZATION_FUNCTION_ATTRIBUTE trinary_expression(rmv_lhs&& lhs, rmv_mhs&& mhs, rmv_rhs&& rhs)
        : lhs_(store_operand(std::move(lhs))),
          mhs_(store_operand(std::move(mhs))),
          rhs_(store_operand(std::move(rhs)))
    {
    }

    VECTORIZATION_FUNCTION_ATTRIBUTE trinary_expression(trinary_expression const& e)     = default;
    VECTORIZATION_FUNCTION_ATTRIBUTE trinary_expression(trinary_expression&& e) noexcept = default;

    // Expressions are invariant
    VECTORIZATION_FUNCTION_ATTRIBUTE trinary_expression& operator=(trinary_expression const& e) =
        delete;
    VECTORIZATION_FUNCTION_ATTRIBUTE trinary_expression& operator=(trinary_expression&& e) = delete;

    VECTORIZATION_FUNCTION_ATTRIBUTE const auto& lhs() const { return lhs_; }
    VECTORIZATION_FUNCTION_ATTRIBUTE const auto& mhs() const { return mhs_; };
    VECTORIZATION_FUNCTION_ATTRIBUTE const auto& rhs() const { return rhs_; }

    template <bool vectorize, bool aligned>
    VECTORIZATION_FUNCTION_ATTRIBUTE static auto evaluate(
        trinary_expression const& expr, size_t index) noexcept
    {
        const auto& lhs =
            expression_loader<stored_lhs, vectorize, aligned>::evaluate(expr.lhs(), index);
        const auto& mhs =
            expression_loader<stored_mhs, vectorize, aligned>::evaluate(expr.mhs(), index);
        const auto& rhs =
            expression_loader<stored_rhs, vectorize, aligned>::evaluate(expr.rhs(), index);
        return EVALUATOR::functor(lhs, mhs, rhs);
    }

    template <bool vectorize, bool aligned>
    VECTORIZATION_FUNCTION_ATTRIBUTE static void prefetch(
        trinary_expression const& expr, size_t index) noexcept
    {
        expression_loader<stored_lhs, vectorize, aligned>::prefetch(expr.lhs(), index);
        expression_loader<stored_mhs, vectorize, aligned>::prefetch(expr.mhs(), index);
        expression_loader<stored_rhs, vectorize, aligned>::prefetch(expr.rhs(), index);
    }
};

template <typename E>
VECTORIZATION_FUNCTION_ATTRIBUTE bool expression_operands_aligned(E const& expr)
{
    using expr_t = vectorization::remove_cvref_t<E>;
    if constexpr (is_pure_expression<expr_t>::value)
    {
        if constexpr (VECTORIZATION_EXPR_HAS_MHS(expr))
        {
            return expression_operands_aligned(expr.lhs()) &&
                   expression_operands_aligned(expr.mhs()) &&
                   expression_operands_aligned(expr.rhs());
        }
        else if constexpr (VECTORIZATION_EXPR_HAS_LHS(expr))
        {
            return expression_operands_aligned(expr.lhs()) &&
                   expression_operands_aligned(expr.rhs());
        }
        else if constexpr (VECTORIZATION_EXPR_HAS_RHS(expr))
        {
            return expression_operands_aligned(expr.rhs());
        }
        else
        {
            return true;
        }
    }
    else if constexpr (is_base_expression<expr_t>::value)
    {
        return expr.is_aligned();
    }
    else
    {
        return true;
    }
}

template <typename A, typename B>
VECTORIZATION_FUNCTION_ATTRIBUTE bool buffers_overlap(
    A const* a, std::size_t a_count, B const* b, std::size_t b_count) noexcept
{
    if (a == nullptr || b == nullptr || a_count == 0 || b_count == 0)
    {
        return false;
    }
    auto const a0 = reinterpret_cast<std::uintptr_t>(static_cast<void const*>(a));
    auto const b0 = reinterpret_cast<std::uintptr_t>(static_cast<void const*>(b));
    auto const a1 = a0 + a_count * sizeof(A);
    auto const b1 = b0 + b_count * sizeof(B);
    return a0 < b1 && b0 < a1;
}

template <typename E, typename Ptr>
VECTORIZATION_FUNCTION_ATTRIBUTE bool expression_aliases_ptr(
    E const& expr, Ptr* ptr, std::size_t count)
{
    using expr_t = vectorization::remove_cvref_t<E>;
    if (ptr == nullptr || count == 0)
    {
        return false;
    }
    if constexpr (is_pure_expression<expr_t>::value)
    {
        if constexpr (VECTORIZATION_EXPR_HAS_MHS(expr))
        {
            return expression_aliases_ptr(expr.lhs(), ptr, count) ||
                   expression_aliases_ptr(expr.mhs(), ptr, count) ||
                   expression_aliases_ptr(expr.rhs(), ptr, count);
        }
        else if constexpr (VECTORIZATION_EXPR_HAS_LHS(expr))
        {
            return expression_aliases_ptr(expr.lhs(), ptr, count) ||
                   expression_aliases_ptr(expr.rhs(), ptr, count);
        }
        else if constexpr (VECTORIZATION_EXPR_HAS_RHS(expr))
        {
            return expression_aliases_ptr(expr.rhs(), ptr, count);
        }
        else
        {
            return false;
        }
    }
    else if constexpr (is_base_expression<expr_t>::value)
    {
        return buffers_overlap(expr.data(), expr.size(), ptr, count);
    }
    else
    {
        return false;
    }
}
}  // namespace vectorization
