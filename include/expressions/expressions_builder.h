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

#ifndef VECTORIZATION_EXPRESSIONS_BUILDER_H
#define VECTORIZATION_EXPRESSIONS_BUILDER_H

#include <utility>

#include "expressions/expression_interface.h"
#include "expressions/expressions_functors.h"

#define OPERATOR(s) operator s

//================================================================================================
#define MACRO_EXPRESSION_FUNCTION_1_ARG_CONST(op)                                                  \
    template <typename LHS,                                                                        \
        std::enable_if_t<vectorization::is_expression<LHS>::value, bool> = true>                   \
    VECTORIZATION_FUNCTION_ATTRIBUTE auto op(LHS const& expr)                                      \
    {                                                                                              \
        return vectorization::unary_expression<LHS, vectorization::MACRO_EVALUATOR_SUFIX(op)>(     \
            expr);                                                                                 \
    }

#define MACRO_EXPRESSION_FUNCTION_1_ARG_MOVE(op)                                                   \
    template <typename LHS,                                                                        \
        std::enable_if_t<vectorization::is_expression<LHS>::value, bool> = true>                   \
    VECTORIZATION_FUNCTION_ATTRIBUTE auto op(LHS&& expr)                                           \
    {                                                                                              \
        return vectorization::unary_expression<LHS, vectorization::MACRO_EVALUATOR_SUFIX(op)>(     \
            std::forward<LHS>(expr));                                                              \
    }

#define MACRO_EXPRESSION_FUNCTION_1_ARG(op)                                                        \
    MACRO_EXPRESSION_FUNCTION_1_ARG_CONST(op);                                                     \
    MACRO_EXPRESSION_FUNCTION_1_ARG_MOVE(op);

MACRO_EXPRESSION_FUNCTION_1_ARG(fabs);
MACRO_EXPRESSION_FUNCTION_1_ARG(neg);
MACRO_EXPRESSION_FUNCTION_1_ARG(floor);
MACRO_EXPRESSION_FUNCTION_1_ARG(ceil);
MACRO_EXPRESSION_FUNCTION_1_ARG(sqrt);
MACRO_EXPRESSION_FUNCTION_1_ARG(sqr);
MACRO_EXPRESSION_FUNCTION_1_ARG(exp);
MACRO_EXPRESSION_FUNCTION_1_ARG(expm1);
MACRO_EXPRESSION_FUNCTION_1_ARG(exp2);
MACRO_EXPRESSION_FUNCTION_1_ARG(log);
MACRO_EXPRESSION_FUNCTION_1_ARG(log1p);
MACRO_EXPRESSION_FUNCTION_1_ARG(log2);
MACRO_EXPRESSION_FUNCTION_1_ARG(log10);
MACRO_EXPRESSION_FUNCTION_1_ARG(sin);
MACRO_EXPRESSION_FUNCTION_1_ARG(cos);
MACRO_EXPRESSION_FUNCTION_1_ARG(tan);
MACRO_EXPRESSION_FUNCTION_1_ARG(asin);
MACRO_EXPRESSION_FUNCTION_1_ARG(acos);
MACRO_EXPRESSION_FUNCTION_1_ARG(atan);
MACRO_EXPRESSION_FUNCTION_1_ARG(sinh);
MACRO_EXPRESSION_FUNCTION_1_ARG(cosh);
MACRO_EXPRESSION_FUNCTION_1_ARG(tanh);
MACRO_EXPRESSION_FUNCTION_1_ARG(asinh);
MACRO_EXPRESSION_FUNCTION_1_ARG(acosh);
MACRO_EXPRESSION_FUNCTION_1_ARG(atanh);
MACRO_EXPRESSION_FUNCTION_1_ARG(cbrt);
MACRO_EXPRESSION_FUNCTION_1_ARG(cdf)
MACRO_EXPRESSION_FUNCTION_1_ARG(inv_cdf)
MACRO_EXPRESSION_FUNCTION_1_ARG(trunc)
MACRO_EXPRESSION_FUNCTION_1_ARG(invsqrt)

//================================================================================================
#define MACRO_EXPRESSION_OPERATOR_1_ARG_MOVE(op, symbol, ref)                                      \
    template <typename LHS,                                                                        \
        std::enable_if_t<vectorization::is_expression<LHS>::value, bool> = true>                   \
    VECTORIZATION_FUNCTION_ATTRIBUTE auto OPERATOR(symbol)(LHS && expr)                            \
    {                                                                                              \
        return vectorization::unary_expression<LHS, vectorization::MACRO_EVALUATOR_SUFIX(op)>(     \
            std::forward<LHS>(expr));                                                              \
    }

#define MACRO_EXPRESSION_OPERATOR_1_ARG_COPY(op, symbol, ref)                                      \
    template <typename LHS,                                                                        \
        std::enable_if_t<vectorization::is_expression<LHS>::value, bool> = true>                   \
    VECTORIZATION_FUNCTION_ATTRIBUTE auto OPERATOR(symbol)(LHS const& expr)                        \
    {                                                                                              \
        return vectorization::unary_expression<LHS, vectorization::MACRO_EVALUATOR_SUFIX(op)>(     \
            expr);                                                                                 \
    }

#define MACRO_EXPRESSION_OPERATOR_1_ARG(op, symbol)                                                \
    MACRO_EXPRESSION_OPERATOR_1_ARG_COPY(op, symbol, const&);                                      \
    MACRO_EXPRESSION_OPERATOR_1_ARG_MOVE(op, symbol, &&);
//------------------------------------------------------------------------------------------------
namespace vectorization
{
MACRO_EXPRESSION_OPERATOR_1_ARG(neg, -);
MACRO_EXPRESSION_OPERATOR_1_ARG(lnot, !);
}  // namespace vectorization

//================================================================================================
#define MACRO_EXPRESSION_FUNCTION_2_ARG_(op, ref)                                                  \
    template <typename LHS,                                                                        \
        typename RHS,                                                                              \
        std::enable_if_t<vectorization::is_expression<LHS>::value ||                               \
                             vectorization::is_expression<RHS>::value,                             \
            bool> = true>                                                                          \
    VECTORIZATION_FUNCTION_ATTRIBUTE auto op(LHS ref lhs, RHS ref rhs)                             \
    {                                                                                              \
        return vectorization::                                                                     \
            binary_expression<LHS, RHS, vectorization::MACRO_EVALUATOR_SUFIX(op)>(lhs, rhs);       \
    };

#define MACRO_EXPRESSION_FUNCTION_2_ARG(op)                                                        \
    MACRO_EXPRESSION_FUNCTION_2_ARG_(op, const&);                                                  \
    template <typename LHS,                                                                        \
        typename RHS,                                                                              \
        std::enable_if_t<vectorization::is_expression<LHS>::value ||                               \
                             vectorization::is_expression<RHS>::value,                             \
            bool> = true>                                                                          \
    VECTORIZATION_FUNCTION_ATTRIBUTE auto op(LHS&& lhs, RHS&& rhs)                                 \
    {                                                                                              \
        return vectorization::                                                                     \
            binary_expression<LHS, RHS, vectorization::MACRO_EVALUATOR_SUFIX(op)>(                 \
                std::forward<LHS>(lhs), std::forward<RHS>(rhs));                                   \
    };
//------------------------------------------------------------------------------------------------
// MACRO_EXPRESSION_FUNCTION_2_ARG(gather)
MACRO_EXPRESSION_FUNCTION_2_ARG(max);
MACRO_EXPRESSION_FUNCTION_2_ARG(min);
MACRO_EXPRESSION_FUNCTION_2_ARG(pow);
MACRO_EXPRESSION_FUNCTION_2_ARG(hypot);
MACRO_EXPRESSION_FUNCTION_2_ARG(copysign);

//================================================================================================
#define MACRO_EXPRESSION_FUNCTION_3_ARG_(op, ref)                                                  \
    template <typename LHS,                                                                        \
        typename MHS,                                                                              \
        typename RHS,                                                                              \
        std::enable_if_t<vectorization::is_expression<LHS>::value ||                               \
                             vectorization::is_expression<MHS>::value ||                           \
                             vectorization::is_expression<RHS>::value,                             \
            bool> = true>                                                                          \
    VECTORIZATION_FUNCTION_ATTRIBUTE auto op(LHS ref lhs, MHS ref mhs, RHS ref rhs)                \
    {                                                                                              \
        return vectorization::                                                                     \
            trinary_expression<LHS, MHS, RHS, vectorization::MACRO_EVALUATOR_SUFIX(op)>(           \
                lhs, mhs, rhs);                                                                    \
    }

#define MACRO_EXPRESSION_FUNCTION_3_ARG(op)                                                        \
    MACRO_EXPRESSION_FUNCTION_3_ARG_(op, const&);                                                  \
    template <typename LHS,                                                                        \
        typename MHS,                                                                              \
        typename RHS,                                                                              \
        std::enable_if_t<vectorization::is_expression<LHS>::value ||                               \
                             vectorization::is_expression<MHS>::value ||                           \
                             vectorization::is_expression<RHS>::value,                             \
            bool> = true>                                                                          \
    VECTORIZATION_FUNCTION_ATTRIBUTE auto op(LHS&& lhs, MHS&& mhs, RHS&& rhs)                      \
    {                                                                                              \
        return vectorization::                                                                     \
            trinary_expression<LHS, MHS, RHS, vectorization::MACRO_EVALUATOR_SUFIX(op)>(           \
                std::forward<LHS>(lhs), std::forward<MHS>(mhs), std::forward<RHS>(rhs));           \
    }
//------------------------------------------------------------------------------------------------
MACRO_EXPRESSION_FUNCTION_3_ARG(fma);
MACRO_EXPRESSION_FUNCTION_3_ARG(if_else);

//================================================================================================
#define EXPRESSION_ARG2_OPERATOR_HELPER(op, symbol, ref)                                           \
    template <typename LHS,                                                                        \
        typename RHS,                                                                              \
        std::enable_if_t<vectorization::is_expression<LHS>::value ||                               \
                             vectorization::is_expression<RHS>::value,                             \
            bool> = true>                                                                          \
    VECTORIZATION_FUNCTION_ATTRIBUTE auto OPERATOR(symbol)(LHS ref lhs, RHS ref rhs)               \
    {                                                                                              \
        return vectorization::                                                                     \
            binary_expression<LHS, RHS, vectorization::MACRO_EVALUATOR_SUFIX(op)>(lhs, rhs);       \
    }

#define MACRO_EXPRESSION_OPERATOR_2_ARG(op, symbol)                                                \
    EXPRESSION_ARG2_OPERATOR_HELPER(op, symbol, const&);                                           \
    template <typename LHS,                                                                        \
        typename RHS,                                                                              \
        std::enable_if_t<vectorization::is_expression<LHS>::value ||                               \
                             vectorization::is_expression<RHS>::value,                             \
            bool> = true>                                                                          \
    VECTORIZATION_FUNCTION_ATTRIBUTE auto OPERATOR(symbol)(LHS && lhs, RHS && rhs)                 \
    {                                                                                              \
        return vectorization::                                                                     \
            binary_expression<LHS, RHS, vectorization::MACRO_EVALUATOR_SUFIX(op)>(                 \
                std::forward<LHS>(lhs), std::forward<RHS>(rhs));                                   \
    }
//------------------------------------------------------------------------------------------------
namespace vectorization
{
namespace detail
{
template <typename E> struct is_mul_node : std::false_type
{
};

template <typename L, typename R>
struct is_mul_node<binary_expression<L, R, mul_evaluator>> : std::true_type
{
};

// True when `E` is a floating-point `x * y` node that `E + z` contracts into
// fma(x, y, z); see VECTORIZATION_FUSE_MUL_ADD.
template <typename E, typename = void> struct fusable_mul : std::false_type
{
};

template <typename E>
struct fusable_mul<E, std::enable_if_t<is_mul_node<remove_cvref_t<E>>::value>>
    : std::bool_constant<VECTORIZATION_FUSE_MUL_ADD &&
                         std::is_floating_point_v<
                             typename scalar_type<remove_cvref_t<E>, remove_cvref_t<E>>::value>>
{
};

// fma(x, y, z) from the `x * y` node `m` and addend `z`. An rvalue `m` gives up its
// stored (already retained) leaves instead of copying them.
template <typename M, typename Z> VECTORIZATION_FUNCTION_ATTRIBUTE auto fuse_mul_add(M&& m, Z&& z)
{
    using x_t    = remove_cvref_t<decltype(m.lhs())>;
    using y_t    = remove_cvref_t<decltype(m.rhs())>;
    using result = trinary_expression<x_t, y_t, remove_cvref_t<Z>, fma_evaluator>;
    if constexpr (std::is_lvalue_reference_v<M> || std::is_const_v<std::remove_reference_t<M>>)
    {
        return result(forward_operands_t{}, m.lhs(), m.rhs(), std::forward<Z>(z));
    }
    else
    {
        return result(forward_operands_t{},
            std::move(m.mutable_lhs()),
            std::move(m.mutable_rhs()),
            std::forward<Z>(z));
    }
}

// `lhs + rhs`. LHS/RHS are the operator's deduced template arguments, so the unfused
// node has the same type as every other binary operator builds.
template <typename LHS, typename RHS, typename A, typename B>
VECTORIZATION_FUNCTION_ATTRIBUTE auto make_add(A&& lhs, B&& rhs)
{
    if constexpr (fusable_mul<A>::value)
    {
        return fuse_mul_add(std::forward<A>(lhs), std::forward<B>(rhs));
    }
    else if constexpr (fusable_mul<B>::value)
    {
        return fuse_mul_add(std::forward<B>(rhs), std::forward<A>(lhs));
    }
    else
    {
        return binary_expression<LHS, RHS, add_evaluator>(
            std::forward<A>(lhs), std::forward<B>(rhs));
    }
}
}  // namespace detail

// land/lor/lxor evaluators (and thus these operators) are only compiled when
// VECTORIZATION_VECTORIZED=1; see expressions_functors.h for the full rationale.
#if VECTORIZATION_VECTORIZED
MACRO_EXPRESSION_OPERATOR_2_ARG(land, &&);
MACRO_EXPRESSION_OPERATOR_2_ARG(lor, ||);
MACRO_EXPRESSION_OPERATOR_2_ARG(lxor, ^);
#endif  // VECTORIZATION_VECTORIZED
MACRO_EXPRESSION_OPERATOR_2_ARG(cmpgt, >);
MACRO_EXPRESSION_OPERATOR_2_ARG(cmplt, <);
MACRO_EXPRESSION_OPERATOR_2_ARG(cmpge, >=);
MACRO_EXPRESSION_OPERATOR_2_ARG(cmple, <=);
MACRO_EXPRESSION_OPERATOR_2_ARG(cmpeq, ==);
MACRO_EXPRESSION_OPERATOR_2_ARG(cmpne, !=);

// `+` contracts a floating-point `x * y` operand into fma (see detail::make_add).
template <typename LHS,
    typename RHS,
    std::enable_if_t<vectorization::is_expression<LHS>::value ||
                         vectorization::is_expression<RHS>::value,
        bool> = true>
VECTORIZATION_FUNCTION_ATTRIBUTE auto operator+(LHS const& lhs, RHS const& rhs)
{
    return detail::make_add<LHS, RHS>(lhs, rhs);
}

template <typename LHS,
    typename RHS,
    std::enable_if_t<vectorization::is_expression<LHS>::value ||
                         vectorization::is_expression<RHS>::value,
        bool> = true>
VECTORIZATION_FUNCTION_ATTRIBUTE auto operator+(LHS&& lhs, RHS&& rhs)
{
    return detail::make_add<LHS, RHS>(std::forward<LHS>(lhs), std::forward<RHS>(rhs));
}

MACRO_EXPRESSION_OPERATOR_2_ARG(sub, -);
MACRO_EXPRESSION_OPERATOR_2_ARG(mul, *);
MACRO_EXPRESSION_OPERATOR_2_ARG(div, /);

}  // namespace vectorization

//================================================================================================
#define EXPRESSION_ARG2_MOPERATOR_HELPER(op, symbol, ref)                                          \
    VECTORIZATION_EXEC_CHECK_DISABLE                                                               \
    template <typename LHS,                                                                        \
        typename RHS,                                                                              \
        std::enable_if_t<vectorization::is_base_expression<LHS>::value, int> = 0>                  \
    VECTORIZATION_FUNCTION_ATTRIBUTE void OPERATOR(symbol)(LHS ref lhs, RHS const ref rhs)         \
    {                                                                                              \
        /* `z += x * y` contracts like `z = z + x * y` (see VECTORIZATION_FUSE_MUL_ADD). */        \
        if constexpr (std::is_same_v<vectorization::MACRO_EVALUATOR_SUFIX(op),                     \
                          vectorization::madd_evaluator> &&                                        \
                      vectorization::detail::fusable_mul<RHS>::value)                              \
        {                                                                                          \
            lhs = vectorization::detail::fuse_mul_add(                                             \
                static_cast<RHS const&>(rhs), static_cast<LHS const&>(lhs));                       \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            lhs = vectorization::                                                                  \
                binary_expression<LHS, RHS, vectorization::MACRO_EVALUATOR_SUFIX(op)>(             \
                    static_cast<LHS const&>(lhs), static_cast<RHS const&>(rhs));                   \
        }                                                                                          \
    }

#define MACRO_EXPRESSION_MOPERATOR_2_ARG(op, symbol)                                               \
    EXPRESSION_ARG2_MOPERATOR_HELPER(op, symbol, &);                                               \
    EXPRESSION_ARG2_MOPERATOR_HELPER(op, symbol, &&);

namespace vectorization
{
MACRO_EXPRESSION_MOPERATOR_2_ARG(madd, +=);
MACRO_EXPRESSION_MOPERATOR_2_ARG(msub, -=);
MACRO_EXPRESSION_MOPERATOR_2_ARG(mmul, *=);
MACRO_EXPRESSION_MOPERATOR_2_ARG(mdiv, /=);
}  // namespace vectorization
#endif
