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

#include <array>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "backend/simd.h"

namespace memory
{
template <typename value_t>
struct data_view;
}

// ---------------------------------------------------------------------------
// is_packet
// ---------------------------------------------------------------------------
namespace vectorization
{

template <typename T>
struct is_packet
{
    static constexpr bool value = false;
};

#if VECTORIZATION_VECTORIZED
// Specializing on hardware vector types (__m256d/__m256/etc.) as template
// arguments is well-defined but GCC/Clang ignore the vector_size attribute
// for template-argument matching purposes and warn about it; the attribute
// isn't needed here since these specializations only key off the type.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

template <typename value_t>
using scalar_type_simd_t = typename simd<value_t>::simd_t;

template <>
struct is_packet<scalar_type_simd_t<double>>
{
    static constexpr bool value = true;
};

template <>
struct is_packet<scalar_type_simd_t<float>>
{
    static constexpr bool value = true;
};

#if VECTORIZATION_HAS_AVX512 || VECTORIZATION_HAS_NEON || VECTORIZATION_HAS_SVE
template <typename value_t>
using scalar_type_mask_t = typename simd<value_t>::mask_t;

template <>
struct is_packet<scalar_type_mask_t<double>>
{
    static constexpr bool value = true;
};

template <>
struct is_packet<scalar_type_mask_t<float>>
{
    static constexpr bool value = true;
};
#endif  // AVX512 / NEON / SVE mask packet traits

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
#endif  // VECTORIZATION_VECTORIZED

}  // namespace vectorization

// ---------------------------------------------------------------------------
// remove_cvref
// ---------------------------------------------------------------------------
namespace vectorization
{
template <typename T>
struct remove_cvref
{
    using type = std::remove_cv_t<std::remove_reference_t<T>>;
};

template <typename T>
using remove_cvref_t = typename remove_cvref<T>::type;

#if __cplusplus >= 202002L
#define VECTORIZATION_EXPR_HAS_MHS(e) (requires { (e).mhs(); })
#define VECTORIZATION_EXPR_HAS_LHS(e) (requires { (e).lhs(); })
#define VECTORIZATION_EXPR_HAS_RHS(e) (requires { (e).rhs(); })
#define VECTORIZATION_EXPR_HAS_RECORD_STREAM(e, s) (requires { (e).record_stream(s); })
#else
// C++17 detection for expression operand accessors (C++20 `requires` equivalent).
template <typename T, typename = void>
struct has_mhs : std::false_type
{
};

template <typename T>
struct has_mhs<T, std::void_t<decltype(std::declval<T const&>().mhs())>> : std::true_type
{
};

template <typename T, typename = void>
struct has_lhs : std::false_type
{
};

template <typename T>
struct has_lhs<T, std::void_t<decltype(std::declval<T const&>().lhs())>> : std::true_type
{
};

template <typename T, typename = void>
struct has_rhs : std::false_type
{
};

template <typename T>
struct has_rhs<T, std::void_t<decltype(std::declval<T const&>().rhs())>> : std::true_type
{
};

template <typename T, typename Stream, typename = void>
struct has_record_stream : std::false_type
{
};

template <typename T, typename Stream>
struct has_record_stream<
    T,
    Stream,
    std::void_t<decltype(std::declval<T const&>().record_stream(std::declval<Stream>()))>>
    : std::true_type
{
};

#define VECTORIZATION_EXPR_HAS_MHS(e) \
    (::vectorization::has_mhs<::vectorization::remove_cvref_t<decltype(e)>>::value)
#define VECTORIZATION_EXPR_HAS_LHS(e) \
    (::vectorization::has_lhs<::vectorization::remove_cvref_t<decltype(e)>>::value)
#define VECTORIZATION_EXPR_HAS_RHS(e) \
    (::vectorization::has_rhs<::vectorization::remove_cvref_t<decltype(e)>>::value)
#define VECTORIZATION_EXPR_HAS_RECORD_STREAM(e, s)    \
    (::vectorization::has_record_stream<              \
        ::vectorization::remove_cvref_t<decltype(e)>, \
        ::vectorization::remove_cvref_t<decltype(s)>>::value)
#endif
}  // namespace vectorization

// ---------------------------------------------------------------------------
// is_fundamental (extends std::is_fundamental with SIMD packet types)
// ---------------------------------------------------------------------------
namespace vectorization
{
template <typename T>
struct is_fundamental
{
    static constexpr bool value =
        (std::is_fundamental<vectorization::remove_cvref_t<T>>::value ||
         is_packet<vectorization::remove_cvref_t<T>>::value);
};
}  // namespace vectorization

// ---------------------------------------------------------------------------
// Forward declarations
//
// tensor<T> is the sole container type.  vector<T> and matrix<T> are
// transparent type aliases so that existing code continues to compile while
// emitting a deprecation diagnostic.  All trait specialisations are written
// once for tensor<T> and automatically cover the aliases.
// ---------------------------------------------------------------------------
namespace vectorization
{

template <typename value_t>
class tensor;

// Expression node types
template <typename LHS, typename EVALUATOR>
class unary_expression;

template <typename LHS, typename RHS, typename EVALUATOR>
class binary_expression;

template <typename LHS, typename MHS, typename RHS, typename EVALUATOR>
class trinary_expression;

}  // namespace vectorization

// ---------------------------------------------------------------------------
// is_base_expression / is_pure_expression / is_expression
// ---------------------------------------------------------------------------
namespace vectorization
{

template <typename T>
struct is_base_expression
{
    static constexpr bool value = false;
};

template <typename value_t>
struct is_base_expression<tensor<value_t>>
{
    static constexpr bool value = true;
};

template <typename value_t>
struct is_base_expression<memory::data_view<value_t>>
{
    static constexpr bool value = true;
};

template <typename T>
struct is_tensor
{
    static constexpr bool value = false;
};

template <typename value_t>
struct is_tensor<tensor<value_t>>
{
    static constexpr bool value = true;
};

template <typename T>
struct is_pure_expression
{
    static constexpr bool value = false;
};

template <typename T, typename E>
struct is_pure_expression<unary_expression<T, E>>
{
    static constexpr bool value = true;
};

template <typename T1, typename T2, typename E>
struct is_pure_expression<binary_expression<T1, T2, E>>
{
    static constexpr bool value = true;
};

template <typename T1, typename T2, typename T3, typename E>
struct is_pure_expression<trinary_expression<T1, T2, T3, E>>
{
    static constexpr bool value = true;
};

template <typename T>
struct is_expression
{
    static constexpr bool value = (is_pure_expression<T>::value || is_base_expression<T>::value);
};

}  // namespace vectorization

// ---------------------------------------------------------------------------
// scalar_type — maps an expression type to its underlying scalar (float/double)
// ---------------------------------------------------------------------------
namespace vectorization
{

template <typename T, typename V, typename Dummy = void>
struct scalar_type
{
};

template <>
struct scalar_type<float, float>
{
    using value = float;
};

template <>
struct scalar_type<double, double>
{
    using value = double;
};

#if VECTORIZATION_VECTORIZED
// See the matching comment on the is_packet specializations above: these
// specializations key off hardware vector types, and the vector_size
// attribute GCC/Clang ignore for template matching isn't needed here.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

template <typename T>
struct scalar_type<scalar_type_simd_t<double>, T>
{
    using value = double;
};

template <typename T>
struct scalar_type<scalar_type_simd_t<float>, T>
{
    using value = float;
};

#if VECTORIZATION_HAS_AVX512 || VECTORIZATION_HAS_NEON || VECTORIZATION_HAS_SVE
template <typename T>
struct scalar_type<scalar_type_mask_t<double>, T>
{
    using value = double;
};

template <typename T>
struct scalar_type<scalar_type_mask_t<float>, T>
{
    using value = float;
};
#endif  // AVX512 / NEON / SVE mask scalar_type

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
#endif  // VECTORIZATION_VECTORIZED

// tensor<value_t> — covers former vector<T> and matrix<T> since they alias tensor
template <typename value_t, typename T>
struct scalar_type<tensor<value_t>, T>
{
    using value = value_t;
};

template <typename value_t, typename T>
struct scalar_type<memory::data_view<value_t>, T>
{
    using value = value_t;
};

template <typename T, typename value_t>
struct scalar_type<
    T,
    tensor<value_t>,
    typename std::enable_if_t<vectorization::is_fundamental<T>::value>>
{
    using value = value_t;
};

template <typename E, typename EVALUATOR, typename T>
struct scalar_type<unary_expression<E, EVALUATOR>, T>
{
    using value = typename scalar_type<E, E>::value;
};

template <typename T, typename E, typename EVALUATOR>
struct scalar_type<
    T,
    unary_expression<E, EVALUATOR>,
    typename std::enable_if_t<vectorization::is_fundamental<T>::value>>
{
    using value =
        typename scalar_type<vectorization::remove_cvref_t<E>, vectorization::remove_cvref_t<E>>::
            value;
};

template <typename LHS, typename RHS, typename EVALUATOR, typename T>
struct scalar_type<binary_expression<LHS, RHS, EVALUATOR>, T>
{
    using value = typename scalar_type<
        vectorization::remove_cvref_t<LHS>,
        vectorization::remove_cvref_t<RHS>>::value;
};

template <typename T, typename LHS, typename RHS, typename EVALUATOR>
struct scalar_type<
    T,
    binary_expression<LHS, RHS, EVALUATOR>,
    typename std::enable_if_t<vectorization::is_fundamental<T>::value>>
{
    using value = typename scalar_type<
        vectorization::remove_cvref_t<LHS>,
        vectorization::remove_cvref_t<RHS>>::value;
};

template <typename LHS, typename MHS, typename RHS, typename EVALUATOR, typename T>
struct scalar_type<trinary_expression<LHS, MHS, RHS, EVALUATOR>, T>
{
    using value = typename scalar_type<
        vectorization::remove_cvref_t<MHS>,
        vectorization::remove_cvref_t<RHS>>::value;
};

template <typename T, typename LHS, typename MHS, typename RHS, typename EVALUATOR>
struct scalar_type<
    T,
    trinary_expression<LHS, MHS, RHS, EVALUATOR>,
    typename std::enable_if_t<vectorization::is_fundamental<T>::value>>
{
    using value = typename scalar_type<
        vectorization::remove_cvref_t<MHS>,
        vectorization::remove_cvref_t<RHS>>::value;
};

}  // namespace vectorization
