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

// Stack alignment for evaluator loops that call SVML.
//
// SVML's vector entry points spill vector registers to the stack. With the
// stack only 16-byte aligned at the call, as the x64 ABIs guarantee, those
// 32/64-byte spills can split cache lines, depending on where the caller's
// frame happens to land. On a Windows clang-cl build that made float
// sin/exp/log up to 1.8x slower from one call site to the next. An
// over-aligned local makes the compiler realign the frame once, at entry, so
// every call made from the loop sees a 64-byte-aligned stack pointer.
//
// VECTORIZATION_MATH_CALL_FRAME(E) goes at the top of each function that holds
// an evaluator loop over expression type E. It does nothing unless E contains
// a node whose SIMD path is an SVML call, so loops of plain arithmetic keep
// their frame pointer register. Define VECTORIZATION_ALIGN_MATH_CALL_FRAME to 0
// to turn the guard off, or to 1 to force it on.

#include <type_traits>

#include "common/vectorization_type_traits.h"

#ifndef VECTORIZATION_ALIGN_MATH_CALL_FRAME
// Measured with clang-cl on x64. The asm statement that keeps the local alive
// is GNU syntax, so MSVC is excluded; MSVC also uses its own SVML intrinsics
// rather than Scalars' SVML wrappers.
#if VECTORIZATION_VECTORIZED && defined(SCALARS_HAS_SVML) && SCALARS_HAS_SVML &&                   \
    (defined(__x86_64__) || defined(_M_X64)) && (defined(__clang__) || defined(__GNUC__))
#define VECTORIZATION_ALIGN_MATH_CALL_FRAME 1
#else
#define VECTORIZATION_ALIGN_MATH_CALL_FRAME 0
#endif
#endif

namespace vectorization
{
// Evaluators defined in expressions_functors.h.
struct exp_evaluator;
struct expm1_evaluator;
struct exp2_evaluator;
struct log_evaluator;
struct log1p_evaluator;
struct log2_evaluator;
struct log10_evaluator;
struct sin_evaluator;
struct cos_evaluator;
struct tan_evaluator;
struct asin_evaluator;
struct acos_evaluator;
struct atan_evaluator;
struct sinh_evaluator;
struct cosh_evaluator;
struct tanh_evaluator;
struct asinh_evaluator;
struct acosh_evaluator;
struct atanh_evaluator;
struct cbrt_evaluator;
struct cdf_evaluator;
struct inv_cdf_evaluator;
struct trunc_evaluator;
struct invsqrt_evaluator;
struct pow_evaluator;
struct hypot_evaluator;

namespace detail
{
// True for evaluators whose x86 SIMD path goes through the vector math
// library (svml.h in Scalars) rather than inline instructions. Some of these
// are inline in Scalars for some types (float exp, for example); the guard
// then costs a frame realignment per call of the loop function and nothing
// per element.
template <typename Ev> struct is_vector_math_call : std::false_type
{
};

#define VECTORIZATION_VECTOR_MATH_CALL(op)                                                         \
    template <> struct is_vector_math_call<op##_evaluator> : std::true_type                        \
    {                                                                                              \
    };
VECTORIZATION_VECTOR_MATH_CALL(exp)
VECTORIZATION_VECTOR_MATH_CALL(expm1)
VECTORIZATION_VECTOR_MATH_CALL(exp2)
VECTORIZATION_VECTOR_MATH_CALL(log)
VECTORIZATION_VECTOR_MATH_CALL(log1p)
VECTORIZATION_VECTOR_MATH_CALL(log2)
VECTORIZATION_VECTOR_MATH_CALL(log10)
VECTORIZATION_VECTOR_MATH_CALL(sin)
VECTORIZATION_VECTOR_MATH_CALL(cos)
VECTORIZATION_VECTOR_MATH_CALL(tan)
VECTORIZATION_VECTOR_MATH_CALL(asin)
VECTORIZATION_VECTOR_MATH_CALL(acos)
VECTORIZATION_VECTOR_MATH_CALL(atan)
VECTORIZATION_VECTOR_MATH_CALL(sinh)
VECTORIZATION_VECTOR_MATH_CALL(cosh)
VECTORIZATION_VECTOR_MATH_CALL(tanh)
VECTORIZATION_VECTOR_MATH_CALL(asinh)
VECTORIZATION_VECTOR_MATH_CALL(acosh)
VECTORIZATION_VECTOR_MATH_CALL(atanh)
VECTORIZATION_VECTOR_MATH_CALL(cbrt)
VECTORIZATION_VECTOR_MATH_CALL(cdf)
VECTORIZATION_VECTOR_MATH_CALL(inv_cdf)
VECTORIZATION_VECTOR_MATH_CALL(trunc)
VECTORIZATION_VECTOR_MATH_CALL(invsqrt)
VECTORIZATION_VECTOR_MATH_CALL(pow)
VECTORIZATION_VECTOR_MATH_CALL(hypot)
#undef VECTORIZATION_VECTOR_MATH_CALL

// True when any node of expression type E calls the vector math library.
// Leaves (tensors, refs, scalars) are false.
template <typename E> struct calls_vector_math : std::false_type
{
};

template <typename L, typename Ev>
struct calls_vector_math<unary_expression<L, Ev>>
    : std::bool_constant<is_vector_math_call<Ev>::value ||
                         calls_vector_math<remove_cvref_t<L>>::value>
{
};

template <typename L, typename R, typename Ev>
struct calls_vector_math<binary_expression<L, R, Ev>>
    : std::bool_constant<is_vector_math_call<Ev>::value ||
                         calls_vector_math<remove_cvref_t<L>>::value ||
                         calls_vector_math<remove_cvref_t<R>>::value>
{
};

template <typename L, typename M, typename R, typename Ev>
struct calls_vector_math<trinary_expression<L, M, R, Ev>>
    : std::bool_constant<is_vector_math_call<Ev>::value ||
                         calls_vector_math<remove_cvref_t<L>>::value ||
                         calls_vector_math<remove_cvref_t<M>>::value ||
                         calls_vector_math<remove_cvref_t<R>>::value>
{
};
}  // namespace detail
}  // namespace vectorization

#if VECTORIZATION_ALIGN_MATH_CALL_FRAME
// The empty asm takes the local's address, so the compiler can neither drop
// it nor its alignment. 64 bytes covers AVX-512 spills as well as AVX2.
#define VECTORIZATION_MATH_CALL_FRAME(E)                                                           \
    if constexpr (::vectorization::detail::calls_vector_math<                                      \
                      ::vectorization::remove_cvref_t<E>>::value)                                  \
    {                                                                                              \
        alignas(64) char vectorization_math_frame_[64];                                            \
        __asm__ volatile("" ::"r"(vectorization_math_frame_) : "memory");                          \
    }
#else
#define VECTORIZATION_MATH_CALL_FRAME(E) static_cast<void>(0)
#endif
