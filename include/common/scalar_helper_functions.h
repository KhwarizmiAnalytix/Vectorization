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

#include <cmath>        // for fma, sqrt, erfc, erfinv
#include <type_traits>  // for enable_if_t

#include "common/normal_cdf.h"  // for normalcdf / inv_normalcdf (host only)
#include "common/vectorization_macros.h"

// All functions below are injected into namespace std so that the expression
// functor macros can call std::op(...) uniformly for both CPU and GPU scalar
// paths.  Every function carries VECTORIZATION_FUNCTION_ATTRIBUTE so that
// NVCC / HIPCC emits __host__ __device__ versions, making them callable from
// both host and GPU device code.

namespace std
{
template <typename T, std::enable_if_t<std::is_fundamental<T>::value, int> = 0>
VECTORIZATION_FUNCTION_ATTRIBUTE auto signcopy(T a, T b)
{
#ifdef copysign
    return copysign(a, b);
#else
    return std::copysign(a, b);
#endif
}

template <typename T, std::enable_if_t<std::is_fundamental<T>::value, int> = 0>
VECTORIZATION_FUNCTION_ATTRIBUTE T fma(T a, T b, T c)
{
    return std::fma(a, b, c);
}

template <typename T, std::enable_if_t<std::is_fundamental<T>::value, int> = 0>
VECTORIZATION_FUNCTION_ATTRIBUTE T neg(T a)
{
    return -a;
}

template <typename T, std::enable_if_t<std::is_fundamental<T>::value, int> = 0>
VECTORIZATION_FUNCTION_ATTRIBUTE T invsqrt(T a)
{
    return T(1) / std::sqrt(a);
}

template <typename T, std::enable_if_t<std::is_fundamental<T>::value, int> = 0>
VECTORIZATION_FUNCTION_ATTRIBUTE T sqr(T a)
{
    return a * a;
}

template <typename T, std::enable_if_t<std::is_fundamental<T>::value, int> = 0>
VECTORIZATION_FUNCTION_ATTRIBUTE T cdf(T a)
{
    // erfc is available in both <cmath> (CPU) and CUDA/HIP device math.
    // Formula: N(x) = 0.5 * erfc(-x / sqrt(2))
#if VECTORIZATION_ON_GPU_DEVICE
    return static_cast<T>(0.5) * erfc(-a * static_cast<T>(0.7071067811865475244));
#else
    return static_cast<T>(vectorization::normalcdf(static_cast<double>(a)));
#endif
}

template <typename T, std::enable_if_t<std::is_fundamental<T>::value, int> = 0>
VECTORIZATION_FUNCTION_ATTRIBUTE T inv_cdf(T a)
{
    // erfinv is available in CUDA (>= 7.5) and ROCm HIP device math.
    // Formula: N^{-1}(p) = -sqrt(2) * erfinv(1 - 2p)
#if VECTORIZATION_ON_GPU_DEVICE
    return static_cast<T>(-1.4142135623730950488) * erfinv(T(1) - T(2) * a);
#else
    return static_cast<T>(vectorization::inv_normalcdf(static_cast<double>(a)));
#endif
}

template <
    typename T1,
    typename T2,
    std::enable_if_t<std::is_fundamental<T1>::value && std::is_fundamental<T2>::value, int> = 0>
VECTORIZATION_FUNCTION_ATTRIBUTE auto add(T1 a, T2 b)
{
    return a + b;
}

template <
    typename T1,
    typename T2,
    std::enable_if_t<std::is_fundamental<T1>::value && std::is_fundamental<T2>::value, int> = 0>
VECTORIZATION_FUNCTION_ATTRIBUTE auto sub(T1 a, T2 b)
{
    return a - b;
}

template <
    typename T1,
    typename T2,
    std::enable_if_t<std::is_fundamental<T1>::value && std::is_fundamental<T2>::value, int> = 0>
VECTORIZATION_FUNCTION_ATTRIBUTE auto mul(T1 a, T2 b)
{
    return a * b;
}

template <
    typename T1,
    typename T2,
    std::enable_if_t<std::is_fundamental<T1>::value && std::is_fundamental<T2>::value, int> = 0>
VECTORIZATION_FUNCTION_ATTRIBUTE auto div(T1 a, T2 b)
{
    return a / b;
}

template <
    typename T1,
    typename T2,
    std::enable_if_t<std::is_fundamental<T1>::value && std::is_fundamental<T2>::value, int> = 0>
VECTORIZATION_FUNCTION_ATTRIBUTE auto eq(T1 a, T2 b)
{
    return a == b;
}

template <
    typename T1,
    typename T2,
    std::enable_if_t<std::is_fundamental<T1>::value && std::is_fundamental<T2>::value, int> = 0>
VECTORIZATION_FUNCTION_ATTRIBUTE auto neq(T1 a, T2 b)
{
    return a != b;
}

template <
    typename T1,
    typename T2,
    std::enable_if_t<std::is_fundamental<T1>::value && std::is_fundamental<T2>::value, int> = 0>
VECTORIZATION_FUNCTION_ATTRIBUTE auto gt(T1 a, T2 b)
{
    return a > b;
}

template <
    typename T1,
    typename T2,
    std::enable_if_t<std::is_fundamental<T1>::value && std::is_fundamental<T2>::value, int> = 0>
VECTORIZATION_FUNCTION_ATTRIBUTE auto ge(T1 a, T2 b)
{
    return a >= b;
}

template <
    typename T1,
    typename T2,
    std::enable_if_t<std::is_fundamental<T1>::value && std::is_fundamental<T2>::value, int> = 0>
VECTORIZATION_FUNCTION_ATTRIBUTE auto lt(T1 a, T2 b)
{
    return a < b;
}

template <
    typename T1,
    typename T2,
    std::enable_if_t<std::is_fundamental<T1>::value && std::is_fundamental<T2>::value, int> = 0>
VECTORIZATION_FUNCTION_ATTRIBUTE auto le(T1 a, T2 b)
{
    return a <= b;
}

template <
    typename T1,
    typename T2,
    std::enable_if_t<std::is_fundamental<T1>::value && std::is_fundamental<T2>::value, int> = 0>
VECTORIZATION_FUNCTION_ATTRIBUTE auto cmple(T1 a, T2 b)
{
    return a <= b;
}

template <typename T1, std::enable_if_t<std::is_fundamental<T1>::value, int> = 0>
VECTORIZATION_FUNCTION_ATTRIBUTE auto lnot(T1 a)
{
    return !a;
}

template <
    typename T1,
    typename T2,
    std::enable_if_t<std::is_fundamental<T1>::value && std::is_fundamental<T2>::value, int> = 0>
VECTORIZATION_FUNCTION_ATTRIBUTE auto lxor(T1 a, T2 b)
{
    return a ^ b;
}

template <
    typename T1,
    typename T2,
    std::enable_if_t<std::is_fundamental<T1>::value && std::is_fundamental<T2>::value, int> = 0>
VECTORIZATION_FUNCTION_ATTRIBUTE auto land(T1 a, T2 b)
{
    return a && b;
}

template <
    typename T1,
    typename T2,
    std::enable_if_t<std::is_fundamental<T1>::value && std::is_fundamental<T2>::value, int> = 0>
VECTORIZATION_FUNCTION_ATTRIBUTE auto lor(T1 a, T2 b)
{
    return a || b;
}

template <
    typename T1,
    typename T2,
    std::enable_if_t<std::is_fundamental<T1>::value && std::is_fundamental<T2>::value, int> = 0>
VECTORIZATION_FUNCTION_ATTRIBUTE auto if_else(bool a, T1 b, T2 c)
{
    return a ? b : c;
}
}  // namespace std
