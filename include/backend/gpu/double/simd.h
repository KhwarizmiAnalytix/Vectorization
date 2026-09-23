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

// GPU scalar backend for double — shared by CUDA (nvcc) and HIP (hipcc).
// See backend/gpu/float/simd.h for design notes.
//
// The two runtimes expose identical unsuffixed double math (::fma, ::pow,
// ::erfc, ::erfinv, ...) in the global namespace, so a single implementation
// serves both; only the runtime header include differs.

#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#elif defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

#include "common/vectorization_macros.h"

// Primary template forward declaration required before the explicit specialization.
template <typename T>
struct simd;

template <>
struct simd<double>
{
    using simd_t  = double;
    using mask_t  = uint64_t;
    using value_t = double;
    using int_t   = uint64_t;

    static constexpr int size = 1;

    VECTORIZATION_SIMD_RETURN_TYPE prefetch(const value_t*) {}

    // -------------------------------------------------------------------------
    // Load / store / set
    // -------------------------------------------------------------------------

    VECTORIZATION_SIMD_METHOD simd_t load(const value_t* addr) { return *addr; }
    VECTORIZATION_SIMD_METHOD simd_t loadu(const value_t* addr) { return *addr; }

    VECTORIZATION_SIMD_RETURN_TYPE store(simd_t from, value_t* to) { *to = from; }
    VECTORIZATION_SIMD_RETURN_TYPE storeu(simd_t from, value_t* to) { *to = from; }

    template <
        typename scalar_t,
        typename std::enable_if<std::is_fundamental<scalar_t>::value, bool>::type = true>
    VECTORIZATION_SIMD_METHOD simd_t set(scalar_t alpha)
    {
        return static_cast<value_t>(alpha);
    }

    VECTORIZATION_SIMD_METHOD simd_t setzero() { return 0.0; }

    // -------------------------------------------------------------------------
    // Arithmetic
    // -------------------------------------------------------------------------

    VECTORIZATION_SIMD_METHOD simd_t add(simd_t x, simd_t y) { return x + y; }
    VECTORIZATION_SIMD_METHOD simd_t sub(simd_t x, simd_t y) { return x - y; }
    VECTORIZATION_SIMD_METHOD simd_t mul(simd_t x, simd_t y) { return x * y; }
    VECTORIZATION_SIMD_METHOD simd_t div(simd_t x, simd_t y) { return x / y; }

    VECTORIZATION_SIMD_METHOD simd_t fma(simd_t x, simd_t y, simd_t z) { return ::fma(x, y, z); }

    VECTORIZATION_SIMD_METHOD simd_t pow(simd_t x, simd_t y) { return ::pow(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t hypot(simd_t x, simd_t y) { return ::hypot(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t min(simd_t x, simd_t y) { return ::fmin(x, y); }
    VECTORIZATION_SIMD_METHOD simd_t max(simd_t x, simd_t y) { return ::fmax(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t signcopy(simd_t x, simd_t y) { return ::copysign(x, y); }

    // -------------------------------------------------------------------------
    // Unary math
    // -------------------------------------------------------------------------

    VECTORIZATION_SIMD_METHOD simd_t sqrt(simd_t x) { return ::sqrt(x); }
    // No double-precision rsqrt hardware on CUDA or HIP; division is correct.
    VECTORIZATION_SIMD_METHOD simd_t invsqrt(simd_t x) { return 1.0 / ::sqrt(x); }
    VECTORIZATION_SIMD_METHOD simd_t sqr(simd_t x) { return x * x; }
    VECTORIZATION_SIMD_METHOD simd_t fabs(simd_t x) { return ::fabs(x); }
    VECTORIZATION_SIMD_METHOD simd_t neg(simd_t x) { return -x; }
    VECTORIZATION_SIMD_METHOD simd_t ceil(simd_t x) { return ::ceil(x); }
    VECTORIZATION_SIMD_METHOD simd_t floor(simd_t x) { return ::floor(x); }
    VECTORIZATION_SIMD_METHOD simd_t trunc(simd_t x) { return ::trunc(x); }

    VECTORIZATION_SIMD_METHOD simd_t exp(simd_t x) { return ::exp(x); }
    VECTORIZATION_SIMD_METHOD simd_t expm1(simd_t x) { return ::expm1(x); }
    VECTORIZATION_SIMD_METHOD simd_t exp2(simd_t x) { return ::exp2(x); }
    VECTORIZATION_SIMD_METHOD simd_t exp10(simd_t x) { return ::exp10(x); }
    VECTORIZATION_SIMD_METHOD simd_t log(simd_t x) { return ::log(x); }
    VECTORIZATION_SIMD_METHOD simd_t log1p(simd_t x) { return ::log1p(x); }
    VECTORIZATION_SIMD_METHOD simd_t log2(simd_t x) { return ::log2(x); }
    VECTORIZATION_SIMD_METHOD simd_t log10(simd_t x) { return ::log10(x); }
    VECTORIZATION_SIMD_METHOD simd_t cbrt(simd_t x) { return ::cbrt(x); }

    VECTORIZATION_SIMD_METHOD simd_t sin(simd_t x) { return ::sin(x); }
    VECTORIZATION_SIMD_METHOD simd_t cos(simd_t x) { return ::cos(x); }
    VECTORIZATION_SIMD_METHOD simd_t tan(simd_t x) { return ::tan(x); }
    VECTORIZATION_SIMD_METHOD simd_t asin(simd_t x) { return ::asin(x); }
    VECTORIZATION_SIMD_METHOD simd_t acos(simd_t x) { return ::acos(x); }
    VECTORIZATION_SIMD_METHOD simd_t atan(simd_t x) { return ::atan(x); }
    VECTORIZATION_SIMD_METHOD simd_t sinh(simd_t x) { return ::sinh(x); }
    VECTORIZATION_SIMD_METHOD simd_t cosh(simd_t x) { return ::cosh(x); }
    VECTORIZATION_SIMD_METHOD simd_t tanh(simd_t x) { return ::tanh(x); }
    VECTORIZATION_SIMD_METHOD simd_t asinh(simd_t x) { return ::asinh(x); }
    VECTORIZATION_SIMD_METHOD simd_t acosh(simd_t x) { return ::acosh(x); }
    VECTORIZATION_SIMD_METHOD simd_t atanh(simd_t x) { return ::atanh(x); }

    // N(x) = 0.5 * erfc(-x / sqrt(2))
    VECTORIZATION_SIMD_METHOD simd_t cdf(simd_t x)
    {
        return 0.5 * ::erfc(-x * 0.7071067811865475244);
    }

    // N^{-1}(p) = -sqrt(2) * erfinv(1 - 2p)
    VECTORIZATION_SIMD_METHOD simd_t inv_cdf(simd_t x)
    {
        return -1.4142135623730950488 * ::erfinv(1.0 - 2.0 * x);
    }

    // -------------------------------------------------------------------------
    // Gather / scatter
    // -------------------------------------------------------------------------

    VECTORIZATION_SIMD_METHOD simd_t gather(const value_t* base, const int* offsets)
    {
        return base[offsets[0]];
    }

    // On GPU, stride is the pre-computed absolute element index for this thread.
    VECTORIZATION_SIMD_METHOD simd_t gather(const value_t* base, int stride)
    {
        return base[stride];
    }

    VECTORIZATION_SIMD_RETURN_TYPE scatter(simd_t val, const int* offsets, value_t* to)
    {
        to[offsets[0]] = val;
    }

    // On GPU, stride is the pre-computed absolute element index for this thread.
    VECTORIZATION_SIMD_RETURN_TYPE scatter(simd_t val, int stride, value_t* to)
    {
        to[stride] = val;
    }

    // -------------------------------------------------------------------------
    // Reductions — trivial identity for size=1
    // -------------------------------------------------------------------------

    VECTORIZATION_SIMD_METHOD simd_t accumulate(simd_t x) { return x; }
    VECTORIZATION_SIMD_METHOD simd_t hmin(simd_t x) { return x; }
    VECTORIZATION_SIMD_METHOD simd_t hmax(simd_t x) { return x; }

    // -------------------------------------------------------------------------
    // Comparisons
    // -------------------------------------------------------------------------

    VECTORIZATION_SIMD_METHOD mask_t eq(simd_t x, simd_t y)
    {
        return x == y ? ~uint64_t(0) : uint64_t(0);
    }
    VECTORIZATION_SIMD_METHOD mask_t neq(simd_t x, simd_t y)
    {
        return x != y ? ~uint64_t(0) : uint64_t(0);
    }
    VECTORIZATION_SIMD_METHOD mask_t lt(simd_t x, simd_t y)
    {
        return x < y ? ~uint64_t(0) : uint64_t(0);
    }
    VECTORIZATION_SIMD_METHOD mask_t le(simd_t x, simd_t y)
    {
        return x <= y ? ~uint64_t(0) : uint64_t(0);
    }
    VECTORIZATION_SIMD_METHOD mask_t gt(simd_t x, simd_t y)
    {
        return x > y ? ~uint64_t(0) : uint64_t(0);
    }
    VECTORIZATION_SIMD_METHOD mask_t ge(simd_t x, simd_t y)
    {
        return x >= y ? ~uint64_t(0) : uint64_t(0);
    }

    VECTORIZATION_SIMD_METHOD mask_t and_mask(mask_t x, mask_t y) { return x & y; }
    VECTORIZATION_SIMD_METHOD mask_t or_mask(mask_t x, mask_t y) { return x | y; }
    VECTORIZATION_SIMD_METHOD mask_t xor_mask(mask_t x, mask_t y) { return x ^ y; }
    VECTORIZATION_SIMD_METHOD mask_t not_mask(mask_t x) { return ~x; }

    VECTORIZATION_SIMD_METHOD simd_t if_else(mask_t m, simd_t t, simd_t f) { return m ? t : f; }
};
