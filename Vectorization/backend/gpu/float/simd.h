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

// GPU scalar backend for float — shared by CUDA (nvcc) and HIP (hipcc).
//
// On GPU each thread processes exactly one element (size = 1); the SIMD
// parallelism comes from launching a grid of threads, not from register-level
// lane packing.  All methods are __host__ __device__ (via VECTORIZATION_SIMD_METHOD)
// so expression types that embed simd<float> can be compiled in host code too.
//
// The two runtimes expose identical f-suffixed float math (::fmaf, ::powf,
// ::erfcf, ::erfinvf, ...) in the global namespace, so a single implementation
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
struct simd<float>
{
    using simd_t  = float;
    using mask_t  = uint32_t;
    using value_t = float;
    using int_t   = uint32_t;

    // One lane per thread — GPU-level parallelism is across threads, not within.
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

    VECTORIZATION_SIMD_METHOD simd_t setzero() { return 0.0f; }

    // -------------------------------------------------------------------------
    // Arithmetic
    // -------------------------------------------------------------------------

    VECTORIZATION_SIMD_METHOD simd_t add(simd_t x, simd_t y) { return x + y; }
    VECTORIZATION_SIMD_METHOD simd_t sub(simd_t x, simd_t y) { return x - y; }
    VECTORIZATION_SIMD_METHOD simd_t mul(simd_t x, simd_t y) { return x * y; }
    VECTORIZATION_SIMD_METHOD simd_t div(simd_t x, simd_t y) { return x / y; }

    VECTORIZATION_SIMD_METHOD simd_t fma(simd_t x, simd_t y, simd_t z) { return ::fmaf(x, y, z); }

    VECTORIZATION_SIMD_METHOD simd_t pow(simd_t x, simd_t y) { return ::powf(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t hypot(simd_t x, simd_t y) { return ::hypotf(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t min(simd_t x, simd_t y) { return ::fminf(x, y); }
    VECTORIZATION_SIMD_METHOD simd_t max(simd_t x, simd_t y) { return ::fmaxf(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t signcopy(simd_t x, simd_t y) { return ::copysignf(x, y); }

    // -------------------------------------------------------------------------
    // Unary math
    // -------------------------------------------------------------------------

    VECTORIZATION_SIMD_METHOD simd_t sqrt(simd_t x) { return ::sqrtf(x); }
    VECTORIZATION_SIMD_METHOD simd_t invsqrt(simd_t x) { return ::rsqrtf(x); }
    VECTORIZATION_SIMD_METHOD simd_t sqr(simd_t x) { return x * x; }
    VECTORIZATION_SIMD_METHOD simd_t fabs(simd_t x) { return ::fabsf(x); }
    VECTORIZATION_SIMD_METHOD simd_t neg(simd_t x) { return -x; }
    VECTORIZATION_SIMD_METHOD simd_t ceil(simd_t x) { return ::ceilf(x); }
    VECTORIZATION_SIMD_METHOD simd_t floor(simd_t x) { return ::floorf(x); }
    VECTORIZATION_SIMD_METHOD simd_t trunc(simd_t x) { return ::truncf(x); }

    VECTORIZATION_SIMD_METHOD simd_t exp(simd_t x) { return ::expf(x); }
    VECTORIZATION_SIMD_METHOD simd_t expm1(simd_t x) { return ::expm1f(x); }
    VECTORIZATION_SIMD_METHOD simd_t exp2(simd_t x) { return ::exp2f(x); }
    VECTORIZATION_SIMD_METHOD simd_t exp10(simd_t x) { return ::exp10f(x); }
    VECTORIZATION_SIMD_METHOD simd_t log(simd_t x) { return ::logf(x); }
    VECTORIZATION_SIMD_METHOD simd_t log1p(simd_t x) { return ::log1pf(x); }
    VECTORIZATION_SIMD_METHOD simd_t log2(simd_t x) { return ::log2f(x); }
    VECTORIZATION_SIMD_METHOD simd_t log10(simd_t x) { return ::log10f(x); }
    VECTORIZATION_SIMD_METHOD simd_t cbrt(simd_t x) { return ::cbrtf(x); }

    VECTORIZATION_SIMD_METHOD simd_t sin(simd_t x) { return ::sinf(x); }
    VECTORIZATION_SIMD_METHOD simd_t cos(simd_t x) { return ::cosf(x); }
    VECTORIZATION_SIMD_METHOD simd_t tan(simd_t x) { return ::tanf(x); }
    VECTORIZATION_SIMD_METHOD simd_t asin(simd_t x) { return ::asinf(x); }
    VECTORIZATION_SIMD_METHOD simd_t acos(simd_t x) { return ::acosf(x); }
    VECTORIZATION_SIMD_METHOD simd_t atan(simd_t x) { return ::atanf(x); }
    VECTORIZATION_SIMD_METHOD simd_t sinh(simd_t x) { return ::sinhf(x); }
    VECTORIZATION_SIMD_METHOD simd_t cosh(simd_t x) { return ::coshf(x); }
    VECTORIZATION_SIMD_METHOD simd_t tanh(simd_t x) { return ::tanhf(x); }
    VECTORIZATION_SIMD_METHOD simd_t asinh(simd_t x) { return ::asinhf(x); }
    VECTORIZATION_SIMD_METHOD simd_t acosh(simd_t x) { return ::acoshf(x); }
    VECTORIZATION_SIMD_METHOD simd_t atanh(simd_t x) { return ::atanhf(x); }

    // Normal CDF/inverse CDF — dispatch to device math on GPU, host math on CPU.
    // scalar_helper_functions.h uses VECTORIZATION_ON_GPU_DEVICE to pick the right path.
    // N(x) = 0.5 * erfc(-x / sqrt(2))
    VECTORIZATION_SIMD_METHOD simd_t cdf(simd_t x)
    {
        return 0.5f * ::erfcf(-x * 0.7071067811865475244f);
    }

    // N^{-1}(p) = -sqrt(2) * erfinv(1 - 2p)
    VECTORIZATION_SIMD_METHOD simd_t inv_cdf(simd_t x)
    {
        return -1.4142135623730950488f * ::erfinvf(1.0f - 2.0f * x);
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
    // Reductions — trivial identity for size=1 (single-lane)
    // -------------------------------------------------------------------------

    VECTORIZATION_SIMD_METHOD simd_t accumulate(simd_t x) { return x; }
    VECTORIZATION_SIMD_METHOD simd_t hmin(simd_t x) { return x; }
    VECTORIZATION_SIMD_METHOD simd_t hmax(simd_t x) { return x; }

    // -------------------------------------------------------------------------
    // Comparisons (all-set = true, all-clear = false, matching CPU lane masks)
    // -------------------------------------------------------------------------

    VECTORIZATION_SIMD_METHOD mask_t eq(simd_t x, simd_t y) { return x == y ? ~0u : 0u; }
    VECTORIZATION_SIMD_METHOD mask_t neq(simd_t x, simd_t y) { return x != y ? ~0u : 0u; }
    VECTORIZATION_SIMD_METHOD mask_t lt(simd_t x, simd_t y) { return x < y ? ~0u : 0u; }
    VECTORIZATION_SIMD_METHOD mask_t le(simd_t x, simd_t y) { return x <= y ? ~0u : 0u; }
    VECTORIZATION_SIMD_METHOD mask_t gt(simd_t x, simd_t y) { return x > y ? ~0u : 0u; }
    VECTORIZATION_SIMD_METHOD mask_t ge(simd_t x, simd_t y) { return x >= y ? ~0u : 0u; }

    VECTORIZATION_SIMD_METHOD mask_t and_mask(mask_t x, mask_t y) { return x & y; }
    VECTORIZATION_SIMD_METHOD mask_t or_mask(mask_t x, mask_t y) { return x | y; }
    VECTORIZATION_SIMD_METHOD mask_t xor_mask(mask_t x, mask_t y) { return x ^ y; }
    VECTORIZATION_SIMD_METHOD mask_t not_mask(mask_t x) { return ~x; }

    VECTORIZATION_SIMD_METHOD simd_t if_else(mask_t m, simd_t t, simd_t f) { return m ? t : f; }
};
