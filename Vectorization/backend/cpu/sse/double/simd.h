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

#include <type_traits>

#if VECTORIZATION_HAS_SVML || VECTORIZATION_HAS_SLEEF
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-calling-convention"
#endif

#include "backend/cpu/sse/svml.h"

#ifdef __clang__
#pragma clang diagnostic pop
#endif
#endif

#include "backend/cpu/sse/intrinsics.h"
#include "common/vectorization_macros.h"

template <>
struct simd<double>
{
    using simd_t      = __m128d;
    using mask_t      = __m128d;
    using simd_half_t = __m128d;
    using simd_int_t  = __m128i;
    using value_t     = double;
    using int_t       = uint32_t;

    static constexpr int size      = 2;
    static constexpr int half_size = 1;

    inline static simd_t const sign_mask = _mm_set1_pd(-0.0);

    VECTORIZATION_SIMD_RETURN_TYPE prefetch(const value_t* addr)
    {
        _mm_prefetch(PREFETCH_PTR_TYPE(addr), _MM_HINT_T0);
    }

    VECTORIZATION_SIMD_METHOD simd_t load(const value_t* addr) { return _mm_load_pd(addr); }

    VECTORIZATION_SIMD_METHOD simd_t loadu(const value_t* addr) { return _mm_loadu_pd(addr); }

    VECTORIZATION_SIMD_RETURN_TYPE store(simd_t from, value_t* to) { _mm_store_pd(to, from); }

    VECTORIZATION_SIMD_RETURN_TYPE storeu(simd_t from, value_t* to) { _mm_storeu_pd(to, from); }

    template <
        typename scalar_t,
        typename std::enable_if<std::is_fundamental<scalar_t>::value, bool>::type = true>
    VECTORIZATION_SIMD_METHOD simd_t set(const scalar_t alpha)
    {
        return _mm_set1_pd(static_cast<value_t>(alpha));
    }

    VECTORIZATION_SIMD_METHOD simd_t setzero() { return _mm_setzero_pd(); }

    VECTORIZATION_SIMD_METHOD simd_t add(simd_t x, simd_t y) { return _mm_add_pd(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t sub(simd_t x, simd_t y) { return _mm_sub_pd(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t mul(simd_t x, simd_t y) { return _mm_mul_pd(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t div(simd_t x, simd_t y) { return _mm_div_pd(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t fma(simd_t x, simd_t y, simd_t z)
    {
#ifdef __FMA__
        return _mm_fmadd_pd(x, y, z);
#else
        return _mm_add_pd(_mm_mul_pd(x, y), z);
#endif
    }

    VECTORIZATION_SIMD_METHOD simd_t pow(simd_t x, simd_t y) { return _mm_pow_pd(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t hypot(simd_t x, simd_t y) { return _mm_hypot_pd(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t min(simd_t x, simd_t y) { return _mm_min_pd(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t max(simd_t x, simd_t y) { return _mm_max_pd(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t signcopy(simd_t x, simd_t sign)
    {
        return _mm_signcopy_pd(x, sign);
    }

    VECTORIZATION_SIMD_METHOD simd_t sqrt(simd_t x) { return _mm_sqrt_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t sqr(simd_t x) { return _mm_mul_pd(x, x); }

    VECTORIZATION_SIMD_METHOD simd_t ceil(simd_t x)
    {
#ifdef __SSE4_1__
        return _mm_ceil_pd(x);
#else
        simd_t f    = _mm_cvtepi32_pd(_mm_cvttpd_epi32(x));
        simd_t mask = _mm_cmplt_pd(f, x);
        return _mm_add_pd(f, _mm_and_pd(mask, _mm_set1_pd(1.0)));
#endif
    }

    VECTORIZATION_SIMD_METHOD simd_t floor(simd_t x)
    {
#ifdef __SSE4_1__
        return _mm_floor_pd(x);
#else
        simd_t f    = _mm_cvtepi32_pd(_mm_cvttpd_epi32(x));
        simd_t mask = _mm_cmpgt_pd(f, x);
        return _mm_sub_pd(f, _mm_and_pd(mask, _mm_set1_pd(1.0)));
#endif
    }

    VECTORIZATION_SIMD_METHOD simd_t exp(simd_t x) { return _mm_exp_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t expm1(simd_t x) { return _mm_expm1_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t exp2(simd_t x) { return _mm_exp2_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t exp10(simd_t x) { return _mm_exp10_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t log(simd_t x) { return _mm_log_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t log1p(simd_t x) { return _mm_log1p_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t log2(simd_t x) { return _mm_log2_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t log10(simd_t x) { return _mm_log10_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t sin(simd_t x) { return _mm_sin_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t cos(simd_t x) { return _mm_cos_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t tan(simd_t x) { return _mm_tan_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t asin(simd_t x) { return _mm_asin_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t acos(simd_t x) { return _mm_acos_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t atan(simd_t x) { return _mm_atan_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t sinh(simd_t x) { return _mm_sinh_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t cosh(simd_t x) { return _mm_cosh_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t tanh(simd_t x) { return _mm_tanh_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t asinh(simd_t x) { return _mm_asinh_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t acosh(simd_t x) { return _mm_acosh_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t atanh(simd_t x) { return _mm_atanh_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t cbrt(simd_t x) { return _mm_cbrt_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t cdf(simd_t x) { return _mm_cdfnorm_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t inv_cdf(simd_t x) { return _mm_cdfnorminv_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t trunc(simd_t x) { return _mm_trunc_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t invsqrt(simd_t x) { return _mm_invsqrt_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t fabs(simd_t x) { return _mm_andnot_pd(sign_mask, x); }

    VECTORIZATION_SIMD_METHOD simd_t neg(simd_t x) { return _mm_xor_pd(x, sign_mask); }

    VECTORIZATION_FORCE_INLINE static double accumulate(simd_t x) { return _mm_accumulate_pd(x); }

    VECTORIZATION_FORCE_INLINE static value_t hmax(simd_t x) { return _mm_hmax_pd(x); }

    VECTORIZATION_FORCE_INLINE static value_t hmin(simd_t x) { return _mm_hmin_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t gather(value_t const* from, int stride)
    {
        return _mm_gather_pd(from, stride);
    }

    VECTORIZATION_SIMD_RETURN_TYPE scatter(simd_t from, int stride, value_t* to)
    {
        _mm_scatter_pd(from, to, stride);
    }

    VECTORIZATION_SIMD_METHOD simd_t gather(value_t const* from, const int* strides)
    {
        return _mm_gather_pd(from, strides);
    }

    VECTORIZATION_SIMD_RETURN_TYPE scatter(simd_t from, const int* stride, value_t* to)
    {
        _mm_scatter_pd(from, to, stride);
    }

    VECTORIZATION_SIMD_METHOD simd_t if_else(const mask_t& x, simd_t y, simd_t z)
    {
#ifdef __SSE4_1__
        return _mm_blendv_pd(z, y, x);
#else
        return _mm_or_pd(_mm_andnot_pd(x, z), _mm_and_pd(x, y));
#endif
    }

    VECTORIZATION_SIMD_METHOD mask_t eq(simd_t x, simd_t y) { return _mm_cmpeq_pd(x, y); }

    VECTORIZATION_SIMD_METHOD mask_t neq(simd_t x, simd_t y) { return _mm_cmpneq_pd(x, y); }

    VECTORIZATION_SIMD_METHOD mask_t gt(simd_t x, simd_t y) { return _mm_cmpgt_pd(x, y); }

    VECTORIZATION_SIMD_METHOD mask_t lt(simd_t x, simd_t y) { return _mm_cmplt_pd(x, y); }

    VECTORIZATION_SIMD_METHOD mask_t ge(simd_t x, simd_t y) { return _mm_cmpge_pd(x, y); }

    VECTORIZATION_SIMD_METHOD mask_t le(simd_t x, simd_t y) { return _mm_cmple_pd(x, y); }

    VECTORIZATION_SIMD_METHOD mask_t loadu(int_t const* from) { return _mm_loadu_mask_pd(from); }

    VECTORIZATION_SIMD_METHOD mask_t load(int_t const* from) { return _mm_load_mask_pd(from); }

    VECTORIZATION_SIMD_RETURN_TYPE storeu(const mask_t& from, int_t* to)
    {
        _mm_storeu_mask_pd(from, to);
    }

    VECTORIZATION_SIMD_RETURN_TYPE store(const mask_t& from, int_t* to)
    {
        _mm_store_mask_pd(from, to);
    }

    VECTORIZATION_SIMD_METHOD mask_t not_mask(const mask_t& x) { return _mm_mask_not_pd(x); }

    VECTORIZATION_SIMD_METHOD mask_t and_mask(const mask_t& x, const mask_t& y)
    {
        return _mm_and_pd(x, y);
    }

    VECTORIZATION_SIMD_METHOD mask_t or_mask(const mask_t& x, const mask_t& y)
    {
        return _mm_or_pd(x, y);
    }

    VECTORIZATION_SIMD_METHOD mask_t xor_mask(const mask_t& x, const mask_t& y)
    {
        return _mm_xor_pd(x, y);
    }

    VECTORIZATION_SIMD_RETURN_TYPE broadcast(
        const double* from, simd_t& a0, simd_t& a1, simd_t& a2, simd_t& a3)
    {
#ifdef __SSE3__
        a0 = _mm_loaddup_pd(from);
        a1 = _mm_loaddup_pd(from + 1);
        a2 = _mm_loaddup_pd(from + 2);
        a3 = _mm_loaddup_pd(from + 3);
#else
        a0 = _mm_set1_pd(from[0]);
        a1 = _mm_set1_pd(from[1]);
        a2 = _mm_set1_pd(from[2]);
        a3 = _mm_set1_pd(from[3]);
#endif
    }

    VECTORIZATION_SIMD_RETURN_TYPE copy(const value_t* from, value_t* to)
    {
        _mm_store_pd(to, _mm_loadu_pd(from));
    }

    template <int N>
    VECTORIZATION_SIMD_RETURN_TYPE ptranspose(simd_t simd[N])  // NOLINT
    {
        constexpr int L = size;
        if constexpr (N == 2 && L == 2)
        {
            __m128d t0 = _mm_shuffle_pd(simd[0], simd[1], 0);
            __m128d t1 = _mm_shuffle_pd(simd[0], simd[1], 3);
            simd[0]    = t0;
            simd[1]    = t1;
        }
        else if constexpr (N == L)
        {
            alignas(16) double buf[N * L];
            for (int i = 0; i < N; ++i)
            {
                _mm_store_pd(buf + i * L, simd[i]);
            }
            alignas(16) double out[N * L];
            for (int i = 0; i < N; ++i)
            {
                for (int j = 0; j < L; ++j)
                {
                    out[i * L + j] = buf[j * L + i];
                }
            }
            for (int i = 0; i < N; ++i)
            {
                simd[i] = _mm_load_pd(out + i * L);
            }
        }
        else if constexpr (N == 8 && L == 2)
        {
            alignas(16) double buf[8][2];
            for (int i = 0; i < 8; ++i)
            {
                _mm_store_pd(buf[i], simd[i]);
            }
            alignas(16) double out[2][8];
            for (int i = 0; i < 8; ++i)
            {
                for (int j = 0; j < 2; ++j)
                {
                    out[j][i] = buf[i][j];
                }
            }
            for (int r = 0; r < 2; ++r)
            {
                for (int q = 0; q < 4; ++q)
                {
                    simd[4 * r + q] = _mm_load_pd(&out[r][2 * q]);
                }
            }
        }
        else if constexpr (N == 4 && L == 2)
        {
            alignas(16) double buf[4][2];
            for (int i = 0; i < 4; ++i)
            {
                _mm_store_pd(buf[i], simd[i]);
            }
            alignas(16) double out[2][4];
            for (int i = 0; i < 4; ++i)
            {
                for (int j = 0; j < 2; ++j)
                {
                    out[j][i] = buf[i][j];
                }
            }
            for (int r = 0; r < 2; ++r)
            {
                for (int q = 0; q < 2; ++q)
                {
                    simd[2 * r + q] = _mm_load_pd(&out[r][2 * q]);
                }
            }
        }
        else
        {
            static_assert(sizeof(simd_t) == 0, "ptranspose: unsupported N for SSE double");
        }
    }

    VECTORIZATION_SIMD_METHOD simd_t ploadquad(const double* from) { return _mm_set1_pd(from[0]); }
};
