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
struct simd<float>
{
    using simd_t = __m128;
    using mask_t = __m128;

    // Distinct from simd_t so overloads (gather/add/scatter on half width) are not redeclarations.
    struct simd_half_t
    {
        __m128 v;

        simd_half_t() = default;

        explicit simd_half_t(__m128 x) noexcept : v(x) {}
    };

    using simd_int_t = __m128i;
    using value_t    = float;
    using int_t      = uint32_t;

    static constexpr int size      = 4;
    static constexpr int half_size = 2;

    inline static simd_t const sign_mask = _mm_set1_ps(-0.0F);

    VECTORIZATION_SIMD_RETURN_TYPE prefetch(const value_t* addr)
    {
        _mm_prefetch(PREFETCH_PTR_TYPE(addr), _MM_HINT_T0);
    }

    VECTORIZATION_SIMD_METHOD simd_t load(const value_t* addr) { return _mm_load_ps(addr); }

    VECTORIZATION_SIMD_METHOD simd_t loadu(const value_t* addr) { return _mm_loadu_ps(addr); }

    VECTORIZATION_SIMD_RETURN_TYPE store(simd_t from, value_t* to) { _mm_store_ps(to, from); }

    VECTORIZATION_SIMD_RETURN_TYPE storeu(simd_t from, value_t* to) { _mm_storeu_ps(to, from); }

    template <
        typename scalar_t,
        typename std::enable_if<std::is_fundamental<scalar_t>::value, bool>::type = true>
    VECTORIZATION_SIMD_METHOD simd_t set(scalar_t alpha)
    {
        return _mm_set1_ps(static_cast<value_t>(alpha));
    }

    VECTORIZATION_SIMD_METHOD simd_t setzero() { return _mm_setzero_ps(); }

    VECTORIZATION_SIMD_METHOD simd_t add(simd_t x, simd_t y) { return _mm_add_ps(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t sub(simd_t x, simd_t y) { return _mm_sub_ps(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t mul(simd_t x, simd_t y) { return _mm_mul_ps(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t div(simd_t x, simd_t y) { return _mm_div_ps(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t pow(simd_t x, simd_t y) { return _mm_pow_ps(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t hypot(simd_t x, simd_t y) { return _mm_hypot_ps(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t min(simd_t x, simd_t y) { return _mm_min_ps(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t max(simd_t x, simd_t y) { return _mm_max_ps(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t fma(simd_t x, simd_t y, simd_t z)
    {
#ifdef __FMA__
        return _mm_fmadd_ps(x, y, z);
#else
        return _mm_add_ps(_mm_mul_ps(x, y), z);
#endif
    }

    VECTORIZATION_SIMD_METHOD simd_t signcopy(simd_t x, simd_t sign)
    {
        return _mm_signcopy_ps(x, sign);
    }

    VECTORIZATION_SIMD_METHOD simd_t sqrt(simd_t x) { return _mm_sqrt_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t sqr(simd_t x) { return _mm_mul_ps(x, x); }

    VECTORIZATION_SIMD_METHOD simd_t ceil(simd_t x)
    {
#ifdef __SSE4_1__
        return _mm_ceil_ps(x);
#else
        simd_t f    = _mm_cvtepi32_ps(_mm_cvttps_epi32(x));
        simd_t mask = _mm_cmplt_ps(f, x);
        return _mm_add_ps(f, _mm_and_ps(mask, _mm_set1_ps(1.0f)));
#endif
    }

    VECTORIZATION_SIMD_METHOD simd_t floor(simd_t x)
    {
#ifdef __SSE4_1__
        return _mm_floor_ps(x);
#else
        simd_t f    = _mm_cvtepi32_ps(_mm_cvttps_epi32(x));
        simd_t mask = _mm_cmpgt_ps(f, x);
        return _mm_sub_ps(f, _mm_and_ps(mask, _mm_set1_ps(1.0f)));
#endif
    }

    VECTORIZATION_SIMD_METHOD simd_t exp(simd_t x) { return _mm_exp_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t expm1(simd_t x) { return _mm_expm1_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t exp2(simd_t x) { return _mm_exp2_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t exp10(simd_t x) { return _mm_exp10_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t log(simd_t x) { return _mm_log_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t log1p(simd_t x) { return _mm_log1p_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t log2(simd_t x) { return _mm_log2_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t log10(simd_t x) { return _mm_log10_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t sin(simd_t x) { return _mm_sin_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t cos(simd_t x) { return _mm_cos_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t tan(simd_t x) { return _mm_tan_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t asin(simd_t x) { return _mm_asin_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t acos(simd_t x) { return _mm_acos_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t atan(simd_t x) { return _mm_atan_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t sinh(simd_t x) { return _mm_sinh_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t cosh(simd_t x) { return _mm_cosh_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t tanh(simd_t x) { return _mm_tanh_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t asinh(simd_t x) { return _mm_asinh_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t acosh(simd_t x) { return _mm_acosh_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t atanh(simd_t x) { return _mm_atanh_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t cbrt(simd_t x) { return _mm_cbrt_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t cdf(simd_t x) { return _mm_cdfnorm_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t inv_cdf(simd_t x) { return _mm_cdfnorminv_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t trunc(simd_t x) { return _mm_trunc_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t invsqrt(simd_t x)
    {
#if VECTORIZATION_HAS_SVML || VECTORIZATION_HAS_SLEEF
        return _mm_invsqrt_ps(x);
#else
        // rsqrt_ps gives ~12-bit accuracy; one Newton-Raphson step reaches ~24 bits (full float).
        // r_new = (r / 2) * (3 - x * r^2)
        __m128 r   = _mm_rsqrt_ps(x);
        __m128 xr2 = _mm_mul_ps(x, _mm_mul_ps(r, r));
        return _mm_mul_ps(_mm_mul_ps(r, _mm_set1_ps(0.5f)), _mm_sub_ps(_mm_set1_ps(3.0f), xr2));
#endif
    }

    VECTORIZATION_SIMD_METHOD simd_t fabs(simd_t x) { return _mm_andnot_ps(sign_mask, x); }

    VECTORIZATION_SIMD_METHOD simd_t neg(simd_t x) { return _mm_xor_ps(x, sign_mask); }

    VECTORIZATION_FORCE_INLINE static double accumulate(simd_t x) { return _mm_accumulate_ps(x); }

    VECTORIZATION_FORCE_INLINE static value_t hmax(simd_t x) { return _mm_hmax_ps(x); }

    VECTORIZATION_FORCE_INLINE static value_t hmin(simd_t x) { return _mm_hmin_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t gather(value_t const* from, int stride)
    {
        return _mm_gather_ps(from, stride);
    }

    VECTORIZATION_SIMD_RETURN_TYPE scatter(simd_t from, int stride, value_t* to)
    {
        _mm_scatter_ps(from, to, stride);
    }

    VECTORIZATION_SIMD_METHOD simd_t gather(value_t const* from, const int* strides)
    {
        return _mm_gather_ps(from, strides);
    }

    VECTORIZATION_SIMD_RETURN_TYPE scatter(simd_t from, const int* stride, value_t* to)
    {
        _mm_scatter_ps(from, to, stride);
    }

    VECTORIZATION_SIMD_METHOD simd_t if_else(const mask_t& x, simd_t y, simd_t z)
    {
#ifdef __SSE4_1__
        return _mm_blendv_ps(z, y, x);
#else
        return _mm_or_ps(_mm_andnot_ps(x, z), _mm_and_ps(x, y));
#endif
    }

    VECTORIZATION_SIMD_METHOD mask_t eq(simd_t x, simd_t y) { return _mm_cmpeq_ps(x, y); }

    VECTORIZATION_SIMD_METHOD mask_t neq(simd_t x, simd_t y) { return _mm_cmpneq_ps(x, y); }

    VECTORIZATION_SIMD_METHOD mask_t gt(simd_t x, simd_t y) { return _mm_cmpgt_ps(x, y); }

    VECTORIZATION_SIMD_METHOD mask_t lt(simd_t x, simd_t y) { return _mm_cmplt_ps(x, y); }

    VECTORIZATION_SIMD_METHOD mask_t ge(simd_t x, simd_t y) { return _mm_cmpge_ps(x, y); }

    VECTORIZATION_SIMD_METHOD mask_t le(simd_t x, simd_t y) { return _mm_cmple_ps(x, y); }

    VECTORIZATION_SIMD_METHOD mask_t loadu(int_t const* from) { return _mm_loadu_mask_ps(from); }

    VECTORIZATION_SIMD_METHOD mask_t load(int_t const* from) { return _mm_load_mask_ps(from); }

    VECTORIZATION_SIMD_RETURN_TYPE storeu(const mask_t& from, int_t* to)
    {
        _mm_storeu_mask_ps(from, to);
    }

    VECTORIZATION_SIMD_RETURN_TYPE store(const mask_t& from, int_t* to)
    {
        _mm_store_mask_ps(from, to);
    }

    VECTORIZATION_SIMD_METHOD mask_t not_mask(const mask_t& x) { return _mm_mask_not_ps(x); }

    VECTORIZATION_SIMD_METHOD mask_t and_mask(const mask_t& x, const mask_t& y)
    {
        return _mm_and_ps(x, y);
    }

    VECTORIZATION_SIMD_METHOD mask_t or_mask(const mask_t& x, const mask_t& y)
    {
        return _mm_or_ps(x, y);
    }

    VECTORIZATION_SIMD_METHOD mask_t xor_mask(const mask_t& x, const mask_t& y)
    {
        return _mm_xor_ps(x, y);
    }

    VECTORIZATION_FORCE_INLINE
    static void broadcast(const value_t* from, simd_t& a0, simd_t& a1, simd_t& a2, simd_t& a3)
    {
        a0 = _mm_set1_ps(from[0]);
        a1 = _mm_set1_ps(from[1]);
        a2 = _mm_set1_ps(from[2]);
        a3 = _mm_set1_ps(from[3]);
    }

    template <int N>
    VECTORIZATION_SIMD_RETURN_TYPE ptranspose(simd_t simd[N])  // NOLINT
    {
        constexpr int L = size;
        if constexpr (N == 4 && L == 4)
        {
            transpose_4x4_ps(simd);
        }
        else if constexpr (N == L)
        {
            alignas(16) float buf[N * L];
            for (int i = 0; i < N; ++i)
            {
                _mm_store_ps(buf + i * L, simd[i]);
            }
            alignas(16) float out[N * L];
            for (int i = 0; i < N; ++i)
            {
                for (int j = 0; j < L; ++j)
                {
                    out[i * L + j] = buf[j * L + i];
                }
            }
            for (int i = 0; i < N; ++i)
            {
                simd[i] = _mm_load_ps(out + i * L);
            }
        }
        else if constexpr (N == 8 && L == 4)
        {
            alignas(16) float buf[8][4];
            for (int i = 0; i < 8; ++i)
            {
                _mm_store_ps(buf[i], simd[i]);
            }
            alignas(16) float out[4][8];
            for (int i = 0; i < 8; ++i)
            {
                for (int j = 0; j < 4; ++j)
                {
                    out[j][i] = buf[i][j];
                }
            }
            for (int r = 0; r < 4; ++r)
            {
                for (int q = 0; q < 2; ++q)
                {
                    simd[2 * r + q] = _mm_load_ps(&out[r][4 * q]);
                }
            }
        }
        else if constexpr (N == 16 && L == 4)
        {
            alignas(16) float buf[16][4];
            for (int i = 0; i < 16; ++i)
            {
                _mm_store_ps(buf[i], simd[i]);
            }
            alignas(16) float out[4][16];
            for (int i = 0; i < 16; ++i)
            {
                for (int j = 0; j < 4; ++j)
                {
                    out[j][i] = buf[i][j];
                }
            }
            for (int r = 0; r < 4; ++r)
            {
                for (int q = 0; q < 4; ++q)
                {
                    simd[4 * r + q] = _mm_load_ps(&out[r][4 * q]);
                }
            }
        }
        else
        {
            static_assert(sizeof(simd_t) == 0, "ptranspose: unsupported N for SSE float");
        }
    }

    VECTORIZATION_SIMD_RETURN_TYPE copy(const value_t* from, value_t* to)
    {
        _mm_store_ps(to, _mm_loadu_ps(from));
    }

    VECTORIZATION_SIMD_METHOD simd_t ploadquad(const value_t* from)
    {
        return _mm_set_ps(from[1], from[1], from[0], from[0]);
    }

    VECTORIZATION_SIMD_METHOD simd_half_t gather_half(const value_t* from, int stride)
    {
        return simd_half_t(_mm_set_ps(0.F, 0.F, from[stride], from[0]));
    }

    VECTORIZATION_FORCE_INLINE static simd_half_t loadu_half(const value_t* from)
    {
        __m128 lo = _mm_load_ss(from);
        __m128 hi = _mm_load_ss(from + 1);
        return simd_half_t(_mm_unpacklo_ps(lo, hi));
    }

    VECTORIZATION_FORCE_INLINE static simd_half_t set(const value_t* from)
    {
        return simd_half_t(_mm_set1_ps(*from));
    }

    VECTORIZATION_SIMD_METHOD simd_half_t
    fma(const simd_half_t& x, const simd_half_t& y, const simd_half_t& z)
    {
#ifdef __FMA__
        return simd_half_t(_mm_fmadd_ps(x.v, y.v, z.v));
#else
        return simd_half_t(_mm_add_ps(_mm_mul_ps(x.v, y.v), z.v));
#endif
    }

    VECTORIZATION_SIMD_METHOD simd_half_t add(const simd_half_t& x, const simd_half_t& y)
    {
        return simd_half_t(_mm_add_ps(x.v, y.v));
    }

    VECTORIZATION_FORCE_INLINE static simd_half_t predux_downto4(simd_t x)
    {
        return simd_half_t(_mm_add_ps(x, _mm_movehl_ps(x, x)));
    }

    VECTORIZATION_SIMD_RETURN_TYPE scatter(const simd_half_t& from, int stride, value_t* to)
    {
        to[0]      = _mm_cvtss_f32(from.v);
        to[stride] = _mm_cvtss_f32(_mm_shuffle_ps(from.v, from.v, 1));
    }
};
