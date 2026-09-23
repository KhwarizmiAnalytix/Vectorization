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

#if VECTORIZATION_HAS_SVML || VECTORIZATION_HAS_SLEEF
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-calling-convention"
#endif

#include "backend/cpu/avx/svml.h"

#ifdef __clang__
#pragma clang diagnostic pop
#endif
#endif

#include "common/vectorization_macros.h"

template <>
struct simd<float>
{
    using simd_t      = __m256;
    using mask_t      = __m256;
    using simd_half_t = __m128;
    using simd_int_t  = __m256i;
    using value_t     = float;
    using int_t       = uint32_t;

    static constexpr int size      = 8;
    static constexpr int half_size = 4;

    inline static simd_t const sign_mask = _mm256_set1_ps(-0.0F);

    VECTORIZATION_SIMD_RETURN_TYPE prefetch(const value_t* addr)
    {
        _mm_prefetch(PREFETCH_PTR_TYPE(addr), _MM_HINT_T0);
    }

    //======================================================================================
    // load, store, set functions
    //======================================================================================

    VECTORIZATION_SIMD_METHOD simd_t load(const value_t* addr) { return _mm256_load_ps(addr); }

    VECTORIZATION_SIMD_METHOD simd_t loadu(const value_t* addr) { return _mm256_loadu_ps(addr); }

    VECTORIZATION_SIMD_RETURN_TYPE store(simd_t from, value_t* to) { _mm256_store_ps(to, from); }

    VECTORIZATION_SIMD_RETURN_TYPE storeu(simd_t from, value_t* to) { _mm256_storeu_ps(to, from); }

    template <
        typename scalar_t,
        typename std::enable_if<std::is_fundamental<scalar_t>::value, bool>::type = true>
    VECTORIZATION_SIMD_METHOD simd_t set(scalar_t alpha)
    {
        return _mm256_set1_ps(static_cast<value_t>(alpha));
    }

    VECTORIZATION_SIMD_METHOD simd_t setzero() { return _mm256_setzero_ps(); }
    //======================================================================================
    // +, -, *, / functions
    //======================================================================================
    VECTORIZATION_SIMD_METHOD simd_t add(simd_t x, simd_t y) { return _mm256_add_ps(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t sub(simd_t x, simd_t y) { return _mm256_sub_ps(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t mul(simd_t x, simd_t y) { return _mm256_mul_ps(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t div(simd_t x, simd_t y) { return _mm256_div_ps(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t pow(simd_t x, simd_t y) { return _mm256_pow_ps(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t hypot(simd_t x, simd_t y) { return _mm256_hypot_ps(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t min(simd_t x, simd_t y) { return _mm256_min_ps(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t max(simd_t x, simd_t y) { return _mm256_max_ps(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t fma(simd_t x, simd_t y, simd_t z)
    {
#ifdef __FMA__
        return _mm256_fmadd_ps(x, y, z);
#else
        return _mm256_add_ps(_mm256_mul_ps(x, y), z);
#endif  // __FMA__
    }

    VECTORIZATION_SIMD_METHOD simd_t signcopy(simd_t x, simd_t sign)
    {
        return _mm256_or_ps(_mm256_and_ps(sign_mask, sign), _mm256_andnot_ps(sign_mask, x));
    }

    //======================================================================================
    // one arg functions
    //======================================================================================
    VECTORIZATION_SIMD_METHOD simd_t sqrt(simd_t x) { return _mm256_sqrt_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t sqr(simd_t x) { return _mm256_mul_ps(x, x); }

    VECTORIZATION_SIMD_METHOD simd_t ceil(simd_t x) { return _mm256_ceil_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t floor(simd_t x) { return _mm256_floor_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t exp(simd_t x) { return _mm256_exp_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t expm1(simd_t x) { return _mm256_expm1_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t exp2(simd_t x) { return _mm256_exp2_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t exp10(simd_t x) { return _mm256_exp10_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t log(simd_t x) { return _mm256_log_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t log1p(simd_t x) { return _mm256_log1p_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t log2(simd_t x) { return _mm256_log2_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t log10(simd_t x) { return _mm256_log10_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t sin(simd_t x) { return _mm256_sin_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t cos(simd_t x) { return _mm256_cos_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t tan(simd_t x) { return _mm256_tan_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t asin(simd_t x) { return _mm256_asin_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t acos(simd_t x) { return _mm256_acos_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t atan(simd_t x) { return _mm256_atan_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t sinh(simd_t x) { return _mm256_sinh_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t cosh(simd_t x) { return _mm256_cosh_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t tanh(simd_t x) { return _mm256_tanh_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t asinh(simd_t x) { return _mm256_asinh_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t acosh(simd_t x) { return _mm256_acosh_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t atanh(simd_t x) { return _mm256_atanh_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t cbrt(simd_t x) { return _mm256_cbrt_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t cdf(simd_t x) { return _mm256_cdfnorm_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t inv_cdf(simd_t x) { return _mm256_cdfnorminv_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t trunc(simd_t x) { return _mm256_trunc_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t invsqrt(simd_t x)
    {
#if VECTORIZATION_HAS_SVML || VECTORIZATION_HAS_SLEEF
        return _mm256_invsqrt_ps(x);
#else
        // rsqrt_ps gives ~12-bit accuracy; one Newton-Raphson step reaches ~24 bits (full float).
        // r_new = (r / 2) * (3 - x * r^2)
        __m256 r   = _mm256_rsqrt_ps(x);
        __m256 xr2 = _mm256_mul_ps(x, _mm256_mul_ps(r, r));
        return _mm256_mul_ps(
            _mm256_mul_ps(r, _mm256_set1_ps(0.5f)), _mm256_sub_ps(_mm256_set1_ps(3.0f), xr2));
#endif
    }

    VECTORIZATION_SIMD_METHOD simd_t fabs(simd_t x) { return _mm256_andnot_ps(sign_mask, x); }

    VECTORIZATION_SIMD_METHOD simd_t neg(simd_t x) { return _mm256_xor_ps(x, sign_mask); }

    //======================================================================================
    // horizantal functions
    //======================================================================================
    VECTORIZATION_FORCE_INLINE static double accumulate(simd_t x)
    {
        auto t = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
        t      = _mm_add_ps(t, _mm_movehl_ps(t, t));
        t      = _mm_add_ss(t, _mm_shuffle_ps(t, t, 0x55));

        return static_cast<double>(_mm_cvtss_f32(t));
    }

    VECTORIZATION_FORCE_INLINE static value_t hmax(simd_t x)
    {
        auto t = _mm256_max_ps(x, _mm256_permute2f128_ps(x, x, 1));
        t      = _mm256_max_ps(t, _mm256_shuffle_ps(t, t, _MM_SHUFFLE(1, 0, 3, 2)));
        return _mm_cvtss_f32(_mm256_castps256_ps128(_mm256_max_ps(t, _mm256_shuffle_ps(t, t, 1))));
    }

    VECTORIZATION_FORCE_INLINE static value_t hmin(simd_t x)
    {
        auto t = _mm256_min_ps(x, _mm256_permute2f128_ps(x, x, 1));
        t      = _mm256_min_ps(t, _mm256_shuffle_ps(t, t, _MM_SHUFFLE(1, 0, 3, 2)));
        return _mm_cvtss_f32(
            _mm256_castps256_ps128((_mm256_min_ps(t, _mm256_shuffle_ps(t, t, 1)))));
    }

    //======================================================================================
    // gather/scatter function
    //======================================================================================
    VECTORIZATION_SIMD_METHOD simd_t gather(value_t const* from, int stride)
    {
        return _mm256_set_ps(
            from[7 * stride],
            from[6 * stride],
            from[5 * stride],
            from[4 * stride],
            from[3 * stride],
            from[2 * stride],
            from[stride],
            from[0]);
    }

    VECTORIZATION_SIMD_RETURN_TYPE scatter(simd_t from, int stride, value_t* to)
    {
        auto low       = _mm256_extractf128_ps(from, 0);
        to[0]          = _mm_cvtss_f32(low);
        to[stride]     = _mm_cvtss_f32(_mm_shuffle_ps(low, low, 1));
        to[stride * 2] = _mm_cvtss_f32(_mm_shuffle_ps(low, low, 2));
        to[stride * 3] = _mm_cvtss_f32(_mm_shuffle_ps(low, low, 3));

        auto high      = _mm256_extractf128_ps(from, 1);
        to[stride * 4] = _mm_cvtss_f32(high);
        to[stride * 5] = _mm_cvtss_f32(_mm_shuffle_ps(high, high, 1));
        to[stride * 6] = _mm_cvtss_f32(_mm_shuffle_ps(high, high, 2));
        to[stride * 7] = _mm_cvtss_f32(_mm_shuffle_ps(high, high, 3));
    }

    VECTORIZATION_SIMD_METHOD simd_t gather(value_t const* from, const int* strides)
    {
#ifdef __AVX2__
        auto indices = _mm256_loadu_si256(reinterpret_cast<const simd_int_t*>(strides));
        return _mm256_i32gather_ps(from, indices, 4);
#else
        return _mm256_set_ps(
            from[strides[7]],
            from[strides[6]],
            from[strides[5]],
            from[strides[4]],
            from[strides[3]],
            from[strides[2]],
            from[strides[1]],
            from[strides[0]]);
#endif  // __AVX2__
    }

    VECTORIZATION_SIMD_RETURN_TYPE scatter(simd_t from, const int* stride, value_t* to)
    {
        auto low      = _mm256_extractf128_ps(from, 0);
        to[stride[0]] = _mm_cvtss_f32(low);
        to[stride[1]] = _mm_cvtss_f32(_mm_shuffle_ps(low, low, 1));
        to[stride[2]] = _mm_cvtss_f32(_mm_shuffle_ps(low, low, 2));
        to[stride[3]] = _mm_cvtss_f32(_mm_shuffle_ps(low, low, 3));

        auto high     = _mm256_extractf128_ps(from, 1);
        to[stride[4]] = _mm_cvtss_f32(high);
        to[stride[5]] = _mm_cvtss_f32(_mm_shuffle_ps(high, high, 1));
        to[stride[6]] = _mm_cvtss_f32(_mm_shuffle_ps(high, high, 2));
        to[stride[7]] = _mm_cvtss_f32(_mm_shuffle_ps(high, high, 3));
    }

    //======================================================================================
    // comparison function
    //======================================================================================
    VECTORIZATION_SIMD_METHOD simd_t if_else(const mask_t& x, simd_t y, simd_t z)
    {
        return _mm256_blendv_ps(z, y, x);
    }

    VECTORIZATION_SIMD_METHOD mask_t eq(simd_t x, simd_t y)
    {
        return _mm256_cmp_ps(x, y, _CMP_EQ_OQ);
    }

    VECTORIZATION_SIMD_METHOD mask_t neq(simd_t x, simd_t y)
    {
        return _mm256_cmp_ps(x, y, _CMP_NEQ_UQ);
    }

    VECTORIZATION_SIMD_METHOD mask_t gt(simd_t x, simd_t y)
    {
        return _mm256_cmp_ps(x, y, _CMP_GT_OS);
    }

    VECTORIZATION_SIMD_METHOD mask_t lt(simd_t x, simd_t y)
    {
        return _mm256_cmp_ps(x, y, _CMP_LT_OS);
    }

    VECTORIZATION_SIMD_METHOD mask_t ge(simd_t x, simd_t y)
    {
        return _mm256_cmp_ps(x, y, _CMP_GE_OS);
    }

    VECTORIZATION_SIMD_METHOD mask_t le(simd_t x, simd_t y)
    {
        return _mm256_cmp_ps(x, y, _CMP_LE_OS);
    }

    //======================================================================================
    // comparison function
    //======================================================================================
    VECTORIZATION_SIMD_METHOD mask_t loadu(int_t const* from)
    {
        return _mm256_castsi256_ps(_mm256_loadu_si256(reinterpret_cast<const simd_int_t*>(from)));
    }

    VECTORIZATION_SIMD_METHOD mask_t load(int_t const* from)
    {
        return _mm256_castsi256_ps(_mm256_load_si256(reinterpret_cast<const simd_int_t*>(from)));
    }

    VECTORIZATION_SIMD_RETURN_TYPE storeu(const mask_t& from, int_t* to)
    {
        _mm256_storeu_si256(reinterpret_cast<simd_int_t*>(to), _mm256_castps_si256(from));
    }

    VECTORIZATION_SIMD_RETURN_TYPE store(const mask_t& from, int_t* to)
    {
        _mm256_store_si256(reinterpret_cast<simd_int_t*>(to), _mm256_castps_si256(from));
    }

    VECTORIZATION_SIMD_METHOD mask_t not_mask(const mask_t& x)
    {
        return _mm256_andnot_ps(x, _mm256_cmp_ps(x, x, _CMP_EQ_OQ));
    }

    VECTORIZATION_SIMD_METHOD mask_t and_mask(const mask_t& x, const mask_t& y)
    {
        return _mm256_and_ps(x, y);
    }

    VECTORIZATION_SIMD_METHOD mask_t or_mask(const mask_t& x, const mask_t& y)
    {
        return _mm256_or_ps(x, y);
    }

    VECTORIZATION_SIMD_METHOD mask_t xor_mask(const mask_t& x, const mask_t& y)
    {
        return _mm256_xor_ps(x, y);
    }

    //-----------------------------------------------------------------------------
    VECTORIZATION_FORCE_INLINE
    static void broadcast(const value_t* from, simd_t& a0, simd_t& a1, simd_t& a2, simd_t& a3)
    {
        a0 = _mm256_broadcast_ss(from);
        a1 = _mm256_broadcast_ss(from + 1);
        a2 = _mm256_broadcast_ss(from + 2);
        a3 = _mm256_broadcast_ss(from + 3);
    }

    template <int N>
    VECTORIZATION_SIMD_RETURN_TYPE ptranspose(simd_t simd[N])  // NOLINT
    {
        if constexpr (N == 8)
        {
            auto T0 = _mm256_unpacklo_ps(simd[0], simd[1]);
            auto T1 = _mm256_unpackhi_ps(simd[0], simd[1]);
            auto T2 = _mm256_unpacklo_ps(simd[2], simd[3]);
            auto T3 = _mm256_unpackhi_ps(simd[2], simd[3]);
            auto T4 = _mm256_unpacklo_ps(simd[4], simd[5]);
            auto T5 = _mm256_unpackhi_ps(simd[4], simd[5]);
            auto T6 = _mm256_unpacklo_ps(simd[6], simd[7]);
            auto T7 = _mm256_unpackhi_ps(simd[6], simd[7]);

            auto S0 = _mm256_shuffle_ps(T0, T2, _MM_SHUFFLE(1, 0, 1, 0));
            auto S1 = _mm256_shuffle_ps(T0, T2, _MM_SHUFFLE(3, 2, 3, 2));
            auto S2 = _mm256_shuffle_ps(T1, T3, _MM_SHUFFLE(1, 0, 1, 0));
            auto S3 = _mm256_shuffle_ps(T1, T3, _MM_SHUFFLE(3, 2, 3, 2));
            auto S4 = _mm256_shuffle_ps(T4, T6, _MM_SHUFFLE(1, 0, 1, 0));
            auto S5 = _mm256_shuffle_ps(T4, T6, _MM_SHUFFLE(3, 2, 3, 2));
            auto S6 = _mm256_shuffle_ps(T5, T7, _MM_SHUFFLE(1, 0, 1, 0));
            auto S7 = _mm256_shuffle_ps(T5, T7, _MM_SHUFFLE(3, 2, 3, 2));

            simd[0] = _mm256_permute2f128_ps(S0, S4, 0x20);
            simd[1] = _mm256_permute2f128_ps(S1, S5, 0x20);
            simd[2] = _mm256_permute2f128_ps(S2, S6, 0x20);
            simd[3] = _mm256_permute2f128_ps(S3, S7, 0x20);
            simd[4] = _mm256_permute2f128_ps(S0, S4, 0x31);
            simd[5] = _mm256_permute2f128_ps(S1, S5, 0x31);
            simd[6] = _mm256_permute2f128_ps(S2, S6, 0x31);
            simd[7] = _mm256_permute2f128_ps(S3, S7, 0x31);
        }
        else
        {
            auto T0 = _mm256_unpacklo_ps(simd[0], simd[1]);
            auto T1 = _mm256_unpackhi_ps(simd[0], simd[1]);
            auto T2 = _mm256_unpacklo_ps(simd[2], simd[3]);
            auto T3 = _mm256_unpackhi_ps(simd[2], simd[3]);

            auto S0 = _mm256_shuffle_ps(T0, T2, _MM_SHUFFLE(1, 0, 1, 0));
            auto S1 = _mm256_shuffle_ps(T0, T2, _MM_SHUFFLE(3, 2, 3, 2));
            auto S2 = _mm256_shuffle_ps(T1, T3, _MM_SHUFFLE(1, 0, 1, 0));
            auto S3 = _mm256_shuffle_ps(T1, T3, _MM_SHUFFLE(3, 2, 3, 2));

            simd[0] = _mm256_permute2f128_ps(S0, S1, 0x20);
            simd[1] = _mm256_permute2f128_ps(S2, S3, 0x20);
            simd[2] = _mm256_permute2f128_ps(S0, S1, 0x31);
            simd[3] = _mm256_permute2f128_ps(S2, S3, 0x31);
        }
    }

    VECTORIZATION_SIMD_RETURN_TYPE copy(const value_t* from, value_t* to)
    {
        _mm256_store_ps(to, _mm256_loadu_ps(from));
    }

    VECTORIZATION_SIMD_METHOD simd_t ploadquad(const value_t* from)
    {
        auto t = _mm256_castps128_ps256(_mm_broadcast_ss(from));
        return _mm256_insertf128_ps(t, _mm_broadcast_ss(from + 1), 1);
    }

    VECTORIZATION_SIMD_METHOD simd_half_t gather_half(const value_t* from, int stride)
    {
        return _mm_set_ps(from[3 * stride], from[2 * stride], from[1 * stride], from[0 * stride]);
    }

    VECTORIZATION_FORCE_INLINE static simd_half_t loadu_half(const value_t* from)
    {
        return _mm_loadu_ps(from);
    }

    VECTORIZATION_FORCE_INLINE static simd_half_t set(const value_t* from)
    {
        return _mm_set1_ps(*from);
    }

    VECTORIZATION_SIMD_METHOD simd_half_t
    fma(const simd_half_t& x, const simd_half_t& y, const simd_half_t& z)
    {
#ifdef __FMA__
        return _mm_fmadd_ps(x, y, z);
#else
        return _mm_add_ps(_mm_mul_ps(x, y), z);
#endif  // __FMA__
    }

    VECTORIZATION_SIMD_METHOD simd_half_t add(const simd_half_t& x, const simd_half_t& y)
    {
        return _mm_add_ps(x, y);
    }

    VECTORIZATION_FORCE_INLINE static simd_half_t predux_downto4(simd_t x)
    {
        return _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    }

    VECTORIZATION_SIMD_RETURN_TYPE scatter(const simd_half_t& from, int stride, value_t* to)
    {
        to[0]          = _mm_cvtss_f32(from);
        to[stride]     = _mm_cvtss_f32(_mm_shuffle_ps(from, from, 1));
        to[stride * 2] = _mm_cvtss_f32(_mm_shuffle_ps(from, from, 2));
        to[stride * 3] = _mm_cvtss_f32(_mm_shuffle_ps(from, from, 3));
    }
};