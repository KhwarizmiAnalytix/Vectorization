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
struct simd<double>
{
    using simd_t      = __m256d;
    using mask_t      = __m256d;
    using simd_half_t = __m128d;
    using simd_int_t  = __m128i;
    using value_t     = double;
    using int_t       = uint32_t;

    static constexpr int size      = 4;
    static constexpr int half_size = 2;

    inline static simd_t const sign_mask = _mm256_set1_pd(-0.0);

    VECTORIZATION_SIMD_RETURN_TYPE prefetch(const value_t* addr)
    {
        _mm_prefetch(PREFETCH_PTR_TYPE(addr), _MM_HINT_T0);
    }

    //======================================================================================
    // load, store, set functions
    //======================================================================================
    VECTORIZATION_SIMD_METHOD simd_t load(const value_t* addr) { return _mm256_load_pd(addr); }

    VECTORIZATION_SIMD_METHOD simd_t loadu(const value_t* addr) { return _mm256_loadu_pd(addr); }

    VECTORIZATION_SIMD_RETURN_TYPE store(simd_t from, value_t* to) { _mm256_store_pd(to, from); }

    VECTORIZATION_SIMD_RETURN_TYPE storeu(simd_t from, value_t* to) { _mm256_storeu_pd(to, from); }

    template <
        typename scalar_t,
        typename std::enable_if<std::is_fundamental<scalar_t>::value, bool>::type = true>
    VECTORIZATION_SIMD_METHOD simd_t set(const scalar_t alpha)
    {
        return _mm256_set1_pd(static_cast<value_t>(alpha));
    }

    VECTORIZATION_SIMD_METHOD simd_t setzero() { return _mm256_setzero_pd(); }

    //======================================================================================
    // +, -, *, /, min, max, hypo functions
    //======================================================================================
    VECTORIZATION_SIMD_METHOD simd_t add(simd_t x, simd_t y) { return _mm256_add_pd(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t sub(simd_t x, simd_t y) { return _mm256_sub_pd(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t mul(simd_t x, simd_t y) { return _mm256_mul_pd(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t div(simd_t x, simd_t y) { return _mm256_div_pd(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t fma(simd_t x, simd_t y, simd_t z)
    {
#ifdef __FMA__
        return _mm256_fmadd_pd(x, y, z);
#else
        return _mm256_add_pd(_mm256_mul_pd(x, y), z);
#endif  // __FMA__
    }

    VECTORIZATION_SIMD_METHOD simd_t pow(simd_t x, simd_t y) { return _mm256_pow_pd(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t hypot(simd_t x, simd_t y) { return _mm256_hypot_pd(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t min(simd_t x, simd_t y) { return _mm256_min_pd(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t max(simd_t x, simd_t y) { return _mm256_max_pd(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t signcopy(simd_t x, simd_t sign)
    {
        return _mm256_or_pd(_mm256_and_pd(sign_mask, sign), _mm256_andnot_pd(sign_mask, x));
    }

    //======================================================================================
    // one arg function
    //======================================================================================
    VECTORIZATION_SIMD_METHOD simd_t sqrt(simd_t x) { return _mm256_sqrt_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t sqr(simd_t x) { return _mm256_mul_pd(x, x); }
    VECTORIZATION_SIMD_METHOD simd_t ceil(simd_t x) { return _mm256_ceil_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t floor(simd_t x) { return _mm256_floor_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t exp(simd_t x) { return _mm256_exp_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t expm1(simd_t x) { return _mm256_expm1_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t exp2(simd_t x) { return _mm256_exp2_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t exp10(simd_t x) { return _mm256_exp10_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t log(simd_t x) { return _mm256_log_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t log1p(simd_t x) { return _mm256_log1p_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t log2(simd_t x) { return _mm256_log2_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t log10(simd_t x) { return _mm256_log10_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t sin(simd_t x) { return _mm256_sin_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t cos(simd_t x) { return _mm256_cos_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t tan(simd_t x) { return _mm256_tan_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t asin(simd_t x) { return _mm256_asin_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t acos(simd_t x) { return _mm256_acos_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t atan(simd_t x) { return _mm256_atan_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t sinh(simd_t x) { return _mm256_sinh_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t cosh(simd_t x) { return _mm256_cosh_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t tanh(simd_t x) { return _mm256_tanh_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t asinh(simd_t x) { return _mm256_asinh_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t acosh(simd_t x) { return _mm256_acosh_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t atanh(simd_t x) { return _mm256_atanh_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t cbrt(simd_t x) { return _mm256_cbrt_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t cdf(simd_t x) { return _mm256_cdfnorm_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t inv_cdf(simd_t x) { return _mm256_cdfnorminv_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t trunc(simd_t x) { return _mm256_trunc_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t invsqrt(simd_t x) { return _mm256_invsqrt_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t fabs(simd_t x) { return _mm256_andnot_pd(sign_mask, x); }

    VECTORIZATION_SIMD_METHOD simd_t neg(simd_t x) { return _mm256_xor_pd(x, sign_mask); }

    //======================================================================================
    // horizantal functions
    //======================================================================================
    VECTORIZATION_FORCE_INLINE static double accumulate(simd_t x)
    {
        const auto& t1 = _mm_add_pd(_mm256_extractf128_pd(x, 1), _mm256_castpd256_pd128(x));
        const auto& t2 = _mm_unpackhi_pd(t1, t1);

        return _mm_cvtsd_f64(_mm_add_sd(t1, t2));
    }

    VECTORIZATION_FORCE_INLINE static value_t hmax(simd_t x)
    {
        const auto& t = _mm256_max_pd(x, _mm256_permute2f128_pd(x, x, 1));
        return _mm256_cvtsd_f64(_mm256_max_pd(t, _mm256_permute_pd(t, 5)));
    }

    VECTORIZATION_FORCE_INLINE static value_t hmin(simd_t x)
    {
        const auto& tmp = _mm256_min_pd(x, _mm256_permute2f128_pd(x, x, 1));
        return _mm256_cvtsd_f64(_mm256_min_pd(tmp, _mm256_permute_pd(tmp, 5)));
    }

    //======================================================================================
    // gather/scatter function
    //======================================================================================
    VECTORIZATION_SIMD_METHOD simd_t gather(value_t const* from, int stride)
    {
        return _mm256_set_pd(
            from[3 * stride], from[2 * stride], from[1 * stride], from[0 * stride]);
    }

    VECTORIZATION_SIMD_RETURN_TYPE scatter(simd_t from, int stride, value_t* to)
    {
        auto low   = _mm256_extractf128_pd(from, 0);
        to[0]      = _mm_cvtsd_f64(low);
        to[stride] = _mm_cvtsd_f64(_mm_shuffle_pd(low, low, 1));

        auto high      = _mm256_extractf128_pd(from, 1);
        to[stride * 2] = _mm_cvtsd_f64(high);
        to[stride * 3] = _mm_cvtsd_f64(_mm_shuffle_pd(high, high, 1));
    }

    VECTORIZATION_SIMD_METHOD simd_t gather(value_t const* from, const int* strides)
    {
#ifdef __AVX2__
        auto indices = _mm_loadu_si128(reinterpret_cast<const __m128i*>(strides));
        return _mm256_i32gather_pd(from, indices, 8);
#else
        return _mm256_set_pd(
            from[strides[3]], from[strides[2]], from[strides[1]], from[strides[0]]);
#endif  // __AVX2__
    }

    VECTORIZATION_SIMD_RETURN_TYPE scatter(simd_t from, const int* stride, value_t* to)
    {
        auto low      = _mm256_extractf128_pd(from, 0);
        to[stride[0]] = _mm_cvtsd_f64(low);
        to[stride[1]] = _mm_cvtsd_f64(_mm_shuffle_pd(low, low, 1));

        auto high     = _mm256_extractf128_pd(from, 1);
        to[stride[2]] = _mm_cvtsd_f64(high);
        to[stride[3]] = _mm_cvtsd_f64(_mm_shuffle_pd(high, high, 1));
    }

    //======================================================================================
    // comparison function
    //======================================================================================
    VECTORIZATION_SIMD_METHOD simd_t if_else(const mask_t& x, simd_t y, simd_t z)
    {
        return _mm256_blendv_pd(z, y, x);
    }

    VECTORIZATION_SIMD_METHOD mask_t eq(simd_t x, simd_t y)
    {
        return _mm256_cmp_pd(x, y, _CMP_EQ_OQ);
    }

    VECTORIZATION_SIMD_METHOD mask_t neq(simd_t x, simd_t y)
    {
        return _mm256_cmp_pd(x, y, _CMP_NEQ_UQ);
    }

    VECTORIZATION_SIMD_METHOD mask_t gt(simd_t x, simd_t y)
    {
        return _mm256_cmp_pd(x, y, _CMP_GT_OS);
    }

    VECTORIZATION_SIMD_METHOD mask_t lt(simd_t x, simd_t y)
    {
        return _mm256_cmp_pd(x, y, _CMP_LT_OS);
    }

    VECTORIZATION_SIMD_METHOD mask_t ge(simd_t x, simd_t y)
    {
        return _mm256_cmp_pd(x, y, _CMP_GE_OS);
    }

    VECTORIZATION_SIMD_METHOD mask_t le(simd_t x, simd_t y)
    {
        return _mm256_cmp_pd(x, y, _CMP_LE_OS);
    }

    //======================================================================================
    // comparison function
    //======================================================================================
    VECTORIZATION_SIMD_METHOD mask_t loadu(int_t const* from)
    {
        return _mm256_cvtepi32_pd(_mm_loadu_si128(reinterpret_cast<const simd_int_t*>(from)));
    }

    VECTORIZATION_SIMD_METHOD mask_t load(int_t const* from)
    {
        return _mm256_cvtepi32_pd(_mm_load_si128(reinterpret_cast<const simd_int_t*>(from)));
    }

    VECTORIZATION_SIMD_RETURN_TYPE storeu(const mask_t& from, int_t* to)
    {
        _mm_storeu_si128(reinterpret_cast<simd_int_t*>(to), _mm256_cvtpd_epi32(from));
    }

    VECTORIZATION_SIMD_RETURN_TYPE store(const mask_t& from, int_t* to)
    {
        _mm_store_si128(reinterpret_cast<simd_int_t*>(to), _mm256_cvtpd_epi32(from));
    }

    VECTORIZATION_SIMD_METHOD mask_t not_mask(const mask_t& x)
    {
        return _mm256_andnot_pd(x, _mm256_cmp_pd(x, x, _CMP_EQ_OQ));
    }

    VECTORIZATION_SIMD_METHOD mask_t and_mask(const mask_t& x, const mask_t& y)
    {
        return _mm256_and_pd(x, y);
    }

    VECTORIZATION_SIMD_METHOD mask_t or_mask(const mask_t& x, const mask_t& y)
    {
        return _mm256_or_pd(x, y);
    }

    VECTORIZATION_SIMD_METHOD mask_t xor_mask(const mask_t& x, const mask_t& y)
    {
        return _mm256_xor_pd(x, y);
    }

    //-----------------------------------------------------------------------------
    VECTORIZATION_SIMD_RETURN_TYPE broadcast(
        const double* from, simd_t& a0, simd_t& a1, simd_t& a2, simd_t& a3)
    {
        a0 = _mm256_broadcast_sd(from);
        a1 = _mm256_broadcast_sd(from + 1);
        a2 = _mm256_broadcast_sd(from + 2);
        a3 = _mm256_broadcast_sd(from + 3);
    }

    VECTORIZATION_SIMD_RETURN_TYPE copy(const value_t* from, value_t* to)
    {
        _mm256_store_pd(to, _mm256_loadu_pd(from));
    }

    template <int N>
    VECTORIZATION_SIMD_RETURN_TYPE ptranspose(simd_t x[N])  // NOLINT
    {
        auto T0 = _mm256_shuffle_pd(x[0], x[1], 15);
        auto T1 = _mm256_shuffle_pd(x[0], x[1], 0);
        auto T2 = _mm256_shuffle_pd(x[2], x[3], 15);
        auto T3 = _mm256_shuffle_pd(x[2], x[3], 0);

        x[1] = _mm256_permute2f128_pd(T0, T2, 32);
        x[3] = _mm256_permute2f128_pd(T0, T2, 49);
        x[0] = _mm256_permute2f128_pd(T1, T3, 32);
        x[2] = _mm256_permute2f128_pd(T1, T3, 49);
    }

    VECTORIZATION_SIMD_METHOD simd_t ploadquad(const double* from)
    {
        return _mm256_broadcast_sd(from);
    }
};
