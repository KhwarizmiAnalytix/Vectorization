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

#include "backend/cpu/avx512/svml.h"

#ifdef __clang__
#pragma clang diagnostic pop
#endif
#endif

#include "common/vectorization_macros.h"

template <>
struct simd<double>
{
    using simd_t      = __m512d;
    using simd_half_t = __m256d;
    using simd_int_t  = __m256i;
    using value_t     = double;
    using mask_t      = __mmask8;
    using int_t       = uint32_t;

    static constexpr int size      = 8;
    static constexpr int half_size = 4;

    inline static simd_t const     sign_mask         = _mm512_set1_pd(-0.0);
    inline static simd_int_t const stride_multiplier = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);

    VECTORIZATION_SIMD_RETURN_TYPE prefetch(const value_t* addr)
    {
        _mm_prefetch(PREFETCH_PTR_TYPE(addr), _MM_HINT_T0);
    }

    //======================================================================================
    // load, store, set functions
    //======================================================================================
    VECTORIZATION_SIMD_METHOD simd_t load(const value_t* addr) { return _mm512_load_pd(addr); }

    VECTORIZATION_SIMD_METHOD simd_t loadu(const value_t* addr) { return _mm512_loadu_pd(addr); }

    VECTORIZATION_SIMD_RETURN_TYPE store(simd_t from, value_t* to) { _mm512_store_pd(to, from); }

    VECTORIZATION_SIMD_RETURN_TYPE storeu(simd_t from, value_t* to) { _mm512_storeu_pd(to, from); }

    template <
        typename scalar_t,
        typename std::enable_if<std::is_fundamental<scalar_t>::value, bool>::type = true>
    VECTORIZATION_SIMD_METHOD simd_t set(scalar_t alpha)
    {
        return _mm512_set1_pd(static_cast<value_t>(alpha));
    }

    VECTORIZATION_SIMD_METHOD simd_t setzero() { return _mm512_setzero_pd(); }
    //======================================================================================
    // +, -, *, /, min, max, hypo functions
    //======================================================================================
    VECTORIZATION_SIMD_METHOD simd_t add(simd_t x, simd_t y) { return _mm512_add_pd(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t sub(simd_t x, simd_t y) { return _mm512_sub_pd(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t mul(simd_t x, simd_t y) { return _mm512_mul_pd(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t div(simd_t x, simd_t y) { return _mm512_div_pd(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t fma(simd_t x, simd_t y, simd_t z)
    {
#ifdef __FMA__
        return _mm512_fmadd_pd(x, y, z);
#else
        return _mm512_add_pd(_mm512_mul_pd(x, y), z);
#endif  // __FMA__
    }

    VECTORIZATION_SIMD_METHOD simd_t pow(simd_t x, simd_t y) { return _mm512_pow_pd(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t hypot(simd_t x, simd_t y) { return _mm512_hypot_pd(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t min(simd_t x, simd_t y) { return _mm512_min_pd(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t max(simd_t x, simd_t y) { return _mm512_max_pd(x, y); }

    //======================================================================================
    // one arg function
    //======================================================================================
    VECTORIZATION_SIMD_METHOD simd_t sqrt(simd_t x) { return _mm512_sqrt_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t sqr(simd_t x) { return _mm512_mul_pd(x, x); }
    VECTORIZATION_SIMD_METHOD simd_t ceil(simd_t x) { return _mm512_ceil_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t floor(simd_t x) { return _mm512_floor_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t exp(simd_t x) { return _mm512_exp_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t expm1(simd_t x) { return _mm512_expm1_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t exp2(simd_t x) { return _mm512_exp2_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t exp10(simd_t x) { return _mm512_exp10_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t log(simd_t x) { return _mm512_log_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t log1p(simd_t x) { return _mm512_log1p_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t log2(simd_t x) { return _mm512_log2_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t log10(simd_t x) { return _mm512_log10_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t sin(simd_t x) { return _mm512_sin_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t cos(simd_t x) { return _mm512_cos_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t tan(simd_t x) { return _mm512_tan_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t asin(simd_t x) { return _mm512_asin_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t acos(simd_t x) { return _mm512_acos_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t atan(simd_t x) { return _mm512_atan_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t sinh(simd_t x) { return _mm512_sinh_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t cosh(simd_t x) { return _mm512_cosh_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t tanh(simd_t x) { return _mm512_tanh_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t asinh(simd_t x) { return _mm512_asinh_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t acosh(simd_t x) { return _mm512_acosh_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t atanh(simd_t x) { return _mm512_atanh_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t cbrt(simd_t x) { return _mm512_cbrt_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t cdf(simd_t x) { return _mm512_cdfnorm_pd(x); }
    VECTORIZATION_SIMD_METHOD simd_t inv_cdf(simd_t x) { return _mm512_cdfnorminv_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t trunc(simd_t x) { return _mm512_trunc_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t invsqrt(simd_t x) { return _mm512_invsqrt_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t fabs(simd_t x) { return _mm512_abs_pd(x); }

    VECTORIZATION_SIMD_METHOD simd_t neg(simd_t x) { return _mm512_xor_pd(x, sign_mask); }

    VECTORIZATION_SIMD_METHOD simd_t signcopy(simd_t x, simd_t sign)
    {
        const auto& sign_z = _mm512_and_pd(sign_mask, sign);
        const auto& fabs_x = _mm512_andnot_pd(sign_mask, x);
        return _mm512_xor_pd(sign_z, fabs_x);
    }

    //======================================================================================
    // horizantal functions
    //======================================================================================
    VECTORIZATION_FORCE_INLINE static double accumulate(simd_t x)
    {
        return _mm512_reduce_add_pd(x);
    }

    VECTORIZATION_FORCE_INLINE static value_t hmax(simd_t x) { return _mm512_reduce_max_pd(x); }

    VECTORIZATION_FORCE_INLINE static value_t hmin(simd_t x) { return _mm512_reduce_min_pd(x); }

    //======================================================================================
    // gather/scatter function
    //======================================================================================
    VECTORIZATION_SIMD_METHOD simd_t gather(value_t const* from, int stride)
    {
        const auto stride_vector = _mm256_set1_epi32(stride);
        const auto indices       = _mm256_mullo_epi32(stride_vector, stride_multiplier);
        return _mm512_i32gather_pd(indices, from, 8);
    }

    VECTORIZATION_SIMD_RETURN_TYPE
    scatter(simd_t from, int stride, value_t* to)
    {
        const auto stride_vector = _mm256_set1_epi32(stride);
        const auto indices       = _mm256_mullo_epi32(stride_vector, stride_multiplier);
        _mm512_i32scatter_pd(to, indices, from, 8);
    }

    VECTORIZATION_SIMD_METHOD simd_t gather(value_t const* from, const int* strides)
    {
        auto indices = _mm256_loadu_si256(reinterpret_cast<const simd_int_t*>(strides));
        return _mm512_i32gather_pd(indices, from, 8);
    }

    VECTORIZATION_SIMD_RETURN_TYPE
    scatter(simd_t from, const int* strides, value_t* to)
    {
        auto indices = _mm256_loadu_si256(reinterpret_cast<const simd_int_t*>(strides));
        _mm512_i32scatter_pd(to, indices, from, 8);
    }

    //======================================================================================
    // comparison function
    //======================================================================================
    VECTORIZATION_SIMD_METHOD simd_t if_else(const mask_t& x, simd_t y, simd_t z)
    {
        return _mm512_mask_mov_pd(z, x, y);
    }

    VECTORIZATION_SIMD_METHOD mask_t eq(simd_t x, simd_t y)
    {
        return _mm512_cmp_pd_mask(x, y, _CMP_EQ_OQ);
    }

    VECTORIZATION_SIMD_METHOD mask_t neq(simd_t x, simd_t y)
    {
        return _mm512_cmp_pd_mask(x, y, _CMP_NEQ_UQ);
    }

    VECTORIZATION_SIMD_METHOD mask_t gt(simd_t x, simd_t y)
    {
        return _mm512_cmp_pd_mask(x, y, _CMP_GT_OS);
    }

    VECTORIZATION_SIMD_METHOD mask_t lt(simd_t x, simd_t y)
    {
        return _mm512_cmp_pd_mask(x, y, _CMP_LT_OS);
    }

    VECTORIZATION_SIMD_METHOD mask_t ge(simd_t x, simd_t y)
    {
        return _mm512_cmp_pd_mask(x, y, _CMP_GE_OS);
    }

    VECTORIZATION_SIMD_METHOD mask_t le(simd_t x, simd_t y)
    {
        return _mm512_cmp_pd_mask(x, y, _CMP_LE_OS);
    }

    //======================================================================================
    // comparison function
    //======================================================================================
    VECTORIZATION_SIMD_METHOD mask_t loadu(int_t const* from)
    {
        mask_t ret = 0x00;
        if (from[0] != 0)
            ret |= 0x01;
        if (from[1] != 0)
            ret |= 0x02;
        if (from[2] != 0)
            ret |= 0x04;
        if (from[3] != 0)
            ret |= 0x08;
        if (from[4] != 0)
            ret |= 0x10;
        if (from[5] != 0)
            ret |= 0x20;
        if (from[6] != 0)
            ret |= 0x40;
        if (from[7] != 0)
            ret |= 0x80;
        return ret;
    }

    VECTORIZATION_SIMD_METHOD mask_t load(int_t const* from)
    {
        mask_t ret = 0x00;
        if (from[0] != 0)
            ret |= 0x01;
        if (from[1] != 0)
            ret |= 0x02;
        if (from[2] != 0)
            ret |= 0x04;
        if (from[3] != 0)
            ret |= 0x08;
        if (from[4] != 0)
            ret |= 0x10;
        if (from[5] != 0)
            ret |= 0x20;
        if (from[6] != 0)
            ret |= 0x40;
        if (from[7] != 0)
            ret |= 0x80;
        return ret;
    }

    VECTORIZATION_SIMD_RETURN_TYPE storeu(const mask_t& from, int_t* to)
    {
        to[0] = (from & 0x01);
        to[1] = (from & 0x02);
        to[2] = (from & 0x04);
        to[3] = (from & 0x08);
        to[4] = (from & 0x10);
        to[5] = (from & 0x20);
        to[6] = (from & 0x40);
        to[7] = (from & 0x80);
    }

    VECTORIZATION_SIMD_RETURN_TYPE store(const mask_t& from, int_t* to)
    {
        to[0] = (from & 0x01);
        to[1] = (from & 0x02);
        to[2] = (from & 0x04);
        to[3] = (from & 0x08);
        to[4] = (from & 0x10);
        to[5] = (from & 0x20);
        to[6] = (from & 0x40);
        to[7] = (from & 0x80);
    }

    VECTORIZATION_SIMD_METHOD mask_t set(int_t const from)
    {
        return static_cast<mask_t>(from != 0 ? 0xFF : 0X00);
    }

    VECTORIZATION_SIMD_METHOD mask_t not_mask(const mask_t& x) { return _knot_mask8(x); }

    VECTORIZATION_SIMD_METHOD mask_t and_mask(const mask_t& x, const mask_t& y)
    {
        return _kand_mask8(x, y);
    }

    VECTORIZATION_SIMD_METHOD mask_t or_mask(const mask_t& x, const mask_t& y)
    {
        return _kor_mask8(x, y);
    }

    VECTORIZATION_SIMD_METHOD mask_t xor_mask(const mask_t& x, const mask_t& y)
    {
        return _kxor_mask8(x, y);
    }

    //-----------------------------------------------------------------------------
    VECTORIZATION_SIMD_RETURN_TYPE
    broadcast(const value_t* from, simd_t& a0, simd_t& a1, simd_t& a2, simd_t& a3)
    {
        a0 = _mm512_set1_pd(from[0]);
        a1 = _mm512_set1_pd(from[1]);
        a2 = _mm512_set1_pd(from[2]);
        a3 = _mm512_set1_pd(from[3]);
    }

    VECTORIZATION_SIMD_RETURN_TYPE copy(const value_t* from, value_t* to)
    {
        _mm512_store_pd(to, _mm512_loadu_pd(from));
    }

    VECTORIZATION_FORCE_INLINE static simd_half_t predux_downto4(simd_t a)
    {
        auto lane0 = _mm512_extractf64x4_pd(a, 0);
        auto lane1 = _mm512_extractf64x4_pd(a, 1);
        return _mm256_add_pd(lane0, lane1);
    }

    // Loads 2 doubles from memory a returns the simd
    // {a0, a0  a0, a0, a1, a1, a1, a1}
    VECTORIZATION_FORCE_INLINE static simd_t ploadquad(const double* from)
    {
        return _mm512_set_pd(
            from[1], from[1], from[1], from[1], from[0], from[0], from[0], from[0]);
    }

    VECTORIZATION_FORCE_INLINE static simd_half_t loadu_half(const value_t* from)
    {
        return _mm256_loadu_pd(from);
    }

    VECTORIZATION_FORCE_INLINE static simd_half_t set(const value_t* from)
    {
        return _mm256_set1_pd(*from);
    }

    VECTORIZATION_SIMD_METHOD simd_half_t add(const simd_half_t& x, const simd_half_t& y)
    {
        return _mm256_add_pd(x, y);
    }

    VECTORIZATION_SIMD_METHOD simd_half_t
    fma(const simd_half_t& x, const simd_half_t& y, const simd_half_t& z)
    {
#ifdef __FMA__
        return _mm256_fmadd_pd(x, y, z);
#else
        return _mm256_add_pd(_mm256_mul_pd(x, y), z);
#endif
    }

    VECTORIZATION_SIMD_METHOD simd_half_t gather_half(value_t const* from, int stride)
    {
        return _mm256_set_pd(from[3 * stride], from[2 * stride], from[stride], from[0]);
    }

    VECTORIZATION_SIMD_RETURN_TYPE scatter(const simd_half_t& from, int stride, value_t* to)
    {
        auto low   = _mm256_extractf128_pd(from, 0);
        to[0]      = _mm_cvtsd_f64(low);
        to[stride] = _mm_cvtsd_f64(_mm_shuffle_pd(low, low, 1));

        auto high      = _mm256_extractf128_pd(from, 1);
        to[stride * 2] = _mm_cvtsd_f64(high);
        to[stride * 3] = _mm_cvtsd_f64(_mm_shuffle_pd(high, high, 1));
    }

#define PACK_OUTPUT_SQ_D(OUTPUT, INPUT, INDEX, STRIDE)                  \
    OUTPUT[INDEX] = _mm512_insertf64x4(OUTPUT[INDEX], INPUT[INDEX], 0); \
    OUTPUT[INDEX] = _mm512_insertf64x4(OUTPUT[INDEX], INPUT[INDEX + STRIDE], 1);

#define PACK_OUTPUT_D(OUTPUT, INPUT, INDEX, STRIDE)                           \
    OUTPUT[INDEX] = _mm512_insertf64x4(OUTPUT[INDEX], INPUT[(2 * INDEX)], 0); \
    OUTPUT[INDEX] = _mm512_insertf64x4(OUTPUT[INDEX], INPUT[(2 * INDEX) + STRIDE], 1);

    template <int N>
    VECTORIZATION_SIMD_RETURN_TYPE ptranspose(simd_t x[N])
    {
        if constexpr (N == 8)
        {
            auto T0 = _mm512_unpacklo_pd(x[0], x[1]);
            auto T1 = _mm512_unpackhi_pd(x[0], x[1]);
            auto T2 = _mm512_unpacklo_pd(x[2], x[3]);
            auto T3 = _mm512_unpackhi_pd(x[2], x[3]);
            auto T4 = _mm512_unpacklo_pd(x[4], x[5]);
            auto T5 = _mm512_unpackhi_pd(x[4], x[5]);
            auto T6 = _mm512_unpacklo_pd(x[6], x[7]);
            auto T7 = _mm512_unpackhi_pd(x[6], x[7]);

            __m256d tmp[16];  // NOLINT

            tmp[0] = _mm256_permute2f128_pd(
                _mm512_extractf64x4_pd(T0, 0), _mm512_extractf64x4_pd(T2, 0), 0x20);
            tmp[1] = _mm256_permute2f128_pd(
                _mm512_extractf64x4_pd(T1, 0), _mm512_extractf64x4_pd(T3, 0), 0x20);
            tmp[2] = _mm256_permute2f128_pd(
                _mm512_extractf64x4_pd(T0, 0), _mm512_extractf64x4_pd(T2, 0), 0x31);
            tmp[3] = _mm256_permute2f128_pd(
                _mm512_extractf64x4_pd(T1, 0), _mm512_extractf64x4_pd(T3, 0), 0x31);

            tmp[4] = _mm256_permute2f128_pd(
                _mm512_extractf64x4_pd(T0, 1), _mm512_extractf64x4_pd(T2, 1), 0x20);
            tmp[5] = _mm256_permute2f128_pd(
                _mm512_extractf64x4_pd(T1, 1), _mm512_extractf64x4_pd(T3, 1), 0x20);
            tmp[6] = _mm256_permute2f128_pd(
                _mm512_extractf64x4_pd(T0, 1), _mm512_extractf64x4_pd(T2, 1), 0x31);
            tmp[7] = _mm256_permute2f128_pd(
                _mm512_extractf64x4_pd(T1, 1), _mm512_extractf64x4_pd(T3, 1), 0x31);

            tmp[8] = _mm256_permute2f128_pd(
                _mm512_extractf64x4_pd(T4, 0), _mm512_extractf64x4_pd(T6, 0), 0x20);
            tmp[9] = _mm256_permute2f128_pd(
                _mm512_extractf64x4_pd(T5, 0), _mm512_extractf64x4_pd(T7, 0), 0x20);
            tmp[10] = _mm256_permute2f128_pd(
                _mm512_extractf64x4_pd(T4, 0), _mm512_extractf64x4_pd(T6, 0), 0x31);
            tmp[11] = _mm256_permute2f128_pd(
                _mm512_extractf64x4_pd(T5, 0), _mm512_extractf64x4_pd(T7, 0), 0x31);

            tmp[12] = _mm256_permute2f128_pd(
                _mm512_extractf64x4_pd(T4, 1), _mm512_extractf64x4_pd(T6, 1), 0x20);
            tmp[13] = _mm256_permute2f128_pd(
                _mm512_extractf64x4_pd(T5, 1), _mm512_extractf64x4_pd(T7, 1), 0x20);
            tmp[14] = _mm256_permute2f128_pd(
                _mm512_extractf64x4_pd(T4, 1), _mm512_extractf64x4_pd(T6, 1), 0x31);
            tmp[15] = _mm256_permute2f128_pd(
                _mm512_extractf64x4_pd(T5, 1), _mm512_extractf64x4_pd(T7, 1), 0x31);

            PACK_OUTPUT_SQ_D(x, tmp, 0, 8);
            PACK_OUTPUT_SQ_D(x, tmp, 1, 8);
            PACK_OUTPUT_SQ_D(x, tmp, 2, 8);
            PACK_OUTPUT_SQ_D(x, tmp, 3, 8);

            PACK_OUTPUT_SQ_D(x, tmp, 4, 8);
            PACK_OUTPUT_SQ_D(x, tmp, 5, 8);
            PACK_OUTPUT_SQ_D(x, tmp, 6, 8);
            PACK_OUTPUT_SQ_D(x, tmp, 7, 8);
        }
        else
        {
            auto T0 = _mm512_shuffle_pd(x[0], x[1], 0);
            auto T1 = _mm512_shuffle_pd(x[0], x[1], 0xff);
            auto T2 = _mm512_shuffle_pd(x[2], x[3], 0);
            auto T3 = _mm512_shuffle_pd(x[2], x[3], 0xff);

            __m256d tmp[8];  // NOLINT

            tmp[0] = _mm256_permute2f128_pd(
                _mm512_extractf64x4_pd(T0, 0), _mm512_extractf64x4_pd(T2, 0), 0x20);
            tmp[1] = _mm256_permute2f128_pd(
                _mm512_extractf64x4_pd(T1, 0), _mm512_extractf64x4_pd(T3, 0), 0x20);
            tmp[2] = _mm256_permute2f128_pd(
                _mm512_extractf64x4_pd(T0, 0), _mm512_extractf64x4_pd(T2, 0), 0x31);
            tmp[3] = _mm256_permute2f128_pd(
                _mm512_extractf64x4_pd(T1, 0), _mm512_extractf64x4_pd(T3, 0), 0x31);

            tmp[4] = _mm256_permute2f128_pd(
                _mm512_extractf64x4_pd(T0, 1), _mm512_extractf64x4_pd(T2, 1), 0x20);
            tmp[5] = _mm256_permute2f128_pd(
                _mm512_extractf64x4_pd(T1, 1), _mm512_extractf64x4_pd(T3, 1), 0x20);
            tmp[6] = _mm256_permute2f128_pd(
                _mm512_extractf64x4_pd(T0, 1), _mm512_extractf64x4_pd(T2, 1), 0x31);
            tmp[7] = _mm256_permute2f128_pd(
                _mm512_extractf64x4_pd(T1, 1), _mm512_extractf64x4_pd(T3, 1), 0x31);

            PACK_OUTPUT_D(x, tmp, 0, 1);
            PACK_OUTPUT_D(x, tmp, 1, 1);
            PACK_OUTPUT_D(x, tmp, 2, 1);
            PACK_OUTPUT_D(x, tmp, 3, 1);
        }
    }
};
