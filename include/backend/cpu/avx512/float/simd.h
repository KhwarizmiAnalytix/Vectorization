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
struct simd<float>
{
    using simd_t      = __m512;
    using simd_half_t = __m256;
    using simd_int_t  = __m512i;
    using value_t     = float;
    using mask_t      = __mmask16;
    using int_t       = uint32_t;

    static constexpr int size      = 16;
    static constexpr int half_size = 8;

    inline static simd_t const     sign_mask = _mm512_set1_ps(-0.0F);
    inline static simd_int_t const stride_multiplier =
        _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

    VECTORIZATION_SIMD_RETURN_TYPE prefetch(const value_t* addr)
    {
        _mm_prefetch(PREFETCH_PTR_TYPE(addr), _MM_HINT_T0);
    }

    //======================================================================================
    // load, store, set functions
    //======================================================================================

    VECTORIZATION_SIMD_METHOD simd_t load(const value_t* addr) { return _mm512_load_ps(addr); }

    VECTORIZATION_SIMD_METHOD simd_t loadu(const value_t* addr) { return _mm512_loadu_ps(addr); }

    VECTORIZATION_SIMD_RETURN_TYPE store(simd_t from, value_t* to) { _mm512_store_ps(to, from); }

    VECTORIZATION_SIMD_RETURN_TYPE storeu(simd_t from, value_t* to) { _mm512_storeu_ps(to, from); }

    template <
        typename scalar_t,
        typename std::enable_if<std::is_fundamental<scalar_t>::value, bool>::type = true>
    VECTORIZATION_SIMD_METHOD simd_t set(scalar_t alpha)
    {
        return _mm512_set1_ps(static_cast<value_t>(alpha));
    }

    VECTORIZATION_SIMD_METHOD simd_t setzero() { return _mm512_setzero_ps(); }
    //======================================================================================
    // +, -, *, / functions
    //======================================================================================
    VECTORIZATION_SIMD_METHOD simd_t add(simd_t x, simd_t y) { return _mm512_add_ps(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t sub(simd_t x, simd_t y) { return _mm512_sub_ps(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t mul(simd_t x, simd_t y) { return _mm512_mul_ps(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t div(simd_t x, simd_t y) { return _mm512_div_ps(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t pow(simd_t x, simd_t y) { return _mm512_pow_ps(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t hypot(simd_t x, simd_t y) { return _mm512_hypot_ps(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t min(simd_t x, simd_t y) { return _mm512_min_ps(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t max(simd_t x, simd_t y) { return _mm512_max_ps(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t fma(simd_t x, simd_t y, simd_t z)
    {
#ifdef __FMA__
        return _mm512_fmadd_ps(x, y, z);
#else
        return _mm512_add_ps(_mm512_mul_ps(x, y), z);
#endif  // __FMA__
    }

    VECTORIZATION_SIMD_METHOD simd_t signcopy(simd_t x, simd_t sign)
    {
        const auto& sign_z = _mm512_and_ps(sign_mask, sign);
        const auto& fabs_x = _mm512_andnot_ps(sign_mask, x);
        return _mm512_xor_ps(sign_z, fabs_x);
    }

    //======================================================================================
    // one arg functions
    //======================================================================================
    VECTORIZATION_SIMD_METHOD simd_t sqrt(simd_t x) { return _mm512_sqrt_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t sqr(simd_t x) { return _mm512_mul_ps(x, x); }

    VECTORIZATION_SIMD_METHOD simd_t ceil(simd_t x) { return _mm512_ceil_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t floor(simd_t x) { return _mm512_floor_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t exp(simd_t x) { return _mm512_exp_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t expm1(simd_t x) { return _mm512_expm1_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t exp2(simd_t x) { return _mm512_exp2_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t exp10(simd_t x) { return _mm512_exp10_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t log(simd_t x) { return _mm512_log_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t log1p(simd_t x) { return _mm512_log1p_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t log2(simd_t x) { return _mm512_log2_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t log10(simd_t x) { return _mm512_log10_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t sin(simd_t x) { return _mm512_sin_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t cos(simd_t x) { return _mm512_cos_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t tan(simd_t x) { return _mm512_tan_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t asin(simd_t x) { return _mm512_asin_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t acos(simd_t x) { return _mm512_acos_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t atan(simd_t x) { return _mm512_atan_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t sinh(simd_t x) { return _mm512_sinh_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t cosh(simd_t x) { return _mm512_cosh_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t tanh(simd_t x) { return _mm512_tanh_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t asinh(simd_t x) { return _mm512_asinh_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t acosh(simd_t x) { return _mm512_acosh_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t atanh(simd_t x) { return _mm512_atanh_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t cbrt(simd_t x) { return _mm512_cbrt_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t cdf(simd_t x) { return _mm512_cdfnorm_ps(x); }
    VECTORIZATION_SIMD_METHOD simd_t inv_cdf(simd_t x) { return _mm512_cdfnorminv_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t trunc(simd_t x) { return _mm512_trunc_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t invsqrt(simd_t x)
    {
#if VECTORIZATION_HAS_SVML || VECTORIZATION_HAS_SLEEF
        return _mm512_invsqrt_ps(x);
#else
        // rsqrt14 gives ~14-bit accuracy; one Newton-Raphson step reaches ~28 bits (full float).
        // r_new = (r / 2) * (3 - x * r^2)
        __m512 r   = _mm512_rsqrt14_ps(x);
        __m512 xr2 = _mm512_mul_ps(x, _mm512_mul_ps(r, r));
        return _mm512_mul_ps(
            _mm512_mul_ps(r, _mm512_set1_ps(0.5f)), _mm512_sub_ps(_mm512_set1_ps(3.0f), xr2));
#endif
    }
    VECTORIZATION_SIMD_METHOD simd_t fabs(simd_t x) { return _mm512_abs_ps(x); }

    VECTORIZATION_SIMD_METHOD simd_t neg(simd_t x) { return _mm512_xor_ps(x, sign_mask); }

    //======================================================================================
    // horizantal functions
    //======================================================================================
    VECTORIZATION_FORCE_INLINE static double accumulate(simd_t x)
    {
        return static_cast<double>(_mm512_reduce_add_ps(x));
    }

    VECTORIZATION_FORCE_INLINE static value_t hmax(simd_t x) { return _mm512_reduce_max_ps(x); }

    VECTORIZATION_FORCE_INLINE static value_t hmin(simd_t x) { return _mm512_reduce_min_ps(x); }

    //======================================================================================
    // gather/scatter function
    //======================================================================================
    VECTORIZATION_SIMD_METHOD simd_t gather(const value_t* from, int stride)
    {
        __m512i index = _mm512_mullo_epi32(_mm512_set1_epi32(stride), stride_multiplier);
        return _mm512_i32gather_ps(index, from, sizeof(float));
    }

    VECTORIZATION_SIMD_RETURN_TYPE
    scatter(simd_t from, int stride, value_t* to)
    {
        // Create an index vector based on the stride
        __m512i index = _mm512_mullo_epi32(_mm512_set1_epi32(stride), stride_multiplier);

        // Use AVX-512 scatter instruction
        _mm512_i32scatter_ps(to, index, from, sizeof(float));
    }

    VECTORIZATION_SIMD_METHOD simd_t gather(const value_t* from, const int* strides)
    {
        __m512i index = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(strides));
        return _mm512_i32gather_ps(index, from, sizeof(float));
    }

    VECTORIZATION_SIMD_RETURN_TYPE
    scatter(simd_t from, const int* strides, value_t* to)
    {
        // Load strides into a SIMD register
        __m512i index = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(strides));

        // Use AVX-512 scatter instruction
        _mm512_i32scatter_ps(to, index, from, sizeof(float));
    }

    //======================================================================================
    // comparison function
    //======================================================================================
    VECTORIZATION_SIMD_METHOD simd_t if_else(const mask_t& x, simd_t y, simd_t z)
    {
        return _mm512_mask_mov_ps(z, x, y);
    }

    VECTORIZATION_SIMD_METHOD mask_t eq(simd_t x, simd_t y)
    {
        return _mm512_cmp_ps_mask(x, y, _CMP_EQ_OQ);
    }

    VECTORIZATION_SIMD_METHOD mask_t neq(simd_t x, simd_t y)
    {
        return _mm512_cmp_ps_mask(x, y, _CMP_NEQ_UQ);
    }

    VECTORIZATION_SIMD_METHOD mask_t gt(simd_t x, simd_t y)
    {
        return _mm512_cmp_ps_mask(x, y, _CMP_GT_OS);
    }

    VECTORIZATION_SIMD_METHOD mask_t lt(simd_t x, simd_t y)
    {
        return _mm512_cmp_ps_mask(x, y, _CMP_LT_OS);
    }

    VECTORIZATION_SIMD_METHOD mask_t ge(simd_t x, simd_t y)
    {
        return _mm512_cmp_ps_mask(x, y, _CMP_GE_OS);
    }

    VECTORIZATION_SIMD_METHOD mask_t le(simd_t x, simd_t y)
    {
        return _mm512_cmp_ps_mask(x, y, _CMP_LE_OS);
    }

    //======================================================================================
    // comparison function
    //======================================================================================
    VECTORIZATION_SIMD_METHOD mask_t loadu(int_t const* from)
    {
        __m512i v = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(from));
        return _mm512_cmpneq_epi32_mask(v, _mm512_setzero_si512());
    }

    VECTORIZATION_SIMD_METHOD mask_t load(int_t const* from)
    {
        __m512i v = _mm512_load_si512(reinterpret_cast<const __m512i*>(from));
        return _mm512_cmpneq_epi32_mask(v, _mm512_setzero_si512());
    }

    VECTORIZATION_SIMD_RETURN_TYPE storeu(const mask_t& from, int_t* to)
    {
        to[0]  = (from & 0x0001);
        to[1]  = (from & 0x0002);
        to[2]  = (from & 0x0004);
        to[3]  = (from & 0x0008);
        to[4]  = (from & 0x0010);
        to[5]  = (from & 0x0020);
        to[6]  = (from & 0x0040);
        to[7]  = (from & 0x0080);
        to[8]  = (from & 0x0100);
        to[9]  = (from & 0x0200);
        to[10] = (from & 0x0400);
        to[11] = (from & 0x0800);
        to[12] = (from & 0x1000);
        to[13] = (from & 0x2000);
        to[14] = (from & 0x4000);
        to[15] = (from & 0x8000);
    }

    VECTORIZATION_SIMD_RETURN_TYPE store(const mask_t& from, int_t* to)
    {
        to[0]  = (from & 0x0001);
        to[1]  = (from & 0x0002);
        to[2]  = (from & 0x0004);
        to[3]  = (from & 0x0008);
        to[4]  = (from & 0x0010);
        to[5]  = (from & 0x0020);
        to[6]  = (from & 0x0040);
        to[7]  = (from & 0x0080);
        to[8]  = (from & 0x0100);
        to[9]  = (from & 0x0200);
        to[10] = (from & 0x0400);
        to[11] = (from & 0x0800);
        to[12] = (from & 0x1000);
        to[13] = (from & 0x2000);
        to[14] = (from & 0x4000);
        to[15] = (from & 0x8000);
    }

    VECTORIZATION_SIMD_METHOD mask_t set(int_t const from)
    {
        return static_cast<mask_t>(from != 0 ? 0xFFFF : 0X0000);
    }

    VECTORIZATION_SIMD_METHOD mask_t not_mask(const mask_t& x) { return _mm512_knot(x); }

    VECTORIZATION_SIMD_METHOD mask_t and_mask(const mask_t& x, const mask_t& y)
    {
        return _mm512_kand(x, y);
    }

    VECTORIZATION_SIMD_METHOD mask_t or_mask(const mask_t& x, const mask_t& y)
    {
        return _mm512_kor(x, y);
    }

    VECTORIZATION_SIMD_METHOD mask_t xor_mask(const mask_t& x, const mask_t& y)
    {
        return _mm512_kxor(x, y);
    }

    //-----------------------------------------------------------------------------
    VECTORIZATION_FORCE_INLINE
    static void broadcast(const value_t* from, simd_t& a0, simd_t& a1, simd_t& a2, simd_t& a3)
    {
        a0 = _mm512_broadcastss_ps(_mm_load_ps1(from));
        a1 = _mm512_broadcastss_ps(_mm_load_ps1(from + 1));
        a2 = _mm512_broadcastss_ps(_mm_load_ps1(from + 2));
        a3 = _mm512_broadcastss_ps(_mm_load_ps1(from + 3));
    }

    VECTORIZATION_SIMD_RETURN_TYPE copy(const value_t* from, value_t* to)
    {
        _mm512_store_ps(to, _mm512_loadu_ps(from));
    }

#ifdef __AVX512DQ__
// AVX512F does not define _mm512_extractf32x8_ps to extract _m256 from _m512
#define VECTORIZATION_EXTRACT_8f_FROM_16f(INPUT, OUTPUT)  \
    __m256 OUTPUT##_0 = _mm512_extractf32x8_ps(INPUT, 0); \
    __m256 OUTPUT##_1 = _mm512_extractf32x8_ps(INPUT, 1);
#define VECTORIZATION_INSERT_8f_INTO_16f(OUTPUT, INPUTA, INPUTB) \
    OUTPUT = _mm512_insertf32x8(OUTPUT, INPUTA, 0);              \
    OUTPUT = _mm512_insertf32x8(OUTPUT, INPUTB, 1);

#else
#define VECTORIZATION_EXTRACT_8f_FROM_16f(INPUT, OUTPUT)          \
    __m256 OUTPUT##_0 = _mm256_insertf128_ps(                     \
        _mm256_castps128_ps256(_mm512_extractf32x4_ps(INPUT, 0)), \
        _mm512_extractf32x4_ps(INPUT, 1),                         \
        1);                                                       \
    __m256 OUTPUT##_1 = _mm256_insertf128_ps(                     \
        _mm256_castps128_ps256(_mm512_extractf32x4_ps(INPUT, 2)), \
        _mm512_extractf32x4_ps(INPUT, 3),                         \
        1);

#define VECTORIZATION_INSERT_8f_INTO_16f(OUTPUT, INPUTA, INPUTB)              \
    OUTPUT = _mm512_insertf32x4(OUTPUT, _mm256_extractf128_ps(INPUTA, 0), 0); \
    OUTPUT = _mm512_insertf32x4(OUTPUT, _mm256_extractf128_ps(INPUTA, 1), 1); \
    OUTPUT = _mm512_insertf32x4(OUTPUT, _mm256_extractf128_ps(INPUTB, 0), 2); \
    OUTPUT = _mm512_insertf32x4(OUTPUT, _mm256_extractf128_ps(INPUTB, 1), 3);
#endif

#define PACK_OUTPUT(OUTPUT, INPUT, INDEX, STRIDE) \
    VECTORIZATION_INSERT_8f_INTO_16f(OUTPUT[INDEX], INPUT[INDEX], INPUT[INDEX + STRIDE]);

#define PACK_OUTPUT_2(OUTPUT, INPUT, INDEX, STRIDE) \
    VECTORIZATION_INSERT_8f_INTO_16f(OUTPUT[INDEX], INPUT[2 * INDEX], INPUT[2 * INDEX + STRIDE]);

    template <int N>
    VECTORIZATION_SIMD_RETURN_TYPE ptranspose(simd_t x[N])  // NOLINT
    {
        if constexpr (N == 16)
        {
            auto T0  = _mm512_unpacklo_ps(x[0], x[1]);
            auto T1  = _mm512_unpackhi_ps(x[0], x[1]);
            auto T2  = _mm512_unpacklo_ps(x[2], x[3]);
            auto T3  = _mm512_unpackhi_ps(x[2], x[3]);
            auto T4  = _mm512_unpacklo_ps(x[4], x[5]);
            auto T5  = _mm512_unpackhi_ps(x[4], x[5]);
            auto T6  = _mm512_unpacklo_ps(x[6], x[7]);
            auto T7  = _mm512_unpackhi_ps(x[6], x[7]);
            auto T8  = _mm512_unpacklo_ps(x[8], x[9]);
            auto T9  = _mm512_unpackhi_ps(x[8], x[9]);
            auto T10 = _mm512_unpacklo_ps(x[10], x[11]);
            auto T11 = _mm512_unpackhi_ps(x[10], x[11]);
            auto T12 = _mm512_unpacklo_ps(x[12], x[13]);
            auto T13 = _mm512_unpackhi_ps(x[12], x[13]);
            auto T14 = _mm512_unpacklo_ps(x[14], x[15]);
            auto T15 = _mm512_unpackhi_ps(x[14], x[15]);
            auto S0  = _mm512_shuffle_ps(T0, T2, _MM_SHUFFLE(1, 0, 1, 0));
            auto S1  = _mm512_shuffle_ps(T0, T2, _MM_SHUFFLE(3, 2, 3, 2));
            auto S2  = _mm512_shuffle_ps(T1, T3, _MM_SHUFFLE(1, 0, 1, 0));
            auto S3  = _mm512_shuffle_ps(T1, T3, _MM_SHUFFLE(3, 2, 3, 2));
            auto S4  = _mm512_shuffle_ps(T4, T6, _MM_SHUFFLE(1, 0, 1, 0));
            auto S5  = _mm512_shuffle_ps(T4, T6, _MM_SHUFFLE(3, 2, 3, 2));
            auto S6  = _mm512_shuffle_ps(T5, T7, _MM_SHUFFLE(1, 0, 1, 0));
            auto S7  = _mm512_shuffle_ps(T5, T7, _MM_SHUFFLE(3, 2, 3, 2));
            auto S8  = _mm512_shuffle_ps(T8, T10, _MM_SHUFFLE(1, 0, 1, 0));
            auto S9  = _mm512_shuffle_ps(T8, T10, _MM_SHUFFLE(3, 2, 3, 2));
            auto S10 = _mm512_shuffle_ps(T9, T11, _MM_SHUFFLE(1, 0, 1, 0));
            auto S11 = _mm512_shuffle_ps(T9, T11, _MM_SHUFFLE(3, 2, 3, 2));
            auto S12 = _mm512_shuffle_ps(T12, T14, _MM_SHUFFLE(1, 0, 1, 0));
            auto S13 = _mm512_shuffle_ps(T12, T14, _MM_SHUFFLE(3, 2, 3, 2));
            auto S14 = _mm512_shuffle_ps(T13, T15, _MM_SHUFFLE(1, 0, 1, 0));
            auto S15 = _mm512_shuffle_ps(T13, T15, _MM_SHUFFLE(3, 2, 3, 2));

            VECTORIZATION_EXTRACT_8f_FROM_16f(S0, S0);
            VECTORIZATION_EXTRACT_8f_FROM_16f(S1, S1);
            VECTORIZATION_EXTRACT_8f_FROM_16f(S2, S2);
            VECTORIZATION_EXTRACT_8f_FROM_16f(S3, S3);
            VECTORIZATION_EXTRACT_8f_FROM_16f(S4, S4);
            VECTORIZATION_EXTRACT_8f_FROM_16f(S5, S5);
            VECTORIZATION_EXTRACT_8f_FROM_16f(S6, S6);
            VECTORIZATION_EXTRACT_8f_FROM_16f(S7, S7);
            VECTORIZATION_EXTRACT_8f_FROM_16f(S8, S8);
            VECTORIZATION_EXTRACT_8f_FROM_16f(S9, S9);
            VECTORIZATION_EXTRACT_8f_FROM_16f(S10, S10);
            VECTORIZATION_EXTRACT_8f_FROM_16f(S11, S11);
            VECTORIZATION_EXTRACT_8f_FROM_16f(S12, S12);
            VECTORIZATION_EXTRACT_8f_FROM_16f(S13, S13);
            VECTORIZATION_EXTRACT_8f_FROM_16f(S14, S14);
            VECTORIZATION_EXTRACT_8f_FROM_16f(S15, S15);

            __m256 tmp[32];  // NOLINT
            tmp[0] = _mm256_permute2f128_ps(S0_0, S4_0, 0x20);
            tmp[1] = _mm256_permute2f128_ps(S1_0, S5_0, 0x20);
            tmp[2] = _mm256_permute2f128_ps(S2_0, S6_0, 0x20);
            tmp[3] = _mm256_permute2f128_ps(S3_0, S7_0, 0x20);
            tmp[4] = _mm256_permute2f128_ps(S0_0, S4_0, 0x31);
            tmp[5] = _mm256_permute2f128_ps(S1_0, S5_0, 0x31);
            tmp[6] = _mm256_permute2f128_ps(S2_0, S6_0, 0x31);
            tmp[7] = _mm256_permute2f128_ps(S3_0, S7_0, 0x31);

            tmp[8]  = _mm256_permute2f128_ps(S0_1, S4_1, 0x20);
            tmp[9]  = _mm256_permute2f128_ps(S1_1, S5_1, 0x20);
            tmp[10] = _mm256_permute2f128_ps(S2_1, S6_1, 0x20);
            tmp[11] = _mm256_permute2f128_ps(S3_1, S7_1, 0x20);
            tmp[12] = _mm256_permute2f128_ps(S0_1, S4_1, 0x31);
            tmp[13] = _mm256_permute2f128_ps(S1_1, S5_1, 0x31);
            tmp[14] = _mm256_permute2f128_ps(S2_1, S6_1, 0x31);
            tmp[15] = _mm256_permute2f128_ps(S3_1, S7_1, 0x31);

            // Second set of _m256 outputs
            tmp[16] = _mm256_permute2f128_ps(S8_0, S12_0, 0x20);
            tmp[17] = _mm256_permute2f128_ps(S9_0, S13_0, 0x20);
            tmp[18] = _mm256_permute2f128_ps(S10_0, S14_0, 0x20);
            tmp[19] = _mm256_permute2f128_ps(S11_0, S15_0, 0x20);
            tmp[20] = _mm256_permute2f128_ps(S8_0, S12_0, 0x31);
            tmp[21] = _mm256_permute2f128_ps(S9_0, S13_0, 0x31);
            tmp[22] = _mm256_permute2f128_ps(S10_0, S14_0, 0x31);
            tmp[23] = _mm256_permute2f128_ps(S11_0, S15_0, 0x31);

            tmp[24] = _mm256_permute2f128_ps(S8_1, S12_1, 0x20);
            tmp[25] = _mm256_permute2f128_ps(S9_1, S13_1, 0x20);
            tmp[26] = _mm256_permute2f128_ps(S10_1, S14_1, 0x20);
            tmp[27] = _mm256_permute2f128_ps(S11_1, S15_1, 0x20);
            tmp[28] = _mm256_permute2f128_ps(S8_1, S12_1, 0x31);
            tmp[29] = _mm256_permute2f128_ps(S9_1, S13_1, 0x31);
            tmp[30] = _mm256_permute2f128_ps(S10_1, S14_1, 0x31);
            tmp[31] = _mm256_permute2f128_ps(S11_1, S15_1, 0x31);

            // Pack them into the output
            PACK_OUTPUT(x, tmp, 0, 16);
            PACK_OUTPUT(x, tmp, 1, 16);
            PACK_OUTPUT(x, tmp, 2, 16);
            PACK_OUTPUT(x, tmp, 3, 16);

            PACK_OUTPUT(x, tmp, 4, 16);
            PACK_OUTPUT(x, tmp, 5, 16);
            PACK_OUTPUT(x, tmp, 6, 16);
            PACK_OUTPUT(x, tmp, 7, 16);

            PACK_OUTPUT(x, tmp, 8, 16);
            PACK_OUTPUT(x, tmp, 9, 16);
            PACK_OUTPUT(x, tmp, 10, 16);
            PACK_OUTPUT(x, tmp, 11, 16);

            PACK_OUTPUT(x, tmp, 12, 16);
            PACK_OUTPUT(x, tmp, 13, 16);
            PACK_OUTPUT(x, tmp, 14, 16);
            PACK_OUTPUT(x, tmp, 15, 16);
        }
        else
        {
            auto T0 = _mm512_unpacklo_ps(x[0], x[1]);
            auto T1 = _mm512_unpackhi_ps(x[0], x[1]);
            auto T2 = _mm512_unpacklo_ps(x[2], x[3]);
            auto T3 = _mm512_unpackhi_ps(x[2], x[3]);

            auto S0 = _mm512_shuffle_ps(T0, T2, _MM_SHUFFLE(1, 0, 1, 0));
            auto S1 = _mm512_shuffle_ps(T0, T2, _MM_SHUFFLE(3, 2, 3, 2));
            auto S2 = _mm512_shuffle_ps(T1, T3, _MM_SHUFFLE(1, 0, 1, 0));
            auto S3 = _mm512_shuffle_ps(T1, T3, _MM_SHUFFLE(3, 2, 3, 2));

            VECTORIZATION_EXTRACT_8f_FROM_16f(S0, S0);
            VECTORIZATION_EXTRACT_8f_FROM_16f(S1, S1);
            VECTORIZATION_EXTRACT_8f_FROM_16f(S2, S2);
            VECTORIZATION_EXTRACT_8f_FROM_16f(S3, S3);

            __m256 tmp[8];  // NOLINT

            tmp[0] = _mm256_permute2f128_ps(S0_0, S1_0, 0x20);
            tmp[1] = _mm256_permute2f128_ps(S2_0, S3_0, 0x20);
            tmp[2] = _mm256_permute2f128_ps(S0_0, S1_0, 0x31);
            tmp[3] = _mm256_permute2f128_ps(S2_0, S3_0, 0x31);

            tmp[4] = _mm256_permute2f128_ps(S0_1, S1_1, 0x20);
            tmp[5] = _mm256_permute2f128_ps(S2_1, S3_1, 0x20);
            tmp[6] = _mm256_permute2f128_ps(S0_1, S1_1, 0x31);
            tmp[7] = _mm256_permute2f128_ps(S2_1, S3_1, 0x31);

            PACK_OUTPUT_2(x, tmp, 0, 1);
            PACK_OUTPUT_2(x, tmp, 1, 1);
            PACK_OUTPUT_2(x, tmp, 2, 1);
            PACK_OUTPUT_2(x, tmp, 3, 1);
        }
    }
};
