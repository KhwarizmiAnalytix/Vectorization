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

#include "common/vectorization_macros.h"

#define _mm_mask_and_ps(a, b) _mm_and_ps((a), (b))
#define _mm_mask_and_pd(a, b) _mm_and_pd((a), (b))
#define _mm_mask_or_ps(a, b) _mm_or_ps((a), (b))
#define _mm_mask_or_pd(a, b) _mm_or_pd((a), (b))
#define _mm_mask_xor_ps(a, b) _mm_xor_ps((a), (b))
#define _mm_mask_xor_pd(a, b) _mm_xor_pd((a), (b))

VECTORIZATION_FORCE_INLINE __m128 VECTORIZATION_VECTORCALL _mm_fabs_ps(__m128 x) noexcept
{
    const auto mask =
        _mm_castsi128_ps(_mm_setr_epi32(0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF));
    return _mm_and_ps(x, mask);
}

VECTORIZATION_FORCE_INLINE __m128d VECTORIZATION_VECTORCALL _mm_fabs_pd(__m128d x) noexcept
{
    const auto mask =
        _mm_castsi128_pd(_mm_setr_epi32(0xFFFFFFFF, 0x7FFFFFFF, 0xFFFFFFFF, 0x7FFFFFFF));  // NOLINT
    return _mm_and_pd(x, mask);
}

VECTORIZATION_FORCE_INLINE __m128 VECTORIZATION_VECTORCALL _mm_neg_ps(__m128 x) noexcept
{
    const auto zero = _mm_set1_ps(-0.F);
    return _mm_xor_ps(x, zero);
}

VECTORIZATION_FORCE_INLINE __m128d VECTORIZATION_VECTORCALL _mm_neg_pd(__m128d x) noexcept
{
    const auto zero = _mm_set1_pd(-0.);
    return _mm_xor_pd(x, zero);
}

VECTORIZATION_FORCE_INLINE __m128 VECTORIZATION_VECTORCALL _mm_sqr_ps(__m128 x) noexcept
{
    return _mm_mul_ps(x, x);
}

VECTORIZATION_FORCE_INLINE __m128d VECTORIZATION_VECTORCALL _mm_sqr_pd(__m128d x) noexcept
{
    return _mm_mul_pd(x, x);
}

VECTORIZATION_FORCE_INLINE double _mm_accumulate_ps(__m128 a) noexcept
{
    auto tmp = _mm_add_ps(a, _mm_movehl_ps(a, a));
    return _mm_cvtss_f32(_mm_add_ss(tmp, _mm_shuffle_ps(tmp, tmp, 1)));
}

VECTORIZATION_FORCE_INLINE double _mm_accumulate_pd(__m128d a) noexcept
{
    return _mm_cvtsd_f64(_mm_add_sd(a, _mm_unpackhi_pd(a, a)));
}

VECTORIZATION_FORCE_INLINE float _mm_hmax_ps(__m128 x) noexcept
{
    auto max1 = _mm_shuffle_ps(x, x, _MM_SHUFFLE(0, 0, 3, 2));
    auto max2 = _mm_max_ps(x, max1);
    auto max3 = _mm_shuffle_ps(max2, max2, _MM_SHUFFLE(0, 0, 0, 1));
    auto max4 = _mm_max_ps(max2, max3);

    return _mm_cvtss_f32(max4);
}

VECTORIZATION_FORCE_INLINE double _mm_hmax_pd(__m128d a) noexcept
{
    return _mm_cvtsd_f64(_mm_max_sd(a, _mm_unpackhi_pd(a, a)));
}

VECTORIZATION_FORCE_INLINE float _mm_hmin_ps(__m128 x) noexcept
{
    auto min1 = _mm_shuffle_ps(x, x, _MM_SHUFFLE(0, 0, 3, 2));
    auto min2 = _mm_min_ps(x, min1);
    auto min3 = _mm_shuffle_ps(min2, min2, _MM_SHUFFLE(0, 0, 0, 1));
    auto min4 = _mm_min_ps(min2, min3);
    return _mm_cvtss_f32(min4);
}

VECTORIZATION_FORCE_INLINE double _mm_hmin_pd(__m128d a) noexcept
{
    return _mm_cvtsd_f64(_mm_min_sd(a, _mm_unpackhi_pd(a, a)));
}

VECTORIZATION_FORCE_INLINE __m128 VECTORIZATION_VECTORCALL
_mm_signcopy_ps(__m128 x, __m128 sign) noexcept
{
    const __m128 mask0 = _mm_set1_ps(-0.F);
    return _mm_or_ps(_mm_and_ps(sign, mask0), _mm_andnot_ps(mask0, x));
}

VECTORIZATION_FORCE_INLINE __m128d VECTORIZATION_VECTORCALL
_mm_signcopy_pd(__m128d x, __m128d sign) noexcept
{
    const __m128d mask0 = _mm_set1_pd(-0.);
    return _mm_or_pd(_mm_and_pd(sign, mask0), _mm_andnot_pd(mask0, x));
}

VECTORIZATION_FORCE_INLINE __m128 VECTORIZATION_VECTORCALL
_mm_gather_ps(float const* from, int stride) noexcept
{
    return _mm_set_ps(from[3 * stride], from[2 * stride], from[stride], from[0]);
}

VECTORIZATION_FORCE_INLINE __m128d VECTORIZATION_VECTORCALL
_mm_gather_pd(double const* from, int stride) noexcept
{
    return _mm_set_pd(from[stride], from[0]);
}

VECTORIZATION_FORCE_INLINE __m128 VECTORIZATION_VECTORCALL
_mm_gather_ps(float const* from, int const* idxs) noexcept
{
    return _mm_set_ps(from[idxs[3]], from[idxs[2]], from[idxs[1]], from[idxs[0]]);
}

VECTORIZATION_FORCE_INLINE __m128d VECTORIZATION_VECTORCALL
_mm_gather_pd(double const* from, int const* idxs) noexcept
{
    return _mm_set_pd(from[idxs[1]], from[idxs[0]]);
}

VECTORIZATION_FORCE_INLINE void VECTORIZATION_VECTORCALL
_mm_scatter_ps(__m128 from, float* to, int stride) noexcept
{
    to[0]          = _mm_cvtss_f32(from);
    to[stride]     = _mm_cvtss_f32(_mm_shuffle_ps(from, from, 1));
    to[stride * 2] = _mm_cvtss_f32(_mm_shuffle_ps(from, from, 2));
    to[stride * 3] = _mm_cvtss_f32(_mm_shuffle_ps(from, from, 3));
}

VECTORIZATION_FORCE_INLINE void VECTORIZATION_VECTORCALL
_mm_scatter_pd(__m128d from, double* to, int stride) noexcept
{
    to[0]      = _mm_cvtsd_f64(from);
    to[stride] = _mm_cvtsd_f64(_mm_shuffle_pd(from, from, 1));
}

VECTORIZATION_FORCE_INLINE void VECTORIZATION_VECTORCALL
_mm_scatter_ps(__m128 from, float* to, int const* stride) noexcept
{
    to[stride[0]] = _mm_cvtss_f32(from);
    to[stride[1]] = _mm_cvtss_f32(_mm_shuffle_ps(from, from, 1));
    to[stride[2]] = _mm_cvtss_f32(_mm_shuffle_ps(from, from, 2));
    to[stride[3]] = _mm_cvtss_f32(_mm_shuffle_ps(from, from, 3));
}

VECTORIZATION_FORCE_INLINE void VECTORIZATION_VECTORCALL
_mm_scatter_pd(__m128d from, double* to, int const* stride) noexcept
{
    to[stride[0]] = _mm_cvtsd_f64(from);
    to[stride[1]] = _mm_cvtsd_f64(_mm_shuffle_pd(from, from, 1));
}

VECTORIZATION_FORCE_INLINE auto VECTORIZATION_VECTORCALL _mm_eq_ps(__m128 a, __m128 b) noexcept
{
    return _mm_cmpeq_ps(a, b);
}
VECTORIZATION_FORCE_INLINE auto VECTORIZATION_VECTORCALL _mm_neq_ps(__m128 a, __m128 b) noexcept
{
    return _mm_cmpneq_ps(a, b);
}
VECTORIZATION_FORCE_INLINE auto VECTORIZATION_VECTORCALL _mm_gt_ps(__m128 a, __m128 b) noexcept
{
    return _mm_cmpgt_ps(a, b);
}
VECTORIZATION_FORCE_INLINE auto VECTORIZATION_VECTORCALL _mm_lt_ps(__m128 a, __m128 b) noexcept
{
    return _mm_cmplt_ps(a, b);
}
VECTORIZATION_FORCE_INLINE auto VECTORIZATION_VECTORCALL _mm_ge_ps(__m128 a, __m128 b) noexcept
{
    return _mm_cmpge_ps(a, b);
}
VECTORIZATION_FORCE_INLINE auto VECTORIZATION_VECTORCALL _mm_le_ps(__m128 a, __m128 b) noexcept
{
    return _mm_cmple_ps(a, b);
}

VECTORIZATION_FORCE_INLINE auto VECTORIZATION_VECTORCALL _mm_eq_pd(__m128d a, __m128d b) noexcept
{
    return _mm_cmpeq_pd(a, b);
}
VECTORIZATION_FORCE_INLINE auto VECTORIZATION_VECTORCALL _mm_neq_pd(__m128d a, __m128d b) noexcept
{
    return _mm_cmpneq_pd(a, b);
}
VECTORIZATION_FORCE_INLINE auto VECTORIZATION_VECTORCALL _mm_gt_pd(__m128d a, __m128d b) noexcept
{
    return _mm_cmpgt_pd(a, b);
}
VECTORIZATION_FORCE_INLINE auto VECTORIZATION_VECTORCALL _mm_lt_pd(__m128d a, __m128d b) noexcept
{
    return _mm_cmplt_pd(a, b);
}
VECTORIZATION_FORCE_INLINE auto VECTORIZATION_VECTORCALL _mm_ge_pd(__m128d a, __m128d b) noexcept
{
    return _mm_cmpge_pd(a, b);
}
VECTORIZATION_FORCE_INLINE auto VECTORIZATION_VECTORCALL _mm_le_pd(__m128d a, __m128d b) noexcept
{
    return _mm_cmple_pd(a, b);
}

VECTORIZATION_FORCE_INLINE void transpose_4x4_ps(__m128* simd) noexcept
{
    _MM_TRANSPOSE4_PS(simd[0], simd[1], simd[2], simd[3]);  // NOLINT
}

VECTORIZATION_FORCE_INLINE void transpose_2x2_pd(__m128d* simd) noexcept
{
    __m128d tmp = _mm_unpackhi_pd(simd[0], simd[1]);
    simd[0]     = _mm_unpacklo_pd(simd[0], simd[1]);
    simd[1]     = tmp;
}

VECTORIZATION_FORCE_INLINE auto _mm_loadu_mask_ps(uint32_t const* from)
{
    return _mm_castsi128_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(from)));
}

VECTORIZATION_FORCE_INLINE auto _mm_loadu_mask_pd(uint32_t const* from)
{
    alignas(16) uint32_t buf[4] = {from[0], from[1], 0u, 0u};
    return _mm_cvtepi32_pd(_mm_load_si128(reinterpret_cast<const __m128i*>(buf)));
}

VECTORIZATION_FORCE_INLINE auto _mm_load_mask_ps(uint32_t const* from)
{
    return _mm_castsi128_ps(_mm_load_si128(reinterpret_cast<const __m128i*>(from)));
}

VECTORIZATION_FORCE_INLINE auto _mm_load_mask_pd(uint32_t const* from)
{
    return _mm_cvtepi32_pd(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(from)));
}

VECTORIZATION_FORCE_INLINE void _mm_storeu_mask_ps(__m128 from, uint32_t* to)
{
    _mm_storeu_si128(reinterpret_cast<__m128i*>(to), _mm_castps_si128(from));
}

VECTORIZATION_FORCE_INLINE void _mm_storeu_mask_pd(__m128d from, uint32_t* to)
{
    _mm_storel_epi64(reinterpret_cast<__m128i*>(to), _mm_cvtpd_epi32(from));
}

VECTORIZATION_FORCE_INLINE void _mm_store_mask_ps(__m128 from, uint32_t* to)
{
    _mm_store_si128(reinterpret_cast<__m128i*>(to), _mm_castps_si128(from));
}

VECTORIZATION_FORCE_INLINE void _mm_store_mask_pd(__m128d from, uint32_t* to)
{
    _mm_storel_epi64(reinterpret_cast<__m128i*>(to), _mm_cvtpd_epi32(from));
}

VECTORIZATION_FORCE_INLINE auto _mm_set1_mask_ps(uint32_t const from)
{
    return _mm_set1_ps(static_cast<float>(from));
}

VECTORIZATION_FORCE_INLINE auto _mm_set1_mask_pd(uint32_t const from)
{
    return _mm_set1_pd(static_cast<double>(from));
}

VECTORIZATION_FORCE_INLINE auto _mm_mask_not_ps(__m128 a)
{
    return _mm_andnot_ps(a, _mm_cmpeq_ps(a, a));
}
VECTORIZATION_FORCE_INLINE auto _mm_mask_not_pd(__m128d a)
{
    return _mm_andnot_pd(a, _mm_cmpeq_pd(a, a));
}
