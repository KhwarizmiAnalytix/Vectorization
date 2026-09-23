/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

#include "backend/cpu/neon/svml.h"
#include "backend/simd.h"
#include "common/normal_cdf.h"
#include "common/vectorization_macros.h"

namespace vectorization
{
namespace detail_neon
{
VECTORIZATION_FORCE_INLINE float64x2_t map1_f64(float64x2_t v, double (*fn)(double))
{
    alignas(16) double b[2];
    vst1q_f64(b, v);
    b[0] = fn(b[0]);
    b[1] = fn(b[1]);
    return vld1q_f64(b);
}
}  // namespace detail_neon
}  // namespace vectorization

template <>
struct simd<double>
{
    using simd_t      = float64x2_t;
    using mask_t      = uint64x2_t;
    using simd_half_t = float64x2_t;
    using simd_int_t  = uint32x2_t;
    using value_t     = double;
    using int_t       = uint32_t;

    static constexpr int size      = 2;
    static constexpr int half_size = 1;

    inline static uint64x2_t const sign_mask = vdupq_n_u64(0x8000000000000000ULL);

    VECTORIZATION_SIMD_RETURN_TYPE prefetch(const value_t* addr) { __builtin_prefetch(addr, 0, 3); }

    VECTORIZATION_SIMD_METHOD simd_t load(const value_t* addr) { return vld1q_f64(addr); }

    VECTORIZATION_SIMD_METHOD simd_t loadu(const value_t* addr) { return vld1q_f64(addr); }

    VECTORIZATION_SIMD_RETURN_TYPE store(simd_t from, value_t* to) { vst1q_f64(to, from); }

    VECTORIZATION_SIMD_RETURN_TYPE storeu(simd_t from, value_t* to) { vst1q_f64(to, from); }

    template <
        typename scalar_t,
        typename std::enable_if<std::is_fundamental<scalar_t>::value, bool>::type = true>
    VECTORIZATION_SIMD_METHOD simd_t set(const scalar_t alpha)
    {
        return vdupq_n_f64(static_cast<value_t>(alpha));
    }

    VECTORIZATION_SIMD_METHOD mask_t set(int_t const from)
    {
        return vdupq_n_u64(from != 0 ? ~0ULL : 0ULL);
    }

    VECTORIZATION_SIMD_METHOD simd_t setzero() { return vdupq_n_f64(0.); }

    VECTORIZATION_SIMD_METHOD simd_t add(simd_t x, simd_t y) { return vaddq_f64(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t sub(simd_t x, simd_t y) { return vsubq_f64(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t mul(simd_t x, simd_t y) { return vmulq_f64(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t div(simd_t x, simd_t y) { return vdivq_f64(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t fma(simd_t x, simd_t y, simd_t z)
    {
        return vfmaq_f64(z, x, y);
    }

    VECTORIZATION_SIMD_METHOD simd_t pow(simd_t x, simd_t y) { return vpowq_f64(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t hypot(simd_t x, simd_t y)
    {
        return vsqrtq_f64(vaddq_f64(vmulq_f64(x, x), vmulq_f64(y, y)));
    }

    VECTORIZATION_SIMD_METHOD simd_t min(simd_t x, simd_t y) { return vminq_f64(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t max(simd_t x, simd_t y) { return vmaxq_f64(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t signcopy(simd_t x, simd_t sign)
    {
        uint64x2_t xs = vreinterpretq_u64_f64(x);
        uint64x2_t ss = vreinterpretq_u64_f64(sign);
        return vreinterpretq_f64_u64(vorrq_u64(vandq_u64(sign_mask, ss), vbicq_u64(xs, sign_mask)));
    }

    VECTORIZATION_SIMD_METHOD simd_t sqrt(simd_t x) { return vsqrtq_f64(x); }

    VECTORIZATION_SIMD_METHOD simd_t sqr(simd_t x) { return vmulq_f64(x, x); }

    VECTORIZATION_SIMD_METHOD simd_t ceil(simd_t x) { return vrndpq_f64(x); }

    VECTORIZATION_SIMD_METHOD simd_t floor(simd_t x) { return vrndmq_f64(x); }

    VECTORIZATION_SIMD_METHOD simd_t exp(simd_t x) { return vexpq_f64(x); }
    VECTORIZATION_SIMD_METHOD simd_t expm1(simd_t x) { return vexpm1q_f64(x); }
    VECTORIZATION_SIMD_METHOD simd_t exp2(simd_t x) { return vexp2q_f64(x); }
    VECTORIZATION_SIMD_METHOD simd_t exp10(simd_t x) { return vexp10q_f64(x); }
    VECTORIZATION_SIMD_METHOD simd_t log(simd_t x) { return vlogq_f64(x); }
    VECTORIZATION_SIMD_METHOD simd_t log1p(simd_t x) { return vlog1pq_f64(x); }
    VECTORIZATION_SIMD_METHOD simd_t log2(simd_t x) { return vlog2q_f64(x); }
    VECTORIZATION_SIMD_METHOD simd_t log10(simd_t x) { return vlog10q_f64(x); }
    VECTORIZATION_SIMD_METHOD simd_t sin(simd_t x) { return vsinq_f64(x); }
    VECTORIZATION_SIMD_METHOD simd_t cos(simd_t x) { return vcosq_f64(x); }
    VECTORIZATION_SIMD_METHOD simd_t tan(simd_t x) { return vtanq_f64(x); }
    VECTORIZATION_SIMD_METHOD simd_t asin(simd_t x) { return vasinq_f64(x); }
    VECTORIZATION_SIMD_METHOD simd_t acos(simd_t x) { return vacosq_f64(x); }
    VECTORIZATION_SIMD_METHOD simd_t atan(simd_t x) { return vatanq_f64(x); }
    VECTORIZATION_SIMD_METHOD simd_t sinh(simd_t x) { return vsinhq_f64(x); }
    VECTORIZATION_SIMD_METHOD simd_t cosh(simd_t x) { return vcoshq_f64(x); }
    VECTORIZATION_SIMD_METHOD simd_t tanh(simd_t x) { return vtanhq_f64(x); }
    VECTORIZATION_SIMD_METHOD simd_t asinh(simd_t x) { return vasinhq_f64(x); }
    VECTORIZATION_SIMD_METHOD simd_t acosh(simd_t x) { return vacoshq_f64(x); }
    VECTORIZATION_SIMD_METHOD simd_t atanh(simd_t x) { return vatanhq_f64(x); }
    VECTORIZATION_SIMD_METHOD simd_t cbrt(simd_t x) { return vcbrtq_f64(x); }

    VECTORIZATION_SIMD_METHOD simd_t cdf(simd_t x)
    {
        alignas(16) double b[2];
        vst1q_f64(b, x);
        b[0] = vectorization::normalcdf(b[0]);
        b[1] = vectorization::normalcdf(b[1]);
        return vld1q_f64(b);
    }

    VECTORIZATION_SIMD_METHOD simd_t inv_cdf(simd_t x)
    {
        alignas(16) double b[2];
        vst1q_f64(b, x);
        b[0] = vectorization::inv_normalcdf(b[0]);
        b[1] = vectorization::inv_normalcdf(b[1]);
        return vld1q_f64(b);
    }

    VECTORIZATION_SIMD_METHOD simd_t trunc(simd_t x)
    {
        return vectorization::detail_neon::map1_f64(
            x, static_cast<double (*)(double)>(&std::trunc));
    }

    VECTORIZATION_SIMD_METHOD simd_t invsqrt(simd_t x)
    {
        float64x2_t est = vrsqrteq_f64(x);
        est             = vmulq_f64(vrsqrtsq_f64(vmulq_f64(x, est), est), est);
        est             = vmulq_f64(vrsqrtsq_f64(vmulq_f64(x, est), est), est);
        return est;
    }

    VECTORIZATION_SIMD_METHOD simd_t fabs(simd_t x)
    {
        return vreinterpretq_f64_u64(vbicq_u64(vreinterpretq_u64_f64(x), sign_mask));
    }

    VECTORIZATION_SIMD_METHOD simd_t neg(simd_t x)
    {
        return vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(x), sign_mask));
    }

    VECTORIZATION_FORCE_INLINE static double accumulate(simd_t x)
    {
        return vgetq_lane_f64(x, 0) + vgetq_lane_f64(x, 1);
    }

    VECTORIZATION_FORCE_INLINE static value_t hmax(simd_t x)
    {
        return (std::max)(vgetq_lane_f64(x, 0), vgetq_lane_f64(x, 1));
    }

    VECTORIZATION_FORCE_INLINE static value_t hmin(simd_t x)
    {
        return (std::min)(vgetq_lane_f64(x, 0), vgetq_lane_f64(x, 1));
    }

    VECTORIZATION_SIMD_METHOD simd_t gather(value_t const* from, int stride)
    {
        alignas(16) value_t tmp[2] = {from[0], from[stride]};
        return vld1q_f64(tmp);
    }

    VECTORIZATION_SIMD_RETURN_TYPE scatter(simd_t from, int stride, value_t* to)
    {
        alignas(16) value_t b[2];
        vst1q_f64(b, from);
        to[0]      = b[0];
        to[stride] = b[1];
    }

    VECTORIZATION_SIMD_METHOD simd_t gather(value_t const* from, const int* strides)
    {
        alignas(16) value_t tmp[2] = {from[strides[0]], from[strides[1]]};
        return vld1q_f64(tmp);
    }

    VECTORIZATION_SIMD_RETURN_TYPE scatter(simd_t from, const int* stride, value_t* to)
    {
        alignas(16) value_t b[2];
        vst1q_f64(b, from);
        to[stride[0]] = b[0];
        to[stride[1]] = b[1];
    }

    VECTORIZATION_SIMD_METHOD simd_t if_else(const mask_t& x, simd_t y, simd_t z)
    {
        return vbslq_f64(x, y, z);
    }

    VECTORIZATION_SIMD_METHOD mask_t eq(simd_t x, simd_t y) { return vceqq_f64(x, y); }

    VECTORIZATION_SIMD_METHOD mask_t neq(simd_t x, simd_t y)
    {
        return veorq_u64(vceqq_f64(x, y), vdupq_n_u64(~0ULL));
    }

    VECTORIZATION_SIMD_METHOD mask_t gt(simd_t x, simd_t y) { return vcgtq_f64(x, y); }

    VECTORIZATION_SIMD_METHOD mask_t lt(simd_t x, simd_t y) { return vcltq_f64(x, y); }

    VECTORIZATION_SIMD_METHOD mask_t ge(simd_t x, simd_t y) { return vcgeq_f64(x, y); }

    VECTORIZATION_SIMD_METHOD mask_t le(simd_t x, simd_t y) { return vcleq_f64(x, y); }

    VECTORIZATION_FORCE_INLINE static uint64x2_t u32_mask_to_u64(uint32_t a, uint32_t b)
    {
        uint64_t   la = a ? ~0ULL : 0ULL;
        uint64_t   lb = b ? ~0ULL : 0ULL;
        uint64x2_t r  = vdupq_n_u64(0);
        r             = vsetq_lane_u64(la, r, 0);
        r             = vsetq_lane_u64(lb, r, 1);
        return r;
    }

    VECTORIZATION_SIMD_METHOD mask_t loadu(int_t const* from)
    {
        return u32_mask_to_u64(from[0], from[1]);
    }

    VECTORIZATION_SIMD_METHOD mask_t load(int_t const* from)
    {
        return u32_mask_to_u64(from[0], from[1]);
    }

    VECTORIZATION_SIMD_RETURN_TYPE storeu(const mask_t& from, int_t* to)
    {
        to[0] = vgetq_lane_u64(from, 0) ? 1u : 0u;
        to[1] = vgetq_lane_u64(from, 1) ? 1u : 0u;
    }

    VECTORIZATION_SIMD_RETURN_TYPE store(const mask_t& from, int_t* to) { storeu(from, to); }

    VECTORIZATION_SIMD_METHOD mask_t not_mask(const mask_t& x)
    {
        return veorq_u64(x, vdupq_n_u64(~0ULL));
    }

    VECTORIZATION_SIMD_METHOD mask_t and_mask(const mask_t& x, const mask_t& y)
    {
        return vandq_u64(x, y);
    }

    VECTORIZATION_SIMD_METHOD mask_t or_mask(const mask_t& x, const mask_t& y)
    {
        return vorrq_u64(x, y);
    }

    VECTORIZATION_SIMD_METHOD mask_t xor_mask(const mask_t& x, const mask_t& y)
    {
        return veorq_u64(x, y);
    }

    VECTORIZATION_SIMD_RETURN_TYPE broadcast(
        const double* from, simd_t& a0, simd_t& a1, simd_t& a2, simd_t& a3)
    {
        a0 = vdupq_n_f64(from[0]);
        a1 = vdupq_n_f64(from[1]);
        a2 = vdupq_n_f64(from[2]);
        a3 = vdupq_n_f64(from[3]);
    }

    template <int N>
    VECTORIZATION_SIMD_RETURN_TYPE ptranspose(simd_t simd[N])  // NOLINT
    {
        constexpr int L = size;
        if constexpr (N == 2 && L == 2)
        {
            float64x2_t t0 = vzip1q_f64(simd[0], simd[1]);
            float64x2_t t1 = vzip2q_f64(simd[0], simd[1]);
            simd[0]        = t0;
            simd[1]        = t1;
        }
        else if constexpr (N == L)
        {
            alignas(16) double buf[N * L];
            for (int i = 0; i < N; ++i)
            {
                vst1q_f64(buf + i * L, simd[i]);
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
                simd[i] = vld1q_f64(out + i * L);
            }
        }
        else if constexpr (N == 8 && L == 2)
        {
            alignas(16) double buf[8][2];
            for (int i = 0; i < 8; ++i)
            {
                vst1q_f64(buf[i], simd[i]);
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
                    simd[4 * r + q] = vld1q_f64(&out[r][2 * q]);
                }
            }
        }
        else if constexpr (N == 4 && L == 2)
        {
            alignas(16) double buf[4][2];
            for (int i = 0; i < 4; ++i)
            {
                vst1q_f64(buf[i], simd[i]);
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
                    simd[2 * r + q] = vld1q_f64(&out[r][2 * q]);
                }
            }
        }
        else
        {
            static_assert(sizeof(simd_t) == 0, "ptranspose: unsupported N for NEON double");
        }
    }

    VECTORIZATION_SIMD_RETURN_TYPE copy(const value_t* from, value_t* to)
    {
        vst1q_f64(to, vld1q_f64(from));
    }

    VECTORIZATION_FORCE_INLINE static simd_t ploadquad(const double* from)
    {
        return vdupq_n_f64(from[0]);
    }
};
