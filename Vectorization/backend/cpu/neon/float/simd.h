/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

#include <arm_neon.h>

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
VECTORIZATION_FORCE_INLINE float32x4_t map1_f32(float32x4_t v, float (*fn)(float))
{
    alignas(16) float b[4];
    vst1q_f32(b, v);
    b[0] = fn(b[0]);
    b[1] = fn(b[1]);
    b[2] = fn(b[2]);
    b[3] = fn(b[3]);
    return vld1q_f32(b);
}
}  // namespace detail_neon
}  // namespace vectorization

template <>
struct simd<float>
{
    using simd_t      = float32x4_t;
    using mask_t      = uint32x4_t;
    using simd_half_t = float32x2_t;
    using simd_int_t  = uint32x4_t;
    using value_t     = float;
    using int_t       = uint32_t;

    static constexpr int size      = 4;
    static constexpr int half_size = 2;

    inline static uint32x4_t const sign_mask = vdupq_n_u32(0x80000000u);

    VECTORIZATION_SIMD_RETURN_TYPE prefetch(const value_t* addr) { __builtin_prefetch(addr, 0, 3); }

    VECTORIZATION_SIMD_METHOD simd_t load(const value_t* addr) { return vld1q_f32(addr); }

    VECTORIZATION_SIMD_METHOD simd_t loadu(const value_t* addr) { return vld1q_f32(addr); }

    VECTORIZATION_SIMD_RETURN_TYPE store(simd_t from, value_t* to) { vst1q_f32(to, from); }

    VECTORIZATION_SIMD_RETURN_TYPE storeu(simd_t from, value_t* to) { vst1q_f32(to, from); }

    template <
        typename scalar_t,
        typename std::enable_if<std::is_fundamental<scalar_t>::value, bool>::type = true>
    VECTORIZATION_SIMD_METHOD simd_t set(scalar_t alpha)
    {
        return vdupq_n_f32(static_cast<value_t>(alpha));
    }

    VECTORIZATION_SIMD_METHOD mask_t set(int_t const from)
    {
        return vdupq_n_u32(from != 0 ? ~0u : 0u);
    }

    VECTORIZATION_SIMD_METHOD simd_t setzero() { return vdupq_n_f32(0.f); }

    VECTORIZATION_SIMD_METHOD simd_t add(simd_t x, simd_t y) { return vaddq_f32(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t sub(simd_t x, simd_t y) { return vsubq_f32(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t mul(simd_t x, simd_t y) { return vmulq_f32(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t div(simd_t x, simd_t y) { return vdivq_f32(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t pow(simd_t x, simd_t y) { return vpowq_f32(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t hypot(simd_t x, simd_t y)
    {
        return vsqrtq_f32(vaddq_f32(vmulq_f32(x, x), vmulq_f32(y, y)));
    }

    VECTORIZATION_SIMD_METHOD simd_t min(simd_t x, simd_t y) { return vminq_f32(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t max(simd_t x, simd_t y) { return vmaxq_f32(x, y); }

    VECTORIZATION_SIMD_METHOD simd_t fma(simd_t x, simd_t y, simd_t z)
    {
        return vfmaq_f32(z, x, y);
    }

    VECTORIZATION_SIMD_METHOD simd_t signcopy(simd_t x, simd_t sign)
    {
        uint32x4_t xs = vreinterpretq_u32_f32(x);
        uint32x4_t ss = vreinterpretq_u32_f32(sign);
        return vreinterpretq_f32_u32(vorrq_u32(vandq_u32(sign_mask, ss), vbicq_u32(xs, sign_mask)));
    }

    VECTORIZATION_SIMD_METHOD simd_t sqrt(simd_t x) { return vsqrtq_f32(x); }

    VECTORIZATION_SIMD_METHOD simd_t sqr(simd_t x) { return vmulq_f32(x, x); }

    VECTORIZATION_SIMD_METHOD simd_t ceil(simd_t x) { return vrndpq_f32(x); }

    VECTORIZATION_SIMD_METHOD simd_t floor(simd_t x) { return vrndmq_f32(x); }

    VECTORIZATION_SIMD_METHOD simd_t exp(simd_t x) { return vexpq_f32(x); }
    VECTORIZATION_SIMD_METHOD simd_t expm1(simd_t x) { return vexpm1q_f32(x); }
    VECTORIZATION_SIMD_METHOD simd_t exp2(simd_t x) { return vexp2q_f32(x); }
    VECTORIZATION_SIMD_METHOD simd_t exp10(simd_t x) { return vexp10q_f32(x); }
    VECTORIZATION_SIMD_METHOD simd_t log(simd_t x) { return vlogq_f32(x); }
    VECTORIZATION_SIMD_METHOD simd_t log1p(simd_t x) { return vlog1pq_f32(x); }
    VECTORIZATION_SIMD_METHOD simd_t log2(simd_t x) { return vlog2q_f32(x); }
    VECTORIZATION_SIMD_METHOD simd_t log10(simd_t x) { return vlog10q_f32(x); }
    VECTORIZATION_SIMD_METHOD simd_t sin(simd_t x) { return vsinq_f32(x); }
    VECTORIZATION_SIMD_METHOD simd_t cos(simd_t x) { return vcosq_f32(x); }
    VECTORIZATION_SIMD_METHOD simd_t tan(simd_t x) { return vtanq_f32(x); }
    VECTORIZATION_SIMD_METHOD simd_t asin(simd_t x) { return vasinq_f32(x); }
    VECTORIZATION_SIMD_METHOD simd_t acos(simd_t x) { return vacosq_f32(x); }
    VECTORIZATION_SIMD_METHOD simd_t atan(simd_t x) { return vatanq_f32(x); }
    VECTORIZATION_SIMD_METHOD simd_t sinh(simd_t x) { return vsinhq_f32(x); }
    VECTORIZATION_SIMD_METHOD simd_t cosh(simd_t x) { return vcoshq_f32(x); }
    VECTORIZATION_SIMD_METHOD simd_t tanh(simd_t x) { return vtanhq_f32(x); }
    VECTORIZATION_SIMD_METHOD simd_t asinh(simd_t x) { return vasinhq_f32(x); }
    VECTORIZATION_SIMD_METHOD simd_t acosh(simd_t x) { return vacoshq_f32(x); }
    VECTORIZATION_SIMD_METHOD simd_t atanh(simd_t x) { return vatanhq_f32(x); }
    VECTORIZATION_SIMD_METHOD simd_t cbrt(simd_t x) { return vcbrtq_f32(x); }

    VECTORIZATION_SIMD_METHOD simd_t cdf(simd_t x)
    {
        alignas(16) float b[4];
        vst1q_f32(b, x);
        b[0] = static_cast<float>(vectorization::normalcdf(static_cast<double>(b[0])));
        b[1] = static_cast<float>(vectorization::normalcdf(static_cast<double>(b[1])));
        b[2] = static_cast<float>(vectorization::normalcdf(static_cast<double>(b[2])));
        b[3] = static_cast<float>(vectorization::normalcdf(static_cast<double>(b[3])));
        return vld1q_f32(b);
    }

    VECTORIZATION_SIMD_METHOD simd_t inv_cdf(simd_t x)
    {
        alignas(16) float b[4];
        vst1q_f32(b, x);
        b[0] = static_cast<float>(vectorization::inv_normalcdf(static_cast<double>(b[0])));
        b[1] = static_cast<float>(vectorization::inv_normalcdf(static_cast<double>(b[1])));
        b[2] = static_cast<float>(vectorization::inv_normalcdf(static_cast<double>(b[2])));
        b[3] = static_cast<float>(vectorization::inv_normalcdf(static_cast<double>(b[3])));
        return vld1q_f32(b);
    }

    VECTORIZATION_SIMD_METHOD simd_t trunc(simd_t x)
    {
        return vectorization::detail_neon::map1_f32(x, static_cast<float (*)(float)>(&std::trunc));
    }

    VECTORIZATION_SIMD_METHOD simd_t invsqrt(simd_t x)
    {
        float32x4_t est = vrsqrteq_f32(x);
        est             = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x, est), est), est);
        return est;
    }

    VECTORIZATION_SIMD_METHOD simd_t fabs(simd_t x)
    {
        return vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(x), sign_mask));
    }

    VECTORIZATION_SIMD_METHOD simd_t neg(simd_t x)
    {
        return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(x), sign_mask));
    }

    VECTORIZATION_FORCE_INLINE static double accumulate(simd_t x)
    {
        float32x2_t t = vadd_f32(vget_high_f32(x), vget_low_f32(x));
        t             = vpadd_f32(t, t);
        return static_cast<double>(vget_lane_f32(t, 0));
    }

    VECTORIZATION_FORCE_INLINE static value_t hmax(simd_t x)
    {
        float32x2_t t = vmax_f32(vget_high_f32(x), vget_low_f32(x));
        t             = vpmax_f32(t, t);
        return vget_lane_f32(t, 0);
    }

    VECTORIZATION_FORCE_INLINE static value_t hmin(simd_t x)
    {
        float32x2_t t = vmin_f32(vget_high_f32(x), vget_low_f32(x));
        t             = vpmin_f32(t, t);
        return vget_lane_f32(t, 0);
    }

    VECTORIZATION_SIMD_METHOD simd_t gather(value_t const* from, int stride)
    {
        alignas(16) value_t tmp[4] = {
            from[0],
            from[1 * stride],
            from[2 * stride],
            from[3 * stride],
        };
        return vld1q_f32(tmp);
    }

    VECTORIZATION_SIMD_RETURN_TYPE scatter(simd_t from, int stride, value_t* to)
    {
        alignas(16) float b[4];
        vst1q_f32(b, from);
        to[0]          = b[0];
        to[stride]     = b[1];
        to[stride * 2] = b[2];
        to[stride * 3] = b[3];
    }

    VECTORIZATION_SIMD_METHOD simd_t gather(value_t const* from, const int* strides)
    {
        alignas(16) value_t tmp[4] = {
            from[strides[0]],
            from[strides[1]],
            from[strides[2]],
            from[strides[3]],
        };
        return vld1q_f32(tmp);
    }

    VECTORIZATION_SIMD_RETURN_TYPE scatter(simd_t from, const int* stride, value_t* to)
    {
        alignas(16) float b[4];
        vst1q_f32(b, from);
        to[stride[0]] = b[0];
        to[stride[1]] = b[1];
        to[stride[2]] = b[2];
        to[stride[3]] = b[3];
    }

    VECTORIZATION_SIMD_METHOD simd_t if_else(const mask_t& x, simd_t y, simd_t z)
    {
        return vbslq_f32(x, y, z);
    }

    VECTORIZATION_SIMD_METHOD mask_t eq(simd_t x, simd_t y) { return vceqq_f32(x, y); }

    VECTORIZATION_SIMD_METHOD mask_t neq(simd_t x, simd_t y) { return vmvnq_u32(vceqq_f32(x, y)); }

    VECTORIZATION_SIMD_METHOD mask_t gt(simd_t x, simd_t y) { return vcgtq_f32(x, y); }

    VECTORIZATION_SIMD_METHOD mask_t lt(simd_t x, simd_t y) { return vcltq_f32(x, y); }

    VECTORIZATION_SIMD_METHOD mask_t ge(simd_t x, simd_t y) { return vcgeq_f32(x, y); }

    VECTORIZATION_SIMD_METHOD mask_t le(simd_t x, simd_t y) { return vcleq_f32(x, y); }

    VECTORIZATION_SIMD_METHOD mask_t loadu(int_t const* from) { return vld1q_u32(from); }

    VECTORIZATION_SIMD_METHOD mask_t load(int_t const* from) { return vld1q_u32(from); }

    VECTORIZATION_SIMD_RETURN_TYPE storeu(const mask_t& from, int_t* to) { vst1q_u32(to, from); }

    VECTORIZATION_SIMD_RETURN_TYPE store(const mask_t& from, int_t* to) { vst1q_u32(to, from); }

    VECTORIZATION_SIMD_METHOD mask_t not_mask(const mask_t& x) { return vmvnq_u32(x); }

    VECTORIZATION_SIMD_METHOD mask_t and_mask(const mask_t& x, const mask_t& y)
    {
        return vandq_u32(x, y);
    }

    VECTORIZATION_SIMD_METHOD mask_t or_mask(const mask_t& x, const mask_t& y)
    {
        return vorrq_u32(x, y);
    }

    VECTORIZATION_SIMD_METHOD mask_t xor_mask(const mask_t& x, const mask_t& y)
    {
        return veorq_u32(x, y);
    }

    VECTORIZATION_FORCE_INLINE
    static void broadcast(const value_t* from, simd_t& a0, simd_t& a1, simd_t& a2, simd_t& a3)
    {
        a0 = vdupq_n_f32(from[0]);
        a1 = vdupq_n_f32(from[1]);
        a2 = vdupq_n_f32(from[2]);
        a3 = vdupq_n_f32(from[3]);
    }

    template <int N>
    VECTORIZATION_SIMD_RETURN_TYPE ptranspose(simd_t simd[N])  // NOLINT
    {
        constexpr int L = size;
        if constexpr (N == 4 && L == 4)
        {
            float32x4x2_t t01 = vtrnq_f32(simd[0], simd[1]);
            float32x4x2_t t23 = vtrnq_f32(simd[2], simd[3]);
            simd[0]           = vcombine_f32(vget_low_f32(t01.val[0]), vget_low_f32(t23.val[0]));
            simd[1]           = vcombine_f32(vget_low_f32(t01.val[1]), vget_low_f32(t23.val[1]));
            simd[2]           = vcombine_f32(vget_high_f32(t01.val[0]), vget_high_f32(t23.val[0]));
            simd[3]           = vcombine_f32(vget_high_f32(t01.val[1]), vget_high_f32(t23.val[1]));
        }
        else if constexpr (N == L)
        {
            alignas(16) float buf[N * L];
            for (int i = 0; i < N; ++i)
            {
                vst1q_f32(buf + i * L, simd[i]);
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
                simd[i] = vld1q_f32(out + i * L);
            }
        }
        else if constexpr (N == 8 && L == 4)
        {
            alignas(16) float buf[8][4];
            for (int i = 0; i < 8; ++i)
            {
                vst1q_f32(buf[i], simd[i]);
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
                    simd[2 * r + q] = vld1q_f32(&out[r][4 * q]);
                }
            }
        }
        else if constexpr (N == 16 && L == 4)
        {
            alignas(16) float buf[16][4];
            for (int i = 0; i < 16; ++i)
            {
                vst1q_f32(buf[i], simd[i]);
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
                    simd[4 * r + q] = vld1q_f32(&out[r][4 * q]);
                }
            }
        }
        else
        {
            static_assert(sizeof(simd_t) == 0, "ptranspose: unsupported N for NEON float");
        }
    }

    VECTORIZATION_SIMD_RETURN_TYPE copy(const value_t* from, value_t* to)
    {
        vst1q_f32(to, vld1q_f32(from));
    }

    VECTORIZATION_FORCE_INLINE static simd_t ploadquad(const value_t* from)
    {
        float32x2_t lo = vld1_dup_f32(from);
        float32x2_t hi = vld1_dup_f32(from + 1);
        return vcombine_f32(lo, hi);
    }

    VECTORIZATION_SIMD_METHOD simd_half_t gather_half(const value_t* from, int stride)
    {
        float32x2_t r = vdup_n_f32(0.f);
        r             = vset_lane_f32(from[0], r, 0);
        r             = vset_lane_f32(from[stride], r, 1);
        return r;
    }

    VECTORIZATION_FORCE_INLINE static simd_half_t loadu_half(const value_t* from)
    {
        return vld1_f32(from);
    }

    VECTORIZATION_FORCE_INLINE static simd_half_t set(const value_t* from)
    {
        return vld1_dup_f32(from);
    }

    VECTORIZATION_SIMD_METHOD simd_half_t
    fma(const simd_half_t& x, const simd_half_t& y, const simd_half_t& z)
    {
        return vfma_f32(z, x, y);
    }

    VECTORIZATION_SIMD_METHOD simd_half_t add(const simd_half_t& x, const simd_half_t& y)
    {
        return vadd_f32(x, y);
    }

    VECTORIZATION_FORCE_INLINE static simd_half_t predux_downto4(simd_t x)
    {
        return vadd_f32(vget_high_f32(x), vget_low_f32(x));
    }

    VECTORIZATION_SIMD_RETURN_TYPE scatter(const simd_half_t& from, int stride, value_t* to)
    {
        to[0]      = vget_lane_f32(from, 0);
        to[stride] = vget_lane_f32(from, 1);
    }
};
