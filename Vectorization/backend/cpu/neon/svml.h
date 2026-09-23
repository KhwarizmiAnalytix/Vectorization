/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

// backend/cpu/sve/{float,double}/simd.h reuse the NEON specializations below
// unconditionally (same lane layout at -msve-vector-bits=128), so these
// wrappers must also be emitted when only VECTORIZATION_HAS_SVE is set.
#if VECTORIZATION_HAS_NEON || VECTORIZATION_HAS_SVE

#include <arm_neon.h>

#include <cmath>

#include "common/vectorization_macros.h"

#if VECTORIZATION_HAS_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif
#if VECTORIZATION_HAS_SLEEF
#include <sleef.h>
#endif

// clang-format off

// ---------------------------------------------------------------------------
// Internal dispatch macros — generate vXXXq_f32 / vXXXq_f64 wrappers.
// Priority: Accelerate vForce  >  SLEEF  >  scalar std:: fallback.
//
// NEON_MATHF(fn, vForce_fn, sleef_fn, std_fn)   — float32x4_t unary
// NEON_MATHF2(fn, vForce_fn, sleef_fn)           — float32x4_t binary (pow)
// NEON_MATHD(fn, vForce_fn, sleef_fn, std_fn)   — float64x2_t unary
// NEON_MATHD2(fn, vForce_fn, sleef_fn)           — float64x2_t binary (pow)
//
// Accelerate vvpow* argument order:
//   vvpowf(out, exponent, base, &n)  →  out[i] = base[i]^exponent[i]
// ---------------------------------------------------------------------------

// --- float32x4_t unary ---
#if VECTORIZATION_HAS_ACCELERATE
#  define NEON_MATHF(fn, vv, sl, sc)                                           \
     VECTORIZATION_FORCE_INLINE float32x4_t fn(float32x4_t x) {               \
         alignas(16) float b[4]; const int n = 4;                              \
         vst1q_f32(b, x); vv(b, b, &n); return vld1q_f32(b); }
#elif VECTORIZATION_HAS_SLEEF
#  define NEON_MATHF(fn, vv, sl, sc)                                           \
     VECTORIZATION_FORCE_INLINE float32x4_t fn(float32x4_t x) { return sl(x); }
#else
#  define NEON_MATHF(fn, vv, sl, sc)                                           \
     VECTORIZATION_FORCE_INLINE float32x4_t fn(float32x4_t x) {               \
         alignas(16) float b[4]; vst1q_f32(b, x);                             \
         b[0]=sc(b[0]); b[1]=sc(b[1]); b[2]=sc(b[2]); b[3]=sc(b[3]);        \
         return vld1q_f32(b); }
#endif

// --- float32x4_t binary (pow) ---
#if VECTORIZATION_HAS_ACCELERATE
#  define NEON_MATHF2(fn, vv, sl)                                              \
     VECTORIZATION_FORCE_INLINE float32x4_t fn(float32x4_t x, float32x4_t y) { \
         alignas(16) float bx[4], by[4], br[4]; const int n = 4;               \
         vst1q_f32(bx, x); vst1q_f32(by, y);                                  \
         vv(br, by, bx, &n); return vld1q_f32(br); }
#elif VECTORIZATION_HAS_SLEEF
#  define NEON_MATHF2(fn, vv, sl)                                              \
     VECTORIZATION_FORCE_INLINE float32x4_t fn(float32x4_t x, float32x4_t y) { return sl(x, y); }
#else
#  define NEON_MATHF2(fn, vv, sl)                                              \
     VECTORIZATION_FORCE_INLINE float32x4_t fn(float32x4_t x, float32x4_t y) { \
         alignas(16) float bx[4], by[4]; vst1q_f32(bx, x); vst1q_f32(by, y); \
         bx[0]=std::pow(bx[0],by[0]); bx[1]=std::pow(bx[1],by[1]);            \
         bx[2]=std::pow(bx[2],by[2]); bx[3]=std::pow(bx[3],by[3]);            \
         return vld1q_f32(bx); }
#endif

// --- float64x2_t unary ---
#if VECTORIZATION_HAS_ACCELERATE
#  define NEON_MATHD(fn, vv, sl, sc)                                           \
     VECTORIZATION_FORCE_INLINE float64x2_t fn(float64x2_t x) {               \
         alignas(16) double b[2]; const int n = 2;                             \
         vst1q_f64(b, x); vv(b, b, &n); return vld1q_f64(b); }
#elif VECTORIZATION_HAS_SLEEF
#  define NEON_MATHD(fn, vv, sl, sc)                                           \
     VECTORIZATION_FORCE_INLINE float64x2_t fn(float64x2_t x) { return sl(x); }
#else
#  define NEON_MATHD(fn, vv, sl, sc)                                           \
     VECTORIZATION_FORCE_INLINE float64x2_t fn(float64x2_t x) {               \
         alignas(16) double b[2]; vst1q_f64(b, x);                            \
         b[0]=sc(b[0]); b[1]=sc(b[1]); return vld1q_f64(b); }
#endif

// --- float64x2_t binary (pow) ---
#if VECTORIZATION_HAS_ACCELERATE
#  define NEON_MATHD2(fn, vv, sl)                                              \
     VECTORIZATION_FORCE_INLINE float64x2_t fn(float64x2_t x, float64x2_t y) { \
         alignas(16) double bx[2], by[2], br[2]; const int n = 2;              \
         vst1q_f64(bx, x); vst1q_f64(by, y);                                  \
         vv(br, by, bx, &n); return vld1q_f64(br); }
#elif VECTORIZATION_HAS_SLEEF
#  define NEON_MATHD2(fn, vv, sl)                                              \
     VECTORIZATION_FORCE_INLINE float64x2_t fn(float64x2_t x, float64x2_t y) { return sl(x, y); }
#else
#  define NEON_MATHD2(fn, vv, sl)                                              \
     VECTORIZATION_FORCE_INLINE float64x2_t fn(float64x2_t x, float64x2_t y) { \
         alignas(16) double bx[2], by[2]; vst1q_f64(bx, x); vst1q_f64(by, y); \
         bx[0]=std::pow(bx[0],by[0]); bx[1]=std::pow(bx[1],by[1]);            \
         return vld1q_f64(bx); }
#endif

// ============================================================================
// float32x4_t — vXXXq_f32
//                       fn               vForce          SLEEF                 scalar
// ============================================================================
NEON_MATHF(  vexpq_f32,    vvexpf,   Sleef_expf4_u10,   std::exp  )
NEON_MATHF(  vexpm1q_f32,  vvexpm1f, Sleef_expm1f4_u10, std::expm1)
NEON_MATHF(  vexp2q_f32,   vvexp2f,  Sleef_exp2f4_u10,  std::exp2 )
NEON_MATHF(  vlogq_f32,    vvlogf,   Sleef_logf4_u10,   std::log  )
NEON_MATHF(  vlog1pq_f32,  vvlog1pf, Sleef_log1pf4_u10, std::log1p)
NEON_MATHF(  vlog2q_f32,   vvlog2f,  Sleef_log2f4_u10,  std::log2 )
NEON_MATHF(  vlog10q_f32,  vvlog10f, Sleef_log10f4_u10, std::log10)
NEON_MATHF(  vsinq_f32,    vvsinf,   Sleef_sinf4_u10,   std::sin  )
NEON_MATHF(  vcosq_f32,    vvcosf,   Sleef_cosf4_u10,   std::cos  )
NEON_MATHF(  vtanq_f32,    vvtanf,   Sleef_tanf4_u10,   std::tan  )
NEON_MATHF(  vasinq_f32,   vvasinf,  Sleef_asinf4_u10,  std::asin )
NEON_MATHF(  vacosq_f32,   vvacosf,  Sleef_acosf4_u10,  std::acos )
NEON_MATHF(  vatanq_f32,   vvatanf,  Sleef_atanf4_u10,  std::atan )
NEON_MATHF(  vsinhq_f32,   vvsinhf,  Sleef_sinhf4_u10,  std::sinh )
NEON_MATHF(  vcoshq_f32,   vvcoshf,  Sleef_coshf4_u10,  std::cosh )
NEON_MATHF(  vtanhq_f32,   vvtanhf,  Sleef_tanhf4_u10,  std::tanh )
NEON_MATHF(  vasinhq_f32,  vvasinhf, Sleef_asinhf4_u10, std::asinh)
NEON_MATHF(  vacoshq_f32,  vvacoshf, Sleef_acoshf4_u10, std::acosh)
NEON_MATHF(  vatanhq_f32,  vvatanhf, Sleef_atanhf4_u10, std::atanh)
NEON_MATHF(  vcbrtq_f32,   vvcbrtf,  Sleef_cbrtf4_u10,  std::cbrt )
NEON_MATHF2( vpowq_f32,    vvpowf,   Sleef_powf4_u10               )

// exp10: no native Accelerate equivalent — use SLEEF or scalar
VECTORIZATION_FORCE_INLINE float32x4_t vexp10q_f32(float32x4_t x)
{
#if VECTORIZATION_HAS_SLEEF
    return Sleef_exp10f4_u10(x);
#else
    alignas(16) float b[4]; vst1q_f32(b, x);
    b[0] = std::pow(10.f, b[0]); b[1] = std::pow(10.f, b[1]);
    b[2] = std::pow(10.f, b[2]); b[3] = std::pow(10.f, b[3]);
    return vld1q_f32(b);
#endif
}

// ============================================================================
// float64x2_t — vXXXq_f64
//                       fn               vForce         SLEEF                 scalar
// ============================================================================
NEON_MATHD(  vexpq_f64,    vvexp,    Sleef_expd2_u10,   std::exp  )
NEON_MATHD(  vexpm1q_f64,  vvexpm1,  Sleef_expm1d2_u10, std::expm1)
NEON_MATHD(  vexp2q_f64,   vvexp2,   Sleef_exp2d2_u10,  std::exp2 )
NEON_MATHD(  vlogq_f64,    vvlog,    Sleef_logd2_u10,   std::log  )
NEON_MATHD(  vlog1pq_f64,  vvlog1p,  Sleef_log1pd2_u10, std::log1p)
NEON_MATHD(  vlog2q_f64,   vvlog2,   Sleef_log2d2_u10,  std::log2 )
NEON_MATHD(  vlog10q_f64,  vvlog10,  Sleef_log10d2_u10, std::log10)
NEON_MATHD(  vsinq_f64,    vvsin,    Sleef_sind2_u10,   std::sin  )
NEON_MATHD(  vcosq_f64,    vvcos,    Sleef_cosd2_u10,   std::cos  )
NEON_MATHD(  vtanq_f64,    vvtan,    Sleef_tand2_u10,   std::tan  )
NEON_MATHD(  vasinq_f64,   vvasin,   Sleef_asind2_u10,  std::asin )
NEON_MATHD(  vacosq_f64,   vvacos,   Sleef_acosd2_u10,  std::acos )
NEON_MATHD(  vatanq_f64,   vvatan,   Sleef_atand2_u10,  std::atan )
NEON_MATHD(  vsinhq_f64,   vvsinh,   Sleef_sinhd2_u10,  std::sinh )
NEON_MATHD(  vcoshq_f64,   vvcosh,   Sleef_coshd2_u10,  std::cosh )
NEON_MATHD(  vtanhq_f64,   vvtanh,   Sleef_tanhd2_u10,  std::tanh )
NEON_MATHD(  vasinhq_f64,  vvasinh,  Sleef_asinhd2_u10, std::asinh)
NEON_MATHD(  vacoshq_f64,  vvacosh,  Sleef_acoshd2_u10, std::acosh)
NEON_MATHD(  vatanhq_f64,  vvatanh,  Sleef_atanhd2_u10, std::atanh)
NEON_MATHD(  vcbrtq_f64,   vvcbrt,   Sleef_cbrtd2_u10,  std::cbrt )
NEON_MATHD2( vpowq_f64,    vvpow,    Sleef_powd2_u10               )

// exp10: no native Accelerate equivalent — use SLEEF or scalar
VECTORIZATION_FORCE_INLINE float64x2_t vexp10q_f64(float64x2_t x)
{
#if VECTORIZATION_HAS_SLEEF
    return Sleef_exp10d2_u10(x);
#else
    alignas(16) double b[2]; vst1q_f64(b, x);
    b[0] = std::pow(10., b[0]); b[1] = std::pow(10., b[1]);
    return vld1q_f64(b);
#endif
}

#undef NEON_MATHF
#undef NEON_MATHF2
#undef NEON_MATHD
#undef NEON_MATHD2

// clang-format on

#endif  // VECTORIZATION_HAS_NEON || VECTORIZATION_HAS_SVE
