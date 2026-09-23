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

// ---------------------------------------------------------------------------
// MKL VML batch-function wrappers
//
// Provides overloaded free functions in namespace vectorization::mkl_vml
// that dispatch to the correct vsXxx (float) or vdXxx (double) VML call.
// All functions follow the VML signature: (MKL_INT n, const T* a, T* y).
// Binary functions: (MKL_INT n, const T* a, const T* b, T* y).
//
// Only compiled when VECTORIZATION_HAS_MKL=1.
// ---------------------------------------------------------------------------

#if VECTORIZATION_HAS_MKL

#include <mkl_vml.h>

namespace vectorization::mkl_vml
{

// =============================================================================
// Unary — float (single precision)
// =============================================================================

inline void fabs(MKL_INT n, const float* __restrict__ a, float* __restrict__ y)
{
    vsAbs(n, a, y);
}
inline void floor(MKL_INT n, const float* __restrict__ a, float* __restrict__ y)
{
    vsFloor(n, a, y);
}
inline void ceil(MKL_INT n, const float* __restrict__ a, float* __restrict__ y)
{
    vsCeil(n, a, y);
}
inline void trunc(MKL_INT n, const float* __restrict__ a, float* __restrict__ y)
{
    vsTrunc(n, a, y);
}
inline void sqrt(MKL_INT n, const float* __restrict__ a, float* __restrict__ y)
{
    vsSqrt(n, a, y);
}
inline void sqr(MKL_INT n, const float* __restrict__ a, float* __restrict__ y)
{
    vsSqr(n, a, y);
}
inline void invsqrt(MKL_INT n, const float* __restrict__ a, float* __restrict__ y)
{
    vsInvSqrt(n, a, y);
}
inline void exp(MKL_INT n, const float* __restrict__ a, float* __restrict__ y)
{
    vsExp(n, a, y);
}
inline void expm1(MKL_INT n, const float* __restrict__ a, float* __restrict__ y)
{
    vsExpm1(n, a, y);
}
inline void exp2(MKL_INT n, const float* __restrict__ a, float* __restrict__ y)
{
    vsExp2(n, a, y);
}
inline void exp10(MKL_INT n, const float* __restrict__ a, float* __restrict__ y)
{
    vsExp10(n, a, y);
}
inline void log(MKL_INT n, const float* __restrict__ a, float* __restrict__ y)
{
    vsLn(n, a, y);
}
inline void log1p(MKL_INT n, const float* __restrict__ a, float* __restrict__ y)
{
    vsLog1p(n, a, y);
}
inline void log2(MKL_INT n, const float* __restrict__ a, float* __restrict__ y)
{
    vsLog2(n, a, y);
}
inline void log10(MKL_INT n, const float* __restrict__ a, float* __restrict__ y)
{
    vsLog10(n, a, y);
}
inline void sin(MKL_INT n, const float* __restrict__ a, float* __restrict__ y)
{
    vsSin(n, a, y);
}
inline void cos(MKL_INT n, const float* __restrict__ a, float* __restrict__ y)
{
    vsCos(n, a, y);
}
inline void tan(MKL_INT n, const float* __restrict__ a, float* __restrict__ y)
{
    vsTan(n, a, y);
}
inline void asin(MKL_INT n, const float* __restrict__ a, float* __restrict__ y)
{
    vsAsin(n, a, y);
}
inline void acos(MKL_INT n, const float* __restrict__ a, float* __restrict__ y)
{
    vsAcos(n, a, y);
}
inline void atan(MKL_INT n, const float* __restrict__ a, float* __restrict__ y)
{
    vsAtan(n, a, y);
}
inline void sinh(MKL_INT n, const float* __restrict__ a, float* __restrict__ y)
{
    vsSinh(n, a, y);
}
inline void cosh(MKL_INT n, const float* __restrict__ a, float* __restrict__ y)
{
    vsCosh(n, a, y);
}
inline void tanh(MKL_INT n, const float* __restrict__ a, float* __restrict__ y)
{
    vsTanh(n, a, y);
}
inline void asinh(MKL_INT n, const float* __restrict__ a, float* __restrict__ y)
{
    vsAsinh(n, a, y);
}
inline void acosh(MKL_INT n, const float* __restrict__ a, float* __restrict__ y)
{
    vsAcosh(n, a, y);
}
inline void atanh(MKL_INT n, const float* __restrict__ a, float* __restrict__ y)
{
    vsAtanh(n, a, y);
}
inline void cbrt(MKL_INT n, const float* __restrict__ a, float* __restrict__ y)
{
    vsCbrt(n, a, y);
}
inline void cdf(MKL_INT n, const float* __restrict__ a, float* __restrict__ y)
{
    vsCdfNorm(n, a, y);
}
inline void inv_cdf(MKL_INT n, const float* __restrict__ a, float* __restrict__ y)
{
    vsCdfNormInv(n, a, y);
}

// =============================================================================
// Unary — double (double precision)
// =============================================================================

inline void fabs(MKL_INT n, const double* __restrict__ a, double* __restrict__ y)
{
    vdAbs(n, a, y);
}
inline void floor(MKL_INT n, const double* __restrict__ a, double* __restrict__ y)
{
    vdFloor(n, a, y);
}
inline void ceil(MKL_INT n, const double* __restrict__ a, double* __restrict__ y)
{
    vdCeil(n, a, y);
}
inline void trunc(MKL_INT n, const double* __restrict__ a, double* __restrict__ y)
{
    vdTrunc(n, a, y);
}
inline void sqrt(MKL_INT n, const double* __restrict__ a, double* __restrict__ y)
{
    vdSqrt(n, a, y);
}
inline void sqr(MKL_INT n, const double* __restrict__ a, double* __restrict__ y)
{
    vdSqr(n, a, y);
}
inline void invsqrt(MKL_INT n, const double* __restrict__ a, double* __restrict__ y)
{
    vdInvSqrt(n, a, y);
}
inline void exp(MKL_INT n, const double* __restrict__ a, double* __restrict__ y)
{
    vdExp(n, a, y);
}
inline void expm1(MKL_INT n, const double* __restrict__ a, double* __restrict__ y)
{
    vdExpm1(n, a, y);
}
inline void exp2(MKL_INT n, const double* __restrict__ a, double* __restrict__ y)
{
    vdExp2(n, a, y);
}
inline void exp10(MKL_INT n, const double* __restrict__ a, double* __restrict__ y)
{
    vdExp10(n, a, y);
}
inline void log(MKL_INT n, const double* __restrict__ a, double* __restrict__ y)
{
    vdLn(n, a, y);
}
inline void log1p(MKL_INT n, const double* __restrict__ a, double* __restrict__ y)
{
    vdLog1p(n, a, y);
}
inline void log2(MKL_INT n, const double* __restrict__ a, double* __restrict__ y)
{
    vdLog2(n, a, y);
}
inline void log10(MKL_INT n, const double* __restrict__ a, double* __restrict__ y)
{
    vdLog10(n, a, y);
}
inline void sin(MKL_INT n, const double* __restrict__ a, double* __restrict__ y)
{
    vdSin(n, a, y);
}
inline void cos(MKL_INT n, const double* __restrict__ a, double* __restrict__ y)
{
    vdCos(n, a, y);
}
inline void tan(MKL_INT n, const double* __restrict__ a, double* __restrict__ y)
{
    vdTan(n, a, y);
}
inline void asin(MKL_INT n, const double* __restrict__ a, double* __restrict__ y)
{
    vdAsin(n, a, y);
}
inline void acos(MKL_INT n, const double* __restrict__ a, double* __restrict__ y)
{
    vdAcos(n, a, y);
}
inline void atan(MKL_INT n, const double* __restrict__ a, double* __restrict__ y)
{
    vdAtan(n, a, y);
}
inline void sinh(MKL_INT n, const double* __restrict__ a, double* __restrict__ y)
{
    vdSinh(n, a, y);
}
inline void cosh(MKL_INT n, const double* __restrict__ a, double* __restrict__ y)
{
    vdCosh(n, a, y);
}
inline void tanh(MKL_INT n, const double* __restrict__ a, double* __restrict__ y)
{
    vdTanh(n, a, y);
}
inline void asinh(MKL_INT n, const double* __restrict__ a, double* __restrict__ y)
{
    vdAsinh(n, a, y);
}
inline void acosh(MKL_INT n, const double* __restrict__ a, double* __restrict__ y)
{
    vdAcosh(n, a, y);
}
inline void atanh(MKL_INT n, const double* __restrict__ a, double* __restrict__ y)
{
    vdAtanh(n, a, y);
}
inline void cbrt(MKL_INT n, const double* __restrict__ a, double* __restrict__ y)
{
    vdCbrt(n, a, y);
}
inline void cdf(MKL_INT n, const double* __restrict__ a, double* __restrict__ y)
{
    vdCdfNorm(n, a, y);
}
inline void inv_cdf(MKL_INT n, const double* __restrict__ a, double* __restrict__ y)
{
    vdCdfNormInv(n, a, y);
}

// =============================================================================
// Binary — float
// =============================================================================

inline void add(
    MKL_INT n, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ y)
{
    vsAdd(n, a, b, y);
}
inline void sub(
    MKL_INT n, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ y)
{
    vsSub(n, a, b, y);
}
inline void mul(
    MKL_INT n, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ y)
{
    vsMul(n, a, b, y);
}
inline void div(
    MKL_INT n, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ y)
{
    vsDiv(n, a, b, y);
}
inline void max(
    MKL_INT n, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ y)
{
    vsFmax(n, a, b, y);
}
inline void min(
    MKL_INT n, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ y)
{
    vsFmin(n, a, b, y);
}
inline void pow(
    MKL_INT n, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ y)
{
    vsPow(n, a, b, y);
}
inline void hypot(
    MKL_INT n, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ y)
{
    vsHypot(n, a, b, y);
}

// =============================================================================
// Binary — double
// =============================================================================

inline void add(
    MKL_INT n, const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ y)
{
    vdAdd(n, a, b, y);
}
inline void sub(
    MKL_INT n, const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ y)
{
    vdSub(n, a, b, y);
}
inline void mul(
    MKL_INT n, const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ y)
{
    vdMul(n, a, b, y);
}
inline void div(
    MKL_INT n, const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ y)
{
    vdDiv(n, a, b, y);
}
inline void max(
    MKL_INT n, const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ y)
{
    vdFmax(n, a, b, y);
}
inline void min(
    MKL_INT n, const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ y)
{
    vdFmin(n, a, b, y);
}
inline void pow(
    MKL_INT n, const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ y)
{
    vdPow(n, a, b, y);
}
inline void hypot(
    MKL_INT n, const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ y)
{
    vdHypot(n, a, b, y);
}

}  // namespace vectorization::mkl_vml

#endif  // VECTORIZATION_HAS_MKL
