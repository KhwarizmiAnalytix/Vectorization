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
#include <cstdint>

#include "common/vectorization_macros.h"

#if VECTORIZATION_VECTORIZED
#if defined(_MSC_VER) && (_MSC_VER >= 1400)
/* Microsoft C/C++-compatible compiler */
#include <intrin.h>
#if _MSC_VER <= 1900
#define _mm256_extract_epi64(X, Y) (_mm_extract_epi64(_mm256_extractf128_si256(X, Y >> 1), Y % 2))
#define _mm256_extract_epi32(X, Y) (_mm_extract_epi32(_mm256_extractf128_si256(X, Y >> 2), Y % 4))
#define _mm256_extract_epi16(X, Y) (_mm_extract_epi16(_mm256_extractf128_si256(X, Y >> 3), Y % 8))
#define _mm256_extract_epi8(X, Y) (_mm_extract_epi8(_mm256_extractf128_si256(X, Y >> 4), Y % 16))
#endif
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
/* GCC-compatible compiler, targeting x86/x86-64 */
#include <x86intrin.h>
#elif defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && VECTORIZATION_HAS_SVE
#include <arm_sve.h>
#elif defined(__GNUC__) && defined(__ARM_NEON__)
/* GCC-compatible compiler, targeting ARM with NEON */
#include <arm_neon.h>
#elif defined(__GNUC__) && defined(__IWMMXT__)
/* GCC-compatible compiler, targeting ARM with WMMX */
#include <mmintrin.h>
#elif (defined(__GNUC__) || defined(__xlC__)) && (defined(__VEC__) || defined(__ALTIVEC__))
/* XLC or GCC-compatible compiler, targeting PowerPC with VMX/VSX */
#include <altivec.h>
#elif defined(__GNUC__) && defined(__SPE__)
/* GCC-compatible compiler, targeting PowerPC with SPE */
#include <spe.h>
#endif

#if __VECTORIZATION_COMPILER_PGI__
using PREFETCH_PTR_TYPE = const void*;
#else
using PREFETCH_PTR_TYPE = const char*;
#endif
#endif

template <typename T>
struct simd
{
};

// Backend selection strategy:
//
//  GPU device pass (nvcc/hipcc compiling __device__ code):
//    Always use the scalar GPU backend so AVX intrinsics (unavailable on device)
//    are never included in device translation units.
//
//  CPU host pass:
//    Use the selected CPU SIMD backend when VECTORIZATION_VECTORIZED=1.
//    Fall back to the GPU scalar backend as a host-side shim when CPU_TYPE=no
//    but a GPU backend is active (keeps simd<T> well-defined for tensor members).

#if VECTORIZATION_ON_GPU_DEVICE
// --- GPU device code: scalar simd<T> ---
// Shared by CUDA and HIP (identical scalar semantics); VECTORIZATION_ON_GPU_DEVICE
// being set already guarantees nvcc or hipcc is doing the device pass.
#include "backend/gpu/double/simd.h"
#include "backend/gpu/float/simd.h"

#else
// --- CPU host code: SIMD or scalar fallback ---
#if VECTORIZATION_HAS_AVX512
#include "backend/cpu/avx512/double/simd.h"
#include "backend/cpu/avx512/float/simd.h"
#elif VECTORIZATION_HAS_AVX2 || VECTORIZATION_HAS_AVX
#include "backend/cpu/avx/double/simd.h"
#include "backend/cpu/avx/float/simd.h"
#elif VECTORIZATION_HAS_SSE
#include "backend/cpu/sse/double/simd.h"
#include "backend/cpu/sse/float/simd.h"
#elif VECTORIZATION_HAS_SVE
#include "backend/cpu/sve/double/simd.h"
#include "backend/cpu/sve/float/simd.h"
#elif VECTORIZATION_HAS_NEON
#include "backend/cpu/neon/double/simd.h"
#include "backend/cpu/neon/float/simd.h"
#endif

#endif  // VECTORIZATION_ON_GPU_DEVICE
