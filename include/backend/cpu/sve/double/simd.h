/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * AArch64 SVE backend (fixed 128-bit vectors). See backend/sve/float/simd.h.
 */

#pragma once

#include <arm_sve.h>

#if !defined(__ARM_FEATURE_SVE_BITS)
#error \
    "VECTORIZATION_HAS_SVE: compile with SVE enabled and a fixed vector length, e.g. \
-march=armv8-a+sve (or armv9-a) and -msve-vector-bits=128"
#endif

#if __ARM_FEATURE_SVE_BITS != 128
#error \
    "VECTORIZATION_HAS_SVE: only -msve-vector-bits=128 is supported; use vectorization_type=neon for variable-length or wider builds"
#endif

#include "../../neon/double/simd.h"
