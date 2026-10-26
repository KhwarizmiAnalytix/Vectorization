// SPDX-License-Identifier: GPL-3.0-or-later OR Commercial

#pragma once

// Test-only VML references. Enabling MKL comparisons does not change Scalars'
// math backend or the public Vectorization target.
#if VECTORIZATION_HAS_MKL
#include <mkl_vml.h>

namespace vectorization::mkl_reference
{
#define VECTORIZATION_MKL_REFERENCE(name, suffix)                                                  \
    inline void name(MKL_INT n, const float* input, float* output)                                 \
    {                                                                                              \
        vs##suffix(n, input, output);                                                              \
    }                                                                                              \
    inline void name(MKL_INT n, const double* input, double* output)                               \
    {                                                                                              \
        vd##suffix(n, input, output);                                                              \
    }

VECTORIZATION_MKL_REFERENCE(exp, Exp)
VECTORIZATION_MKL_REFERENCE(log, Ln)
VECTORIZATION_MKL_REFERENCE(sqrt, Sqrt)
VECTORIZATION_MKL_REFERENCE(sin, Sin)
VECTORIZATION_MKL_REFERENCE(cos, Cos)
VECTORIZATION_MKL_REFERENCE(tanh, Tanh)
VECTORIZATION_MKL_REFERENCE(fabs, Abs)

#undef VECTORIZATION_MKL_REFERENCE
}  // namespace vectorization::mkl_reference
#endif
