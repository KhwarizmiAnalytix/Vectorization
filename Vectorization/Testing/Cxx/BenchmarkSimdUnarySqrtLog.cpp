/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * Unary SIMD benchmarks: roots, exponentials, logarithms, norm CDF / inverse, rounding, fabs.
 *
 * Target: benchmark_simdunarysqrtlog
 */

#include "BenchmarkSimdUnaryShared.h"

UNARY_BENCH_PAIR(sqrt, 0.25, 4.0)
UNARY_BENCH_PAIR(sqr, -4.0, 4.0)
UNARY_BENCH_PAIR(invsqrt, 0.25, 4.0)
UNARY_BENCH_PAIR(exp, -4.0, 4.0)
UNARY_BENCH_PAIR(exp2, -4.0, 4.0)
UNARY_BENCH_PAIR(expm1, -0.75, 0.75)
UNARY_BENCH_PAIR(log, 0.05, 4.0)
UNARY_BENCH_PAIR(log1p, -0.75, 3.0)
UNARY_BENCH_PAIR(log2, 0.05, 4.0)

#undef UNARY_BENCH_PAIR

BENCHMARK_MAIN();
