/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * Unary SIMD benchmarks: cbrt and neg.
 *
 * Target: benchmark_simdunarycbrtneg
 */

#include "BenchmarkSimdUnaryShared.h"

UNARY_BENCH_PAIR(cbrt, -8.0, 8.0)
UNARY_BENCH_PAIR(cdf, -5.0, 5.0)
UNARY_BENCH_PAIR(inv_cdf, 0.02, 0.98)
UNARY_BENCH_PAIR(ceil, -5.0, 5.0)
UNARY_BENCH_PAIR(floor, -5.0, 5.0)
UNARY_BENCH_PAIR(trunc, -5.0, 5.0)
UNARY_BENCH_PAIR(fabs, -5.0, 5.0)
UNARY_BENCH_PAIR(neg, -5.0, 5.0)

#undef UNARY_BENCH_PAIR

BENCHMARK_MAIN();
