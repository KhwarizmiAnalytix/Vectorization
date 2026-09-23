/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * Unary SIMD benchmarks: trigonometric functions.
 *
 * Target: benchmark_simdunarytrig
 */

#include "BenchmarkSimdUnaryShared.h"

UNARY_BENCH_PAIR(sin, -3.0, 3.0)
UNARY_BENCH_PAIR(cos, -3.0, 3.0)
UNARY_BENCH_PAIR(tan, -1.2, 1.2)
UNARY_BENCH_PAIR(asin, -0.99, 0.99)
UNARY_BENCH_PAIR(acos, -0.99, 0.99)
UNARY_BENCH_PAIR(atan, -3.0, 3.0)

#undef UNARY_BENCH_PAIR

BENCHMARK_MAIN();
