/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * Unary SIMD benchmarks: hyperbolic functions.
 *
 * Target: benchmark_simdunaryhyperbolic
 */

#include "BenchmarkSimdUnaryShared.h"

UNARY_BENCH_PAIR(sinh, -3.0, 3.0)
UNARY_BENCH_PAIR(cosh, -3.0, 3.0)
UNARY_BENCH_PAIR(tanh, -3.0, 3.0)
UNARY_BENCH_PAIR(asinh, -3.0, 3.0)
UNARY_BENCH_PAIR(acosh, 1.01, 4.0)
UNARY_BENCH_PAIR(atanh, -0.9, 0.9)

#undef UNARY_BENCH_PAIR

BENCHMARK_MAIN();
