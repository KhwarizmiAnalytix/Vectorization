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

#include <benchmark/benchmark.h>

#include "terminals/tensor.h"

// Preallocated input, four cache/streaming sizes, and a fused expression.
// Include inputs above 2^24 so integer counting and the bounded GPU grid can
// be profiled with the same workload (GPU registrations live in BenchmarkTensorGpu).
#define REDUCTION_BENCHMARK(NAME)                                                                  \
    template <typename T> void CPU_Reduce_##NAME(benchmark::State& state)                          \
    {                                                                                              \
        const auto               n = static_cast<size_t>(state.range(0));                          \
        vectorization::tensor<T> x(n);                                                             \
        for (size_t i = 0; i < n; ++i)                                                             \
        {                                                                                          \
            x[i] = i % 7 == 0 ? T(0) : T(1);                                                       \
        }                                                                                          \
        for (auto _ : state)                                                                       \
        {                                                                                          \
            benchmark::DoNotOptimize(vectorization::NAME(x * T(0.5) + T(0.5)));                    \
        }                                                                                          \
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));                     \
        state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(n * sizeof(T)));         \
    }                                                                                              \
    BENCHMARK_TEMPLATE(CPU_Reduce_##NAME, float)                                                   \
        ->Arg(257)                                                                                 \
        ->Arg(4096)                                                                                \
        ->Arg(1 << 20)                                                                             \
        ->Arg((1 << 24) + 3);                                                                      \
    BENCHMARK_TEMPLATE(CPU_Reduce_##NAME, double)                                                  \
        ->Arg(257)                                                                                 \
        ->Arg(4096)                                                                                \
        ->Arg(1 << 20)                                                                             \
        ->Arg((1 << 24) + 3);

REDUCTION_BENCHMARK(sum)
REDUCTION_BENCHMARK(prod)
REDUCTION_BENCHMARK(mean)
REDUCTION_BENCHMARK(all)
REDUCTION_BENCHMARK(any)
REDUCTION_BENCHMARK(count_nonzero)
REDUCTION_BENCHMARK(nansum)
REDUCTION_BENCHMARK(nanmean)
#undef REDUCTION_BENCHMARK
