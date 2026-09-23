/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * Shared helpers for BenchmarkSimdUnary*.cpp: vectorized tensor ops (::op) vs scalar loops.
 * Scalar path uses <cmath> where defined, plus std::sqr / std::invsqrt / std::cdf / std::inv_cdf
 * from scalar_helper_functions.h (same reference as expression templates).
 */

#pragma once

#include <benchmark/benchmark.h>

#include <cstddef>
#include <random>

#include "common/vectorization_macros.h"
#include "expressions/expressions.h"
#include "terminals/tensor.h"

template <typename T>
void fill_tensor_uniform(vectorization::tensor<T>& a, double lo, double hi, unsigned seed = 5489u)
{
    std::mt19937                           gen(seed);
    std::uniform_real_distribution<double> dist(lo, hi);
    for (std::size_t i = 0; i < a.size(); ++i)
        a[i] = static_cast<T>(dist(gen));
}

// Vectorized: one expression evaluation over the full buffer.
// Scalar: same buffer, element-wise std::op (reference loop).
#define UNARY_BENCH_PAIR(OP, lo, hi)                                                 \
    template <typename T>                                                            \
    static void Vectorized_##OP(benchmark::State& state)                             \
    {                                                                                \
        constexpr std::size_t    n = (2u << 16) + 3;                                 \
        vectorization::tensor<T> a(n);                                               \
        vectorization::tensor<T> out(n);                                             \
        fill_tensor_uniform(a, (lo), (hi));                                          \
        for (auto _ : state)                                                         \
            benchmark::DoNotOptimize(out = ::OP(a));                                 \
        state.SetItemsProcessed(                                                     \
            static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));     \
        state.SetBytesProcessed(                                                     \
            static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n) * 2 * \
            static_cast<int64_t>(sizeof(T)));                                        \
    }                                                                                \
    template <typename T>                                                            \
    static void Scalar_loop_##OP(benchmark::State& state)                            \
    {                                                                                \
        constexpr std::size_t    n = (2u << 16) + 3;                                 \
        vectorization::tensor<T> a(n);                                               \
        vectorization::tensor<T> out(n);                                             \
        fill_tensor_uniform(a, (lo), (hi));                                          \
        for (auto _ : state)                                                         \
        {                                                                            \
            for (std::size_t i = 0; i < n; ++i)                                      \
                benchmark::DoNotOptimize(out[i] = std::OP(a[i]));                    \
        }                                                                            \
        state.SetItemsProcessed(                                                     \
            static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));     \
        state.SetBytesProcessed(                                                     \
            static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n) * 2 * \
            static_cast<int64_t>(sizeof(T)));                                        \
    }                                                                                \
    BENCHMARK_TEMPLATE(Vectorized_##OP, float)->MeasureProcessCPUTime();             \
    BENCHMARK_TEMPLATE(Scalar_loop_##OP, float)->MeasureProcessCPUTime();            \
    BENCHMARK_TEMPLATE(Vectorized_##OP, double)->MeasureProcessCPUTime();            \
    BENCHMARK_TEMPLATE(Scalar_loop_##OP, double)->MeasureProcessCPUTime();
