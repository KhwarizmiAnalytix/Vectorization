/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * SIMD micro-benchmarks: composite two-input kernels (sum, mixed_formula, mixed_if_else).
 * Target: benchmark_simdbinaryop
 */

#include <benchmark/benchmark.h>

#include <cmath>
#include <limits>
#include <random>

#include "common/vectorization_macros.h"
#include "expressions/expressions.h"
#include "terminals/tensor.h"

#define BENCHMARK_DONOT_OPTIMIZE(X) benchmark::DoNotOptimize(X)

namespace vectorization
{
class func_sum
{
public:
    template <typename T>
    VECTORIZATION_FORCE_INLINE static void run(T const& x, T const& y, T& z)
    {
        BENCHMARK_DONOT_OPTIMIZE(z = x * y - x + x * y - x);
    };
};

class func_mixed_if_else
{
public:
    template <typename T>
    VECTORIZATION_FORCE_INLINE static void run(T const& a, T const& b, T& c)
    {
        double dcf = 0.25;
        if constexpr (std::is_arithmetic<T>::value)
        {
            auto x = -dcf * fabs(-a + b);
            c      = a * dcf *
                ((fabs(x) < std::numeric_limits<double>::epsilon()) ? 1. - 0.5 * x : expm1(x) / x);
        }
        else
        {
            c      = 0.;
            auto x = -dcf * fabs(-a + b);
            c += a * dcf *
                 if_else(
                     fabs(x) < std::numeric_limits<double>::epsilon(), 1. - 0.5 * x, expm1(x) / x);
        }
    };
};
}  // namespace vectorization

#define SIMD_BENCHMARK(op)                                                           \
    template <typename scalar_t>                                                     \
    static void Vectorized_##op(benchmark::State& state)                             \
    {                                                                                \
        const size_t n = (2 << 16) + 3;                                              \
                                                                                     \
        vectorization::tensor<scalar_t> a(n);                                        \
        vectorization::tensor<scalar_t> b(n);                                        \
        vectorization::tensor<scalar_t> c(n);                                        \
        std::default_random_engine      generator;                                   \
                                                                                     \
        std::uniform_real_distribution<scalar_t> distribution(-5., 5.);              \
                                                                                     \
        for (size_t i = 0; i < n; ++i)                                               \
        {                                                                            \
            auto tmp_x = distribution(generator);                                    \
            auto tmp_y = distribution(generator);                                    \
            a[i]       = tmp_x;                                                      \
            b[i]       = tmp_y;                                                      \
        }                                                                            \
                                                                                     \
        for (auto _ : state)                                                         \
            vectorization::func_##op::run(a, b, c);                                  \
        state.SetItemsProcessed(                                                     \
            static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));     \
        state.SetBytesProcessed(                                                     \
            static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n) * 3 * \
            static_cast<int64_t>(sizeof(scalar_t)));                                 \
    }                                                                                \
    template <typename scalar_t>                                                     \
    static void Scalar_##op(benchmark::State& state)                                 \
    {                                                                                \
        const size_t n = (2 << 16) + 3;                                              \
                                                                                     \
        vectorization::tensor<scalar_t> a(n);                                        \
        vectorization::tensor<scalar_t> b(n);                                        \
        vectorization::tensor<scalar_t> c(n);                                        \
        std::default_random_engine      generator;                                   \
                                                                                     \
        std::uniform_real_distribution<scalar_t> distribution(-5., 5.);              \
                                                                                     \
        for (size_t i = 0; i < n; ++i)                                               \
        {                                                                            \
            auto tmp_x = distribution(generator);                                    \
            auto tmp_y = distribution(generator);                                    \
            a[i]       = tmp_x;                                                      \
            b[i]       = tmp_y;                                                      \
        }                                                                            \
                                                                                     \
        for (auto _ : state)                                                         \
            for (size_t i = 0; i < n; ++i)                                           \
                vectorization::func_##op::run(a[i], b[i], c[i]);                     \
        state.SetItemsProcessed(                                                     \
            static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));     \
        state.SetBytesProcessed(                                                     \
            static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n) * 3 * \
            static_cast<int64_t>(sizeof(scalar_t)));                                 \
    }                                                                                \
    BENCHMARK_TEMPLATE(Vectorized_##op, float)->MeasureProcessCPUTime();             \
    BENCHMARK_TEMPLATE(Scalar_##op, float)->MeasureProcessCPUTime();                 \
    BENCHMARK_TEMPLATE(Vectorized_##op, double)->MeasureProcessCPUTime();            \
    BENCHMARK_TEMPLATE(Scalar_##op, double)->MeasureProcessCPUTime();

SIMD_BENCHMARK(sum);
SIMD_BENCHMARK(mixed_if_else);

BENCHMARK_MAIN();
