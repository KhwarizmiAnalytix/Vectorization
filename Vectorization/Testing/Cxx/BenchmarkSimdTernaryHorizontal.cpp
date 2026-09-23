/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * SIMD micro-benchmarks: fused multiply-add (three tensor inputs) and horizontal reductions.
 * Target: benchmark_simdternaryhorizontal
 */

#include <benchmark/benchmark.h>

#include <random>

#include "common/vectorization_macros.h"
#include "expressions/expressions.h"
#include "terminals/tensor.h"

#define BENCHMARK_DONOT_OPTIMIZE(X) benchmark::DoNotOptimize(X)

namespace std
{
template <typename T>
inline T accumulate(T a, double c)
{
    return a + c;
}
}  // namespace std

namespace vectorization
{
class func_fma
{
public:
    template <typename T>
    VECTORIZATION_FORCE_INLINE static void run(T const& a, T const& b, T const& d, T& c)
    {
        if constexpr (vectorization::is_fundamental<T>::value)
            BENCHMARK_DONOT_OPTIMIZE(c = std::fma(a, b, d));
        else
            BENCHMARK_DONOT_OPTIMIZE(c = ::fma(a, b, d));
    };
};
}  // namespace vectorization

#define SIMD_BENCHMARK_TERNARY(op)                                                   \
    template <typename scalar_t>                                                     \
    static void Vectorized_##op(benchmark::State& state)                             \
    {                                                                                \
        const size_t n = (2 << 16) + 3;                                              \
                                                                                     \
        vectorization::tensor<scalar_t> a(n);                                        \
        vectorization::tensor<scalar_t> b(n);                                        \
        vectorization::tensor<scalar_t> d(n);                                        \
        vectorization::tensor<scalar_t> c(n);                                        \
        std::default_random_engine      generator;                                   \
                                                                                     \
        std::uniform_real_distribution<scalar_t> dist(-5., 5.);                      \
                                                                                     \
        for (size_t i = 0; i < n; ++i)                                               \
        {                                                                            \
            a[i] = dist(generator);                                                  \
            b[i] = dist(generator);                                                  \
            d[i] = dist(generator);                                                  \
        }                                                                            \
                                                                                     \
        for (auto _ : state)                                                         \
            vectorization::func_##op::run(a, b, d, c);                               \
        state.SetItemsProcessed(                                                     \
            static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));     \
        state.SetBytesProcessed(                                                     \
            static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n) * 4 * \
            static_cast<int64_t>(sizeof(scalar_t)));                                 \
    }                                                                                \
    template <typename scalar_t>                                                     \
    static void Scalar_##op(benchmark::State& state)                                 \
    {                                                                                \
        const size_t n = (2 << 16) + 3;                                              \
                                                                                     \
        vectorization::tensor<scalar_t> a(n);                                        \
        vectorization::tensor<scalar_t> b(n);                                        \
        vectorization::tensor<scalar_t> d(n);                                        \
        vectorization::tensor<scalar_t> c(n);                                        \
        std::default_random_engine      generator;                                   \
                                                                                     \
        std::uniform_real_distribution<scalar_t> dist(-5., 5.);                      \
                                                                                     \
        for (size_t i = 0; i < n; ++i)                                               \
        {                                                                            \
            a[i] = dist(generator);                                                  \
            b[i] = dist(generator);                                                  \
            d[i] = dist(generator);                                                  \
        }                                                                            \
                                                                                     \
        for (auto _ : state)                                                         \
            for (size_t i = 0; i < n; ++i)                                           \
                vectorization::func_##op::run(a[i], b[i], d[i], c[i]);               \
        state.SetItemsProcessed(                                                     \
            static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n));     \
        state.SetBytesProcessed(                                                     \
            static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n) * 4 * \
            static_cast<int64_t>(sizeof(scalar_t)));                                 \
    }                                                                                \
    BENCHMARK_TEMPLATE(Vectorized_##op, float)->MeasureProcessCPUTime();             \
    BENCHMARK_TEMPLATE(Scalar_##op, float)->MeasureProcessCPUTime();                 \
    BENCHMARK_TEMPLATE(Vectorized_##op, double)->MeasureProcessCPUTime();            \
    BENCHMARK_TEMPLATE(Scalar_##op, double)->MeasureProcessCPUTime();

SIMD_BENCHMARK_TERNARY(fma);

#define SIMDHORIZANTALBENCHMARK(op1, op2)                                        \
    template <typename scalar_t>                                                 \
    static void Vectorized_h##op1(benchmark::State& state)                       \
    {                                                                            \
        const size_t n = (2 << 12) + 3;                                          \
                                                                                 \
        vectorization::tensor<scalar_t> a(n);                                    \
        std::default_random_engine      generator;                               \
                                                                                 \
        std::uniform_real_distribution<scalar_t> distribution(-5., 5.);          \
                                                                                 \
        for (size_t i = 0; i < n; ++i)                                           \
        {                                                                        \
            auto tmp_x = distribution(generator);                                \
            a[i]       = tmp_x;                                                  \
        }                                                                        \
                                                                                 \
        for (auto _ : state)                                                     \
            BENCHMARK_DONOT_OPTIMIZE(vectorization::op1(a));                     \
        state.SetItemsProcessed(                                                 \
            static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n)); \
        state.SetBytesProcessed(                                                 \
            static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n) * \
            static_cast<int64_t>(sizeof(scalar_t)));                             \
    }                                                                            \
    template <typename scalar_t>                                                 \
    static void Scalar_h##op2(benchmark::State& state)                           \
    {                                                                            \
        const size_t                    n = (2 << 12) + 3;                       \
        vectorization::tensor<scalar_t> a(n);                                    \
        std::default_random_engine      generator;                               \
                                                                                 \
        std::uniform_real_distribution<scalar_t> distribution(-5., 5.);          \
                                                                                 \
        for (size_t i = 0; i < n; ++i)                                           \
        {                                                                        \
            auto tmp_x = distribution(generator);                                \
            a[i]       = tmp_x;                                                  \
        }                                                                        \
                                                                                 \
        for (auto _ : state)                                                     \
        {                                                                        \
            scalar_t c = 0;                                                      \
            for (size_t i = 0; i < n; ++i)                                       \
                BENCHMARK_DONOT_OPTIMIZE(c = std::op2(a[i], c));                 \
        }                                                                        \
        state.SetItemsProcessed(                                                 \
            static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n)); \
        state.SetBytesProcessed(                                                 \
            static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(n) * \
            static_cast<int64_t>(sizeof(scalar_t)));                             \
    }                                                                            \
    BENCHMARK_TEMPLATE(Vectorized_h##op1, float)->MeasureProcessCPUTime();       \
    BENCHMARK_TEMPLATE(Scalar_h##op2, float)->MeasureProcessCPUTime();           \
    BENCHMARK_TEMPLATE(Vectorized_h##op1, double)->MeasureProcessCPUTime();      \
    BENCHMARK_TEMPLATE(Scalar_h##op2, double)->MeasureProcessCPUTime();

SIMDHORIZANTALBENCHMARK(hmin, min);
SIMDHORIZANTALBENCHMARK(hmax, max);
SIMDHORIZANTALBENCHMARK(accumulate, accumulate);

BENCHMARK_MAIN();
