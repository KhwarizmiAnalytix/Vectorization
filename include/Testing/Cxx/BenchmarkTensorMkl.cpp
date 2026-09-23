/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * Tensor throughput benchmarks: XSigma tensor<T> vs Intel MKL VML.
 *
 * Measures elements/sec for common unary element-wise ops.
 * Allocation is excluded from the hot loop (pre-allocated buffers).
 *
 * Each benchmark pair is named:
 *   XSigma_<Op><Type>  — SIMD expression-template path
 *   MklVml_<Op><Type>  — direct MKL VML batch call
 *
 * Guard: compiled only when VECTORIZATION_HAS_MKL=1.
 *
 * Target: benchmark_tensormkl
 */

#include <benchmark/benchmark.h>

#if VECTORIZATION_HAS_MKL

#include <cstddef>
#include <random>

#include "backend/cpu/mkl/mkl_vml.h"
#include "terminals/tensor.h"

// ── shared constants & helpers ────────────────────────────────────────────────

namespace
{

// ~131k elements: exceeds L2 on most CPUs, not a multiple of any common SIMD width.
constexpr std::size_t kN = (2u << 16) + 3;

template <typename T>
static void fill_uniform(T* p, std::size_t n, T lo, T hi, unsigned seed)
{
    std::mt19937                      gen(seed);
    std::uniform_real_distribution<T> dist(lo, hi);
    for (std::size_t i = 0; i < n; ++i)
        p[i] = dist(gen);
}

}  // namespace

// ── benchmark macro ───────────────────────────────────────────────────────────
//
// BENCH_UNARY_MKL(NAME, XS_OP, MKL_FN, LO, HI)
//   XS_OP  — called as XS_OP(a)  on a tensor
//   MKL_FN — called as mkl_vml::MKL_FN(n, a, y) on raw pointers
//
// ─────────────────────────────────────────────────────────────────────────────

#define BENCH_UNARY_MKL(NAME, XS_OP, MKL_FN, LO, HI)                                        \
    template <typename T>                                                                   \
    static void XSigma_##NAME(benchmark::State& state)                                      \
    {                                                                                       \
        vectorization::tensor<T> a(kN), out(kN);                                            \
        fill_uniform(a.data(), kN, static_cast<T>(LO), static_cast<T>(HI), 42u);            \
        for (auto _ : state)                                                                \
            benchmark::DoNotOptimize(out = XS_OP(a));                                       \
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(kN));             \
    }                                                                                       \
    template <typename T>                                                                   \
    static void MklVml_##NAME(benchmark::State& state)                                      \
    {                                                                                       \
        std::vector<T> a(kN), out(kN);                                                      \
        fill_uniform(a.data(), kN, static_cast<T>(LO), static_cast<T>(HI), 42u);            \
        for (auto _ : state)                                                                \
        {                                                                                   \
            vectorization::mkl_vml::MKL_FN(static_cast<MKL_INT>(kN), a.data(), out.data()); \
            benchmark::DoNotOptimize(out.data());                                           \
        }                                                                                   \
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(kN));             \
    }                                                                                       \
    BENCHMARK_TEMPLATE(XSigma_##NAME, float)                                                \
        ->MeasureProcessCPUTime()                                                           \
        ->Unit(benchmark::kMicrosecond);                                                    \
    BENCHMARK_TEMPLATE(MklVml_##NAME, float)                                                \
        ->MeasureProcessCPUTime()                                                           \
        ->Unit(benchmark::kMicrosecond);                                                    \
    BENCHMARK_TEMPLATE(XSigma_##NAME, double)                                               \
        ->MeasureProcessCPUTime()                                                           \
        ->Unit(benchmark::kMicrosecond);                                                    \
    BENCHMARK_TEMPLATE(MklVml_##NAME, double)                                               \
        ->MeasureProcessCPUTime()                                                           \
        ->Unit(benchmark::kMicrosecond);

// ── unary transcendentals ─────────────────────────────────────────────────────

BENCH_UNARY_MKL(Exp, ::exp, exp, -1, 1)
BENCH_UNARY_MKL(Log, ::log, log, 0.1, 4)
BENCH_UNARY_MKL(Sqrt, ::sqrt, sqrt, 0.1, 4)
BENCH_UNARY_MKL(Sin, ::sin, sin, -4, 4)
BENCH_UNARY_MKL(Cos, ::cos, cos, -4, 4)
BENCH_UNARY_MKL(Tanh, ::tanh, tanh, -4, 4)

#endif  // VECTORIZATION_HAS_MKL

BENCHMARK_MAIN();
