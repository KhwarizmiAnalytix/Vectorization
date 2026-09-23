/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * Tensor throughput benchmarks: XSigma tensor<T> vs LibTorch torch::Tensor.
 *
 * Measures elements/sec for common element-wise ops on pre-allocated outputs
 * so that allocation overhead is excluded from the hot loop.
 *
 * Each benchmark pair is named:
 *   XSigma_<Op><Type>   — our SIMD expression-template path
 *   LibTorch_<Op><Type> — equivalent LibTorch _out kernel
 *
 * Guard: compiled only when VECTORIZATION_HAS_LIBTORCH=1 (CMake finds LibTorch under
 *        VECTORIZATION_LIBTORCH_ROOT / ThirdParty/libtorch).
 *
 * Target: benchmark_tensorlibtorch
 */

#include <benchmark/benchmark.h>

#include <cstddef>
#include <random>

#include "terminals/tensor.h"

#if VECTORIZATION_HAS_LIBTORCH

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996 4100)
#endif
#include <torch/torch.h>
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif

#endif  // VECTORIZATION_HAS_LIBTORCH

// ── shared constants & helpers ────────────────────────────────────────────────

namespace
{

// ~131k elements: large enough to exceed L2 on most CPUs, not a multiple of any
// common SIMD width — exercises both the vectorised and scalar tail paths.
constexpr std::size_t kN = (2u << 16) + 3;

template <typename T>
static void fill_uniform(T* p, std::size_t n, T lo, T hi, unsigned seed)
{
    std::mt19937                      gen(seed);
    std::uniform_real_distribution<T> dist(lo, hi);
    for (std::size_t i = 0; i < n; ++i)
        p[i] = dist(gen);
}

#if VECTORIZATION_HAS_LIBTORCH

template <typename T>
struct TorchDtype
{
    static constexpr auto value = torch::kFloat32;
};
template <>
struct TorchDtype<double>
{
    static constexpr auto value = torch::kFloat64;
};

// Create an owned LibTorch CPU tensor from a range.
template <typename T>
static torch::Tensor make_th(T lo, T hi, unsigned seed = 42u)
{
    std::vector<T> buf(kN);
    fill_uniform(buf.data(), kN, lo, hi, seed);
    auto opts = torch::TensorOptions().dtype(TorchDtype<T>::value).device(torch::kCPU);
    return torch::from_blob(buf.data(), {static_cast<int64_t>(kN)}, opts).clone();
}

#endif  // VECTORIZATION_HAS_LIBTORCH

}  // namespace

// ── benchmark macros ──────────────────────────────────────────────────────────
//
// BENCH_UNARY(NAME, XS_OP, TH_OP, LO, HI)
//   XS_OP  — called as XS_OP(a)        e.g. ::exp
//   TH_OP  — called as torch::TH_OP_out(out, a)  e.g. exp
//
// BENCH_BINARY(NAME, XS_EXPR, TH_OP, LO, HI)
//   XS_EXPR — expression using a, b    e.g. a + b
//   TH_OP   — called as torch::TH_OP_out(out, a, b)  e.g. add
// ─────────────────────────────────────────────────────────────────────────────

#define BENCH_UNARY_XS(NAME, XS_OP, LO, HI)                                      \
    template <typename T>                                                        \
    static void XSigma_##NAME(benchmark::State& state)                           \
    {                                                                            \
        vectorization::tensor<T> a(kN), out(kN);                                 \
        fill_uniform(a.data(), kN, static_cast<T>(LO), static_cast<T>(HI), 42u); \
        for (auto _ : state)                                                     \
            benchmark::DoNotOptimize(out = XS_OP(a));                            \
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(kN));  \
    }                                                                            \
    BENCHMARK_TEMPLATE(XSigma_##NAME, float)                                     \
        ->MeasureProcessCPUTime()                                                \
        ->Unit(benchmark::kMicrosecond);                                         \
    BENCHMARK_TEMPLATE(XSigma_##NAME, double)                                    \
        ->MeasureProcessCPUTime()                                                \
        ->Unit(benchmark::kMicrosecond);

#if VECTORIZATION_HAS_LIBTORCH
#define BENCH_UNARY_TORCH(NAME, TH_OP, LO, HI)                                  \
    template <typename T>                                                       \
    static void LibTorch_##NAME(benchmark::State& state)                        \
    {                                                                           \
        auto ta    = make_th<T>(static_cast<T>(LO), static_cast<T>(HI));        \
        auto out_t = torch::empty_like(ta);                                     \
        for (auto _ : state)                                                    \
            benchmark::DoNotOptimize(torch::TH_OP##_out(out_t, ta));            \
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(kN)); \
    }                                                                           \
    BENCHMARK_TEMPLATE(LibTorch_##NAME, float)                                  \
        ->MeasureProcessCPUTime()                                               \
        ->Unit(benchmark::kMicrosecond);                                        \
    BENCHMARK_TEMPLATE(LibTorch_##NAME, double)                                 \
        ->MeasureProcessCPUTime()                                               \
        ->Unit(benchmark::kMicrosecond);
#else
#define BENCH_UNARY_TORCH(NAME, TH_OP, LO, HI)
#endif

#define BENCH_UNARY(NAME, XS_OP, TH_OP, LO, HI) \
    BENCH_UNARY_XS(NAME, XS_OP, LO, HI)         \
    BENCH_UNARY_TORCH(NAME, TH_OP, LO, HI)

#define BENCH_BINARY_XS(NAME, XS_EXPR, LO, HI)                                  \
    template <typename T>                                                       \
    static void XSigma_##NAME(benchmark::State& state)                          \
    {                                                                           \
        vectorization::tensor<T> a(kN), b(kN), out(kN);                         \
        fill_uniform(a.data(), kN, static_cast<T>(LO), static_cast<T>(HI), 1u); \
        fill_uniform(b.data(), kN, static_cast<T>(LO), static_cast<T>(HI), 2u); \
        for (auto _ : state)                                                    \
            benchmark::DoNotOptimize(out = (XS_EXPR));                          \
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(kN)); \
    }                                                                           \
    BENCHMARK_TEMPLATE(XSigma_##NAME, float)                                    \
        ->MeasureProcessCPUTime()                                               \
        ->Unit(benchmark::kMicrosecond);                                        \
    BENCHMARK_TEMPLATE(XSigma_##NAME, double)                                   \
        ->MeasureProcessCPUTime()                                               \
        ->Unit(benchmark::kMicrosecond);

#if VECTORIZATION_HAS_LIBTORCH
#define BENCH_BINARY_TORCH(NAME, TH_OP, LO, HI)                                 \
    template <typename T>                                                       \
    static void LibTorch_##NAME(benchmark::State& state)                        \
    {                                                                           \
        auto ta    = make_th<T>(static_cast<T>(LO), static_cast<T>(HI), 1u);    \
        auto tb    = make_th<T>(static_cast<T>(LO), static_cast<T>(HI), 2u);    \
        auto out_t = torch::empty_like(ta);                                     \
        for (auto _ : state)                                                    \
            benchmark::DoNotOptimize(torch::TH_OP##_out(out_t, ta, tb));        \
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(kN)); \
    }                                                                           \
    BENCHMARK_TEMPLATE(LibTorch_##NAME, float)                                  \
        ->MeasureProcessCPUTime()                                               \
        ->Unit(benchmark::kMicrosecond);                                        \
    BENCHMARK_TEMPLATE(LibTorch_##NAME, double)                                 \
        ->MeasureProcessCPUTime()                                               \
        ->Unit(benchmark::kMicrosecond);
#else
#define BENCH_BINARY_TORCH(NAME, TH_OP, LO, HI)
#endif

#define BENCH_BINARY(NAME, XS_EXPR, TH_OP, LO, HI) \
    BENCH_BINARY_XS(NAME, XS_EXPR, LO, HI)         \
    BENCH_BINARY_TORCH(NAME, TH_OP, LO, HI)

// ── binary ops ────────────────────────────────────────────────────────────────

BENCH_BINARY(Add, a + b, add, -4, 4)
BENCH_BINARY(Sub, a - b, sub, -4, 4)
BENCH_BINARY(Mul, a* b, mul, -4, 4)

// ── unary transcendentals ─────────────────────────────────────────────────────

BENCH_UNARY(Exp, ::exp, exp, -1, 1)      // restricted range to avoid overflow
BENCH_UNARY(Log, ::log, log, 0.1, 4)     // positive domain
BENCH_UNARY(Sqrt, ::sqrt, sqrt, 0.1, 4)  // positive domain
BENCH_UNARY(Sin, ::sin, sin, -4, 4)
BENCH_UNARY(Cos, ::cos, cos, -4, 4)
BENCH_UNARY(Tanh, ::tanh, tanh, -4, 4)

// ── fused multiply-add: a * b + c ────────────────────────────────────────────
// LibTorch equivalent: addcmul(c, a, b) = c + 1 * a * b

template <typename T>
static void XSigma_Fma(benchmark::State& state)
{
    vectorization::tensor<T> a(kN), b(kN), c(kN), out(kN);
    fill_uniform(a.data(), kN, static_cast<T>(-4), static_cast<T>(4), 1u);
    fill_uniform(b.data(), kN, static_cast<T>(-4), static_cast<T>(4), 2u);
    fill_uniform(c.data(), kN, static_cast<T>(-4), static_cast<T>(4), 3u);
    for (auto _ : state)
        benchmark::DoNotOptimize(out = (a * b + c));
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(kN));
}

BENCHMARK_TEMPLATE(XSigma_Fma, float)->MeasureProcessCPUTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(XSigma_Fma, double)->MeasureProcessCPUTime()->Unit(benchmark::kMicrosecond);

#if VECTORIZATION_HAS_LIBTORCH

template <typename T>
static void LibTorch_Fma(benchmark::State& state)
{
    auto ta    = make_th<T>(static_cast<T>(-4), static_cast<T>(4), 1u);
    auto tb    = make_th<T>(static_cast<T>(-4), static_cast<T>(4), 2u);
    auto tc    = make_th<T>(static_cast<T>(-4), static_cast<T>(4), 3u);
    auto out_t = torch::empty_like(ta);
    for (auto _ : state)
        benchmark::DoNotOptimize(torch::addcmul_out(out_t, tc, ta, tb));
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(kN));
}

BENCHMARK_TEMPLATE(LibTorch_Fma, float)->MeasureProcessCPUTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(LibTorch_Fma, double)->MeasureProcessCPUTime()->Unit(benchmark::kMicrosecond);

#endif  // VECTORIZATION_HAS_LIBTORCH

// Custom main so we can pin LibTorch to one thread before benchmarks run.
// torch::set_num_interop_threads must be called before any parallel work starts,
// hence it cannot live inside individual benchmark functions.
int main(int argc, char** argv)
{
#if VECTORIZATION_HAS_LIBTORCH
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);
#endif
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
