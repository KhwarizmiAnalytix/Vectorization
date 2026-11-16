/*
 * Vectorization: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * Baseline comparison benchmarks: Vectorization tensor<T> vs a plain scalar
 * loop, Eigen arrays and LibTorch tensors.
 *
 * Every case runs the same operation on the same inputs through each
 * implementation, so rows with a common prefix are directly comparable:
 *
 *   <Op>/<f32|f64>/<n>/Vectorization  — SIMD expression-template path
 *   <Op>/<f32|f64>/<n>/Loop           — plain scalar loop (whatever the compiler
 *                                       auto-vectorizes with the active flags)
 *   <Op>/<f32|f64>/<n>/Eigen          — Eigen::Array expression (bundled Eigen)
 *   <Op>/<f32|f64>/<n>/LibTorch       — torch::*_out kernel, one intra-op thread
 *                                       (only when VECTORIZATION_HAS_LIBTORCH=1)
 *
 * Outputs are preallocated so the hot loop excludes allocation, except where an
 * implementation needs temporaries for a fused expression (LibTorch Composite),
 * which is part of what is being compared. Before timing, each implementation's
 * result is checked against a double-precision reference; a mismatch skips the
 * case with an error instead of reporting a misleading time.
 *
 * Compare, for example:
 *   benchmark_tensorbaselines --benchmark_filter='Exp/f32/'
 *
 * Target: benchmark_tensorbaselines
 */

#include <benchmark/benchmark.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "terminals/tensor.h"

#ifndef EIGEN_DONT_PARALLELIZE
#define EIGEN_DONT_PARALLELIZE
#endif
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push, 0)
#endif
#include <Eigen/Core>
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif

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

namespace
{

// L1-resident, L2-resident and DRAM-bound working sets. Odd sizes keep every
// implementation on its scalar tail path as well as its vector body.
constexpr std::array<std::size_t, 3> kSizes = {(1u << 10) + 3, (1u << 15) + 3, (1u << 20) + 3};

// Element type of a tensor or Eigen array, for typed scalar literals in expressions.
template <typename A>
using elem_t = std::remove_cv_t<std::remove_pointer_t<decltype(std::declval<const A&>().data())>>;

template <typename T> constexpr const char* type_name()
{
    return std::is_same_v<T, float> ? "f32" : "f64";
}

// Element-wise tolerance against the double-precision reference, scaled by
// max(1, |ref|). Loose enough for any correctly rounded-ish SIMD math library,
// tight enough to catch a wrong kernel.
template <typename T> constexpr double elementwise_tolerance()
{
    return std::is_same_v<T, float> ? 2e-5 : 1e-11;
}

// Reduction tolerance, scaled by sum(|term|): summation order differs between
// implementations, so this bounds accumulated rounding rather than per-element error.
template <typename T> constexpr double reduction_tolerance()
{
    return std::is_same_v<T, float> ? 1e-4 : 1e-10;
}

template <typename T> std::vector<T> make_input(std::size_t n, double lo, double hi, unsigned seed)
{
    std::mt19937                      gen(seed);
    std::uniform_real_distribution<T> dist(static_cast<T>(lo), static_cast<T>(hi));
    std::vector<T>                    v(n);
    for (auto& x : v)
    {
        x = dist(gen);
    }
    return v;
}

// One input set per (op, type, size) case, shared by all implementations.
template <typename T> struct inputs
{
    std::vector<T> a, b, c;
};

template <typename Op, typename T> inputs<T> make_inputs(std::size_t n)
{
    return {make_input<T>(n, Op::lo, Op::hi, 1u),
        make_input<T>(n, Op::lo, Op::hi, 2u),
        make_input<T>(n, Op::lo, Op::hi, 3u)};
}

template <typename Op, typename T> bool elementwise_matches(const T* got, const inputs<T>& in)
{
    for (std::size_t i = 0; i < in.a.size(); ++i)
    {
        const double ref = Op::scalar(static_cast<double>(in.a[i]),
            static_cast<double>(in.b[i]),
            static_cast<double>(in.c[i]));
        const double err = std::abs(static_cast<double>(got[i]) - ref);
        if (!(err <= elementwise_tolerance<T>() * std::max(1.0, std::abs(ref))))
        {
            return false;  // also rejects NaN
        }
    }
    return true;
}

template <typename Op, typename T> bool reduction_matches(T got, const inputs<T>& in)
{
    double ref = 0.0, scale = 0.0;
    for (std::size_t i = 0; i < in.a.size(); ++i)
    {
        const double term = Op::scalar(static_cast<double>(in.a[i]),
            static_cast<double>(in.b[i]),
            static_cast<double>(in.c[i]));
        ref += term;
        scale += std::abs(term);
    }
    const double err = std::abs(static_cast<double>(got) - ref);
    return err <= reduction_tolerance<T>() * std::max(1.0, scale);
}

template <typename Op, typename T> void set_counters(benchmark::State& state, std::size_t n)
{
    const auto items = state.iterations() * static_cast<int64_t>(n);
    state.SetItemsProcessed(items);
    state.SetBytesProcessed(items * static_cast<int64_t>(sizeof(T) * Op::streams));
}

void skip_mismatch(benchmark::State& state)
{
    state.SkipWithError("result does not match the double-precision reference");
}

}  // namespace

// ── operations ────────────────────────────────────────────────────────────────
//
// Each op describes itself once per implementation:
//   scalar(a, b, c)   — per-element value (element-wise) or term (reduction);
//                       drives both the Loop baseline and the reference check
//   vec(a, b, c)      — Vectorization expression (or reduced value)
//   eigen(a, b, c)    — Eigen::Array expression (or reduced value)
//   torch(o, a, b, c) — LibTorch kernel writing into o
//   streams           — arrays read + written per element, for bytes/second
// ─────────────────────────────────────────────────────────────────────────────

#if VECTORIZATION_HAS_LIBTORCH
#define BASELINE_TORCH(BODY)                                                                       \
    static void torch(                                                                             \
        at::Tensor& o, const at::Tensor& a, const at::Tensor& b, const at::Tensor& c)              \
    {                                                                                              \
        (void)a;                                                                                   \
        (void)b;                                                                                   \
        (void)c;                                                                                   \
        BODY;                                                                                      \
    }
#else
#define BASELINE_TORCH(BODY)
#endif

#define BASELINE_ELEMENTWISE(NAME, ARITY, LO, HI, SCALAR, VEC, EIGEN, TORCH)                       \
    struct NAME##Op                                                                                \
    {                                                                                              \
        static constexpr const char*   name      = #NAME;                                          \
        static constexpr bool          reduction = false;                                          \
        static constexpr std::size_t   streams   = (ARITY) + 1;                                    \
        static constexpr double        lo = LO, hi = HI;                                           \
        template <typename U> static U scalar(U a, U b, U c)                                       \
        {                                                                                          \
            (void)b;                                                                               \
            (void)c;                                                                               \
            return SCALAR;                                                                         \
        }                                                                                          \
        template <typename A, typename B, typename C>                                              \
        static auto vec(const A& a, const B& b, const C& c)                                        \
        {                                                                                          \
            (void)b;                                                                               \
            (void)c;                                                                               \
            return VEC;                                                                            \
        }                                                                                          \
        template <typename A, typename B, typename C>                                              \
        static auto eigen(const A& a, const B& b, const C& c)                                      \
        {                                                                                          \
            (void)b;                                                                               \
            (void)c;                                                                               \
            return EIGEN;                                                                          \
        }                                                                                          \
        BASELINE_TORCH(TORCH)                                                                      \
    };

#define BASELINE_UNARY(NAME, FN, LO, HI)                                                           \
    BASELINE_ELEMENTWISE(NAME, 1, LO, HI, std::FN(a), ::FN(a), a.FN(), torch::FN##_out(o, a))

// Memory-bound arithmetic.
BASELINE_ELEMENTWISE(Add, 2, -4, 4, a + b, a + b, a + b, torch::add_out(o, a, b))
BASELINE_ELEMENTWISE(Mul, 2, -4, 4, a* b, a* b, a* b, torch::mul_out(o, a, b))
BASELINE_ELEMENTWISE(Saxpy,
    2,
    -4,
    4,
    a* U(2.5) + b,
    a* elem_t<A>(2.5) + b,
    a* elem_t<A>(2.5) + b,
    torch::add_out(o, b, a, 2.5))  // o = b + 2.5 * a
BASELINE_ELEMENTWISE(Fma, 3, -4, 4, a* b + c, a* b + c, a* b + c, torch::addcmul_out(o, c, a, b))

// Fused multi-op expression: one pass for expression templates, several
// kernels plus temporaries for an eager library.
BASELINE_ELEMENTWISE(Composite,
    3,
    -4,
    4,
    std::sqrt(a* a + b * b) + c,
    ::sqrt(a* a + b * b) + c,
    (a * a + b * b).sqrt() + c,
    torch::add_out(o, torch::sqrt(a* a + b * b), c))

// Compute-bound transcendentals, with inputs inside each function's domain.
BASELINE_UNARY(Exp, exp, -1, 1)
BASELINE_UNARY(Log, log, 0.1, 4)
BASELINE_UNARY(Sqrt, sqrt, 0.1, 4)
BASELINE_UNARY(Sin, sin, -4, 4)
BASELINE_UNARY(Cos, cos, -4, 4)
BASELINE_UNARY(Tanh, tanh, -4, 4)

#undef BASELINE_UNARY
#undef BASELINE_ELEMENTWISE

// Reductions: scalar() is the per-element term the Loop baseline accumulates.
struct SumOp
{
    static constexpr const char*                              name      = "Sum";
    static constexpr bool                                     reduction = true;
    static constexpr std::size_t                              streams   = 1;
    static constexpr double                                   lo = -4, hi = 4;
    template <typename U> static U                            scalar(U a, U, U) { return a; }
    template <typename A, typename B, typename C> static auto vec(const A& a, const B&, const C&)
    {
        return vectorization::sum(a);
    }
    template <typename A, typename B, typename C> static auto eigen(const A& a, const B&, const C&)
    {
        return a.sum();
    }
    BASELINE_TORCH(torch::sum_out(o, a, std::vector<int64_t>{0}))
};

struct DotOp
{
    static constexpr const char*                              name      = "Dot";
    static constexpr bool                                     reduction = true;
    static constexpr std::size_t                              streams   = 2;
    static constexpr double                                   lo = -4, hi = 4;
    template <typename U> static U                            scalar(U a, U b, U) { return a * b; }
    template <typename A, typename B, typename C> static auto vec(const A& a, const B& b, const C&)
    {
        return vectorization::sum(a * b);
    }
    template <typename A, typename B, typename C>
    static auto eigen(const A& a, const B& b, const C&)
    {
        return a.matrix().dot(b.matrix());
    }
    BASELINE_TORCH(torch::dot_out(o, a, b))
};

#undef BASELINE_TORCH

// ── implementations ───────────────────────────────────────────────────────────

namespace
{

template <typename Op, typename T> void bench_vectorization(benchmark::State& state, std::size_t n)
{
    const auto               in = make_inputs<Op, T>(n);
    vectorization::tensor<T> a(n), b(n), c(n);
    std::copy(in.a.begin(), in.a.end(), a.data());
    std::copy(in.b.begin(), in.b.end(), b.data());
    std::copy(in.c.begin(), in.c.end(), c.data());

    if constexpr (Op::reduction)
    {
        if (!reduction_matches<Op>(static_cast<T>(Op::vec(a, b, c)), in))
        {
            return skip_mismatch(state);
        }
        for (auto _ : state)
        {
            benchmark::DoNotOptimize(Op::vec(a, b, c));
        }
    }
    else
    {
        vectorization::tensor<T> out(n);
        out = Op::vec(a, b, c);
        if (!elementwise_matches<Op>(out.data(), in))
        {
            return skip_mismatch(state);
        }
        for (auto _ : state)
        {
            out = Op::vec(a, b, c);
            benchmark::DoNotOptimize(out.data());
            benchmark::ClobberMemory();
        }
    }
    set_counters<Op, T>(state, n);
}

template <typename Op, typename T> void bench_loop(benchmark::State& state, std::size_t n)
{
    const auto in = make_inputs<Op, T>(n);
    const T*   a  = in.a.data();
    const T*   b  = in.b.data();
    const T*   c  = in.c.data();
    // Escape the inputs so ClobberMemory() below stops the compiler from
    // hoisting the loop-invariant computation out of the timed loop. The
    // lambdas capture by value: reference captures of these escaped locals
    // would force a reload after every store and block auto-vectorization.
    benchmark::DoNotOptimize(a);
    benchmark::DoNotOptimize(b);
    benchmark::DoNotOptimize(c);

    if constexpr (Op::reduction)
    {
        const auto run = [a, b, c, n]
        {
            T acc = T(0);
            for (std::size_t i = 0; i < n; ++i)
            {
                acc += Op::scalar(a[i], b[i], c[i]);
            }
            return acc;
        };
        if (!reduction_matches<Op>(run(), in))
        {
            return skip_mismatch(state);
        }
        for (auto _ : state)
        {
            benchmark::DoNotOptimize(run());
            benchmark::ClobberMemory();
        }
    }
    else
    {
        std::vector<T> out(n);
        T*             o   = out.data();
        const auto     run = [a, b, c, o, n]
        {
            for (std::size_t i = 0; i < n; ++i)
            {
                o[i] = Op::scalar(a[i], b[i], c[i]);
            }
        };
        run();
        if (!elementwise_matches<Op>(o, in))
        {
            return skip_mismatch(state);
        }
        for (auto _ : state)
        {
            run();
            benchmark::DoNotOptimize(o);
            benchmark::ClobberMemory();
        }
    }
    set_counters<Op, T>(state, n);
}

template <typename Op, typename T> void bench_eigen(benchmark::State& state, std::size_t n)
{
    using array_t  = Eigen::Array<T, Eigen::Dynamic, 1>;
    using map_t    = Eigen::Map<const array_t>;
    const auto in  = make_inputs<Op, T>(n);
    const auto len = static_cast<Eigen::Index>(n);
    // Owning (aligned) arrays rather than Maps over the shared buffers, so Eigen
    // runs on its preferred layout.
    const array_t a = map_t(in.a.data(), len);
    const array_t b = map_t(in.b.data(), len);
    const array_t c = map_t(in.c.data(), len);

    if constexpr (Op::reduction)
    {
        if (!reduction_matches<Op>(static_cast<T>(Op::eigen(a, b, c)), in))
        {
            return skip_mismatch(state);
        }
        for (auto _ : state)
        {
            benchmark::DoNotOptimize(Op::eigen(a, b, c));
        }
    }
    else
    {
        array_t out(len);
        out = Op::eigen(a, b, c);
        if (!elementwise_matches<Op>(out.data(), in))
        {
            return skip_mismatch(state);
        }
        for (auto _ : state)
        {
            out = Op::eigen(a, b, c);
            benchmark::DoNotOptimize(out.data());
            benchmark::ClobberMemory();
        }
    }
    set_counters<Op, T>(state, n);
}

#if VECTORIZATION_HAS_LIBTORCH

template <typename Op, typename T> void bench_libtorch(benchmark::State& state, std::size_t n)
{
    // Single intra-op thread: the other baselines are single-threaded too.
    torch::set_num_threads(1);
    const auto in   = make_inputs<Op, T>(n);
    const auto opts = torch::TensorOptions()
                          .dtype(std::is_same_v<T, float> ? torch::kFloat32 : torch::kFloat64)
                          .device(torch::kCPU);
    const auto wrap = [&](const std::vector<T>& v)
    { return torch::from_blob(const_cast<T*>(v.data()), {static_cast<int64_t>(n)}, opts).clone(); };
    const at::Tensor a = wrap(in.a), b = wrap(in.b), c = wrap(in.c);

    at::Tensor out =
        Op::reduction ? torch::empty({}, opts) : torch::empty({static_cast<int64_t>(n)}, opts);
    Op::torch(out, a, b, c);
    const bool ok = Op::reduction ? reduction_matches<Op>(out.item<T>(), in)
                                  : elementwise_matches<Op>(out.data_ptr<T>(), in);
    if (!ok)
    {
        return skip_mismatch(state);
    }
    for (auto _ : state)
    {
        Op::torch(out, a, b, c);
        benchmark::DoNotOptimize(out.data_ptr());
        benchmark::ClobberMemory();
    }
    set_counters<Op, T>(state, n);
}

#endif  // VECTORIZATION_HAS_LIBTORCH

// ── registration ──────────────────────────────────────────────────────────────

// Wall time drives the iteration count (process CPU time is coarse on Windows,
// ~15.6 ms ticks); the CPU column still reports all threads' work, which is the
// fair comparison if the tensor evaluator runs in parallel.
void configure(benchmark::Benchmark* b)
{
    b->MeasureProcessCPUTime()->UseRealTime()->Unit(benchmark::kMicrosecond);
}

// Registered per size so the implementations of one case print adjacently.
template <typename Op, typename T> void register_case()
{
    for (const std::size_t n : kSizes)
    {
        const std::string prefix =
            std::string(Op::name) + "/" + type_name<T>() + "/" + std::to_string(n) + "/";
        configure(benchmark::RegisterBenchmark(prefix + "Vectorization",
            [n](benchmark::State& s) { bench_vectorization<Op, T>(s, n); }));
        configure(benchmark::RegisterBenchmark(
            prefix + "Loop", [n](benchmark::State& s) { bench_loop<Op, T>(s, n); }));
        configure(benchmark::RegisterBenchmark(
            prefix + "Eigen", [n](benchmark::State& s) { bench_eigen<Op, T>(s, n); }));
#if VECTORIZATION_HAS_LIBTORCH
        configure(benchmark::RegisterBenchmark(
            prefix + "LibTorch", [n](benchmark::State& s) { bench_libtorch<Op, T>(s, n); }));
#endif
    }
}

template <typename... Ops> void register_ops()
{
    (register_case<Ops, float>(), ...);
    (register_case<Ops, double>(), ...);
}

[[maybe_unused]] const bool kRegistered = []
{
    register_ops<AddOp,
        MulOp,
        SaxpyOp,
        FmaOp,
        CompositeOp,
        ExpOp,
        LogOp,
        SqrtOp,
        SinOp,
        CosOp,
        TanhOp,
        SumOp,
        DotOp>();
    return true;
}();

}  // namespace
