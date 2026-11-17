/*
 * Vectorization: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * CPU baselines for design phases P0/P2/P3/P5. All inputs and outputs are
 * preallocated except CPU_TensorAllocFree and the staged CPU_ProductEpilogue
 * references, which allocate the product temporary they measure. Bytes are logical reads/writes,
 * not hardware traffic; broadcast inputs are counted once per output element.
 * Use wall time so parallel evaluator work is included. The raw triad is a
 * compiler-vectorized reference, not a result from the STREAM benchmark.
 */

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "terminals/tensor.h"

namespace
{
using vectorization::tensor;

template <typename T>
void report_traffic(benchmark::State& state, std::size_t n, std::size_t arrays)
{
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
    state.SetBytesProcessed(state.iterations() * static_cast<int64_t>(n * arrays * sizeof(T)));
}

template <typename T>
void check_output(benchmark::State& state, tensor<T> const& output, T expected)
{
    if (!std::all_of(
            output.begin(), output.end(), [expected](T value) { return value == expected; }))
    {
        state.SkipWithError("incorrect tensor benchmark output");
    }
}

template <typename T, bool Raw, bool NoAlias = false> void CPU_Triad(benchmark::State& state)
{
    const auto n = static_cast<std::size_t>(state.range(0));
    tensor<T>  a(n), b(n), output(n);
    a = T(1);
    b = T(2);
    // Escape the buffers before timing so constant inputs cannot be folded.
    benchmark::DoNotOptimize(a.data());
    benchmark::DoNotOptimize(b.data());
    for (auto _ : state)
    {
        if constexpr (Raw)
        {
            const auto* x = a.data();
            const auto* y = b.data();
            auto*       z = output.data();
            for (std::size_t i = 0; i < n; ++i)
            {
                z[i] = x[i] + T(3) * y[i];
            }
        }
        else if constexpr (NoAlias)
        {
            output.noalias() = a + T(3) * b;
        }
        else
        {
            output = a + T(3) * b;
        }
        benchmark::ClobberMemory();
    }
    check_output(state, output, T(7));
    report_traffic<T>(state, n, 3);
}

// Sum of use_count() over the retained tensor leaves of an expression tree, so the
// count does not depend on how the builder shapes the tree (e.g. `a + 3 * b`
// contracting into fma(3, b, a) under VECTORIZATION_FUSE_MUL_ADD).
template <typename T> struct is_retained_ref : std::false_type
{
};

template <typename T> struct is_retained_ref<vectorization::retained_ref<T>> : std::true_type
{
};

template <typename E> int64_t retained_use_count(E const& expr)
{
    using raw = vectorization::remove_cvref_t<E>;
    if constexpr (vectorization::is_pure_expression<raw>::value)
    {
        int64_t count = retained_use_count(expr.rhs());
        if constexpr (VECTORIZATION_EXPR_HAS_LHS(expr))
        {
            count += retained_use_count(expr.lhs());
        }
        if constexpr (VECTORIZATION_EXPR_HAS_MHS(expr))
        {
            count += retained_use_count(expr.mhs());
        }
        return count;
    }
    else if constexpr (is_retained_ref<raw>::value)
    {
        return expr.storage.use_count();
    }
    else
    {
        return 0;
    }
}

template <typename T> void CPU_ExpressionCapture(benchmark::State& state)
{
    const auto n = static_cast<std::size_t>(state.range(0));
    tensor<T>  a(n), b(n);
    // P2 gate: capture with tensor<T> leaves should add no atomic operations.
    // Each storage handle a live expression keeps on top of the owners' own
    // reference was one atomic increment at capture and is one atomic
    // decrement at destruction, so 2 * retains is a lower bound on atomic RMWs.
    int64_t retains = 0;
    {
        const auto base_a     = a.as_view();
        auto       expression = a + T(3) * b;
        retains               = retained_use_count(expression);
        // Subtract the owners' references and the probe view held above.
        retains -= 3;
        benchmark::DoNotOptimize(base_a);
    }
    // Includes descriptor capture and retained-handle release, without evaluation.
    for (auto _ : state)
    {
        auto expression = a + T(3) * b;
        benchmark::DoNotOptimize(expression);
    }
    state.counters["retains"]        = static_cast<double>(retains);
    state.counters["atomic_rmw_min"] = static_cast<double>(2 * retains);
}

// P3 gate: `t().contiguous()` >= 50% of tier-0 copy bandwidth. Both cases
// include allocation and view construction; the packed reference is clone(),
// the tier-0 copy with the same allocation cost.
template <typename T, bool Transposed> void CPU_Materialize(benchmark::State& state)
{
    const auto n = static_cast<std::size_t>(state.range(0));
    tensor<T>  a(n, n);
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        a[i] = static_cast<T>(i % 251);
    }
    for (auto _ : state)
    {
        if constexpr (Transposed)
        {
            auto packed = a.t().contiguous();
            benchmark::DoNotOptimize(packed.data());
        }
        else
        {
            auto packed = a.clone();
            benchmark::DoNotOptimize(packed.data());
        }
        benchmark::ClobberMemory();
    }
    auto       check = Transposed ? tensor<T>(a.t().contiguous()) : a.clone();
    const auto data  = check.data();
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const auto source_index = Transposed ? (i % n) * n + i / n : i;
        if (data[i] != a[source_index])
        {
            state.SkipWithError("incorrect materialization output");
            break;
        }
    }
    report_traffic<T>(state, a.size(), 2);
}

template <typename T> void CPU_TensorAllocFree(benchmark::State& state)
{
    const auto n = static_cast<std::size_t>(state.range(0));
    for (auto _ : state)
    {
        tensor<T> value(n);
        benchmark::DoNotOptimize(value.data());
    }
}

enum class Epilogue : std::uint8_t
{
    staged_accumulate,  // product temporary, then c = tmp + c (the general path)
    accumulate,         // c = matmul(a, b) + c: one GEMM with beta = 1
    staged_bias,        // product temporary, then c = tmp + bias
    bias                // c = matmul(a, b) + bias: broadcast copy, then beta = 1
};

// P5 epilogues: an n x n product combined with a leaf. The staged cases
// reproduce the product-temporary lowering for comparison.
template <typename T, Epilogue Mode> void CPU_ProductEpilogue(benchmark::State& state)
{
    const auto n = static_cast<std::size_t>(state.range(0));
    tensor<T>  a(n, n), b(n, n), c(n, n), bias(n);
    a    = T(1) / static_cast<T>(n);
    b    = T(1);
    c    = T(0);
    bias = T(1);
    for (auto _ : state)
    {
        if constexpr (Mode == Epilogue::staged_accumulate)
        {
            tensor<T> product(matmul(a, b));
            c = product + c;
        }
        else if constexpr (Mode == Epilogue::accumulate)
        {
            c = matmul(a, b) + c;
        }
        else if constexpr (Mode == Epilogue::staged_bias)
        {
            tensor<T> product(matmul(a, b));
            c = product + bias;
        }
        else
        {
            c = matmul(a, b) + bias;
        }
        benchmark::DoNotOptimize(c.data());
        benchmark::ClobberMemory();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(2 * n * n * n));
}

enum class Layout : std::uint8_t
{
    row_broadcast,
    column_broadcast,
    stepped,
    packed_copy,
    transpose_copy
};

template <typename T, Layout Mode> void CPU_Layout(benchmark::State& state)
{
    constexpr std::size_t rows    = 256;
    const auto            columns = static_cast<std::size_t>(state.range(0));
    tensor<T>             a(rows, columns);
    a = T(1);
    if constexpr (Mode == Layout::row_broadcast || Mode == Layout::column_broadcast)
    {
        tensor<T> output(rows, columns);
        tensor<T> b(
            Mode == Layout::row_broadcast ? 1 : rows, Mode == Layout::row_broadcast ? columns : 1);
        b = T(2);
        for (auto _ : state)
        {
            output = a + T(3) * b;
            benchmark::ClobberMemory();
        }
        check_output(state, output, T(7));
        report_traffic<T>(state, a.size(), 3);
    }
    else if constexpr (Mode == Layout::stepped)
    {
        tensor<T> source(rows, 2 * columns), destination(rows, 2 * columns);
        source      = T(2);
        destination = T(0);
        auto input  = source.slice(1, 0, 2 * columns, 2);
        auto out    = destination.slice(1, 0, 2 * columns, 2);
        for (auto _ : state)
        {
            out = a + T(3) * input;
            benchmark::ClobberMemory();
        }
        for (std::size_t i = 0; i < destination.size(); ++i)
        {
            if (destination[i] != (i % 2 == 0 ? T(7) : T(0)))
            {
                state.SkipWithError("incorrect stepped benchmark output");
                break;
            }
        }
        report_traffic<T>(state, a.size(), 3);
    }
    else
    {
        // View construction and allocation are excluded; this measures the
        // logical copy used to materialize a transpose into reusable storage.
        tensor<T> copied_output(Mode == Layout::transpose_copy ? columns : rows,
            Mode == Layout::transpose_copy ? rows : columns);
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            a[i] = static_cast<T>(i % 251);
        }
        auto input = Mode == Layout::transpose_copy ? a.t() : a.as_view();
        for (auto _ : state)
        {
            // Owning tensor operator= clones; assign() evaluates into this buffer.
            copied_output.assign(input);
            benchmark::ClobberMemory();
        }
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            const auto source_index =
                Mode == Layout::transpose_copy ? (i % rows) * columns + i / rows : i;
            if (copied_output[i] != a[source_index])
            {
                state.SkipWithError("incorrect copy benchmark output");
                break;
            }
        }
        report_traffic<T>(state, a.size(), 2);
    }
}

// Include small-tensor latency, both sides of the planner's threading grain,
// cache-resident inputs, and a 192/384 MiB float/double triad working set.
const auto triad_sizes = [](auto* benchmark_case)
{
    for (int64_t n : {1, 16, 64, 256, 32767, 32768, 32769, 1 << 20, 1 << 24})
    {
        benchmark_case->Arg(n);
    }
    benchmark_case->UseRealTime();
};

BENCHMARK_TEMPLATE(CPU_Triad, float, true)->Apply(triad_sizes);
BENCHMARK_TEMPLATE(CPU_Triad, float, false)->Apply(triad_sizes);
BENCHMARK_TEMPLATE(CPU_Triad, float, false, true)->Apply(triad_sizes);
BENCHMARK_TEMPLATE(CPU_Triad, double, true)->Apply(triad_sizes);
BENCHMARK_TEMPLATE(CPU_Triad, double, false)->Apply(triad_sizes);
BENCHMARK_TEMPLATE(CPU_Triad, double, false, true)->Apply(triad_sizes);

BENCHMARK_TEMPLATE(CPU_ExpressionCapture, float)->Arg(1)->Arg(64)->Arg(256)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_ExpressionCapture, double)->Arg(1)->Arg(64)->Arg(256)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_TensorAllocFree, float)->Arg(1)->Arg(64)->Arg(256)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_TensorAllocFree, double)->Arg(1)->Arg(64)->Arg(256)->UseRealTime();

// 256 and 1,024 are cache-resident; 4,096 (64/128 MiB per float/double array)
// is memory-bound.
BENCHMARK_TEMPLATE(CPU_Materialize, float, false)->Arg(256)->Arg(1024)->Arg(4096)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_Materialize, float, true)->Arg(256)->Arg(1024)->Arg(4096)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_Materialize, double, false)->Arg(256)->Arg(1024)->Arg(4096)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_Materialize, double, true)->Arg(256)->Arg(1024)->Arg(4096)->UseRealTime();

BENCHMARK_TEMPLATE(CPU_Layout, float, Layout::row_broadcast)->Arg(256)->Arg(1024)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_Layout, float, Layout::column_broadcast)->Arg(256)->Arg(1024)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_Layout, float, Layout::stepped)->Arg(256)->Arg(1024)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_Layout, float, Layout::packed_copy)->Arg(256)->Arg(1024)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_Layout, float, Layout::transpose_copy)->Arg(256)->Arg(1024)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_Layout, double, Layout::row_broadcast)->Arg(256)->Arg(1024)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_Layout, double, Layout::column_broadcast)
    ->Arg(256)
    ->Arg(1024)
    ->UseRealTime();
BENCHMARK_TEMPLATE(CPU_Layout, double, Layout::stepped)->Arg(256)->Arg(1024)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_Layout, double, Layout::packed_copy)->Arg(256)->Arg(1024)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_Layout, double, Layout::transpose_copy)->Arg(256)->Arg(1024)->UseRealTime();
BENCHMARK_TEMPLATE(CPU_ProductEpilogue, float, Epilogue::staged_accumulate)
    ->Arg(16)
    ->Arg(64)
    ->Arg(256)
    ->UseRealTime();
BENCHMARK_TEMPLATE(CPU_ProductEpilogue, float, Epilogue::accumulate)
    ->Arg(16)
    ->Arg(64)
    ->Arg(256)
    ->UseRealTime();
BENCHMARK_TEMPLATE(CPU_ProductEpilogue, float, Epilogue::staged_bias)
    ->Arg(16)
    ->Arg(64)
    ->Arg(256)
    ->UseRealTime();
BENCHMARK_TEMPLATE(CPU_ProductEpilogue, float, Epilogue::bias)
    ->Arg(16)
    ->Arg(64)
    ->Arg(256)
    ->UseRealTime();
BENCHMARK_TEMPLATE(CPU_ProductEpilogue, double, Epilogue::staged_accumulate)
    ->Arg(16)
    ->Arg(64)
    ->Arg(256)
    ->UseRealTime();
BENCHMARK_TEMPLATE(CPU_ProductEpilogue, double, Epilogue::accumulate)
    ->Arg(16)
    ->Arg(64)
    ->Arg(256)
    ->UseRealTime();
}  // namespace
