/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * CPU vs GPU tensor throughput benchmarks.
 *
 * Compares vectorization::tensor<T> expression evaluation on the CPU SIMD
 * backend against the GPU backend (run_gpu / fill_gpu for CUDA/HIP,
 * run_metal / fill_metal for Metal), across problem sizes ranging from
 * latency-bound (1K elements) to throughput-bound (4M elements). Metal is
 * float-only (MSL has no double) — GPU_*<double> benchmarks skip themselves
 * under Metal (see kMetalOnlyBackend).
 *
 * Naming / methodology:
 *   CPU_<Op><T>            — host SIMD path (expressions_evaluator::run on CPU)
 *   GPU_<Op><T>             — device-resident: operands already on the GPU,
 *                            times kernel launch + execution only
 *                            (cudaDeviceSynchronize() inside the timed loop).
 *   GPU_<Op>_Transfer<T>    — same op, but re-uploads inputs / downloads the
 *                            result every iteration: the realistic cost when
 *                            the tensors do not already live on the device.
 *   GPU_TensorAllocFree<T>  — isolates the cost of constructing/destroying a
 *                            device tensor (XSigma metal/cuda caching
 *                            allocator path via Memory/allocator.h).
 *   LibTorch_MPS_TensorAllocFree<T> — same alloc/free loop on torch::kMPS
 *                            (Metal builds with LibTorch only), for a direct
 *                            comparison against PyTorch's MPS caching allocator.
 *   CPU/GPU_MonteCarloPath<T> — multi-factor Monte Carlo path update
 *                            X += sigma_0*Z_0 + ... + sigma_3*Z_3, repeated
 *                            over kMcSteps time steps per iteration. X holds
 *                            one running value per path; Z_i are per-factor
 *                            shocks (fixed across steps); sigma_i are scalar
 *                            loadings. GPU variant syncs once after all
 *                            kMcSteps launches, not per step.
 *   GPU_<Op>InPlace<T>      — same op as GPU_<Op>, but writes into one of the
 *                            operands (`a += b`) instead of a freshly
 *                            constructed result tensor `c`, isolating the
 *                            cost of the extra allocation that GPU_<Op> pays
 *                            on every construction of the benchmark fixture
 *                            (not per iteration — see GPU_TensorAllocFree
 *                            note below for why that construction itself is
 *                            cheap after warm-up).
 *   GPU_SingleStream_<Op><T> /
 *   GPU_MultiStream_<Op><T> — same kNumStreams independent Op invocations,
 *                            issued sequentially on the default stream vs. one
 *                            per stream via stream_guard
 *                            (terminals/stream_guard.h); isolates the
 *                            concurrency benefit of directing independent work
 *                            onto separate streams. CUDA/HIP only — Metal has
 *                            no stream concept.
 *
 * GPU benchmarks use wall-clock time (not MeasureProcessCPUTime): most of the
 * "time" is the device executing, not the host CPU, so process-CPU time would
 * understate the cost and make CPU/GPU numbers non-comparable.
 *
 * Clock/link ramp-up: all GPU_* and LibTorch_MPS_* cases are registered with
 * an explicit ->MinWarmUpTime()->MinTime() (see GPU_BENCH_SIZES below), which
 * google-benchmark treats as an override of the process-wide
 * --benchmark_min_time flag (BenchmarkRunner::GetMinTimeToApply() prefers a
 * benchmark's own min_time() the moment it's non-zero — see
 * ThirdParty/benchmark/src/benchmark_runner.cc). This matters specifically on
 * this project's CI/setup.py invocation, which passes a very short
 * --benchmark_min_time=0.01s to keep the overall test run fast: without a
 * per-benchmark override, GPU kernels would be measured while the driver is
 * still at its idle power state (observed on a GeForce card: ~200MHz SM clock
 * against a >3GHz boost clock, PCIe link negotiated at Gen1 x8 instead of the
 * card's Gen4 x8 max) because a 10ms run is too short and bursty for the
 * driver to ramp clocks/link before the timed samples are taken. CPU_* cases
 * keep the short default since host clocks aren't gated the same way.
 *
 * GPU_TensorAllocFree<T> isolates same-size device tensor construct/destroy
 * cost. After the first ("cache miss") iteration the CUDA/HIP caching
 * allocator should serve from a per-size-class free list — see cache_hits /
 * cache_misses around allocate() — so near-identical timing across sizes is
 * the expected result, not a benchmark artifact.
 *
 * This file is compiled as a CUDA or HIP translation unit (CMake sets
 * LANGUAGE CUDA/HIP on it, mirroring TestTensorGpu.cpp) so run_gpu/fill_gpu
 * are instantiated; Metal needs no such CMake language (ordinary host C++,
 * device-count queries routed through the metal_device_probe.mm shim, same
 * as TestTensorGpu.cpp). It degrades to a benchmark stub (no registered
 * benchmarks) when none of VECTORIZATION_HAS_CUDA/_HIP/_METAL is on, or when
 * the CUDA compiler is Clang (see the exclusion note in Testing/Cxx/CMakeLists.txt).
 *
 * Target: benchmark_tensorgpu
 */

#if VECTORIZATION_HAS_CUDA
// nvcc (cicc) forces fmt's FMT_USE_INT128 fallback (fmt::detail::uint128, no
// operator~) even for host-only formatting -- see cuda_fmt_int128_fix.h (in
// Testing/, not Testing/Cxx/) for the full explanation. Included directly
// here rather than via VectorizationTest.h: this file is a Google Benchmark
// target, not GTest, so it doesn't include that header.
#include "cuda_fmt_int128_fix.h"
#endif

#include <benchmark/benchmark.h>

#if VECTORIZATION_HAS_CUDA || VECTORIZATION_HAS_HIP || VECTORIZATION_HAS_METAL

#if VECTORIZATION_HAS_CUDA
#include <cuda_runtime.h>
using gpu_error_t                 = cudaError_t;
constexpr gpu_error_t kGpuSuccess = cudaSuccess;
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamDestroy cudaStreamDestroy
#define gpuStreamSynchronize cudaStreamSynchronize
#elif VECTORIZATION_HAS_HIP
#include <hip/hip_runtime.h>
using gpu_error_t                 = hipError_t;
constexpr gpu_error_t kGpuSuccess = hipSuccess;
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuStreamCreate hipStreamCreate
#define gpuStreamDestroy hipStreamDestroy
#define gpuStreamSynchronize hipStreamSynchronize
#elif VECTORIZATION_HAS_METAL
// metal_dispatch.h is a plain C++ header (no Objective-C types cross its boundary —
// see its own header comment), so this file can call into it directly without
// becoming Objective-C++ itself, unlike TestTensorGpu.cpp's device-count query (which
// needed a separate .mm shim because that file only wants device_enum, not the rest of
// the Metal backend surface).
#include "backend/gpu/metal/metal_dispatch.h"
using gpu_error_t                 = int;
constexpr gpu_error_t kGpuSuccess = 0;
#define gpuGetDeviceCount(pn) \
    (*(pn) = vectorization::metal_backend::device_available() ? 1 : 0, kGpuSuccess)
// Every metal_backend::dispatch()/dispatch_fill() call already blocks on
// waitUntilCompleted internally (see metal_dispatch.mm) — no separate device-sync API.
#define gpuDeviceSynchronize() ((void)0)
#endif

#include <cstddef>
#include <random>
#include <type_traits>
#include <vector>

#include "terminals/tensor.h"

#if VECTORIZATION_HAS_METAL && VECTORIZATION_HAS_LIBTORCH
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <torch/mps.h>
#include <torch/torch.h>
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
#endif

namespace
{
using namespace vectorization;

#if VECTORIZATION_HAS_CUDA
constexpr device_enum kActiveGpuDevice  = device_enum::CUDA;
constexpr bool        kMetalOnlyBackend = false;
#elif VECTORIZATION_HAS_HIP
constexpr device_enum kActiveGpuDevice  = device_enum::HIP;
constexpr bool        kMetalOnlyBackend = false;
#elif VECTORIZATION_HAS_METAL
constexpr device_enum kActiveGpuDevice  = device_enum::METAL;
constexpr bool        kMetalOnlyBackend = true;
#endif

template <typename T>
void fill_uniform(std::vector<T>& v, T lo, T hi, unsigned seed)
{
    std::mt19937                      gen(seed);
    std::uniform_real_distribution<T> dist(lo, hi);
    for (auto& x : v)
        x = dist(gen);
}

bool has_gpu_device()
{
    int ndev = 0;
    return gpuGetDeviceCount(&ndev) == kGpuSuccess && ndev > 0;
}

// Metal is float-only (MSL has no double); tensor<double> on device_enum::METAL throws
// at allocation time (Library/Memory/allocator.h). Every GPU_*<double> benchmark checks
// this before constructing a device tensor so it skips cleanly instead of throwing.
template <typename T>
bool skip_if_unsupported(benchmark::State& state)
{
    if (kMetalOnlyBackend && std::is_same_v<T, double>)
    {
        state.SkipWithError("Metal backend is float-only (no fp64 on Apple GPUs)");
        return true;
    }
    if (!has_gpu_device())
    {
        state.SkipWithError("No GPU device");
        return true;
    }
    return false;
}

}  // namespace

// Explicit sizes: 1K (latency-bound), 64K, 1M, 4M (throughput-bound).
#define BENCH_SIZES \
    ->Arg(1 << 10)->Arg(1 << 16)->Arg(1 << 20)->Arg(1 << 22)->Unit(benchmark::kMicrosecond)

// GPU cases additionally get a warm-up + minimum measurement window so the
// driver has time to leave its idle power state before/while the timed
// samples are taken (see the file header for why this overrides
// --benchmark_min_time rather than relying on it).
#define GPU_BENCH_SIZES BENCH_SIZES->MinWarmUpTime(0.1)->MinTime(0.3)

// ---------------------------------------------------------------------------
// Fill: a = scalar
// ---------------------------------------------------------------------------
template <typename T>
static void CPU_Fill(benchmark::State& state)
{
    const size_t n = static_cast<size_t>(state.range(0));
    tensor<T>    a(n);
    for (auto _ : state)
        a = static_cast<T>(3.14159);
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK_TEMPLATE(CPU_Fill, float) BENCH_SIZES;
BENCHMARK_TEMPLATE(CPU_Fill, double) BENCH_SIZES;

template <typename T>
static void GPU_Fill(benchmark::State& state)
{
    if (skip_if_unsupported<T>(state))
        return;
    const size_t n = static_cast<size_t>(state.range(0));
    tensor<T>    a(n, kActiveGpuDevice);
    for (auto _ : state)
    {
        a = static_cast<T>(3.14159);
        gpuDeviceSynchronize();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK_TEMPLATE(GPU_Fill, float) GPU_BENCH_SIZES;
BENCHMARK_TEMPLATE(GPU_Fill, double) GPU_BENCH_SIZES;

// ---------------------------------------------------------------------------
// Binary add: c = a + b  (device-resident; excludes transfer)
// ---------------------------------------------------------------------------
template <typename T>
static void CPU_Add(benchmark::State& state)
{
    const size_t   n = static_cast<size_t>(state.range(0));
    std::vector<T> ha(n), hb(n);
    fill_uniform(ha, static_cast<T>(-2), static_cast<T>(2), 1u);
    fill_uniform(hb, static_cast<T>(0.5), static_cast<T>(1.5), 2u);

    tensor<T> a(n), b(n), c(n);
    a.copy_from_host(ha);
    b.copy_from_host(hb);

    for (auto _ : state)
        benchmark::DoNotOptimize(c = a + b);
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK_TEMPLATE(CPU_Add, float) BENCH_SIZES;
BENCHMARK_TEMPLATE(CPU_Add, double) BENCH_SIZES;

template <typename T>
static void GPU_Add(benchmark::State& state)
{
    if (skip_if_unsupported<T>(state))
        return;
    const size_t   n = static_cast<size_t>(state.range(0));
    std::vector<T> ha(n), hb(n);
    fill_uniform(ha, static_cast<T>(-2), static_cast<T>(2), 1u);
    fill_uniform(hb, static_cast<T>(0.5), static_cast<T>(1.5), 2u);

    tensor<T> a(n, kActiveGpuDevice), b(n, kActiveGpuDevice), c(n, kActiveGpuDevice);
    a.copy_from_host(ha);
    b.copy_from_host(hb);

    for (auto _ : state)
    {
        c = a + b;
        gpuDeviceSynchronize();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK_TEMPLATE(GPU_Add, float) GPU_BENCH_SIZES;
BENCHMARK_TEMPLATE(GPU_Add, double) GPU_BENCH_SIZES;

template <typename T>
static void GPU_Add_Transfer(benchmark::State& state)
{
    if (skip_if_unsupported<T>(state))
        return;
    const size_t   n = static_cast<size_t>(state.range(0));
    std::vector<T> ha(n), hb(n);
    fill_uniform(ha, static_cast<T>(-2), static_cast<T>(2), 1u);
    fill_uniform(hb, static_cast<T>(0.5), static_cast<T>(1.5), 2u);

    tensor<T> a(n, kActiveGpuDevice), b(n, kActiveGpuDevice), c(n, kActiveGpuDevice);

    for (auto _ : state)
    {
        a.copy_from_host(ha);
        b.copy_from_host(hb);
        c           = a + b;
        auto result = c.to_host_vector();
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK_TEMPLATE(GPU_Add_Transfer, float) GPU_BENCH_SIZES;
BENCHMARK_TEMPLATE(GPU_Add_Transfer, double) GPU_BENCH_SIZES;

#if VECTORIZATION_HAS_CUDA || VECTORIZATION_HAS_HIP
// ---------------------------------------------------------------------------
// Multi-stream throughput: kNumStreams independent c_i = a_i + b_i expressions,
// directed one per stream via stream_guard (terminals/stream_guard.h), compared
// against the same kNumStreams independent ops issued sequentially on the
// default stream. Isolates the benefit of concurrent kernel execution across
// streams (subject to the device having enough free SMs/queues to overlap
// them) from the underlying add kernel's own cost, which GPU_Add already
// measures. CUDA/HIP only -- Metal has no stream concept (see the
// StreamGuard* tests in TestTensorGpu.cpp, same restriction).
//
// Tensors are destroyed before the streams they were directed to (nested
// scope below) since a live tensor can still reference its stream via the
// caching allocator's deferred-reuse bookkeeping even after the stream_guard
// that redirected it goes out of scope -- see the comment on
// StreamGuardRedirectsExpressionConstruction in TestTensorGpu.cpp.
// ---------------------------------------------------------------------------
constexpr int kNumStreams = 4;

template <typename T>
static void GPU_SingleStream_Add(benchmark::State& state)
{
    if (skip_if_unsupported<T>(state))
        return;
    const size_t n = static_cast<size_t>(state.range(0));

    std::vector<tensor<T>> a, b, c;
    for (int i = 0; i < kNumStreams; ++i)
    {
        std::vector<T> ha(n), hb(n);
        fill_uniform(ha, static_cast<T>(-2), static_cast<T>(2), 1u + static_cast<unsigned>(i));
        fill_uniform(hb, static_cast<T>(0.5), static_cast<T>(1.5), 100u + static_cast<unsigned>(i));
        a.emplace_back(n, kActiveGpuDevice);
        b.emplace_back(n, kActiveGpuDevice);
        c.emplace_back(n, kActiveGpuDevice);
        a.back().copy_from_host(ha);
        b.back().copy_from_host(hb);
    }

    for (auto _ : state)
    {
        for (int i = 0; i < kNumStreams; ++i)
            c[i] = a[i] + b[i];
        gpuDeviceSynchronize();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n) * kNumStreams);
}
BENCHMARK_TEMPLATE(GPU_SingleStream_Add, float) GPU_BENCH_SIZES;
BENCHMARK_TEMPLATE(GPU_SingleStream_Add, double) GPU_BENCH_SIZES;

template <typename T>
static void GPU_MultiStream_Add(benchmark::State& state)
{
    if (skip_if_unsupported<T>(state))
        return;
    const size_t n = static_cast<size_t>(state.range(0));

    gpu_stream_t streams[kNumStreams];
    for (auto& s : streams)
        gpuStreamCreate(&s);

    {
        std::vector<tensor<T>> a, b, c;
        for (int i = 0; i < kNumStreams; ++i)
        {
            std::vector<T> ha(n), hb(n);
            fill_uniform(ha, static_cast<T>(-2), static_cast<T>(2), 1u + static_cast<unsigned>(i));
            fill_uniform(
                hb, static_cast<T>(0.5), static_cast<T>(1.5), 100u + static_cast<unsigned>(i));
            a.emplace_back(n, kActiveGpuDevice);
            b.emplace_back(n, kActiveGpuDevice);
            c.emplace_back(n, kActiveGpuDevice);
            a.back().copy_from_host(ha);
            b.back().copy_from_host(hb);
        }

        for (auto _ : state)
        {
            for (int i = 0; i < kNumStreams; ++i)
            {
                stream_guard guard(streams[i], 0);
                c[i] = a[i] + b[i];
            }
            for (auto& s : streams)
                gpuStreamSynchronize(s);
        }
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n) * kNumStreams);
    }  // a, b, c destroyed here, before the streams they referenced

    for (auto& s : streams)
        gpuStreamDestroy(s);
}
BENCHMARK_TEMPLATE(GPU_MultiStream_Add, float) GPU_BENCH_SIZES;
BENCHMARK_TEMPLATE(GPU_MultiStream_Add, double) GPU_BENCH_SIZES;
#endif  // VECTORIZATION_HAS_CUDA || VECTORIZATION_HAS_HIP

// ---------------------------------------------------------------------------
// Binary multiply: c = a * b  (device-resident; excludes transfer)
// ---------------------------------------------------------------------------
template <typename T>
static void CPU_Multiply(benchmark::State& state)
{
    const size_t   n = static_cast<size_t>(state.range(0));
    std::vector<T> ha(n), hb(n);
    fill_uniform(ha, static_cast<T>(-2), static_cast<T>(2), 1u);
    fill_uniform(hb, static_cast<T>(0.5), static_cast<T>(1.5), 2u);

    tensor<T> a(n), b(n), c(n);
    a.copy_from_host(ha);
    b.copy_from_host(hb);

    for (auto _ : state)
        benchmark::DoNotOptimize(c = a * b);
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK_TEMPLATE(CPU_Multiply, float) BENCH_SIZES;
BENCHMARK_TEMPLATE(CPU_Multiply, double) BENCH_SIZES;

template <typename T>
static void GPU_Multiply(benchmark::State& state)
{
    if (skip_if_unsupported<T>(state))
        return;
    const size_t   n = static_cast<size_t>(state.range(0));
    std::vector<T> ha(n), hb(n);
    fill_uniform(ha, static_cast<T>(-2), static_cast<T>(2), 1u);
    fill_uniform(hb, static_cast<T>(0.5), static_cast<T>(1.5), 2u);

    tensor<T> a(n, kActiveGpuDevice), b(n, kActiveGpuDevice), c(n, kActiveGpuDevice);
    a.copy_from_host(ha);
    b.copy_from_host(hb);

    for (auto _ : state)
    {
        c = a * b;
        gpuDeviceSynchronize();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK_TEMPLATE(GPU_Multiply, float) GPU_BENCH_SIZES;
BENCHMARK_TEMPLATE(GPU_Multiply, double) GPU_BENCH_SIZES;

// ---------------------------------------------------------------------------
// Binary divide: c = a / b  (device-resident; excludes transfer). b is kept
// in [0.5, 1.5] (same fill_uniform call as the other binary ops), so this
// never exercises a divide-by-zero path.
// ---------------------------------------------------------------------------
template <typename T>
static void CPU_Divide(benchmark::State& state)
{
    const size_t   n = static_cast<size_t>(state.range(0));
    std::vector<T> ha(n), hb(n);
    fill_uniform(ha, static_cast<T>(-2), static_cast<T>(2), 1u);
    fill_uniform(hb, static_cast<T>(0.5), static_cast<T>(1.5), 2u);

    tensor<T> a(n), b(n), c(n);
    a.copy_from_host(ha);
    b.copy_from_host(hb);

    for (auto _ : state)
        benchmark::DoNotOptimize(c = a / b);
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK_TEMPLATE(CPU_Divide, float) BENCH_SIZES;
BENCHMARK_TEMPLATE(CPU_Divide, double) BENCH_SIZES;

template <typename T>
static void GPU_Divide(benchmark::State& state)
{
    if (skip_if_unsupported<T>(state))
        return;
    const size_t   n = static_cast<size_t>(state.range(0));
    std::vector<T> ha(n), hb(n);
    fill_uniform(ha, static_cast<T>(-2), static_cast<T>(2), 1u);
    fill_uniform(hb, static_cast<T>(0.5), static_cast<T>(1.5), 2u);

    tensor<T> a(n, kActiveGpuDevice), b(n, kActiveGpuDevice), c(n, kActiveGpuDevice);
    a.copy_from_host(ha);
    b.copy_from_host(hb);

    for (auto _ : state)
    {
        c = a / b;
        gpuDeviceSynchronize();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK_TEMPLATE(GPU_Divide, float) GPU_BENCH_SIZES;
BENCHMARK_TEMPLATE(GPU_Divide, double) GPU_BENCH_SIZES;

// ---------------------------------------------------------------------------
// In-place add: a += b  (device-resident; excludes transfer). Unlike
// CPU_Add/GPU_Add, there is no result tensor `c` — writes go directly into
// `a`, isolating the extra-allocation cost that the out-of-place variant
// pays once when the benchmark fixture constructs `c` (see the file header's
// GPU_TensorAllocFree note for why repeated construct/destroy of same-sized
// tensors is cheap after warm-up: this instead measures a fixture with one
// fewer live allocation altogether).
// ---------------------------------------------------------------------------
template <typename T>
static void CPU_AddInPlace(benchmark::State& state)
{
    const size_t   n = static_cast<size_t>(state.range(0));
    std::vector<T> ha(n), hb(n);
    fill_uniform(ha, static_cast<T>(-2), static_cast<T>(2), 1u);
    fill_uniform(hb, static_cast<T>(0.5), static_cast<T>(1.5), 2u);

    tensor<T> a(n), b(n);
    a.copy_from_host(ha);
    b.copy_from_host(hb);

    for (auto _ : state)
        a += b;
    benchmark::DoNotOptimize(a.data());
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK_TEMPLATE(CPU_AddInPlace, float) BENCH_SIZES;
BENCHMARK_TEMPLATE(CPU_AddInPlace, double) BENCH_SIZES;

template <typename T>
static void GPU_AddInPlace(benchmark::State& state)
{
    if (skip_if_unsupported<T>(state))
        return;
    const size_t   n = static_cast<size_t>(state.range(0));
    std::vector<T> ha(n), hb(n);
    fill_uniform(ha, static_cast<T>(-2), static_cast<T>(2), 1u);
    fill_uniform(hb, static_cast<T>(0.5), static_cast<T>(1.5), 2u);

    tensor<T> a(n, kActiveGpuDevice), b(n, kActiveGpuDevice);
    a.copy_from_host(ha);
    b.copy_from_host(hb);

    for (auto _ : state)
    {
        a += b;
        gpuDeviceSynchronize();
    }
    benchmark::DoNotOptimize(a.data());
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK_TEMPLATE(GPU_AddInPlace, float) GPU_BENCH_SIZES;
BENCHMARK_TEMPLATE(GPU_AddInPlace, double) GPU_BENCH_SIZES;

// ---------------------------------------------------------------------------
// Compound expression: exp(a) + sqrt(b) -- fused into a single kernel/loop
// ---------------------------------------------------------------------------
template <typename T>
static void CPU_Compound(benchmark::State& state)
{
    const size_t   n = static_cast<size_t>(state.range(0));
    std::vector<T> ha(n), hb(n);
    fill_uniform(ha, static_cast<T>(-1), static_cast<T>(1), 3u);
    fill_uniform(hb, static_cast<T>(0.5), static_cast<T>(1.5), 4u);

    tensor<T> a(n), b(n), c(n);
    a.copy_from_host(ha);
    b.copy_from_host(hb);

    for (auto _ : state)
        benchmark::DoNotOptimize(c = ::exp(a) + ::sqrt(b));
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK_TEMPLATE(CPU_Compound, float) BENCH_SIZES;
BENCHMARK_TEMPLATE(CPU_Compound, double) BENCH_SIZES;

template <typename T>
static void GPU_Compound(benchmark::State& state)
{
    if (skip_if_unsupported<T>(state))
        return;
    const size_t   n = static_cast<size_t>(state.range(0));
    std::vector<T> ha(n), hb(n);
    fill_uniform(ha, static_cast<T>(-1), static_cast<T>(1), 3u);
    fill_uniform(hb, static_cast<T>(0.5), static_cast<T>(1.5), 4u);

    tensor<T> a(n, kActiveGpuDevice), b(n, kActiveGpuDevice), c(n, kActiveGpuDevice);
    a.copy_from_host(ha);
    b.copy_from_host(hb);

    for (auto _ : state)
    {
        c = ::exp(a) + ::sqrt(b);
        gpuDeviceSynchronize();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n));
}
BENCHMARK_TEMPLATE(GPU_Compound, float) GPU_BENCH_SIZES;
BENCHMARK_TEMPLATE(GPU_Compound, double) GPU_BENCH_SIZES;

// ---------------------------------------------------------------------------
// Monte Carlo path update: X += sigma_0*Z_0 + sigma_1*Z_1 + sigma_2*Z_2 + sigma_3*Z_3,
// repeated over kMcSteps time steps (Z_i held fixed across steps — only the
// running sum X changes). Models the per-step diffusion update of a
// multi-factor Monte Carlo path simulation, where X holds one running value
// per path, Z_i are per-factor shocks, and sigma_i are scalar loadings.
// Device-resident: for GPU, all kMcSteps kernel launches are queued on the
// default stream and synced once at the end, matching how a real engine
// would only synchronize after advancing the full path.
// ---------------------------------------------------------------------------
constexpr int kMcSteps = 64;

template <typename T>
static void CPU_MonteCarloPath(benchmark::State& state)
{
    const size_t   n = static_cast<size_t>(state.range(0));
    std::vector<T> hz0(n), hz1(n), hz2(n), hz3(n);
    fill_uniform(hz0, static_cast<T>(-1), static_cast<T>(1), 1u);
    fill_uniform(hz1, static_cast<T>(-1), static_cast<T>(1), 2u);
    fill_uniform(hz2, static_cast<T>(-1), static_cast<T>(1), 3u);
    fill_uniform(hz3, static_cast<T>(-1), static_cast<T>(1), 4u);

    tensor<T> x(n), z0(n), z1(n), z2(n), z3(n);
    x = static_cast<T>(0);
    z0.copy_from_host(hz0);
    z1.copy_from_host(hz1);
    z2.copy_from_host(hz2);
    z3.copy_from_host(hz3);

    const T sigma0 = static_cast<T>(0.10);
    const T sigma1 = static_cast<T>(0.15);
    const T sigma2 = static_cast<T>(0.20);
    const T sigma3 = static_cast<T>(0.25);

    for (auto _ : state)
    {
        for (int step = 0; step < kMcSteps; ++step)
            x = x + sigma0 * z0 + sigma1 * z1 + sigma2 * z2 + sigma3 * z3;
        benchmark::DoNotOptimize(x.data());
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n) * kMcSteps);
}
BENCHMARK_TEMPLATE(CPU_MonteCarloPath, float) BENCH_SIZES;
BENCHMARK_TEMPLATE(CPU_MonteCarloPath, double) BENCH_SIZES;

template <typename T>
static void GPU_MonteCarloPath(benchmark::State& state)
{
    if (skip_if_unsupported<T>(state))
        return;
    const size_t   n = static_cast<size_t>(state.range(0));
    std::vector<T> hz0(n), hz1(n), hz2(n), hz3(n);
    fill_uniform(hz0, static_cast<T>(-1), static_cast<T>(1), 1u);
    fill_uniform(hz1, static_cast<T>(-1), static_cast<T>(1), 2u);
    fill_uniform(hz2, static_cast<T>(-1), static_cast<T>(1), 3u);
    fill_uniform(hz3, static_cast<T>(-1), static_cast<T>(1), 4u);

    tensor<T> x(n, kActiveGpuDevice), z0(n, kActiveGpuDevice), z1(n, kActiveGpuDevice),
        z2(n, kActiveGpuDevice), z3(n, kActiveGpuDevice);
    x = static_cast<T>(0);
    z0.copy_from_host(hz0);
    z1.copy_from_host(hz1);
    z2.copy_from_host(hz2);
    z3.copy_from_host(hz3);

    const T sigma0 = static_cast<T>(0.10);
    const T sigma1 = static_cast<T>(0.15);
    const T sigma2 = static_cast<T>(0.20);
    const T sigma3 = static_cast<T>(0.25);

    for (auto _ : state)
    {
        for (int step = 0; step < kMcSteps; ++step)
            x = x + sigma0 * z0 + sigma1 * z1 + sigma2 * z2 + sigma3 * z3;
        gpuDeviceSynchronize();
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n) * kMcSteps);
}
BENCHMARK_TEMPLATE(GPU_MonteCarloPath, float) GPU_BENCH_SIZES;
BENCHMARK_TEMPLATE(GPU_MonteCarloPath, double) GPU_BENCH_SIZES;

// ---------------------------------------------------------------------------
// Allocation overhead: construct + destroy a device tensor every iteration.
// ---------------------------------------------------------------------------
template <typename T>
static void GPU_TensorAllocFree(benchmark::State& state)
{
    if (skip_if_unsupported<T>(state))
        return;
    const int64_t n = state.range(0);
    {
        tensor<T> warm(static_cast<size_t>(n), kActiveGpuDevice);
        benchmark::DoNotOptimize(warm.data());
    }
    gpuDeviceSynchronize();
    for (auto _ : state)
    {
        tensor<T> t(static_cast<size_t>(n), kActiveGpuDevice);
        benchmark::DoNotOptimize(t.data());
    }
    gpuDeviceSynchronize();
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK_TEMPLATE(GPU_TensorAllocFree, float) GPU_BENCH_SIZES;
BENCHMARK_TEMPLATE(GPU_TensorAllocFree, double) GPU_BENCH_SIZES;

#if VECTORIZATION_HAS_METAL && VECTORIZATION_HAS_LIBTORCH
// ---------------------------------------------------------------------------
// LibTorch MPS allocation overhead — same construct/destroy loop as
// GPU_TensorAllocFree, for a head-to-head against PyTorch's MPS caching path.
// ---------------------------------------------------------------------------
template <typename T>
static void LibTorch_MPS_TensorAllocFree(benchmark::State& state)
{
    if constexpr (std::is_same_v<T, double>)
    {
        state.SkipWithError("Metal/MPS comparison is float-only (no fp64 on Apple GPUs)");
        return;
    }
    if (!torch::mps::is_available())
    {
        state.SkipWithError("LibTorch MPS device not available");
        return;
    }

    const int64_t n = state.range(0);
    auto          opts =
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMPS).requires_grad(false);

    // Warm the MPS caching allocator once so the timed loop measures reuse,
    // matching XSigma's warm caching-allocator path after the first iteration.
    {
        auto warm = torch::empty({n}, opts);
        benchmark::DoNotOptimize(warm.data_ptr());
    }
    torch::mps::synchronize();

    for (auto _ : state)
    {
        auto t = torch::empty({n}, opts);
        benchmark::DoNotOptimize(t.data_ptr());
    }
    torch::mps::synchronize();
    state.SetItemsProcessed(state.iterations() * n);
}
BENCHMARK_TEMPLATE(LibTorch_MPS_TensorAllocFree, float) GPU_BENCH_SIZES;
#endif  // VECTORIZATION_HAS_METAL && VECTORIZATION_HAS_LIBTORCH

#undef BENCH_SIZES

// ---------------------------------------------------------------------------
// Reduction: sum_i (A[i] + B[i] * sin(X[i])) over a fixed, small N=512 — a
// latency-bound size chosen specifically to make the GPU's fixed per-launch
// overhead (command buffer encode + commit + wait) visible against the CPU's
// direct SIMD loop, rather than being amortized away like the throughput-bound
// BENCH_SIZES cases above.
//
// CPU: vectorization::accumulate(a + b * sin(x)) — a single host loop.
// GPU: the same accumulate() call on device-resident tensors (CUDA/HIP fused
// reduce kernel; Metal materializes the fused tree then multi-block reduces).
// ---------------------------------------------------------------------------
constexpr size_t kSumN = 512;

template <typename T>
static void CPU_SumAddMulSin(benchmark::State& state)
{
    std::vector<T> ha(kSumN), hb(kSumN), hx(kSumN);
    fill_uniform(ha, static_cast<T>(-1), static_cast<T>(1), 5u);
    fill_uniform(hb, static_cast<T>(0.5), static_cast<T>(1.5), 6u);
    fill_uniform(hx, static_cast<T>(-3.14159), static_cast<T>(3.14159), 7u);

    tensor<T> a(kSumN), b(kSumN), x(kSumN);
    a.copy_from_host(ha);
    b.copy_from_host(hb);
    x.copy_from_host(hx);

    for (auto _ : state)
        benchmark::DoNotOptimize(vectorization::accumulate(a + b * ::sin(x)));
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(kSumN));
}
BENCHMARK_TEMPLATE(CPU_SumAddMulSin, float)->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(CPU_SumAddMulSin, double)->Unit(benchmark::kMicrosecond);

template <typename T>
static void GPU_SumAddMulSin(benchmark::State& state)
{
    if (skip_if_unsupported<T>(state))
    {
        return;
    }
    std::vector<T> ha(kSumN), hb(kSumN), hx(kSumN);
    fill_uniform(ha, static_cast<T>(-1), static_cast<T>(1), 5u);
    fill_uniform(hb, static_cast<T>(0.5), static_cast<T>(1.5), 6u);
    fill_uniform(hx, static_cast<T>(-3.14159), static_cast<T>(3.14159), 7u);

    tensor<T> a(kSumN, kActiveGpuDevice), b(kSumN, kActiveGpuDevice), x(kSumN, kActiveGpuDevice);
    a.copy_from_host(ha);
    b.copy_from_host(hb);
    x.copy_from_host(hx);

    for (auto _ : state)
    {
        T const sum = vectorization::accumulate(a + b * ::sin(x));
        benchmark::DoNotOptimize(sum);
    }
    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(kSumN));
}
BENCHMARK_TEMPLATE(GPU_SumAddMulSin, float)
    ->Unit(benchmark::kMicrosecond)
    ->MinWarmUpTime(0.1)
    ->MinTime(0.3);
BENCHMARK_TEMPLATE(GPU_SumAddMulSin, double)
    ->Unit(benchmark::kMicrosecond)
    ->MinWarmUpTime(0.1)
    ->MinTime(0.3);

#endif  // VECTORIZATION_HAS_CUDA || VECTORIZATION_HAS_HIP || VECTORIZATION_HAS_METAL

BENCHMARK_MAIN();
