/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * Unit tests for the GPU simd<T> backend (CUDA or HIP).
 *
 * Each test group launches a device kernel that exercises a category of simd<T>
 * ops on-device, copies the results back to the host, and compares against
 * std:: math.  Tests are skipped automatically when no GPU device is present
 * at runtime.  BackendProperties is a compile-time / host-side check that does
 * not require a GPU.
 */

// VectorizationTest.h pulls in the nvcc fmt/int128 ADL shim itself (guarded
// by VECTORIZATION_HAS_CUDA) -- see Testing/VectorizationTest.h.in and
// cuda_fmt_int128_fix.h for the full explanation.
#include "VectorizationTest.h"

#if VECTORIZATION_HAS_CUDA || VECTORIZATION_HAS_HIP

#if VECTORIZATION_HAS_CUDA
#include <cuda_runtime.h>
#elif VECTORIZATION_HAS_HIP
#include <hip/hip_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "backend/gpu/double/simd.h"
#include "backend/gpu/float/simd.h"

// ---- CUDA/HIP runtime API aliases -----------------------------------------
// The two runtimes expose an identical API surface under different names;
// aliasing lets the kernels/tests below be written once for both backends.
#if VECTORIZATION_HAS_CUDA
using gpu_error_t                 = cudaError_t;
constexpr gpu_error_t kGpuSuccess = cudaSuccess;
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuGetErrorString cudaGetErrorString
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuGetDeviceCount cudaGetDeviceCount
#elif VECTORIZATION_HAS_HIP
using gpu_error_t                 = hipError_t;
constexpr gpu_error_t kGpuSuccess = hipSuccess;
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuMemcpy hipMemcpy
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuGetErrorString hipGetErrorString
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuGetDeviceCount hipGetDeviceCount
#endif

namespace
{

// ---- CUDA/HIP error helper -------------------------------------------------

#define GPU_CHECK(expr)                                                                          \
    do                                                                                           \
    {                                                                                            \
        gpu_error_t _e = (expr);                                                                 \
        if (_e != kGpuSuccess)                                                                   \
            ADD_FAILURE() << "GPU error: " << gpuGetErrorString(_e) << " at " << __FILE__ << ":" \
                          << __LINE__;                                                           \
    } while (0)

// ---- RAII device buffer --------------------------------------------------

template <typename T>
struct DevBuf
{
    T*     d = nullptr;
    size_t n = 0;

    explicit DevBuf(size_t count) : n(count) { GPU_CHECK(gpuMalloc(&d, n * sizeof(T))); }
    ~DevBuf() { gpuFree(d); }

    void upload(const std::vector<T>& h) const
    {
        GPU_CHECK(gpuMemcpy(d, h.data(), n * sizeof(T), gpuMemcpyHostToDevice));
    }

    std::vector<T> download() const
    {
        std::vector<T> h(n);
        GPU_CHECK(gpuMemcpy(h.data(), d, n * sizeof(T), gpuMemcpyDeviceToHost));
        return h;
    }
};

// ---- Kernels ------------------------------------------------------------

// Arithmetic: add, sub, mul, div, fma(a,b,b) == a*b+b
template <typename T>
__global__ void k_arith(
    const T* a, const T* b, T* r_add, T* r_sub, T* r_mul, T* r_div, T* r_fma, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    T av = simd<T>::loadu(a + i), bv = simd<T>::loadu(b + i);
    simd<T>::storeu(simd<T>::add(av, bv), r_add + i);
    simd<T>::storeu(simd<T>::sub(av, bv), r_sub + i);
    simd<T>::storeu(simd<T>::mul(av, bv), r_mul + i);
    simd<T>::storeu(simd<T>::div(av, bv), r_div + i);
    simd<T>::storeu(simd<T>::fma(av, bv, bv), r_fma + i);
}

// Unary: sqrt(|a|), sqr, fabs, neg, ceil, floor, trunc
template <typename T>
__global__ void k_unary(
    const T* a,
    T*       r_sqrt,
    T*       r_sqr,
    T*       r_fabs,
    T*       r_neg,
    T*       r_ceil,
    T*       r_floor,
    T*       r_trunc,
    size_t   n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    T av = simd<T>::loadu(a + i);
    simd<T>::storeu(simd<T>::sqrt(simd<T>::fabs(av)), r_sqrt + i);
    simd<T>::storeu(simd<T>::sqr(av), r_sqr + i);
    simd<T>::storeu(simd<T>::fabs(av), r_fabs + i);
    simd<T>::storeu(simd<T>::neg(av), r_neg + i);
    simd<T>::storeu(simd<T>::ceil(av), r_ceil + i);
    simd<T>::storeu(simd<T>::floor(av), r_floor + i);
    simd<T>::storeu(simd<T>::trunc(av), r_trunc + i);
}

// Exp / log: exp, log(|a|+eps), exp2, log2, cbrt
template <typename T>
__global__ void k_explog(const T* a, T* r_exp, T* r_log, T* r_exp2, T* r_log2, T* r_cbrt, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    T av  = simd<T>::loadu(a + i);
    T pos = simd<T>::fabs(av) + static_cast<T>(0.1);
    simd<T>::storeu(simd<T>::exp(av), r_exp + i);
    simd<T>::storeu(simd<T>::log(pos), r_log + i);
    simd<T>::storeu(simd<T>::exp2(av), r_exp2 + i);
    simd<T>::storeu(simd<T>::log2(pos), r_log2 + i);
    simd<T>::storeu(simd<T>::cbrt(av), r_cbrt + i);
}

// Trig: sin, cos, tanh
template <typename T>
__global__ void k_trig(const T* a, T* r_sin, T* r_cos, T* r_tanh, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    T av = simd<T>::loadu(a + i);
    simd<T>::storeu(simd<T>::sin(av), r_sin + i);
    simd<T>::storeu(simd<T>::cos(av), r_cos + i);
    simd<T>::storeu(simd<T>::tanh(av), r_tanh + i);
}

// Comparisons + identity reductions (size==1 → accumulate/hmin/hmax are pass-through)
template <typename T>
__global__ void k_cmp_reduce(
    const T* a, const T* b, T* r_lt, T* r_eq_self, T* r_acc, T* r_hmin, T* r_hmax, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    T av = simd<T>::loadu(a + i), bv = simd<T>::loadu(b + i);
    r_lt[i]      = simd<T>::lt(av, bv) ? static_cast<T>(1) : static_cast<T>(0);
    r_eq_self[i] = simd<T>::eq(av, av) ? static_cast<T>(1) : static_cast<T>(0);
    simd<T>::storeu(simd<T>::accumulate(av), r_acc + i);
    simd<T>::storeu(simd<T>::hmin(av), r_hmin + i);
    simd<T>::storeu(simd<T>::hmax(av), r_hmax + i);
}

// ---- Templated test body -------------------------------------------------

template <typename T>
void run_gpu_simd_test()
{
    constexpr size_t N    = 512;
    constexpr double kRel = std::is_same_v<T, float> ? 5e-5 : 1e-11;
    constexpr double kAbs = kRel;

    // a in [-2, 2), b in [0.5, 1.5) — chosen so neither is zero and lt has a mix
    std::vector<T> ha(N), hb(N);
    for (size_t i = 0; i < N; ++i)
    {
        double t = static_cast<double>(i) / static_cast<double>(N);
        ha[i]    = static_cast<T>(t * 4.0 - 2.0);
        hb[i]    = static_cast<T>(t + 0.5);
    }

    DevBuf<T> da(N), db(N);
    da.upload(ha);
    db.upload(hb);

    constexpr unsigned kBlock = 128u;
    const unsigned     kGrid  = static_cast<unsigned>((N + kBlock - 1u) / kBlock);

    // ---- Arithmetic -------------------------------------------------------
    {
        DevBuf<T> r_add(N), r_sub(N), r_mul(N), r_div(N), r_fma(N);
        k_arith<<<kGrid, kBlock>>>(da.d, db.d, r_add.d, r_sub.d, r_mul.d, r_div.d, r_fma.d, N);
        GPU_CHECK(gpuDeviceSynchronize());

        auto vadd = r_add.download(), vsub = r_sub.download(), vmul = r_mul.download(),
             vdiv = r_div.download(), vfma = r_fma.download();

        for (size_t i = 0; i < N; ++i)
        {
            double a = ha[i], b = hb[i];
            EXPECT_NEAR(vadd[i], a + b, std::abs(a + b) * kRel + kAbs);
            EXPECT_NEAR(vsub[i], a - b, std::abs(a - b) * kRel + kAbs);
            EXPECT_NEAR(vmul[i], a * b, std::abs(a * b) * kRel + kAbs);
            EXPECT_NEAR(vdiv[i], a / b, std::abs(a / b) * kRel + kAbs);
            EXPECT_NEAR(vfma[i], a * b + b, std::abs(a * b + b) * kRel + kAbs);
        }
    }

    // ---- Unary ------------------------------------------------------------
    {
        DevBuf<T> r_sqrt(N), r_sqr(N), r_fabs(N), r_neg(N), r_ceil(N), r_floor(N), r_trunc(N);
        k_unary<<<kGrid, kBlock>>>(
            da.d, r_sqrt.d, r_sqr.d, r_fabs.d, r_neg.d, r_ceil.d, r_floor.d, r_trunc.d, N);
        GPU_CHECK(gpuDeviceSynchronize());

        auto vsqrt = r_sqrt.download(), vsqr = r_sqr.download();
        auto vfabs = r_fabs.download(), vneg = r_neg.download();
        auto vceil = r_ceil.download(), vfloor = r_floor.download();
        auto vtrunc = r_trunc.download();

        for (size_t i = 0; i < N; ++i)
        {
            double a = ha[i];
            EXPECT_NEAR(vsqrt[i], std::sqrt(std::fabs(a)), kAbs);
            EXPECT_NEAR(vsqr[i], a * a, std::abs(a * a) * kRel + kAbs);
            EXPECT_NEAR(vfabs[i], std::fabs(a), kAbs);
            EXPECT_NEAR(vneg[i], -a, kAbs);
            EXPECT_NEAR(vceil[i], std::ceil(a), kAbs);
            EXPECT_NEAR(vfloor[i], std::floor(a), kAbs);
            EXPECT_NEAR(vtrunc[i], std::trunc(a), kAbs);
        }
    }

    // ---- Exp / log --------------------------------------------------------
    {
        DevBuf<T> r_exp(N), r_log(N), r_exp2(N), r_log2(N), r_cbrt(N);
        k_explog<<<kGrid, kBlock>>>(da.d, r_exp.d, r_log.d, r_exp2.d, r_log2.d, r_cbrt.d, N);
        GPU_CHECK(gpuDeviceSynchronize());

        auto vexp = r_exp.download(), vlog = r_log.download();
        auto vexp2 = r_exp2.download(), vlog2 = r_log2.download();
        auto vcbrt = r_cbrt.download();

        for (size_t i = 0; i < N; ++i)
        {
            double a = ha[i], pos = std::fabs(a) + 0.1;
            EXPECT_NEAR(vexp[i], std::exp(a), std::exp(a) * kRel + kAbs);
            EXPECT_NEAR(vlog[i], std::log(pos), std::abs(std::log(pos)) * kRel + kAbs);
            EXPECT_NEAR(vexp2[i], std::exp2(a), std::exp2(a) * kRel + kAbs);
            EXPECT_NEAR(vlog2[i], std::log2(pos), std::abs(std::log2(pos)) * kRel + kAbs);
            EXPECT_NEAR(vcbrt[i], std::cbrt(a), kAbs);
        }
    }

    // ---- Trig -------------------------------------------------------------
    {
        DevBuf<T> r_sin(N), r_cos(N), r_tanh(N);
        k_trig<<<kGrid, kBlock>>>(da.d, r_sin.d, r_cos.d, r_tanh.d, N);
        GPU_CHECK(gpuDeviceSynchronize());

        auto vsin = r_sin.download(), vcos = r_cos.download(), vtanh = r_tanh.download();

        for (size_t i = 0; i < N; ++i)
        {
            double a = ha[i];
            EXPECT_NEAR(vsin[i], std::sin(a), kAbs);
            EXPECT_NEAR(vcos[i], std::cos(a), kAbs);
            EXPECT_NEAR(vtanh[i], std::tanh(a), kAbs);
        }
    }

    // ---- Comparisons and reductions ---------------------------------------
    {
        DevBuf<T> r_lt(N), r_eq(N), r_acc(N), r_hmin(N), r_hmax(N);
        k_cmp_reduce<<<kGrid, kBlock>>>(da.d, db.d, r_lt.d, r_eq.d, r_acc.d, r_hmin.d, r_hmax.d, N);
        GPU_CHECK(gpuDeviceSynchronize());

        auto vlt = r_lt.download(), veq = r_eq.download();
        auto vacc = r_acc.download(), vhmin = r_hmin.download(), vhmax = r_hmax.download();

        for (size_t i = 0; i < N; ++i)
        {
            EXPECT_EQ(vlt[i] != static_cast<T>(0), ha[i] < hb[i]) << "lt at i=" << i;
            EXPECT_EQ(veq[i], static_cast<T>(1)) << "eq(x,x) must be true at i=" << i;
            EXPECT_EQ(vacc[i], ha[i]) << "accumulate is identity for size=1, i=" << i;
            EXPECT_EQ(vhmin[i], ha[i]) << "hmin is identity for size=1, i=" << i;
            EXPECT_EQ(vhmax[i], ha[i]) << "hmax is identity for size=1, i=" << i;
        }
    }
}

}  // namespace

// --------------------------------------------------------------------------
// Compile-time and host-side property checks — no GPU device required.
// --------------------------------------------------------------------------
VECTORIZATIONTEST(GpuSimd, BackendProperties)
{
    static_assert(simd<float>::size == 1, "GPU simd<float>::size must be 1");
    static_assert(simd<double>::size == 1, "GPU simd<double>::size must be 1");
    EXPECT_EQ(simd<float>::size, 1);
    EXPECT_EQ(simd<double>::size, 1);
    END_TEST();
}

// --------------------------------------------------------------------------
// On-device arithmetic and math tests
// --------------------------------------------------------------------------
VECTORIZATIONTEST(GpuSimd, ArithmeticAndMathFloat)
{
    int ndev = 0;
    gpuGetDeviceCount(&ndev);
    if (ndev == 0)
        GTEST_SKIP() << "No GPU device — skipping GpuSimd float test";
    run_gpu_simd_test<float>();
    END_TEST();
}

VECTORIZATIONTEST(GpuSimd, ArithmeticAndMathDouble)
{
    int ndev = 0;
    gpuGetDeviceCount(&ndev);
    if (ndev == 0)
        GTEST_SKIP() << "No GPU device — skipping GpuSimd double test";
    run_gpu_simd_test<double>();
    END_TEST();
}

#else  // !(VECTORIZATION_HAS_CUDA || VECTORIZATION_HAS_HIP)

VECTORIZATIONTEST(GpuSimd, BackendProperties)
{
    END_TEST();
}
VECTORIZATIONTEST(GpuSimd, ArithmeticAndMathFloat)
{
    END_TEST();
}
VECTORIZATIONTEST(GpuSimd, ArithmeticAndMathDouble)
{
    END_TEST();
}

#endif  // VECTORIZATION_HAS_CUDA || VECTORIZATION_HAS_HIP
