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

#pragma once

// GPU expression evaluation — supports both CUDA (nvcc) and HIP (hipcc).
//
// Compiled only when VECTORIZATION_HAS_CUDA=1 (nvcc) or VECTORIZATION_HAS_HIP=1 (hipcc).
//
// Provides:
//   gpu_eval_kernel<E,T>   — __global__ kernel: one thread per element
//   gpu_fill_kernel<T>     — __global__ kernel for fill
//   run_gpu<E,T>           — host launcher for gpu_eval_kernel
//   fill_gpu<T>            — host launcher for gpu_fill_kernel
//   reduce_gpu<E,T,Op>     — fused expression reduction (sum / min / max)
//
// Kernel bodies are identical for CUDA and HIP; only the launch syntax differs
// (<<<>>> for CUDA, hipLaunchKernelGGL for HIP).
//
// All GPU tensor pointers in the expression tree must be valid device pointers
// (allocated with device_enum::CUDA).  The expression struct is passed by value
// to the kernel so captured pointers are copied to device.

#if VECTORIZATION_HAS_CUDA || VECTORIZATION_HAS_HIP

#if VECTORIZATION_HAS_CUDA
#include <cuda_runtime.h>
#elif VECTORIZATION_HAS_HIP
#include <hip/hip_runtime.h>
#endif

#include "allocator.h"
#include "common/device.h"
#include "expressions/expression_interface_loader.h"
#include "expressions/reduce_op.h"

namespace vectorization
{

// ---------------------------------------------------------------------------
// gpu_stream_t — platform-neutral stream handle alias. cudaStream_t/hipStream_t
// are ordinary host-visible types declared by the runtime headers included
// above, so this only needs VECTORIZATION_HAS_CUDA/HIP -- not
// __CUDACC__/__HIPCC__ -- unlike the kernels/launchers below. Plain .cpp
// translation units (e.g. Testing/Cxx tests that call gpuStreamCreate /
// gpuStreamDestroy directly) must see the real type here, not the void*
// placeholder, or raw CUDA/HIP API calls against it fail to compile.
// ---------------------------------------------------------------------------
#if VECTORIZATION_HAS_CUDA
using gpu_stream_t = cudaStream_t;
#elif VECTORIZATION_HAS_HIP
using gpu_stream_t = hipStream_t;
#else
// Host-only compilation unit: provide a dummy type so signatures compile.
using gpu_stream_t = void*;
#endif

// ---------------------------------------------------------------------------
// __global__ kernels — shared body between CUDA and HIP
// ---------------------------------------------------------------------------
#if (VECTORIZATION_HAS_CUDA && defined(__CUDACC__)) || (VECTORIZATION_HAS_HIP && defined(__HIPCC__))

template <typename E, typename T>
__global__ void gpu_eval_kernel(E expr, T* __restrict__ out, size_t n)
{
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < n)
        out[tid] = expression_loader<E, false, false>::evaluate(expr, tid);
}

template <typename T>
__device__ T* gpu_shared_scratch()
{
    extern __shared__ unsigned char smem_raw[];
    return reinterpret_cast<T*>(smem_raw);
}

template <typename T>
__global__ void gpu_fill_kernel(T* __restrict__ out, T value, size_t n)
{
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid < n)
        out[tid] = value;
}

// Block-level reduction of a fused expression: each thread evaluates one
// element, the block tree-reduces in shared memory, thread 0 writes one
// partial per block.
template <typename E, typename T, reduce_op Op>
__global__ void gpu_reduce_expr_kernel(E expr, T* __restrict__ partials, size_t n)
{
    T* const smem = gpu_shared_scratch<T>();

    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    T            val = reduce_identity<T, Op>();
    if (tid < n)
    {
        val = expression_loader<E, false, false>::evaluate(expr, tid);
    }
    smem[threadIdx.x] = val;
    __syncthreads();

    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            smem[threadIdx.x] =
                reduce_combine<T, Op>(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        partials[blockIdx.x] = smem[0];
    }
}

// Same tree-reduce over a raw device buffer (second and later passes).
template <typename T, reduce_op Op>
__global__ void gpu_reduce_buf_kernel(T const* __restrict__ in, T* __restrict__ out, size_t n)
{
    T* const smem = gpu_shared_scratch<T>();

    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    T            val = reduce_identity<T, Op>();
    if (tid < n)
    {
        val = in[tid];
    }
    smem[threadIdx.x] = val;
    __syncthreads();

    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            smem[threadIdx.x] =
                reduce_combine<T, Op>(smem[threadIdx.x], smem[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        out[blockIdx.x] = smem[0];
    }
}

// ---------------------------------------------------------------------------
// Host launchers — differ only in the kernel-launch syntax
// ---------------------------------------------------------------------------

template <typename E, typename T>
void run_gpu(E const& expr, T* data, size_t n, gpu_stream_t stream = nullptr)
{
    if (n == 0)
        return;
    constexpr unsigned block_size = 256;
    size_t const       blocks =
        (n + static_cast<size_t>(block_size) - 1) / static_cast<size_t>(block_size);
    // CUDA/HIP grid.x is a 32-bit dim; wrapping would launch an invalid config
    // that previously completed without checking cudaGetLastError.
    VECTORIZATION_CHECK(
        blocks > 0 && blocks <= static_cast<size_t>(0x7fffffff),
        "run_gpu grid overflow for n={}",
        n);
    const unsigned grid_size = static_cast<unsigned>(blocks);

#if VECTORIZATION_HAS_CUDA && defined(__CUDACC__)
    gpu_eval_kernel<E, T><<<dim3(grid_size), dim3(block_size), 0, stream>>>(expr, data, n);
    cudaError_t const err = cudaGetLastError();
    VECTORIZATION_CHECK(err == cudaSuccess, "run_gpu launch failed: {}", cudaGetErrorString(err));
#else
    hipLaunchKernelGGL(
        (gpu_eval_kernel<E, T>), dim3(grid_size), dim3(block_size), 0, stream, expr, data, n);
    hipError_t const err = hipGetLastError();
    VECTORIZATION_CHECK(err == hipSuccess, "run_gpu launch failed: {}", hipGetErrorString(err));
#endif
}

template <typename T>
void fill_gpu(T* data, T value, size_t n, gpu_stream_t stream = nullptr)
{
    if (n == 0)
        return;
    constexpr unsigned block_size = 256;
    size_t const       blocks =
        (n + static_cast<size_t>(block_size) - 1) / static_cast<size_t>(block_size);
    VECTORIZATION_CHECK(
        blocks > 0 && blocks <= static_cast<size_t>(0x7fffffff),
        "fill_gpu grid overflow for n={}",
        n);
    const unsigned grid_size = static_cast<unsigned>(blocks);

#if VECTORIZATION_HAS_CUDA && defined(__CUDACC__)
    gpu_fill_kernel<T><<<dim3(grid_size), dim3(block_size), 0, stream>>>(data, value, n);
    cudaError_t const err = cudaGetLastError();
    VECTORIZATION_CHECK(err == cudaSuccess, "fill_gpu launch failed: {}", cudaGetErrorString(err));
#else
    hipLaunchKernelGGL(
        (gpu_fill_kernel<T>), dim3(grid_size), dim3(block_size), 0, stream, data, value, n);
    hipError_t const err = hipGetLastError();
    VECTORIZATION_CHECK(err == hipSuccess, "fill_gpu launch failed: {}", hipGetErrorString(err));
#endif
}

inline void gpu_check_last_error(char const* what)
{
#if VECTORIZATION_HAS_CUDA && defined(__CUDACC__)
    cudaError_t const err = cudaGetLastError();
    VECTORIZATION_CHECK(err == cudaSuccess, "{} launch failed: {}", what, cudaGetErrorString(err));
#else
    hipError_t const err = hipGetLastError();
    VECTORIZATION_CHECK(err == hipSuccess, "{} launch failed: {}", what, hipGetErrorString(err));
#endif
}

template <typename T, reduce_op Op>
T reduce_gpu_partials(
    T* buf, size_t n, memory::device_enum device, int device_index, gpu_stream_t stream)
{
    using alloc_t                 = memory::allocator<T>;
    constexpr unsigned block_size = 256;
    while (n > 1)
    {
        size_t const blocks =
            (n + static_cast<size_t>(block_size) - 1) / static_cast<size_t>(block_size);
        VECTORIZATION_CHECK(
            blocks > 0 && blocks <= static_cast<size_t>(0x7fffffff),
            "reduce_gpu grid overflow for n={}",
            n);
        T* const next = alloc_t::allocate(blocks, device, device_index, stream);
        VECTORIZATION_CHECK(next != nullptr, "reduce_gpu: failed to allocate {} partials", blocks);
        const unsigned    grid_size = static_cast<unsigned>(blocks);
        const std::size_t smem      = static_cast<std::size_t>(block_size) * sizeof(T);
#if VECTORIZATION_HAS_CUDA && defined(__CUDACC__)
        gpu_reduce_buf_kernel<T, Op>
            <<<dim3(grid_size), dim3(block_size), smem, stream>>>(buf, next, n);
#else
        hipLaunchKernelGGL(
            (gpu_reduce_buf_kernel<T, Op>),
            dim3(grid_size),
            dim3(block_size),
            smem,
            stream,
            buf,
            next,
            n);
#endif
        gpu_check_last_error("reduce_gpu_buf");
        alloc_t::free(buf, device, device_index, 0, stream);
        buf = next;
        n   = blocks;
    }

    T host = reduce_identity<T, Op>();
    if (n == 1)
    {
        alloc_t::copy(buf, 1, &host, device, memory::device_enum::CPU, device_index, 0, stream);
        if (stream != nullptr)
        {
#if VECTORIZATION_HAS_CUDA && defined(__CUDACC__)
            cudaError_t const sync_err = cudaStreamSynchronize(stream);
            VECTORIZATION_CHECK(
                sync_err == cudaSuccess,
                "reduce_gpu D2H sync failed: {}",
                cudaGetErrorString(sync_err));
#else
            hipError_t const sync_err = hipStreamSynchronize(stream);
            VECTORIZATION_CHECK(
                sync_err == hipSuccess,
                "reduce_gpu D2H sync failed: {}",
                hipGetErrorString(sync_err));
#endif
        }
    }
    alloc_t::free(buf, device, device_index, 0, stream);
    return host;
}

template <typename E, typename T, reduce_op Op>
T reduce_gpu(
    E const&            expr,
    size_t              n,
    memory::device_enum device,
    int                 device_index,
    gpu_stream_t        stream = nullptr)
{
    if (n == 0)
    {
        return reduce_identity<T, Op>();
    }

    using alloc_t                 = memory::allocator<T>;
    constexpr unsigned block_size = 256;
    size_t const       blocks =
        (n + static_cast<size_t>(block_size) - 1) / static_cast<size_t>(block_size);
    VECTORIZATION_CHECK(
        blocks > 0 && blocks <= static_cast<size_t>(0x7fffffff),
        "reduce_gpu grid overflow for n={}",
        n);
    T* const partials = alloc_t::allocate(blocks, device, device_index, stream);
    VECTORIZATION_CHECK(partials != nullptr, "reduce_gpu: failed to allocate {} partials", blocks);

    const unsigned    grid_size = static_cast<unsigned>(blocks);
    const std::size_t smem      = static_cast<std::size_t>(block_size) * sizeof(T);
#if VECTORIZATION_HAS_CUDA && defined(__CUDACC__)
    gpu_reduce_expr_kernel<E, T, Op>
        <<<dim3(grid_size), dim3(block_size), smem, stream>>>(expr, partials, n);
#else
    hipLaunchKernelGGL(
        (gpu_reduce_expr_kernel<E, T, Op>),
        dim3(grid_size),
        dim3(block_size),
        smem,
        stream,
        expr,
        partials,
        n);
#endif
    gpu_check_last_error("reduce_gpu_expr");
    return reduce_gpu_partials<T, Op>(partials, blocks, device, device_index, stream);
}

#endif  // CUDACC || HIPCC

}  // namespace vectorization

#endif  // VECTORIZATION_HAS_CUDA || VECTORIZATION_HAS_HIP
