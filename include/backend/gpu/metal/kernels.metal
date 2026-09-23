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

// Fixed, hand-written Metal kernels for fill, reductions, and metal_backend::dispatch()
// tests. Expression trees such as `x = y + a + 5*d` are emitted as one fused MSL kernel
// at runtime (see expressions_evaluator_metal.h) — there is no per-node lowering.
// float-only: MSL has no double type on any Apple GPU. One thread per element; buffer
// argument order is always (inputs..., out, n).
//
// This source is embedded into the Vectorization library as a C++ string at CMake
// configure time (see the "Metal GPU backend" block in CMakeLists.txt) and compiled at
// runtime via -[MTLDevice newLibraryWithSource:options:error:] — there is no .metallib
// build step.

#include <metal_stdlib>
using namespace metal;

// -----------------------------------------------------------------------------------
// Fill
// -----------------------------------------------------------------------------------

kernel void fill_float(
    device float* out [[buffer(0)]],
    constant float& value [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n)
        out[tid] = value;
}

// -----------------------------------------------------------------------------------
// Binary arithmetic
// -----------------------------------------------------------------------------------

kernel void add_float(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n)
        out[tid] = a[tid] + b[tid];
}

kernel void sub_float(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n)
        out[tid] = a[tid] - b[tid];
}

kernel void mul_float(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n)
        out[tid] = a[tid] * b[tid];
}

kernel void div_float(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n)
        out[tid] = a[tid] / b[tid];
}

// -----------------------------------------------------------------------------------
// Ternary: fused multiply-add
// -----------------------------------------------------------------------------------

kernel void fma_float(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device const float* c [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n)
        out[tid] = fma(a[tid], b[tid], c[tid]);
}

// -----------------------------------------------------------------------------------
// Unary math
// -----------------------------------------------------------------------------------

kernel void sqrt_float(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n)
        out[tid] = sqrt(a[tid]);
}

kernel void exp_float(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n)
        out[tid] = exp(a[tid]);
}

kernel void log_float(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n)
        out[tid] = log(a[tid]);
}

kernel void sin_float(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n)
        out[tid] = sin(a[tid]);
}

kernel void cos_float(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n)
        out[tid] = cos(a[tid]);
}

kernel void tanh_float(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n)
        out[tid] = tanh(a[tid]);
}

kernel void fabs_float(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n)
        out[tid] = fabs(a[tid]);
}

kernel void neg_float(
    device const float* a [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < n)
        out[tid] = -a[tid];
}

// -----------------------------------------------------------------------------------
// Reductions: one threadgroup writes one partial (out[gid]). The host launcher
// walks the partials until a single value remains, so n is not limited to one
// threadgroup.
// -----------------------------------------------------------------------------------
kernel void reduce_sum_float(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]])
{
    shared[lid] = (tid < n) ? in[tid] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1)
    {
        if (lid < stride)
            shared[lid] += shared[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0)
        out[gid] = shared[0];
}

kernel void reduce_min_float(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]])
{
    shared[lid] = (tid < n) ? in[tid] : FLT_MAX;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1)
    {
        if (lid < stride)
            shared[lid] = min(shared[lid], shared[lid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0)
        out[gid] = shared[0];
}

kernel void reduce_max_float(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]])
{
    shared[lid] = (tid < n) ? in[tid] : -FLT_MAX;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1)
    {
        if (lid < stride)
            shared[lid] = max(shared[lid], shared[lid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0)
        out[gid] = shared[0];
}
