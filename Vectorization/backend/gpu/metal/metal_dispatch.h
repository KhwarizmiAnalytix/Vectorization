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

// Plain C++ surface over the fixed Metal kernel set (backend/gpu/metal/kernels.metal) —
// no Objective-C types cross this boundary, so expressions_evaluator_metal.h (header-only,
// compiled by ordinary clang++) can call into it without becoming Objective-C++ itself.
//
// in_buffers/out_buffer are raw host pointers previously returned by
// memory::allocator<float>::allocate(..., device_enum::METAL) — host-visible offsets into
// shared-storage MTLBuffers (possibly packed by the Metal caching allocator). dispatch()
// resolves them via memory::metal::mtl_buffer_handle() + mtl_buffer_offset() before
// binding with setBuffer:offset:.

#include <cstddef>
#include <string>

namespace vectorization::metal_backend
{
bool device_available();

// Dispatches the kernel named kernel_name (e.g. "add" -> "add_float") over n_elems
// threads. in_buffers holds n_in input pointers, consumed in the same order the
// corresponding .metal kernel declares its [[buffer(k)]] arguments. Synchronous
// (waits for GPU completion before returning).
void dispatch(
    const char*        kernel_name,
    const void* const* in_buffers,
    int                n_in,
    void*              out_buffer,
    std::size_t        n_elems);

void dispatch_fill(void* out_buffer, float value, std::size_t n_elems);

// One fused element-wise kernel compiled from `kernel_source` (MSL). Tensor
// leaves bind as device buffers 0..n_in-1, scalars as constant buffers
// n_in..n_in+n_scalars-1 (setBytes), then out and n. The kernel function
// must be named fused_float. Source is compiled once and cached.
void dispatch_fused(
    std::string const& kernel_source,
    const void* const* in_buffers,
    int                n_in,
    float const*       scalars,
    int                n_scalars,
    void*              out_buffer,
    std::size_t        n_elems);

// Multi-block reductions. Each threadgroup writes one partial; the host
// launcher ping-pongs until a single value remains. n_elems is not limited
// to maxTotalThreadsPerThreadgroup. Empty input returns the matching
// accumulate / hmin / hmax identity (0, +FLT_MAX, -FLT_MAX).
float reduce_sum(const void* buffer, std::size_t n_elems);
float reduce_min(const void* buffer, std::size_t n_elems);
float reduce_max(const void* buffer, std::size_t n_elems);

}  // namespace vectorization::metal_backend
