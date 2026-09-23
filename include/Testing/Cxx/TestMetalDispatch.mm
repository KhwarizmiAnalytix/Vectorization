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

// Exercises the fixed kernel set through backend/gpu/metal/metal_dispatch.h
// (fill, add, sin, fma, reduce). Expression-tree fusion is covered by
// TestTensorGpu / MetalDispatch.FusedYPlusAPlusFiveD, not this file.

#include "VectorizationTest.h"

#if VECTORIZATION_HAS_METAL

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "allocator.h"
#include "backend/gpu/metal/metal_dispatch.h"
#include "common/device.h"

namespace
{
using alloc_t = memory::allocator<float>;

std::vector<float> download(const float* metal_ptr, size_t n)
{
    std::vector<float> host(n);
    alloc_t::copy(metal_ptr, n, host.data(), memory::device_enum::METAL, memory::device_enum::CPU);
    return host;
}

float* upload(const std::vector<float>& host)
{
    float* ptr = alloc_t::allocate(host.size(), memory::device_enum::METAL);
    alloc_t::copy(
        host.data(), host.size(), ptr, memory::device_enum::CPU, memory::device_enum::METAL);
    return ptr;
}
}  // namespace

VECTORIZATIONTEST(MetalDispatch, DeviceAvailable)
{
    EXPECT_TRUE(vectorization::metal_backend::device_available());
    END_TEST();
}

VECTORIZATIONTEST(MetalDispatch, Fill)
{
    constexpr size_t n   = 512;
    float*           out = alloc_t::allocate(n, memory::device_enum::METAL);

    vectorization::metal_backend::dispatch_fill(out, 3.5f, n);

    auto host = download(out, n);
    for (float v : host)
        EXPECT_FLOAT_EQ(v, 3.5f);

    alloc_t::free(out, memory::device_enum::METAL);
    END_TEST();
}

VECTORIZATIONTEST(MetalDispatch, Add)
{
    constexpr size_t   n = 300;
    std::vector<float> a(n), b(n);
    for (size_t i = 0; i < n; ++i)
    {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i);
    }

    float* da    = upload(a);
    float* db    = upload(b);
    float* doubt = alloc_t::allocate(n, memory::device_enum::METAL);

    const void* ins[] = {da, db};
    vectorization::metal_backend::dispatch("add", ins, 2, doubt, n);

    auto host = download(doubt, n);
    for (size_t i = 0; i < n; ++i)
        EXPECT_FLOAT_EQ(host[i], a[i] + b[i]);

    alloc_t::free(da, memory::device_enum::METAL);
    alloc_t::free(db, memory::device_enum::METAL);
    alloc_t::free(doubt, memory::device_enum::METAL);
    END_TEST();
}

VECTORIZATIONTEST(MetalDispatch, Sin)
{
    constexpr size_t   n = 200;
    std::vector<float> a(n);
    for (size_t i = 0; i < n; ++i)
        a[i] = static_cast<float>(i) * 0.01f;

    float* da    = upload(a);
    float* doubt = alloc_t::allocate(n, memory::device_enum::METAL);

    const void* ins[] = {da};
    vectorization::metal_backend::dispatch("sin", ins, 1, doubt, n);

    auto host = download(doubt, n);
    for (size_t i = 0; i < n; ++i)
        EXPECT_NEAR(host[i], std::sin(a[i]), 1e-5f);

    alloc_t::free(da, memory::device_enum::METAL);
    alloc_t::free(doubt, memory::device_enum::METAL);
    END_TEST();
}

VECTORIZATIONTEST(MetalDispatch, Fma)
{
    constexpr size_t   n = 128;
    std::vector<float> a(n), b(n), c(n);
    for (size_t i = 0; i < n; ++i)
    {
        a[i] = static_cast<float>(i) * 0.5f;
        b[i] = static_cast<float>(i) * 0.25f;
        c[i] = 1.0f;
    }

    float* da    = upload(a);
    float* db    = upload(b);
    float* dc    = upload(c);
    float* doubt = alloc_t::allocate(n, memory::device_enum::METAL);

    const void* ins[] = {da, db, dc};
    vectorization::metal_backend::dispatch("fma", ins, 3, doubt, n);

    auto host = download(doubt, n);
    for (size_t i = 0; i < n; ++i)
        EXPECT_NEAR(host[i], a[i] * b[i] + c[i], 1e-5f);

    alloc_t::free(da, memory::device_enum::METAL);
    alloc_t::free(db, memory::device_enum::METAL);
    alloc_t::free(dc, memory::device_enum::METAL);
    alloc_t::free(doubt, memory::device_enum::METAL);
    END_TEST();
}

VECTORIZATIONTEST(MetalDispatch, ReduceSum)
{
    constexpr size_t   n = 512;
    std::vector<float> a(n);
    float              expected = 0.0f;
    for (size_t i = 0; i < n; ++i)
    {
        a[i] = static_cast<float>(i) * 0.01f - 1.0f;
        expected += a[i];
    }

    float* da  = upload(a);
    float  got = vectorization::metal_backend::reduce_sum(da, n);
    EXPECT_NEAR(got, expected, 1e-2f);

    alloc_t::free(da, memory::device_enum::METAL);
    END_TEST();
}

VECTORIZATIONTEST(MetalDispatch, ReduceSumNonPowerOfTwo)
{
    constexpr size_t   n = 300;
    std::vector<float> a(n);
    float              expected = 0.0f;
    for (size_t i = 0; i < n; ++i)
    {
        a[i] = 1.0f;
        expected += a[i];
    }

    float* da  = upload(a);
    float  got = vectorization::metal_backend::reduce_sum(da, n);
    EXPECT_NEAR(got, expected, 1e-3f);

    alloc_t::free(da, memory::device_enum::METAL);
    END_TEST();
}

VECTORIZATIONTEST(MetalDispatch, ReduceSumLarge)
{
    constexpr size_t   n = 10000;
    std::vector<float> a(n, 1.0f);
    float*             da  = upload(a);
    float              got = vectorization::metal_backend::reduce_sum(da, n);
    EXPECT_NEAR(got, static_cast<float>(n), 1e-2f);
    alloc_t::free(da, memory::device_enum::METAL);
    END_TEST();
}

VECTORIZATIONTEST(MetalDispatch, ReduceMinMax)
{
    constexpr size_t   n = 3000;
    std::vector<float> a(n);
    for (size_t i = 0; i < n; ++i)
        a[i] = static_cast<float>(i) * 0.01f - 10.0f;
    float* da   = upload(a);
    float  gmin = vectorization::metal_backend::reduce_min(da, n);
    float  gmax = vectorization::metal_backend::reduce_max(da, n);
    EXPECT_NEAR(gmin, a.front(), 1e-5f);
    EXPECT_NEAR(gmax, a.back(), 1e-5f);
    alloc_t::free(da, memory::device_enum::METAL);
    END_TEST();
}

VECTORIZATIONTEST(MetalDispatch, ReduceEmpty)
{
    EXPECT_EQ(vectorization::metal_backend::reduce_sum(nullptr, 0), 0.0f);
    EXPECT_EQ(
        vectorization::metal_backend::reduce_min(nullptr, 0), std::numeric_limits<float>::max());
    EXPECT_EQ(
        vectorization::metal_backend::reduce_max(nullptr, 0), -std::numeric_limits<float>::max());
    END_TEST();
}

VECTORIZATIONTEST(MetalDispatch, FusedYPlusAPlusFiveD)
{
    constexpr size_t   n = 256;
    std::vector<float> y(n), a(n), d(n), ref(n);
    for (size_t i = 0; i < n; ++i)
    {
        y[i]   = static_cast<float>(i);
        a[i]   = 2.0f;
        d[i]   = 0.5f;
        ref[i] = y[i] + a[i] + 5.0f * d[i];
    }
    float* dy = upload(y);
    float* da = upload(a);
    float* dd = upload(d);
    float* dx = alloc_t::allocate(n, memory::device_enum::METAL);

    std::string const src =
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void fused_float(\n"
        "  device const float* in0 [[buffer(0)]],\n"
        "  device const float* in1 [[buffer(1)]],\n"
        "  device const float* in2 [[buffer(2)]],\n"
        "  constant float& c0 [[buffer(3)]],\n"
        "  device float* out [[buffer(4)]],\n"
        "  constant uint& n [[buffer(5)]],\n"
        "  uint tid [[thread_position_in_grid]])\n"
        "{\n  if (tid < n)\n    out[tid] = ((in0[tid]+in1[tid])+(c0*in2[tid]));\n}\n";
    const void* ins[] = {dy, da, dd};
    float       five  = 5.0f;
    vectorization::metal_backend::dispatch_fused(src, ins, 3, &five, 1, dx, n);

    auto got = download(dx, n);
    for (size_t i = 0; i < n; ++i)
        EXPECT_FLOAT_EQ(got[i], ref[i]) << " at i=" << i;

    alloc_t::free(dx, memory::device_enum::METAL);
    alloc_t::free(dd, memory::device_enum::METAL);
    alloc_t::free(da, memory::device_enum::METAL);
    alloc_t::free(dy, memory::device_enum::METAL);
    END_TEST();
}

#endif  // VECTORIZATION_HAS_METAL
