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

// Backend-neutral gpu*/cuda*/hip* aliases: gpuGetDeviceCount, gpuStreamCreate,
// gpuStreamDestroy, gpuStreamSynchronize, gpuSuccess. Lets callers write one
// code path instead of their own CUDA/HIP/Metal #if ladder.
//
// Metal has no stream concept, so it only gets gpuGetDeviceCount here (routed
// through the metal_device_probe.mm shim used by Testing/Cxx, since callers
// like TestTensorGpu.cpp must stay plain .cpp) — guard stream-based code with
// VECTORIZATION_HAS_CUDA || VECTORIZATION_HAS_HIP yourself.

#if VECTORIZATION_HAS_CUDA
#include <cuda_runtime.h>
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamDestroy cudaStreamDestroy
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuSuccess cudaSuccess
#elif VECTORIZATION_HAS_HIP
#include <hip/hip_runtime.h>
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuStreamCreate hipStreamCreate
#define gpuStreamDestroy hipStreamDestroy
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuSuccess hipSuccess
#elif VECTORIZATION_HAS_METAL
// Implemented in Testing/Cxx/metal_device_probe.mm (Objective-C++) — callers
// needing this branch (e.g. TestTensorGpu.cpp) must stay plain .cpp themselves.
extern "C" int xsigma_metal_device_count();
#define gpuGetDeviceCount(pn) (*(pn) = xsigma_metal_device_count())
#endif
