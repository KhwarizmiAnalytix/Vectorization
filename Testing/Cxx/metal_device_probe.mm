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

// Tiny Objective-C++ shim so TestTensorGpu.cpp — which must stay a plain .cpp, shared
// textually with the CUDA/HIP path — can query Metal device presence via a C-linkage
// call instead of becoming Objective-C++ itself.

#import <Metal/Metal.h>

extern "C" int xsigma_metal_device_count()
{
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    return dev != nil ? 1 : 0;
}
