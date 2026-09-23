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

// nvcc's device-code frontend (cicc) always defines __NVCC__, which forces
// fmt/include/fmt/base.h to set FMT_USE_INT128=0 and fall back to fmt's own
// fmt::detail::uint128 class instead of the compiler-native __uint128_t —
// even for translation units, like this one, that only ever format on the
// host. fmt::detail::uint128 does not define operator~, so
// fmt::detail::format_hexfloat's `f.f &= ~(inc - 1);` (carrier_uint for
// `long double`, ThirdParty/fmt/include/fmt/format.h) fails to compile under
// nvcc with "no operator '~' matches these operands", even though the code
// path never executes on device.
//
// ThirdParty/ is vendored and must never be edited (see CLAUDE.md). Supply
// the missing operator here instead, in fmt's own namespace so ADL finds it
// at the point format_hexfloat<long double> is instantiated.
//
// Not force-included via -include on the nvcc command line: nvcc reapplies
// -include when it recompiles its own flattened cudafe1.cpp intermediate,
// which would duplicate all of <fmt> with no header guards left and corrupt
// the parser's namespace state. Instead this header is pulled in, guarded by
// `#if VECTORIZATION_HAS_CUDA`, from the top of VectorizationTest.h.in (see
// Testing/VectorizationTest.h.in) so every GTest-based file under Testing/Cxx
// gets it for free via its existing `#include "VectorizationTest.h"` instead
// of each one repeating this block. BenchmarkTensorGpu.cpp doesn't use
// VectorizationTest.h (it's Google Benchmark, not GTest), so it still
// includes this header directly.
#include <fmt/format.h>

namespace fmt
{
namespace detail
{
constexpr uint128 operator~(const uint128& x)
{
    return uint128(~x.high(), ~x.low());
}
}  // namespace detail
}  // namespace fmt
