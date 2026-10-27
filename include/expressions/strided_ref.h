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

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "common/data_view.h"
#include "common/device.h"
#include "common/vectorization_macros.h"

namespace vectorization
{

static constexpr int kMaxRank = 8;

// Trivially-copyable tensor descriptor stored as expression leaves.
// Replaces data_view<T>, preserving shape and strides so the planner and
// strided evaluation can read them. The embedded data_view handles device
// placement and stream bookkeeping unchanged.
template <typename T, int MaxRank = kMaxRank>
struct strided_ref
{
    memory::data_view<T> view_{};
    int     rank{0};
    int64_t sizes[MaxRank]{};
    int64_t strides[MaxRank]{};
    size_t  numel{0};
    bool    contiguous{false};
    size_t  align_start{0};
    size_t  align_end{0};

    VECTORIZATION_FUNCTION_ATTRIBUTE T*          data()          const noexcept { return view_.data(); }
    VECTORIZATION_FUNCTION_ATTRIBUTE size_t      size()          const noexcept { return numel; }
    VECTORIZATION_FUNCTION_ATTRIBUTE device_enum device()        const noexcept { return view_.device(); }
    VECTORIZATION_FUNCTION_ATTRIBUTE int         device_index()  const noexcept { return view_.device_index(); }
    VECTORIZATION_FUNCTION_ATTRIBUTE auto        stream()        const noexcept { return view_.stream(); }
    VECTORIZATION_FUNCTION_ATTRIBUTE bool        is_aligned()    const noexcept { return view_.is_aligned(); }
    VECTORIZATION_FUNCTION_ATTRIBUTE bool        is_contiguous() const noexcept { return contiguous; }
    // Delegation to data_view for allocation-identity checks used by tests.
    VECTORIZATION_FUNCTION_ATTRIBUTE auto        base()          const noexcept { return view_.base(); }
    VECTORIZATION_FUNCTION_ATTRIBUTE auto        id()            const noexcept { return view_.id(); }

    template <typename Stream>
    VECTORIZATION_HOST_FUNCTION_ATTRIBUTE void record_stream(Stream s) { view_.record_stream(s); }
};

// Compute the element offset for flat index `flat` using the strided layout.
// Returns `flat` directly when the tensor is contiguous.
template <typename T, int MaxRank>
VECTORIZATION_FUNCTION_ATTRIBUTE std::size_t strided_offset(
    strided_ref<T, MaxRank> const& s, std::size_t flat) noexcept
{
    if (s.contiguous)
        return flat;
    const int n   = s.rank;
    size_t    off = 0;
    size_t    rem = flat;
    for (int k = n - 1; k >= 0; --k)
    {
        const auto   dk  = static_cast<size_t>(s.sizes[k]);
        if (dk == 0) return 0;
        const size_t idx = rem % dk;
        rem /= dk;
        off += idx * static_cast<size_t>(s.strides[k]);
    }
    return off;
}

// ADL-friendly leaf_offset overload for strided_ref — used by expression_loader.
template <typename T, int MaxRank>
VECTORIZATION_FUNCTION_ATTRIBUTE size_t leaf_offset(
    strided_ref<T, MaxRank> const& s, size_t idx) noexcept
{
    return strided_offset(s, idx);
}

// One operand's current row inside the CPU planner's loop2d kernel
// (iteration_plan.h). `ptr` is the row's first element; `stride` is the
// operand's innermost stride after planning: 1 (unit stride), 0 (broadcast
// along the row) or anything else (strided, scalar loop only). The planner
// rewrites `ptr` once per row, so element i of the row is ptr[i * stride] and
// no index is ever divided.
template <typename T>
struct row_ref
{
    using value_type = T;

    T const* ptr{nullptr};
    int64_t  stride{0};
};

template <typename T>
struct is_strided_ref : std::false_type
{
};

template <typename T, int MaxRank>
struct is_strided_ref<strided_ref<T, MaxRank>> : std::true_type
{
};

template <typename T>
struct is_row_ref : std::false_type
{
};

template <typename T>
struct is_row_ref<row_ref<T>> : std::true_type
{
};

}  // namespace vectorization
