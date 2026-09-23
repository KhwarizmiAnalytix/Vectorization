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

#include "common/vectorization_macros.h"

// C++20 and later: vectorization::span is std::span.
// C++17: a minimal stand-in with the same construction / size / indexing /
// iterator surface used by tensor shape accessors.
#if __cplusplus >= 202002L
#include <span>

namespace vectorization
{
using std::span;
}

#else

#include <cstddef>
#include <type_traits>

namespace vectorization
{
template <typename T>
class span
{
public:
    using element_type = T;
    using value_type   = std::remove_cv_t<T>;
    using size_type    = std::size_t;
    using pointer      = T*;
    using reference    = T&;
    using iterator     = pointer;

    constexpr span() noexcept : data_(nullptr), size_(0) {}

    constexpr span(pointer ptr, size_type count) noexcept : data_(ptr), size_(count) {}

    constexpr iterator                     begin() const noexcept { return data_; }
    constexpr iterator                     end() const noexcept { return data_ + size_; }
    constexpr size_type                    size() const noexcept { return size_; }
    constexpr pointer                      data() const noexcept { return data_; }
    VECTORIZATION_NODISCARD constexpr bool empty() const noexcept { return size_ == 0; }

    constexpr reference operator[](size_type idx) const noexcept { return data_[idx]; }

private:
    pointer   data_;
    size_type size_;
};
}  // namespace vectorization

#endif
