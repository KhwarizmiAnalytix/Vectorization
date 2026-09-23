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

#include <algorithm>
#include <cstdint>

#include "common/span.h"
#include "common/vectorization_export.h"
#include "common/vectorization_macros.h"

namespace vectorization
{

// Packed container for tensor sizes and strides.
// This design improves on the previous approach of using a pair of
// SmallVector<int64_t, 5> by specializing for the operations we
// actually use and enforcing that the number of sizes is the same as
// the number of strides. The memory layout is as follows:
//
// 1 size_t for the size
// 5 eightbytes of inline sizes and 5 eightbytes of inline strides, OR pointer
// to out-of-line array
class VECTORIZATION_API sizes_and_strides
{
public:
    static constexpr size_t MAX_INLINE_SIZE = 5;

    // TODO: different iterator types for sizes & strides to prevent
    // mixing the two accidentally.
    using sizes_iterator         = int64_t*;
    using sizes_const_iterator   = const int64_t*;
    using strides_iterator       = int64_t*;
    using strides_const_iterator = const int64_t*;

    sizes_and_strides()
    {
        size_at_unchecked(0)   = 0;
        stride_at_unchecked(0) = 1;
    }

    ~sizes_and_strides()
    {
        if (VECTORIZATION_UNLIKELY(!isInline()))
        {
            // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
            free(outOfLineStorage_);
        }
    }

    sizes_and_strides(const sizes_and_strides& rhs) : size_(rhs.size_)
    {
        if (VECTORIZATION_LIKELY(rhs.isInline()))
        {
            copyDataInline(rhs);
        }
        else
        {
            allocateOutOfLineStorage(size_);
            copyDataOutline(rhs);
        }
    }

    bool operator==(const sizes_and_strides& other) const
    {
        if (size_ != other.size_)
        {
            return false;
        }
        return !(
            isInline()
                ? std::memcmp(inlineStorage_, other.inlineStorage_, sizeof(inlineStorage_))
                : std::memcmp(outOfLineStorage_, other.outOfLineStorage_, storageBytes(size_)));
    }

    bool operator!=(const sizes_and_strides& other) const { return !(*this == other); }

    sizes_and_strides& operator=(const sizes_and_strides& rhs)
    {
        if (this == &rhs)
        {
            return *this;
        }
        if (VECTORIZATION_LIKELY(rhs.isInline()))
        {
            if (VECTORIZATION_UNLIKELY(!isInline()))
            {
                // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
                free(outOfLineStorage_);
            }
            copyDataInline(rhs);
        }
        else
        {
            if (isInline())
            {
                allocateOutOfLineStorage(rhs.size_);
            }
            else
            {
                resizeOutOfLineStorage(rhs.size_);
            }
            copyDataOutline(rhs);
        }
        size_ = rhs.size_;
        return *this;
    }

    // Move from rhs. rhs.size() == 0 afterwards.
    sizes_and_strides(sizes_and_strides&& rhs) noexcept : size_(rhs.size_)
    {
        if (VECTORIZATION_LIKELY(isInline()))
        {
            memcpy(inlineStorage_, rhs.inlineStorage_, sizeof(inlineStorage_));
        }
        else
        {
            outOfLineStorage_     = rhs.outOfLineStorage_;
            rhs.outOfLineStorage_ = nullptr;
        }

        rhs.size_ = 0;
    }

    // Move from rhs. rhs.size() == 0 afterwards.
    sizes_and_strides& operator=(sizes_and_strides&& rhs) noexcept
    {
        if (this == &rhs)
        {
            return *this;
        }
        if (VECTORIZATION_LIKELY(rhs.isInline()))
        {
            if (VECTORIZATION_UNLIKELY(!isInline()))
            {
                // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
                free(outOfLineStorage_);
            }
            copyDataInline(rhs);
        }
        else
        {
            // They're outline. We're going to steal their vector.
            if (!isInline())
            {
                // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
                free(outOfLineStorage_);
            }
            outOfLineStorage_     = rhs.outOfLineStorage_;
            rhs.outOfLineStorage_ = nullptr;
        }
        size_     = rhs.size_;
        rhs.size_ = 0;

        return *this;
    }

    size_t size() const noexcept { return size_; }

    const int64_t* sizes_data() const noexcept
    {
        if (VECTORIZATION_LIKELY(isInline()))
        {
            return &inlineStorage_[0];
        }
        else
        {
            return &outOfLineStorage_[0];
        }
    }

    int64_t* sizes_data() noexcept
    {
        if (VECTORIZATION_LIKELY(isInline()))
        {
            return &inlineStorage_[0];
        }
        else
        {
            return &outOfLineStorage_[0];
        }
    }

    sizes_const_iterator sizes_begin() const noexcept { return sizes_data(); }

    sizes_iterator sizes_begin() noexcept { return sizes_data(); }

    sizes_const_iterator sizes_end() const noexcept { return sizes_begin() + size(); }

    sizes_iterator sizes_end() noexcept { return sizes_begin() + size(); }

    span<const int64_t> sizes_arrayref() const noexcept { return {sizes_data(), size()}; }

    void set_sizes(span<const int64_t> newSizes)
    {
        resize(newSizes.size());
        std::copy(newSizes.begin(), newSizes.end(), sizes_begin());
    }

    void set_strides(span<const int64_t> strides)
    {
        VECTORIZATION_CHECK(strides.size() == size());
        std::copy(strides.begin(), strides.end(), strides_begin());
    }

    const int64_t* strides_data() const noexcept
    {
        if (VECTORIZATION_LIKELY(isInline()))
        {
            return &inlineStorage_[MAX_INLINE_SIZE];
        }
        else
        {
            return &outOfLineStorage_[size()];
        }
    }

    int64_t* strides_data() noexcept
    {
        if (VECTORIZATION_LIKELY(isInline()))
        {
            return &inlineStorage_[MAX_INLINE_SIZE];
        }
        else
        {
            return &outOfLineStorage_[size()];
        }
    }

    strides_const_iterator strides_begin() const noexcept
    {
        if (VECTORIZATION_LIKELY(isInline()))
        {
            return &inlineStorage_[MAX_INLINE_SIZE];
        }
        else
        {
            return &outOfLineStorage_[size()];
        }
    }

    strides_iterator strides_begin() noexcept
    {
        if (VECTORIZATION_LIKELY(isInline()))
        {
            return &inlineStorage_[MAX_INLINE_SIZE];
        }
        else
        {
            return &outOfLineStorage_[size()];
        }
    }

    strides_const_iterator strides_end() const noexcept { return strides_begin() + size(); }

    strides_iterator strides_end() noexcept { return strides_begin() + size(); }

    span<const int64_t> strides_arrayref() const noexcept { return {strides_data(), size()}; }

    // Size accessors.
    int64_t size_at(size_t idx) const noexcept
    {
        assert(idx < size());
        return sizes_data()[idx];
    }

    int64_t& size_at(size_t idx) noexcept
    {
        assert(idx < size());
        return sizes_data()[idx];
    }

    int64_t size_at_unchecked(size_t idx) const noexcept { return sizes_data()[idx]; }

    int64_t& size_at_unchecked(size_t idx) noexcept { return sizes_data()[idx]; }

    // Stride accessors.
    int64_t stride_at(size_t idx) const noexcept
    {
        assert(idx < size());
        return strides_data()[idx];
    }

    int64_t& stride_at(size_t idx) noexcept
    {
        assert(idx < size());
        return strides_data()[idx];
    }

    int64_t stride_at_unchecked(size_t idx) const noexcept { return strides_data()[idx]; }

    int64_t& stride_at_unchecked(size_t idx) noexcept { return strides_data()[idx]; }

    void resize(size_t newSize)
    {
        const auto oldSize = size();
        if (newSize == oldSize)
        {
            return;
        }
        if (VECTORIZATION_LIKELY(newSize <= MAX_INLINE_SIZE && isInline()))
        {
            if (oldSize < newSize)
            {
                const auto bytesToZero = (newSize - oldSize) * sizeof(inlineStorage_[0]);
                memset(&inlineStorage_[oldSize], 0, bytesToZero);
                memset(&inlineStorage_[MAX_INLINE_SIZE + oldSize], 0, bytesToZero);
            }
            size_ = newSize;
        }
        else
        {
            resizeSlowPath(newSize, oldSize);
        }
    }

    void resizeSlowPath(size_t newSize, size_t oldSize);

private:
    bool isInline() const noexcept { return size_ <= MAX_INLINE_SIZE; }

    void copyDataInline(const sizes_and_strides& rhs)
    {
        VECTORIZATION_CHECK_DEBUG(rhs.isInline());
        memcpy(inlineStorage_, rhs.inlineStorage_, sizeof(inlineStorage_));
    }

    static size_t storageBytes(size_t size) noexcept { return size * 2 * sizeof(int64_t); }

    void allocateOutOfLineStorage(size_t size)
    {
        // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
        outOfLineStorage_ = static_cast<int64_t*>(malloc(storageBytes(size)));
        VECTORIZATION_CHECK(
            outOfLineStorage_, "Could not allocate memory for Tensor sizes_and_strides!");
    }

    void resizeOutOfLineStorage(size_t newSize)
    {
        VECTORIZATION_CHECK_DEBUG(!isInline());
        outOfLineStorage_ = static_cast<int64_t*>(
            // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
            realloc(outOfLineStorage_, storageBytes(newSize)));
        VECTORIZATION_CHECK(
            outOfLineStorage_, "Could not allocate memory for Tensor sizes_and_strides!");
    }

    void copyDataOutline(const sizes_and_strides& rhs) noexcept
    {
        memcpy(outOfLineStorage_, rhs.outOfLineStorage_, storageBytes(rhs.size_));
    }

    size_t size_{1};
    union
    {
        int64_t* outOfLineStorage_;
        // NOLINTNEXTLINE(*c-array*)
        int64_t inlineStorage_[MAX_INLINE_SIZE * 2]{};
    };
};

}  // namespace vectorization
