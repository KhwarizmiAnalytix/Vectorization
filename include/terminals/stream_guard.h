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

// Ambient "current stream" for GPU tensor operations, mirroring PyTorch's
// c10::cuda::CUDAStreamGuard / at::cuda::getCurrentCUDAStream(). Lets ordinary tensor
// syntax (operator=, expression construction, scalar fill) transparently run on a
// caller-scoped stream instead of requiring the differently-named *_async API for
// every call. Keyed by device_index (thread_local, one entry per device) since this
// library has no single ambient "current device" the way PyTorch does.
//
// Requires gpu_stream_t to already be visible (tensor.h includes this after
// "expressions/expressions.h", which defines it for every backend configuration).

#include <cstddef>
#include <unordered_map>

namespace vectorization
{
namespace detail
{
// One entry per device with an explicit stream or active guards on this thread.
// A guard pins its entry so restoration never needs to allocate. Empty on every
// thread until a stream is selected or a guard is constructed.
struct stream_state
{
    gpu_stream_t stream{nullptr};
    std::size_t  guards{0};
};

inline thread_local std::unordered_map<int, stream_state> g_current_streams;
}  // namespace detail

// Returns the ambient stream set by the innermost active stream_guard for device_index on
// this thread, or nullptr (the default/null stream) if none is active.
VECTORIZATION_HOST_ONLY inline gpu_stream_t current_stream(int device_index = 0)
{
    auto& streams = detail::g_current_streams;
    if (streams.empty())
    {
        return nullptr;
    }
    auto it = streams.find(device_index);
    return it == streams.end() ? nullptr : it->second.stream;
}

// Directly sets the ambient stream for device_index on this thread; passing nullptr clears
// it back to the default stream. Prefer stream_guard (RAII) over calling this directly so
// the previous ambient stream is restored automatically.
// Not noexcept: the underlying unordered_map insert can throw std::bad_alloc under OOM,
// and letting that propagate (rather than terminating) matches how allocation failures are
// handled everywhere else in this library (e.g. tensor's own constructors).
VECTORIZATION_HOST_ONLY inline void set_current_stream(gpu_stream_t stream, int device_index = 0)
{
    if (stream == nullptr)
    {
        auto& streams = detail::g_current_streams;
        auto  it      = streams.find(device_index);
        if (it != streams.end())
        {
            if (it->second.guards == 0)
            {
                streams.erase(it);
            }
            else
            {
                it->second.stream = nullptr;
            }
        }
    }
    else
    {
        detail::g_current_streams[device_index].stream = stream;
    }
}

// RAII guard: makes @p stream the ambient current stream for @p device_index for its
// lifetime, restoring whatever was current before on destruction (nested guards compose
// correctly). Not copyable — same RAII style as std::scoped_lock.
//
//   {
//       vectorization::stream_guard guard(my_stream, device_index);
//       tensor<float> c = a + b;   // runs on my_stream
//       d = c * 2.0f;               // also runs on my_stream
//   }
//   // ambient stream reverts here
class stream_guard
{
public:
    VECTORIZATION_HOST_FUNCTION_ATTRIBUTE explicit stream_guard(
        gpu_stream_t stream, int device_index = 0)
        : device_index_(device_index), prev_stream_(current_stream(device_index))
    {
        // Reserve the entry before changing ambient state. Active guards keep
        // it alive even when a nested guard or direct setter selects nullptr.
        auto& state = detail::g_current_streams[device_index_];
        ++state.guards;
        state.stream = stream;
    }

    VECTORIZATION_HOST_FUNCTION_ATTRIBUTE ~stream_guard() noexcept
    {
        auto& streams     = detail::g_current_streams;
        auto  it          = streams.find(device_index_);
        it->second.stream = prev_stream_;
        if (--it->second.guards == 0 && prev_stream_ == nullptr)
        {
            streams.erase(it);
        }
    }

    stream_guard(const stream_guard&)            = delete;
    stream_guard& operator=(const stream_guard&) = delete;
    stream_guard(stream_guard&&)                 = delete;
    stream_guard& operator=(stream_guard&&)      = delete;

private:
    int          device_index_;
    gpu_stream_t prev_stream_;
};

}  // namespace vectorization
