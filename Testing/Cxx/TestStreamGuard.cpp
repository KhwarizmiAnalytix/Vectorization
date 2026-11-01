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

#include <stdexcept>
#include <type_traits>

#include "VectorizationTest.h"
#include "terminals/tensor.h"

namespace
{
// These opaque tokens are never passed to a GPU API; stream selection is host state.
vectorization::gpu_stream_t stream_token(int& token)
{
    return reinterpret_cast<vectorization::gpu_stream_t>(&token);
}
}  // namespace

VECTORIZATIONTEST(StreamGuard, NestedDefaultAndDeviceStreams)
{
    using namespace vectorization;
    static_assert(std::is_nothrow_destructible_v<stream_guard>);
    int        first  = 0;
    int        second = 0;
    const auto a      = stream_token(first);
    const auto b      = stream_token(second);
    EXPECT_EQ(current_stream(0), nullptr);
    EXPECT_EQ(current_stream(1), nullptr);
    {
        stream_guard outer(a);
        EXPECT_EQ(current_stream(), a);
        {
            stream_guard default_stream(nullptr);
            EXPECT_EQ(current_stream(), nullptr);
            {
                stream_guard nested(b);
                stream_guard other_device(b, 1);
                EXPECT_EQ(current_stream(), b);
                EXPECT_EQ(current_stream(1), b);
            }
            EXPECT_EQ(current_stream(), nullptr);
            EXPECT_EQ(current_stream(1), nullptr);
        }
        EXPECT_EQ(current_stream(), a);
    }
    EXPECT_EQ(current_stream(), nullptr);
    EXPECT_EQ(current_stream(1), nullptr);
    END_TEST();
}

VECTORIZATIONTEST(StreamGuard, DirectSetterAndExceptionUnwinding)
{
    using namespace vectorization;
    int        first  = 0;
    int        second = 0;
    const auto a      = stream_token(first);
    const auto b      = stream_token(second);
    // A direct selection made before the guard must survive a nested reset.
    set_current_stream(a);
    try
    {
        stream_guard outer(b);
        set_current_stream(nullptr);
        EXPECT_EQ(current_stream(), nullptr);
        {
            stream_guard nested(a);
            set_current_stream(b);
            EXPECT_EQ(current_stream(), b);
        }
        EXPECT_EQ(current_stream(), nullptr);
        throw std::runtime_error("unwind the stream guard");
    }
    catch (std::runtime_error const&)
    {
        EXPECT_EQ(current_stream(), a);
    }
    set_current_stream(nullptr);
    EXPECT_EQ(current_stream(), nullptr);
    END_TEST();
}
