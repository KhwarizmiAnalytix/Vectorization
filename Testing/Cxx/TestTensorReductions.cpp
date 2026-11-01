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

#include <algorithm>
#include <limits>
#include <type_traits>
#include <vector>

#include "VectorizationTest.h"
#include "terminals/tensor.h"

namespace
{
template <typename T> void check_reductions()
{
    using namespace vectorization;
    using tensor_t = tensor<T>;
    static_assert(std::is_same_v<decltype(mean(std::declval<tensor_t const&>())), T>);
    static_assert(std::is_same_v<decltype(all(std::declval<tensor_t const&>())), bool>);
    static_assert(
        std::is_same_v<decltype(count_nonzero(std::declval<tensor_t const&>())), std::uint64_t>);
    for (size_t n : {size_t(0),
             size_t(1),
             size_t(3),
             size_t(17),
             size_t(255),
             size_t(257),
             size_t(1025),
             size_t(66001)})
    {
        SCOPED_TRACE(n);
        tensor_t x(n);
        x = T(1);
        EXPECT_EQ(prod(x), T(1));
        EXPECT_EQ(sum(x), T(n));
        EXPECT_EQ(count_nonzero(x), n);
        EXPECT_TRUE(all(x));
        EXPECT_EQ(any(x), n != 0);
        EXPECT_EQ(nansum(x), T(n));
        if (n == 0)
        {
            EXPECT_TRUE(std::isnan(mean(x)));
            EXPECT_TRUE(std::isnan(nanmean(x)));
        }
        else
        {
            EXPECT_EQ(mean(x), T(1));
            EXPECT_EQ(nanmean(x), T(1));
            x[n - 1] = T(0);
            EXPECT_FALSE(all(x));
            EXPECT_EQ(count_nonzero(x), n - 1);
            EXPECT_EQ(prod(x), T(0));
            x = T(0);
            EXPECT_FALSE(any(x));
            EXPECT_EQ(count_nonzero(x), 0u);
        }
    }
    tensor_t x(67);
    x     = T(1);
    x[3]  = T(-2);
    x[66] = T(3);
    EXPECT_EQ(prod(x), T(-6));
    EXPECT_EQ(mean(x + T(2)), T(200) / T(67));
    EXPECT_TRUE(all(x != T(0)));
    EXPECT_TRUE(any(x < T(0)));
    EXPECT_EQ(count_nonzero(x < T(0)), 1u);
    EXPECT_EQ(sum(x < T(0)), T(1));
    EXPECT_EQ(mean(x < T(0)), T(1) / T(67));
    EXPECT_EQ(count_nonzero(x * T(0)), 0u);

    const T nan = std::numeric_limits<T>::quiet_NaN();
    x           = nan;
    EXPECT_EQ(nansum(x), T(0));
    EXPECT_TRUE(std::isnan(nanmean(x)));
    EXPECT_EQ(count_nonzero(x), 67u);  // NaN is truthy, as in PyTorch
    EXPECT_TRUE(all(x));
    EXPECT_TRUE(any(x));
    x[0]  = T(2);
    x[66] = T(4);
    EXPECT_EQ(nansum(x), T(6));
    EXPECT_EQ(nanmean(x), T(3));
    EXPECT_EQ(nansum(x * T(2)), T(12));
    EXPECT_TRUE(std::isnan(mean(x)));
    EXPECT_TRUE(std::isnan(prod(x)));
    x[0] = std::numeric_limits<T>::infinity();
    EXPECT_EQ(nansum(x), std::numeric_limits<T>::infinity());
    x[66] = -std::numeric_limits<T>::infinity();
    EXPECT_TRUE(std::isnan(nansum(x)));  // only input NaNs are ignored
}

template <typename T> void check_strides_and_shapes()
{
    using namespace vectorization;
    using tensor_t = tensor<T>;
    using dims_t   = typename tensor_t::dimensions_type;
    tensor_t x(dims_t{3, 5});
    for (size_t i = 0; i < x.size(); ++i)
    {
        x[i] = T(i + 1);
    }
    auto transposed = x.t();
    auto columns    = x.slice(1, 0, 5, 2);
    EXPECT_EQ(mean(transposed), T(8));
    EXPECT_EQ(count_nonzero(columns), 9u);
    EXPECT_EQ(nansum(columns * T(2)), T(144));
    EXPECT_TRUE(all(transposed));
    EXPECT_TRUE(any(columns - T(1)));
    tensor_t row(dims_t{5});
    row = T(1);
    EXPECT_EQ(mean(x + row), T(9));
    EXPECT_EQ(count_nonzero(x + row), 15u);
    EXPECT_EQ(nanmean(x + row), T(9));
    tensor_t bad(dims_t{7});
    EXPECT_ANY_THROW((void)prod(x + bad));
    EXPECT_ANY_THROW((void)mean(x + bad));
    EXPECT_ANY_THROW((void)count_nonzero(x + bad));
    EXPECT_ANY_THROW((void)nanmean(x + bad));
    const auto& readonly = x;
    EXPECT_EQ(count_nonzero(readonly.as_view()), 15u);
}
}  // namespace

VECTORIZATIONTEST(TensorReductions, Float)
{
    check_reductions<float>();
    check_strides_and_shapes<float>();
    END_TEST();
}
VECTORIZATIONTEST(TensorReductions, Double)
{
    check_reductions<double>();
    check_strides_and_shapes<double>();
    END_TEST();
}
VECTORIZATIONTEST(TensorReductions, LargeExactCount)
{
    constexpr size_t             n = (size_t(1) << 24) + 3;
    vectorization::tensor<float> x(n);
    x        = 1.0f;
    x[n - 1] = 0.0f;
    EXPECT_EQ(vectorization::count_nonzero(x), n - 1);
    EXPECT_EQ(vectorization::count_nonzero(x + 1.0f), n);
    END_TEST();
}
