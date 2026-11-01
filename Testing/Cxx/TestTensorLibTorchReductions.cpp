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

#include "VectorizationTest.h"

#if VECTORIZATION_HAS_LIBTORCH
#include "TestTensorLibTorchHelpers.h"

namespace
{
template <typename T> void compare_reductions()
{
    using namespace vectorization;
    constexpr double tol = std::is_same_v<T, float> ? 2e-4 : 1e-11;
    for (size_t n : {size_t(0), size_t(1), size_t(17), size_t(1025), size_t(66001)})
    {
        auto      host = libtorch_test::rand_vec<T>(n, T(-2), T(2), 17);
        tensor<T> x(n);
        x.copy_from_host(host);
        auto tx = libtorch_test::to_torch(host.data(), n);
        EXPECT_EQ(all(x), torch::all(tx).template item<bool>());
        EXPECT_EQ(any(x), torch::any(tx).template item<bool>());
        EXPECT_EQ(count_nonzero(x), torch::count_nonzero(tx).template item<int64_t>());
        EXPECT_NEAR(sum(x),
            torch::sum(tx).template item<T>(),
            tol * std::sqrt(static_cast<double>(std::max(size_t(1), n))));
        if (n == 0)
        {
            EXPECT_TRUE(std::isnan(mean(x)));
            EXPECT_TRUE(std::isnan(nanmean(x)));
            EXPECT_EQ(prod(x), torch::prod(tx).template item<T>());
        }
        else
        {
            EXPECT_NEAR(mean(x), torch::mean(tx).template item<T>(), tol);
            host[n / 2] = std::numeric_limits<T>::quiet_NaN();
            x.copy_from_host(host);
            tx = libtorch_test::to_torch(host.data(), n);
            EXPECT_NEAR(nansum(x),
                torch::nansum(tx).template item<T>(),
                tol * std::sqrt(static_cast<double>(n)));
            if (n == 1)
            {
                EXPECT_TRUE(std::isnan(nanmean(x)));
            }
            else
            {
                EXPECT_NEAR(nanmean(x), torch::nanmean(tx).template item<T>(), tol);
            }
        }
    }
    // Products near one stay finite while still exposing rounding and tails.
    auto host = libtorch_test::rand_vec<T>(1025, T(0.999), T(1.001), 7);
    host[5]   = T(-1);
    tensor<T> x(host.size());
    x.copy_from_host(host);
    auto tx = libtorch_test::to_torch(host.data(), host.size());
    EXPECT_NEAR(prod(x), torch::prod(tx).template item<T>(), tol);
    EXPECT_EQ(count_nonzero(x > T(1)), torch::count_nonzero(tx > 1).template item<int64_t>());
    EXPECT_NEAR(mean(x * T(2) + T(1)), torch::mean(tx * 2 + 1).template item<T>(), tol);
}
}  // namespace

VECTORIZATIONTEST(LibTorch, ReductionsFloat)
{
    compare_reductions<float>();
    END_TEST();
}
VECTORIZATIONTEST(LibTorch, ReductionsDouble)
{
    compare_reductions<double>();
    END_TEST();
}
#endif
