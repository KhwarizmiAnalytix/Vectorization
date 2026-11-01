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

#include <type_traits>

namespace vectorization
{
// The assertion covers source/destination aliasing only. The destination retains
// its shape and must still be injective in every build configuration.
template <typename Destination> class noalias_proxy
{
public:
    explicit noalias_proxy(Destination& dst) : dst_(dst) {}
    noalias_proxy(noalias_proxy const&)            = delete;
    noalias_proxy& operator=(noalias_proxy const&) = delete;

    template <typename E> Destination& operator=(E const& expr)
    {
        if constexpr (std::is_fundamental_v<E>)
        {
            return dst_ = expr;
        }
        else
        {
            return dst_.assign(expr, true);
        }
    }
    template <typename E> Destination& operator+=(E const& expr)
    {
        return dst_.assign(dst_ + expr, true);
    }
    template <typename E> Destination& operator-=(E const& expr)
    {
        return dst_.assign(dst_ - expr, true);
    }

private:
    Destination& dst_;
};
}  // namespace vectorization
