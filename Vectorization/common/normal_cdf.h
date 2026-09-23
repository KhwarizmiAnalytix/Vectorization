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

#include "common/vectorization_export.h"

namespace vectorization
{
/**
    //tex:
    //$$N(x)=\int_{-\infty}^{x}\frac{\exp{(-\frac{u^2}{2})}}{\sqrt{2\pi}}du$$
    **/
VECTORIZATION_API double normalcdf(double z);

/**
    //tex:
    //$$N^{-1}(x)$$
    **/
VECTORIZATION_API double inv_normalcdf(double p);
}  // namespace vectorization
