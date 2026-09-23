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

#include "common/normal_cdf.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace vectorization
{
namespace
{
// 1 / sqrt(2 * pi); same value as xsigma::constants::INVERSE_SQRT_2PI in Core.
inline constexpr double k_inverse_sqrt_2pi = 0.3989422804014326779399460599343818684758576;
}  // namespace

//-----------------------------------------------------------------------------
template <typename T1, typename T2>
constexpr auto POLYNOM_6(T1 coefficients, T2 X)
{
    return (
        coefficients[0] +
        (coefficients[1] +
         (coefficients[2] +
          (coefficients[3] + (coefficients[4] + (coefficients[5] + coefficients[6] * X) * X) * X) *
              X) *
             X) *
            X);
}

//-----------------------------------------------------------------------------
template <typename T1, typename T2>
constexpr auto POLYNOM_7(T1 coefficients, T2 X)
{
    return (
        coefficients[0] +
        (coefficients[1] +
         (coefficients[2] +
          (coefficients[3] +
           (coefficients[4] + (coefficients[5] + (coefficients[6] + coefficients[7] * X) * X) * X) *
               X) *
              X) *
             X) *
            X);
}

//-----------------------------------------------------------------------------
// Based upon
// algorithm 5666 for the error function, from:
/// Hart, J. F. et al, 'Computer Approximations', Wiley 1968
//-----------------------------------------------------------------------------
double normalcdf(double z)
{
    if (z < -7.071067811865475)
    {
        const double x2 = 1. / (z * z);
        const double p =
            -z * z * 0.5 +
            log(std::fabs(
                1. / z *
                (1. -
                 x2 * (1. -
                       3. * x2 *
                           (1. - 5. * x2 * (1. - 7. * x2 * (1. - 9. * x2 * (1. - 11. * x2))))))));

        return k_inverse_sqrt_2pi * std::exp(p);
    }

    const auto x = std::fabs(z);

    if (x > 36.)
    {
        return 1.;
    }

    auto p = exp(-.5 * x * x);

    if (x < 7.071067811865475)
    {
        static std::array<double, 7> A = {
            220.2068679123761,
            221.2135961699311,
            112.0792914978709,
            33.912866078383,
            6.37396220353165,
            .7003830644436881,
            .03526249659989109};

        static std::array<double, 8> B = {
            440.4137358247522,
            793.8265125199484,
            637.3336333788311,
            296.5642487796737,
            86.78073220294608,
            16.06417757920695,
            1.755667163182642,
            .08838834764831844};

        p *= POLYNOM_6(A, x) / POLYNOM_7(B, x);
    }
    else
    {
        const double x2 = 1. / (z * z);
        p *=
            k_inverse_sqrt_2pi *
            std::fabs(
                1. / z *
                (1. - x2 * (1. - 3. * x2 *
                                     (1. - 5. * x2 *
                                               (1. - 7. * x2 * (1. - 9. * x2 * (1. - 11 * x2)))))));
    }

    return (z > 0.) ? 1. - p : p;
}

//-----------------------------------------------------------------------------
// The algorithm in Wichura's 1988 paper on ALGORITHM AS241 APPL. STATIST.
// (1988) VOL. 37, NO. 3 Produces the normal_distribution deviate Z
// corresponding to a given lower tail area of P; Z is accurate to about 1 part
// in 10**16.
//-----------------------------------------------------------------------------
double inv_normalcdf(double p)
{
    static const double SPLIT1 = .425;
    static const double CONST1 = .180625;
    static const double SPLIT2 = 5.;
    static const double CONST2 = 1.6;

    const auto q = p - .5;

    if (std::fabs(q) <= SPLIT1)
    {
        static const double P1[8] = {// NOLINT
                                     3.3871328727963666080259528779427788158739210379764,
                                     133.14166968794228852770808564748670287633346154120,
                                     1971.5910094275027424697683591038339346936953627853,
                                     13731.694456624315217864236408896200772793054228337,
                                     45921.957400032815544819980657702498844637293527101,
                                     67265.778102764018588449157431865730541748168710612,
                                     33430.580423506993022366545337636415809973223808597,
                                     2509.0814069870534781529831560407278731068028274270};

        static const double Q1[8] = {// NOLINT
                                     1.,
                                     42.313331231889500017187287592131981830109606699891,
                                     687.18702654028253804115761174023624065078998751050,
                                     5394.1962710137248305498997611193069508648159382145,
                                     21213.795765973111651970052869305507789573595154341,
                                     39307.899609448792796879833774972167788565304184291,
                                     28729.089491615611535067926442316676533115693351172,
                                     5226.4961724990340848855571296643657843805218376688};

        auto r = CONST1 - q * q;

        return q * POLYNOM_7(P1, r) / POLYNOM_7(Q1, r);
    }

    auto r = (q < 0.) ? p : 1. - p;
    r      = std::sqrt(-log(r));

    double z;
    if (r <= SPLIT2)
    {
        static const double P2[8] = {// NOLINT
                                     1.42343711074968357734e0,
                                     4.63033784615654529590e0,
                                     5.76949722146069140550e0,
                                     3.64784832476320460504e0,
                                     1.27045825245236838258e0,
                                     2.41780725177450611770e-1,
                                     2.27238449892691845833e-2,
                                     7.74545014278341407640e-4};

        static const double Q2[8] = {// NOLINT
                                     1.,
                                     2.05319162663775882187e0,
                                     1.67638483018380384940e0,
                                     6.89767334985100004550e-1,
                                     1.48103976427480074590e-1,
                                     1.51986665636164571966e-2,
                                     5.47593808499534494600e-4,
                                     1.05075007164441684324e-9};
        r -= CONST2;
        z = POLYNOM_7(P2, r) / POLYNOM_7(Q2, r);
    }
    else
    {
        static const double P3[8] = {// NOLINT
                                     6.65790464350110377720e0,
                                     5.46378491116411436990e0,
                                     1.78482653991729133580e0,
                                     2.96560571828504891230e-1,
                                     2.65321895265761230930e-2,
                                     1.24266094738807843860e-3,
                                     2.71155556874348757815e-5,
                                     2.01033439929228813265e-7};

        static const double Q3[8] = {// NOLINT
                                     1.,
                                     5.99832206555887937690e-1,
                                     1.36929880922735805310e-1,
                                     1.48753612908506148525e-2,
                                     7.86869131145613259100e-4,
                                     1.84631831751005468180e-5,
                                     1.42151175831644588870e-7,
                                     2.04426310338993978564e-15};
        r -= SPLIT2;
        z = POLYNOM_7(P3, r) / POLYNOM_7(Q3, r);
    }

    return (q < 0.) ? -z : z;
}

}  // namespace vectorization
