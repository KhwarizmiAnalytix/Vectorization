/*
 * Vectorization: High-Performance Quantitative Library
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

#include "common/vectorization_type_traits.h"

namespace vectorization
{

// Pivots/info are stored as tensor<T> (the factored matrix's own element
// type), not tensor<int32_t>: Scalars' simd<T> has no int32 specialization,
// so a SIMD-enabled build cannot instantiate tensor<int32_t> (tensor_impl's
// simd_t alias hard-fails substitution). Values are small whole numbers and
// round-trip exactly through float/double at the supported matrix orders.
// Integer metadata tensors remain a P5 gap; this is not dtype parity with
// PyTorch. LU rejects orders beyond float's exact integer range below.
template <typename T> struct lu_factor_result
{
    tensor<T> LU;
    tensor<T> pivots;
};

template <typename T> struct lu_factor_ex_result
{
    tensor<T> LU;
    tensor<T> pivots;
    tensor<T> info;
};

template <typename T> struct lu_result
{
    tensor<T> P;
    tensor<T> L;
    tensor<T> U;
};

template <typename T> struct cholesky_ex_result
{
    tensor<T> L;
    tensor<T> info;
};

template <typename T> struct solve_ex_result
{
    tensor<T> result;
    tensor<T> info;
};

template <typename T> struct inv_ex_result
{
    tensor<T> inverse;
    tensor<T> info;
};

template <typename T> struct slogdet_result
{
    tensor<T> sign;
    tensor<T> logabsdet;
};

namespace linalg
{
using vectorization::cholesky_ex_result;
using vectorization::inv_ex_result;
using vectorization::lu_factor_ex_result;
using vectorization::lu_factor_result;
using vectorization::lu_result;
using vectorization::slogdet_result;
using vectorization::solve_ex_result;
}  // namespace linalg

}  // namespace vectorization
