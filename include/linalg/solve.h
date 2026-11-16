/*
 * Vectorization: High-Performance Quantitative Library
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "expressions/product_expression.h"
#include "include/matrix_operation/linear_solver.h"
#include "include/matrix_operation/triangular_solver.h"
#include "linalg/factorizations.h"
#include "linalg/linalg_results.h"

namespace vectorization
{
namespace detail
{

// Every operand reaching these helpers has already been packed into a
// contiguous tensor<T> (via .clone() or a fresh allocation), so an operand's
// own stride for a batch dimension is just its ordinary packed row-major
// stride -- unlike product_expression.h's general strided_ref case, no
// separate stride bookkeeping is needed here.
template <typename A> size_t solve_batch_rank(A const& a, size_t matrix_dims)
{
    const size_t rank = a.rank();
    return rank > matrix_dims ? rank - matrix_dims : 0;
}

template <typename A, typename B>
std::vector<int64_t> combine_solve_batch_dims(
    A const& a, size_t a_matrix_dims, B const& b, size_t b_matrix_dims)
{
    const size_t         a_batch = solve_batch_rank(a, a_matrix_dims);
    const size_t         b_batch = solve_batch_rank(b, b_matrix_dims);
    const size_t         rank    = std::max(a_batch, b_batch);
    std::vector<int64_t> dims(rank, 1);
    for (size_t i = 0; i < rank; ++i)
    {
        const int64_t a_size = i < a_batch ? static_cast<int64_t>(a.dimension(a_batch - 1 - i)) : 1;
        const int64_t b_size = i < b_batch ? static_cast<int64_t>(b.dimension(b_batch - 1 - i)) : 1;
        VECTORIZATION_CHECK(a_size == b_size || a_size == 1 || b_size == 1,
            "solve batch dimensions cannot broadcast");
        dims[rank - 1 - i] = a_size == 1 ? b_size : a_size;
    }
    return dims;
}

// Element offset into `a`'s own packed storage for the batch coordinate that
// `flat_index` denotes under the combined (broadcast) batch shape, scaled by
// `matrix_block` (the element count of a's trailing matrix_dims). Size-1 dims
// on `a` contribute zero offset (broadcast); mirrors
// product_expression.h's detail::product_batch_offset, generalized to a
// method-based tensor<T> operand and an arbitrary matrix_dims.
template <typename A>
size_t solve_operand_offset(A const& a,
    size_t                           matrix_dims,
    std::vector<int64_t> const&      combined,
    size_t                           flat_index,
    size_t                           matrix_block)
{
    const size_t a_batch    = solve_batch_rank(a, matrix_dims);
    size_t       own_stride = matrix_block;
    size_t       offset     = 0;
    size_t       index      = flat_index;
    for (size_t i = 0; i < combined.size(); ++i)
    {
        const size_t  pos        = combined.size() - 1 - i;
        const int64_t size       = combined[pos];
        const size_t  coordinate = index % static_cast<size_t>(size);
        index /= static_cast<size_t>(size);
        if (i < a_batch)
        {
            const int64_t a_size = static_cast<int64_t>(a.dimension(a_batch - 1 - i));
            if (a_size != 1)
            {
                offset += coordinate * own_stride;
            }
            own_stride *= static_cast<size_t>(a_size);
        }
    }
    return offset;
}

// Transposes the trailing two dimensions of a packed tensor, batch dims
// untouched. Used for solve(..., left=false)'s XA=B -> A^T X^T = B^T trick
// and for solve_triangular's right-side RHS reshape.
template <typename T> tensor<T> transpose_last_two(tensor<T> const& src)
{
    const size_t rank = src.rank();
    VECTORIZATION_CHECK(rank >= 2, "transpose_last_two requires rank >= 2");
    const int64_t r = static_cast<int64_t>(src.size(rank - 2));
    const int64_t c = static_cast<int64_t>(src.size(rank - 1));

    typename tensor<T>::dimensions_type dims;
    for (size_t d = 0; d + 2 < rank; ++d)
    {
        dims.push_back(src.dimension(d));
    }
    dims.push_back(static_cast<size_t>(c));
    dims.push_back(static_cast<size_t>(r));
    tensor<T> dst(dims, device_enum::CPU, src.device_index());

    size_t batch = 1;
    for (size_t d = 0; d + 2 < rank; ++d)
    {
        batch *= src.dimension(d);
    }
    const auto block = static_cast<size_t>(r) * static_cast<size_t>(c);
    for (size_t b = 0; b < batch; ++b)
    {
        T const* in  = src.data() + b * block;
        T*       out = dst.data() + b * block;
        for (int64_t i = 0; i < r; ++i)
        {
            for (int64_t j = 0; j < c; ++j)
            {
                out[j * r + i] = in[i * c + j];
            }
        }
    }
    return dst;
}

template <typename T> thread_local memory::data_ptr<T> linear_solver_workspace_cache;
template <typename T> thread_local memory::data_ptr<T> triangular_solver_workspace_cache;

// Solves A X = B (A already LU-factored) one RHS column at a time: upstream
// linear_solver::apply has no nrhs/ldx parameter, same gap
// product_expression.h's matmul_out alpha/beta path documents for
// multiplication::apply -- loop until U2 adds multi-RHS.
template <typename T>
void lu_solve_standard_block(T const* lu_block,
    T const*                          pivot_block,
    T const*                          rhs_block,
    T*                                out_block,
    int64_t                           n,
    int64_t                           nrhs)
{
    using kernel        = ::linalg::linear_solver<T, detail::product_cpu_backend>;
    const auto required = kernel::workspace_size(
        static_cast<linalg_int>(n), ::linalg::linear_solver_type::LU_UPFRONT_LINEAR_SOLVER);
    T* const workspace = grow_workspace(linear_solver_workspace_cache<T>, required);

    std::vector<T> lu_copy(lu_block, lu_block + static_cast<size_t>(n) * static_cast<size_t>(n));
    std::vector<linalg_int> pivot_ints(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i)
    {
        pivot_ints[static_cast<size_t>(i)] = static_cast<linalg_int>(pivot_block[i]);
    }

    std::vector<T> column(static_cast<size_t>(n));
    for (int64_t j = 0; j < nrhs; ++j)
    {
        for (int64_t i = 0; i < n; ++i)
        {
            column[static_cast<size_t>(i)] = rhs_block[i * nrhs + j];
        }
        kernel::apply(lu_copy.data(),
            pivot_ints.data(),
            static_cast<linalg_int>(n),
            column.data(),
            ::linalg::linear_solver_type::LU_UPFRONT_LINEAR_SOLVER,
            workspace,
            required);
        for (int64_t i = 0; i < n; ++i)
        {
            out_block[i * nrhs + j] = column[static_cast<size_t>(i)];
        }
    }
}

// Solves the CHOLESKY_UPFRONT system A X = B given an existing triangular
// factor, same single-RHS-column loop as lu_solve_standard_block.
template <typename T>
void cholesky_solve_block(
    T const* factor_block, T const* rhs_block, T* out_block, int64_t n, int64_t nrhs)
{
    using kernel        = ::linalg::linear_solver<T, detail::product_cpu_backend>;
    const auto required = kernel::workspace_size(
        static_cast<linalg_int>(n), ::linalg::linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER);
    T* const workspace = grow_workspace(linear_solver_workspace_cache<T>, required);

    std::vector<T> factor_copy(
        factor_block, factor_block + static_cast<size_t>(n) * static_cast<size_t>(n));

    std::vector<T> column(static_cast<size_t>(n));
    for (int64_t j = 0; j < nrhs; ++j)
    {
        for (int64_t i = 0; i < n; ++i)
        {
            column[static_cast<size_t>(i)] = rhs_block[i * nrhs + j];
        }
        kernel::apply(factor_copy.data(),
            nullptr,
            static_cast<linalg_int>(n),
            column.data(),
            ::linalg::linear_solver_type::CHOLESKY_UPFRONT_LINEAR_SOLVER,
            workspace,
            required);
        for (int64_t i = 0; i < n; ++i)
        {
            out_block[i * nrhs + j] = column[static_cast<size_t>(i)];
        }
    }
}

// Solves A^T X = B directly from A's existing LU factors (no refactorization
// of A^T): since A = P_lapack^T * L * U (see factorizations.h's lu() unpack
// comment), A^T = U^T * L^T * P_lapack, so the system splits into two
// triangular solves against the stored L/U blocks -- each already
// multi-RHS-capable via triangular_solver, unlike the single-RHS
// linear_solver path above -- followed by undoing the forward pivot
// sequence (replaying it in reverse order inverts a permutation built by
// applying swaps in increasing-i order).
template <typename T>
void lu_solve_transposed_block(T const* lu_block,
    T const*                            pivot_block,
    T const*                            rhs_block,
    T*                                  out_block,
    int64_t                             n,
    int64_t                             nrhs)
{
    using kernel = ::linalg::triangular_solver<T, detail::product_cpu_backend>;
    const auto required =
        kernel::workspace_size(static_cast<linalg_int>(n), static_cast<linalg_int>(nrhs));
    T* const workspace = grow_workspace(triangular_solver_workspace_cache<T>, required);

    std::vector<T> stage1(static_cast<size_t>(n) * static_cast<size_t>(nrhs));
    kernel::apply(/*upper=*/true,
        /*transpose=*/true,
        /*unit_diagonal=*/false,
        static_cast<linalg_int>(n),
        static_cast<linalg_int>(nrhs),
        lu_block,
        static_cast<linalg_int>(n),
        rhs_block,
        static_cast<linalg_int>(nrhs),
        stage1.data(),
        static_cast<linalg_int>(nrhs),
        workspace,
        required);
    kernel::apply(/*upper=*/false,
        /*transpose=*/true,
        /*unit_diagonal=*/true,
        static_cast<linalg_int>(n),
        static_cast<linalg_int>(nrhs),
        lu_block,
        static_cast<linalg_int>(n),
        stage1.data(),
        static_cast<linalg_int>(nrhs),
        out_block,
        static_cast<linalg_int>(nrhs),
        workspace,
        required);

    std::vector<T> column(static_cast<size_t>(n));
    for (int64_t j = 0; j < nrhs; ++j)
    {
        for (int64_t i = 0; i < n; ++i)
        {
            column[static_cast<size_t>(i)] = out_block[i * nrhs + j];
        }
        for (int64_t i = n - 1; i >= 0; --i)
        {
            const int64_t p = static_cast<int64_t>(pivot_block[i]) - 1;
            if (p != i)
            {
                std::swap(column[static_cast<size_t>(i)], column[static_cast<size_t>(p)]);
            }
        }
        for (int64_t i = 0; i < n; ++i)
        {
            out_block[i * nrhs + j] = column[static_cast<size_t>(i)];
        }
    }
}

// PyTorch's solve treats B as vectors only when it is rank 1 or its entire
// shape equals A.shape[:-1]. Rank alone confuses a broadcast (n, k) RHS
// with a batch of vectors when A is rank 3.
template <typename A, typename B> bool solve_rhs_is_vector(A const& a, B const& b)
{
    if (b.rank() == 1)
    {
        return true;
    }
    if (b.rank() + 1 != a.rank())
    {
        return false;
    }
    for (size_t d = 0; d < b.rank(); ++d)
    {
        if (a.dimension(d) != b.dimension(d))
        {
            return false;
        }
    }
    return true;
}

template <typename A, typename B>
bool check_solve_rhs(A const& a, B const& b, bool left, bool allow_vector)
{
    VECTORIZATION_CHECK(a.device() == device_enum::CPU && b.device() == device_enum::CPU &&
                            a.device_index() == b.device_index(),
        "solve requires CPU operands on the same device index");
    const auto n      = factor_matrix_order(a);
    const bool vector = allow_vector && solve_rhs_is_vector(a, b);
    VECTORIZATION_CHECK(b.rank() >= (vector ? 1u : 2u),
        "solve requires a matrix right-hand side for this operation");
    VECTORIZATION_CHECK(left || !vector, "solve: vector RHS requires left=true");
    const size_t axis = b.rank() - (vector || !left ? 1 : 2);
    VECTORIZATION_CHECK(b.size(axis) == n,
        "solve: right-hand side contraction dimension must match the matrix order");
    if (!vector)
    {
        const auto nrhs = b.size(b.rank() - (left ? 1 : 2));
        VECTORIZATION_CHECK(nrhs <= std::numeric_limits<int>::max(),
            "solve: right-hand side count exceeds the backend integer range");
    }
    return vector;
}

template <typename LU, typename P> void check_solve_pivots(LU const& lu, P const& pivots)
{
    VECTORIZATION_CHECK(
        pivots.device() == device_enum::CPU && pivots.device_index() == lu.device_index(),
        "lu_solve: pivots must share the factors' CPU device index");
    VECTORIZATION_CHECK(
        pivots.rank() + 1 == lu.rank(), "lu_solve: pivot shape must equal LU.shape[:-1]");
    for (size_t d = 0; d < pivots.rank(); ++d)
    {
        VECTORIZATION_CHECK(pivots.dimension(d) == lu.dimension(d),
            "lu_solve: pivot shape must equal LU.shape[:-1]");
    }
    const auto n = lu.size(lu.rank() - 1);
    for (auto value : pivots.to_host_vector())
    {
        VECTORIZATION_CHECK(
            std::isfinite(value) && value >= 1 && value <= n && std::trunc(value) == value,
            "lu_solve: pivots must be whole 1-based row indices in [1, n]");
    }
}

template <typename A, typename B, std::enable_if_t<are_tensor_operands_v<A, B>, bool> = true>
solve_ex_result<operand_value_t<A>> solve_impl(
    A const& a, B const& b, bool left, bool check_errors, char const* op)
{
    using T = operand_value_t<A>;
    static_assert(
        std::is_same_v<T, float> || std::is_same_v<T, double>, "solve supports float and double");
    const bool is_vector = detail::check_solve_rhs(a, b, left, true);

    if (!left)
    {
        VECTORIZATION_CHECK(
            b.rank() >= 2, "solve: a matrix right-hand side is required when left=false");
        tensor<T> a_t = detail::transpose_last_two(tensor<T>(a.clone()));
        tensor<T> b_t = detail::transpose_last_two(tensor<T>(b.clone()));
        auto      sub = solve_impl(a_t, b_t, /*left=*/true, check_errors, op);
        return solve_ex_result<T>{detail::transpose_last_two(sub.result), std::move(sub.info)};
    }

    const int64_t n    = detail::factor_matrix_order(a);
    const int64_t nrhs = is_vector ? 1 : static_cast<int64_t>(b.size(b.rank() - 1));
    if (!is_vector)
    {
        VECTORIZATION_CHECK(static_cast<int64_t>(b.size(b.rank() - 2)) == n,
            "solve: right-hand side row count must match the coefficient matrix order");
    }
    else
    {
        VECTORIZATION_CHECK(static_cast<int64_t>(b.size(b.rank() - 1)) == n,
            "solve: right-hand side length must match the coefficient matrix order");
    }

    auto factored = detail::lu_factor_impl(a, check_errors, op);

    const size_t b_matrix_dims = is_vector ? 1 : 2;
    auto combined = detail::combine_solve_batch_dims(factored.LU, size_t(2), b, b_matrix_dims);

    typename tensor<T>::dimensions_type out_dims;
    for (auto d : combined)
    {
        out_dims.push_back(static_cast<size_t>(d));
    }
    out_dims.push_back(static_cast<size_t>(n));
    if (!is_vector)
    {
        out_dims.push_back(static_cast<size_t>(nrhs));
    }
    tensor<T> result(out_dims, device_enum::CPU, a.device_index());

    if (result.size() == 0)
    {
        return solve_ex_result<T>{std::move(result), std::move(factored.info)};
    }

    size_t total = 1;
    for (auto d : combined)
    {
        total *= static_cast<size_t>(d);
    }
    const auto nn        = static_cast<size_t>(n) * static_cast<size_t>(n);
    const auto b_block   = static_cast<size_t>(n) * static_cast<size_t>(nrhs);
    const auto out_block = b_block;
    tensor<T>  b_clone   = b.clone();
    for (size_t idx = 0; idx < total; ++idx)
    {
        const size_t lu_off =
            detail::solve_operand_offset(factored.LU, size_t(2), combined, idx, nn);
        const size_t pivot_off = detail::solve_operand_offset(
            factored.pivots, size_t(1), combined, idx, static_cast<size_t>(n));
        const size_t b_off =
            detail::solve_operand_offset(b_clone, b_matrix_dims, combined, idx, b_block);
        detail::lu_solve_standard_block(factored.LU.data() + lu_off,
            factored.pivots.data() + pivot_off,
            b_clone.data() + b_off,
            result.data() + idx * out_block,
            n,
            nrhs);
    }
    return solve_ex_result<T>{std::move(result), std::move(factored.info)};
}
}  // namespace detail

template <typename A, typename B, std::enable_if_t<are_tensor_operands_v<A, B>, bool> = true>
auto solve_ex(A const& a, B const& b, bool left = true, bool check_errors = false)
{
    return detail::solve_impl(a, b, left, check_errors, "solve_ex");
}

template <typename A, typename B, std::enable_if_t<are_tensor_operands_v<A, B>, bool> = true>
auto solve(A const& a, B const& b, bool left = true)
{
    return detail::solve_impl(a, b, left, /*check_errors=*/true, "solve").result;
}

template <typename A, typename B, std::enable_if_t<are_tensor_operands_v<A, B>, bool> = true>
auto solve_triangular(
    A const& a, B const& b, bool upper, bool left = true, bool unitriangular = false)
{
    using T = operand_value_t<A>;
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
        "solve_triangular supports float and double");
    (void)detail::check_solve_rhs(a, b, left, false);
    const int64_t n = detail::factor_matrix_order(a);

    // left=false (XA=B, B/X shaped (...,k,n)) has no native right-side mode
    // in triangular_solver; BLAS trsm's uplo describes the stored triangle
    // independently of its trans flag, so transposing only the RHS/result
    // (not A's storage) and requesting transpose=true solves A^T Y = B^T,
    // i.e. exactly (XA=B)^T, without re-factoring or re-packing A.
    tensor<T> b_work =
        left ? tensor<T>(b.clone()) : detail::transpose_last_two(tensor<T>(b.clone()));
    VECTORIZATION_CHECK(b_work.rank() >= 2, "solve_triangular requires a matrix right-hand side");
    VECTORIZATION_CHECK(static_cast<int64_t>(b_work.size(b_work.rank() - 2)) == n,
        "solve_triangular: right-hand side row count must match the coefficient matrix order");
    const int64_t nrhs = static_cast<int64_t>(b_work.size(b_work.rank() - 1));

    auto combined = detail::combine_solve_batch_dims(a, size_t(2), b_work, size_t(2));
    typename tensor<T>::dimensions_type out_dims;
    for (auto d : combined)
    {
        out_dims.push_back(static_cast<size_t>(d));
    }
    out_dims.push_back(static_cast<size_t>(n));
    out_dims.push_back(static_cast<size_t>(nrhs));
    tensor<T> x_work(out_dims, device_enum::CPU, a.device_index());
    if (x_work.size() == 0)
    {
        return left ? std::move(x_work) : detail::transpose_last_two(x_work);
    }

    size_t total = 1;
    for (auto d : combined)
    {
        total *= static_cast<size_t>(d);
    }
    const auto nn      = static_cast<size_t>(n) * static_cast<size_t>(n);
    const auto rhs_blk = static_cast<size_t>(n) * static_cast<size_t>(nrhs);
    tensor<T>  a_clone = a.clone();

    using kernel = ::linalg::triangular_solver<T, detail::product_cpu_backend>;
    const auto required =
        kernel::workspace_size(static_cast<linalg_int>(n), static_cast<linalg_int>(nrhs));
    T* const workspace =
        detail::grow_workspace(detail::triangular_solver_workspace_cache<T>, required);

    for (size_t idx = 0; idx < total; ++idx)
    {
        const size_t a_off = detail::solve_operand_offset(a_clone, size_t(2), combined, idx, nn);
        const size_t b_off =
            detail::solve_operand_offset(b_work, size_t(2), combined, idx, rhs_blk);
        kernel::apply(upper,
            /*transpose=*/!left,
            unitriangular,
            static_cast<linalg_int>(n),
            static_cast<linalg_int>(nrhs),
            a_clone.data() + a_off,
            static_cast<linalg_int>(n),
            b_work.data() + b_off,
            static_cast<linalg_int>(nrhs),
            x_work.data() + idx * rhs_blk,
            static_cast<linalg_int>(nrhs),
            workspace,
            required);
    }
    return left ? x_work : detail::transpose_last_two(x_work);
}

template <typename LU,
    typename P,
    typename B,
    std::enable_if_t<are_tensor_operands_v<LU, P, B>, bool> = true>
tensor<operand_value_t<LU>> lu_solve(
    LU const& lu, P const& pivots, B const& b, bool left = true, bool adjoint = false)
{
    using T = operand_value_t<LU>;
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
        "lu_solve supports float and double");
    (void)detail::check_solve_rhs(lu, b, left, false);
    detail::check_solve_pivots(lu, pivots);
    const int64_t n = detail::factor_matrix_order(lu);

    tensor<T> lu_clone     = lu.clone();
    tensor<T> pivots_clone = pivots.clone();

    if (!left)
    {
        // XA=B reduces to (XA=B)^T = A^T X^T = B^T, i.e. the left=true solve
        // of the SAME system with `adjoint` flipped, applied to B^T.
        VECTORIZATION_CHECK(
            b.rank() >= 2, "lu_solve: a matrix right-hand side is required when left=false");
        tensor<T> b_t = detail::transpose_last_two(tensor<T>(b.clone()));
        auto      x_t = lu_solve(lu_clone, pivots_clone, b_t, /*left=*/true, !adjoint);
        return detail::transpose_last_two(x_t);
    }

    // adjoint=true (A^T X = B) has no native support in linear_solver's
    // UPFRONT mode (no trans flag), so it's routed through the triangular
    // substitution in detail::lu_solve_transposed_block instead; see that
    // function's comment for the A = P_lapack^T * L * U derivation.
    const bool     use_transposed_primitive = adjoint;
    constexpr bool is_vector                = false;
    const int64_t  nrhs          = is_vector ? 1 : static_cast<int64_t>(b.size(b.rank() - 1));
    const size_t   b_matrix_dims = is_vector ? 1 : 2;

    auto combined = detail::combine_solve_batch_dims(lu_clone, size_t(2), b, b_matrix_dims);
    typename tensor<T>::dimensions_type out_dims;
    for (auto d : combined)
    {
        out_dims.push_back(static_cast<size_t>(d));
    }
    out_dims.push_back(static_cast<size_t>(n));
    if (!is_vector)
    {
        out_dims.push_back(static_cast<size_t>(nrhs));
    }
    tensor<T> result(out_dims, device_enum::CPU, lu.device_index());
    if (result.size() == 0)
    {
        return result;
    }

    size_t total = 1;
    for (auto d : combined)
    {
        total *= static_cast<size_t>(d);
    }
    const auto nn      = static_cast<size_t>(n) * static_cast<size_t>(n);
    const auto b_block = static_cast<size_t>(n) * static_cast<size_t>(nrhs);
    tensor<T>  b_clone = b.clone();
    for (size_t idx = 0; idx < total; ++idx)
    {
        const size_t lu_off = detail::solve_operand_offset(lu_clone, size_t(2), combined, idx, nn);
        const size_t pivot_off = detail::solve_operand_offset(
            pivots_clone, size_t(1), combined, idx, static_cast<size_t>(n));
        const size_t b_off =
            detail::solve_operand_offset(b_clone, b_matrix_dims, combined, idx, b_block);
        T* const out_ptr = result.data() + idx * b_block;
        if (use_transposed_primitive)
        {
            detail::lu_solve_transposed_block(lu_clone.data() + lu_off,
                pivots_clone.data() + pivot_off,
                b_clone.data() + b_off,
                out_ptr,
                n,
                nrhs);
        }
        else
        {
            detail::lu_solve_standard_block(lu_clone.data() + lu_off,
                pivots_clone.data() + pivot_off,
                b_clone.data() + b_off,
                out_ptr,
                n,
                nrhs);
        }
    }
    return result;
}

template <typename B, typename L, std::enable_if_t<are_tensor_operands_v<B, L>, bool> = true>
auto cholesky_solve(B const& b, L const& l, bool upper = false)
{
    using T = operand_value_t<B>;
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
        "cholesky_solve supports float and double");
    (void)detail::check_solve_rhs(l, b, true, false);
    const int64_t n = detail::factor_matrix_order(l);

    constexpr bool is_vector     = false;
    const int64_t  nrhs          = is_vector ? 1 : static_cast<int64_t>(b.size(b.rank() - 1));
    const size_t   b_matrix_dims = is_vector ? 1 : 2;

    tensor<T> l_clone  = upper ? detail::transpose_last_two(l.clone()) : l.clone();
    auto      combined = detail::combine_solve_batch_dims(l_clone, size_t(2), b, b_matrix_dims);
    typename tensor<T>::dimensions_type out_dims;
    for (auto d : combined)
    {
        out_dims.push_back(static_cast<size_t>(d));
    }
    out_dims.push_back(static_cast<size_t>(n));
    if (!is_vector)
    {
        out_dims.push_back(static_cast<size_t>(nrhs));
    }
    tensor<T> result(out_dims, device_enum::CPU, l.device_index());
    if (result.size() == 0)
    {
        return result;
    }

    size_t total = 1;
    for (auto d : combined)
    {
        total *= static_cast<size_t>(d);
    }
    const auto nn      = static_cast<size_t>(n) * static_cast<size_t>(n);
    const auto b_block = static_cast<size_t>(n) * static_cast<size_t>(nrhs);
    tensor<T>  b_clone = b.clone();
    for (size_t idx = 0; idx < total; ++idx)
    {
        const size_t l_off = detail::solve_operand_offset(l_clone, size_t(2), combined, idx, nn);
        const size_t b_off =
            detail::solve_operand_offset(b_clone, b_matrix_dims, combined, idx, b_block);
        detail::cholesky_solve_block(
            l_clone.data() + l_off, b_clone.data() + b_off, result.data() + idx * b_block, n, nrhs);
    }
    return result;
}

template <typename T>
template <typename B>
auto lu_factorization<T>::solve(B const& b, bool left, bool adjoint) const
{
    return lu_solve(LU(), pivots(), b, left, adjoint);
}

template <typename T> template <typename B> auto cholesky_factorization<T>::solve(B const& b) const
{
    return cholesky_solve(b, L(), upper());
}

namespace linalg
{
using vectorization::cholesky_solve;
using vectorization::lu_solve;
using vectorization::solve;
using vectorization::solve_ex;
using vectorization::solve_triangular;
}  // namespace linalg

}  // namespace vectorization
