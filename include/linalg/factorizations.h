/*
 * Vectorization: High-Performance Quantitative Library
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "expressions/product_expression.h"
#include "include/matrix_operation/cholesky_decomposition.h"
#include "include/matrix_operation/lu_decomposition.h"
#include "linalg/linalg_results.h"

namespace vectorization
{
namespace detail
{
// A, unlike product_expression's two-operand batch loop, has one operand: no
// broadcasting, so each batch matrix is just a contiguous n*n block once `a`
// has been packed (see factor_batch_count/pack_square below).
template <typename A> int64_t factor_matrix_order(A const& a)
{
    const size_t rank = a.rank();
    VECTORIZATION_CHECK(rank >= 2, "factorization requires a matrix or batch of matrices");
    const int64_t n = a.size(rank - 2);
    VECTORIZATION_CHECK(a.size(rank - 1) == n,
        "factorization requires square matrices; rectangular inputs are "
        "not yet supported");
    VECTORIZATION_CHECK(
        n <= std::numeric_limits<int>::max(), "matrix order exceeds the backend integer range");
    return n;
}

template <typename A> size_t factor_batch_count(A const& a)
{
    const size_t rank  = a.rank();
    size_t       batch = 1;
    for (size_t d = 0; d + 2 < rank; ++d)
    {
        batch *= a.dimension(d);
    }
    return batch;
}

template <typename T, typename A> typename tensor<T>::dimensions_type factor_batch_dims(A const& a)
{
    typename tensor<T>::dimensions_type dims;
    const size_t                        rank = a.rank();
    for (size_t d = 0; d + 2 < rank; ++d)
    {
        dims.push_back(a.dimension(d));
    }
    return dims;
}

// Function-local thread_local rather than thread_local variable templates:
// GCC 13 emits a duplicate `__tls_guard` when a TU also has an inline
// thread_local (terminals/stream_guard.h) and rejects the translation unit.
template <typename T> memory::data_ptr<T>& lu_workspace_cache()
{
    thread_local memory::data_ptr<T> cache;
    return cache;
}

template <typename T> memory::data_ptr<T>& cholesky_workspace_cache()
{
    thread_local memory::data_ptr<T> cache;
    return cache;
}

template <typename T> T* grow_workspace(memory::data_ptr<T>& cache, size_t required_bytes)
{
    const size_t elements = required_bytes / sizeof(T) + (required_bytes % sizeof(T) != 0 ? 1 : 0);
    if (cache.size() < elements)
    {
        cache = memory::data_ptr<T>(elements, device_enum::CPU);
    }
    return cache.data();
}
}  // namespace detail

namespace detail
{
// Shared body of lu_factor_ex/lu_factor (and solve-style callers): `op`
// names the public operation in error messages, as PyTorch's
// "linalg.lu_factor: (Batch element b): ..." does. Upstream LU follows
// LAPACK getrf on every backend: a zero pivot does not stop the
// factorization, so LU/pivots are always complete and info[b] is the
// 1-based index of the first exactly-zero U diagonal entry (0 if none).
template <typename A> auto lu_factor_impl(A const& a, bool check_errors, char const* op)
{
    using T = operand_value_t<A>;
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
        "lu_factor supports float and double");
    VECTORIZATION_CHECK(
        a.device() == device_enum::CPU, "{}: requires a CPU tensor in this build", op);
    const int64_t n     = factor_matrix_order(a);
    const size_t  batch = factor_batch_count(a);
    if constexpr (std::is_same_v<T, float>)
    {
        VECTORIZATION_CHECK(n <= (int64_t{1} << std::numeric_limits<T>::digits),
            "LU order exceeds the exact integer range of floating-point pivots");
    }

    tensor<T> LU = a.clone();

    auto batch_dims = factor_batch_dims<T>(a);
    auto pivot_dims = batch_dims;
    pivot_dims.push_back(static_cast<size_t>(n));

    tensor<T> pivots(pivot_dims, device_enum::CPU, a.device_index());
    tensor<T> info(batch_dims, device_enum::CPU, a.device_index());

    // Empty matrices (n == 0) have nothing to factor: PyTorch returns empty
    // LU/pivots and zero info. Only the batch's info entries need writing;
    // the upstream kernel rejects lda == 0, so it must not be called.
    if (n == 0 || batch == 0)
    {
        for (size_t b = 0; b < batch; ++b)
        {
            info.data()[b] = T(0);
        }
        return lu_factor_ex_result<T>{std::move(LU), std::move(pivots), std::move(info)};
    }

    using kernel         = ::linalg::lu_decomposition<T, product_cpu_backend>;
    const auto required  = kernel::workspace_size(static_cast<linalg_int>(n));
    T* const   workspace = grow_workspace(lu_workspace_cache<T>(), required);

    // Upstream writes linalg_int pivots; pivots are stored as tensor<T> (see
    // linalg_results.h). Reuse the thread's cached buffer across calls.
    thread_local std::vector<linalg_int> pivot_buffer;
    pivot_buffer.resize(static_cast<size_t>(n));

    const auto nn            = static_cast<size_t>(n) * static_cast<size_t>(n);
    size_t     first_failed  = batch;
    int        first_bad_row = 0;
    for (size_t b = 0; b < batch; ++b)
    {
        int lapack_info = 0;
        kernel::apply(LU.data() + b * nn,
            static_cast<linalg_int>(n),
            pivot_buffer.data(),
            workspace,
            required,
            &lapack_info);
        T* const piv = pivots.data() + b * static_cast<size_t>(n);
        for (int64_t i = 0; i < n; ++i)
        {
            piv[i] = static_cast<T>(pivot_buffer[static_cast<size_t>(i)]);
        }
        info.data()[b] = static_cast<T>(lapack_info);
        if (lapack_info != 0 && first_failed == batch)
        {
            first_failed  = b;
            first_bad_row = lapack_info;
        }
    }
    VECTORIZATION_CHECK(!check_errors || first_failed == batch,
        "{}: (Batch element {}): U[{},{}] is zero, so the matrix is singular and the "
        "factorization cannot be used to solve",
        op,
        first_failed,
        first_bad_row - 1,
        first_bad_row - 1);
    return lu_factor_ex_result<T>{std::move(LU), std::move(pivots), std::move(info)};
}
}  // namespace detail

// Errors raised by check_errors (and by lu_factor/cholesky) are thrown as
// logging::exception (a std::exception) under the default
// logging::exception_mode::THROW; under LOG_FATAL they abort instead.
template <typename A, std::enable_if_t<are_tensor_operands_v<A>, bool> = true>
auto lu_factor_ex(A const& a, bool check_errors = false)
{
    return detail::lu_factor_impl(a, check_errors, "lu_factor_ex");
}

template <typename A, std::enable_if_t<are_tensor_operands_v<A>, bool> = true>
auto lu_factor(A const& a)
{
    using T     = operand_value_t<A>;
    auto result = detail::lu_factor_impl(a, /*check_errors=*/true, "lu_factor");
    return lu_factor_result<T>{std::move(result.LU), std::move(result.pivots)};
}

// Square inputs only: rectangular lu()/lu_factor() is a deferred P5 part 2
// gap (see docs/vectorization_design_plan.md section 11), not yet confirmed
// supported by the upstream lu_decomposition wrapper.
//
// Like PyTorch's linalg.lu, singular input is not an error: A = P * L * U
// still holds, with a zero on U's diagonal.
template <typename A, std::enable_if_t<are_tensor_operands_v<A>, bool> = true> auto lu(A const& a)
{
    using T                = operand_value_t<A>;
    auto          factored = detail::lu_factor_impl(a, /*check_errors=*/false, "lu");
    const size_t  rank     = factored.LU.rank();
    const int64_t n        = factored.LU.size(rank - 1);
    const size_t  batch    = detail::factor_batch_count(factored.LU);

    // Outputs are written element-for-element below, so allocate them
    // directly rather than cloning LU.
    auto dims = detail::factor_batch_dims<T>(factored.LU);
    dims.push_back(static_cast<size_t>(n));
    dims.push_back(static_cast<size_t>(n));
    tensor<T> P(dims, device_enum::CPU, factored.LU.device_index());
    tensor<T> L(dims, device_enum::CPU, factored.LU.device_index());
    tensor<T> U = std::move(factored.LU);

    // perm[i] = original row that ends up at row i, from replaying getrf's
    // swap(i, pivot[i]-1) in increasing-i order: P_lapack * A = L * U with
    // P_lapack's row i = e_{perm[i]}. PyTorch's convention is A = P * L * U,
    // so P = P_lapack^T, i.e. P[perm[i]][i] = 1.
    std::vector<int64_t> perm(static_cast<size_t>(n));
    const auto           nn = static_cast<size_t>(n) * static_cast<size_t>(n);
    for (size_t b = 0; b < batch; ++b)
    {
        T*       p_block = P.data() + b * nn;
        T*       l_block = L.data() + b * nn;
        T*       u_block = U.data() + b * nn;
        T const* piv     = factored.pivots.data() + b * static_cast<size_t>(n);

        for (int64_t i = 0; i < n; ++i)
        {
            perm[static_cast<size_t>(i)] = i;
        }
        for (int64_t i = 0; i < n; ++i)
        {
            const auto piv_i = static_cast<size_t>(static_cast<int64_t>(piv[i]) - 1);
            std::swap(perm[static_cast<size_t>(i)], perm[piv_i]);
        }
        for (size_t k = 0; k < nn; ++k)
        {
            p_block[k] = T(0);
        }
        for (int64_t i = 0; i < n; ++i)
        {
            p_block[perm[static_cast<size_t>(i)] * n + i] = T(1);
        }

        // U is LU itself: move L's multipliers out of the strict lower
        // triangle, leaving U's upper triangle (and diagonal) in place.
        for (int64_t i = 0; i < n; ++i)
        {
            for (int64_t j = 0; j < n; ++j)
            {
                T& u = u_block[i * n + j];
                if (j < i)
                {
                    l_block[i * n + j] = u;
                    u                  = T(0);
                }
                else
                {
                    l_block[i * n + j] = j == i ? T(1) : T(0);
                }
            }
        }
    }
    return lu_result<T>{std::move(P), std::move(L), std::move(U)};
}

namespace detail
{
template <typename A> auto cholesky_impl(A const& a, bool upper, bool check_errors, char const* op)
{
    using T = operand_value_t<A>;
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
        "cholesky supports float and double");
    VECTORIZATION_CHECK(
        a.device() == device_enum::CPU, "{}: requires a CPU tensor in this build", op);
    const int64_t n     = factor_matrix_order(a);
    const size_t  batch = factor_batch_count(a);

    tensor<T> L = a.clone();
    tensor<T> info(factor_batch_dims<T>(a), device_enum::CPU, a.device_index());

    // Empty matrices: nothing to factor (see lu_factor_impl).
    if (n == 0)
    {
        for (size_t b = 0; b < batch; ++b)
        {
            info.data()[b] = T(0);
        }
        return cholesky_ex_result<T>{std::move(L), std::move(info)};
    }

    using kernel = ::linalg::cholesky_decomposition<T, product_cpu_backend>;
    // The native upper kernel reads the lower input triangle. Normalize the
    // selected upper triangle into the lower one, then use the lower kernel
    // on every backend and transpose its output below.
    const auto type      = ::linalg::cholesky_decomposition_enum::LOWER_TRIANGULAR;
    const auto required  = kernel::workspace_size(static_cast<linalg_int>(n), type);
    T* const   workspace = grow_workspace(cholesky_workspace_cache<T>(), required);

    const auto nn              = static_cast<size_t>(n) * static_cast<size_t>(n);
    size_t     first_failed    = batch;
    int        first_bad_minor = 0;
    for (size_t b = 0; b < batch; ++b)
    {
        T* const block = L.data() + b * nn;
        if (upper)
        {
            for (int64_t i = 0; i < n; ++i)
            {
                for (int64_t j = 0; j < i; ++j)
                {
                    block[i * n + j] = block[j * n + i];
                }
            }
        }
        int lapack_info = 0;
        kernel::apply(block, static_cast<linalg_int>(n), type, workspace, required, &lapack_info);
        info.data()[b] = static_cast<T>(lapack_info);
        if (lapack_info != 0 && first_failed == batch)
        {
            first_failed    = b;
            first_bad_minor = lapack_info;
        }
        for (int64_t i = 0; i < n; ++i)
        {
            for (int64_t j = 0; j < n; ++j)
            {
                if (j > i)
                {
                    block[i * n + j] = upper ? block[j * n + i] : T(0);
                    if (upper)
                    {
                        block[j * n + i] = T(0);
                    }
                }
            }
        }
    }
    VECTORIZATION_CHECK(!check_errors || first_failed == batch,
        "{}: (Batch element {}): the factorization could not be completed because the input is "
        "not positive-definite (the leading minor of order {} is not positive-definite)",
        op,
        first_failed,
        first_bad_minor);
    return cholesky_ex_result<T>{std::move(L), std::move(info)};
}
}  // namespace detail

template <typename A, std::enable_if_t<are_tensor_operands_v<A>, bool> = true>
auto cholesky_ex(A const& a, bool upper = false, bool check_errors = false)
{
    return detail::cholesky_impl(a, upper, check_errors, "cholesky_ex");
}

template <typename A, std::enable_if_t<are_tensor_operands_v<A>, bool> = true>
auto cholesky(A const& a, bool upper = false)
{
    return detail::cholesky_impl(a, upper, /*check_errors=*/true, "cholesky").L;
}

template <typename T> class lu_factorization
{
public:
    template <typename A,
        std::enable_if_t<are_tensor_operands_v<A> && std::is_same_v<operand_value_t<A>, T>, bool> =
            true>
    explicit lu_factorization(A const& a) : factors_(lu_factor(a))
    {
    }

    tensor<T> const& LU() const { return factors_.LU; }
    tensor<T> const& pivots() const { return factors_.pivots; }

    // Defined in linalg/solve.h (which includes this header), avoiding a
    // circular include: lu_solve() lives there and itself builds on
    // lu_factor(), which this class already depends on.
    template <typename B> auto solve(B const& b, bool left = true, bool adjoint = false) const;

private:
    lu_factor_result<T> factors_;
};

template <typename T> class cholesky_factorization
{
public:
    template <typename A,
        std::enable_if_t<are_tensor_operands_v<A> && std::is_same_v<operand_value_t<A>, T>, bool> =
            true>
    explicit cholesky_factorization(A const& a, bool upper = false)
        : factor_(cholesky(a, upper)), upper_(upper)
    {
    }

    tensor<T> const& L() const { return factor_; }
    bool             upper() const { return upper_; }

    // Defined in linalg/solve.h; see lu_factorization::solve's comment.
    template <typename B> auto solve(B const& b) const;

private:
    tensor<T> factor_;
    bool      upper_;
};

namespace linalg
{
using vectorization::cholesky;
using vectorization::cholesky_ex;
using vectorization::cholesky_factorization;
using vectorization::lu;
using vectorization::lu_factor;
using vectorization::lu_factor_ex;
using vectorization::lu_factorization;
}  // namespace linalg

}  // namespace vectorization
