/*
 * Vectorization: High-Performance Quantitative Library
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <exception>
#include <initializer_list>
#include <limits>
#include <string>
#include <vector>

#include "terminals/tensor.h"

namespace
{
using vectorization::tensor;

template <typename T> class LinalgFactorizations : public ::testing::Test
{
};
using FactorTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(LinalgFactorizations, FactorTypes);

// Batch of `batch` n x n matrices from row-major values (all batches
// concatenated). batch == 0 means a plain matrix (no batch dimension).
template <typename T>
tensor<T> make_matrices(size_t batch, size_t n, std::initializer_list<double> values)
{
    std::vector<size_t> dims;
    if (batch > 0)
    {
        dims.push_back(batch);
    }
    dims.push_back(n);
    dims.push_back(n);
    tensor<T> a(dims);
    size_t    k = 0;
    for (double v : values)
    {
        a.data()[k++] = static_cast<T>(v);
    }
    EXPECT_EQ(k, (batch > 0 ? batch : 1) * n * n);
    return a;
}

// max |P * L * U - A| over block b, computed independently of the library.
template <typename T>
double plu_residual(tensor<T> const& P,
    tensor<T> const&                 L,
    tensor<T> const&                 U,
    tensor<T> const&                 A,
    size_t                           n,
    size_t                           b)
{
    const size_t nn  = n * n;
    const T*     p   = P.data() + b * nn;
    const T*     l   = L.data() + b * nn;
    const T*     u   = U.data() + b * nn;
    const T*     a   = A.data() + b * nn;
    double       err = 0;
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            double sum = 0;
            for (size_t k = 0; k < n; ++k)
            {
                double lu_kj = 0;
                for (size_t m = 0; m < n; ++m)
                {
                    lu_kj += static_cast<double>(l[k * n + m]) * static_cast<double>(u[m * n + j]);
                }
                sum += static_cast<double>(p[i * n + k]) * lu_kj;
            }
            err = std::fmax(err, std::fabs(sum - static_cast<double>(a[i * n + j])));
        }
    }
    return err;
}

template <typename T>
void expect_triangular_factors(tensor<T> const& L, tensor<T> const& U, size_t n)
{
    for (size_t i = 0; i < n; ++i)
    {
        EXPECT_EQ(L.data()[i * n + i], T(1));
        for (size_t j = i + 1; j < n; ++j)
        {
            EXPECT_EQ(L.data()[i * n + j], T(0));
            EXPECT_EQ(U.data()[j * n + i], T(0));
        }
    }
}

template <typename F> std::string error_message(F&& f)
{
    try
    {
        f();
    }
    catch (std::exception const& e)
    {
        return e.what();
    }
    return {};
}

template <typename T> double tol()
{
    return 32 * static_cast<double>(std::numeric_limits<T>::epsilon());
}
}  // namespace

TYPED_TEST(LinalgFactorizations, LuReconstructsNonsingular)
{
    using T       = TypeParam;
    const auto A  = make_matrices<T>(0, 3, {1, 2, 3, 4, 5, 6, 7, 8, 10});
    auto       lu = vectorization::lu(A);
    EXPECT_LE(plu_residual(lu.P, lu.L, lu.U, A, 3, 0), 10 * tol<T>());
    expect_triangular_factors(lu.L, lu.U, 3);
}

// PyTorch's linalg.lu does not raise on singular input; A = P L U still holds.
TYPED_TEST(LinalgFactorizations, LuOfSingularMatrixDoesNotThrow)
{
    using T       = TypeParam;
    const auto A  = make_matrices<T>(0, 3, {0, 1, 2, 0, 3, 4, 0, 5, 7});
    auto       lu = vectorization::lu(A);
    EXPECT_LE(plu_residual(lu.P, lu.L, lu.U, A, 3, 0), 10 * tol<T>());
    expect_triangular_factors(lu.L, lu.U, 3);
    EXPECT_EQ(lu.U.data()[0], T(0));
}

TYPED_TEST(LinalgFactorizations, LuFactorExReportsInfoPerBatch)
{
    using T = TypeParam;
    // Batch 0 nonsingular, batch 1 singular at the first pivot.
    const auto A  = make_matrices<T>(2, 2, {2, 1, 1, 3, 0, 1, 0, 4});
    auto       ex = vectorization::lu_factor_ex(A);
    ASSERT_EQ(ex.info.size(), 2u);
    EXPECT_EQ(ex.info.data()[0], T(0));
    EXPECT_EQ(ex.info.data()[1], T(1));

    const std::string msg = error_message([&] { (void)vectorization::lu_factor(A); });
    EXPECT_NE(msg.find("lu_factor: (Batch element 1)"), std::string::npos) << msg;
    EXPECT_NE(msg.find("U[0,0] is zero"), std::string::npos) << msg;

    const std::string ex_msg =
        error_message([&] { (void)vectorization::lu_factor_ex(A, /*check_errors=*/true); });
    EXPECT_NE(ex_msg.find("lu_factor_ex: (Batch element 1)"), std::string::npos) << ex_msg;
}

// Singularity means an exactly-zero pivot, not a pivot below epsilon.
TYPED_TEST(LinalgFactorizations, SmallScaleMatrixIsNotSingular)
{
    using T       = TypeParam;
    const auto A  = make_matrices<T>(0, 2, {1e-8, 0, 0, 1e-8});
    auto       ex = vectorization::lu_factor_ex(A, /*check_errors=*/true);
    EXPECT_EQ(ex.info.data()[0], T(0));
    EXPECT_EQ(ex.pivots.data()[0], T(1));
    EXPECT_EQ(ex.pivots.data()[1], T(2));
}

TYPED_TEST(LinalgFactorizations, EmptyMatricesReturnEmptyResults)
{
    using T = TypeParam;
    for (size_t batch : {size_t(0), size_t(3)})
    {
        std::vector<size_t> dims;
        if (batch > 0)
        {
            dims.push_back(batch);
        }
        dims.push_back(0);
        dims.push_back(0);
        const tensor<T> A(dims);

        auto ex = vectorization::lu_factor_ex(A, /*check_errors=*/true);
        EXPECT_EQ(ex.LU.size(), 0u);
        EXPECT_EQ(ex.pivots.size(), 0u);
        EXPECT_EQ(ex.info.size(), batch > 0 ? batch : 1u);
        for (size_t b = 0; b < ex.info.size(); ++b)
        {
            EXPECT_EQ(ex.info.data()[b], T(0));
        }
        EXPECT_EQ(vectorization::lu_factor(A).LU.size(), 0u);

        auto lu = vectorization::lu(A);
        EXPECT_EQ(lu.P.rank(), dims.size());
        EXPECT_EQ(lu.P.size(), 0u);
        EXPECT_EQ(lu.L.size(), 0u);
        EXPECT_EQ(lu.U.size(), 0u);

        auto ch = vectorization::cholesky_ex(A, /*upper=*/false, /*check_errors=*/true);
        EXPECT_EQ(ch.L.size(), 0u);
        EXPECT_EQ(ch.info.size(), batch > 0 ? batch : 1u);
        EXPECT_EQ(vectorization::cholesky(A).size(), 0u);
    }
}

TYPED_TEST(LinalgFactorizations, CholeskyErrorNamesOperationAndBatch)
{
    using T = TypeParam;
    // Batch 0 SPD, batch 1 indefinite at the second leading minor.
    const auto A  = make_matrices<T>(2, 2, {4, 2, 2, 3, 1, 2, 2, 1});
    auto       ex = vectorization::cholesky_ex(A);
    EXPECT_EQ(ex.info.data()[0], T(0));
    EXPECT_EQ(ex.info.data()[1], T(2));

    const std::string msg = error_message([&] { (void)vectorization::cholesky(A); });
    EXPECT_NE(msg.find("cholesky: (Batch element 1)"), std::string::npos) << msg;
    EXPECT_NE(msg.find("leading minor of order 2"), std::string::npos) << msg;
}
