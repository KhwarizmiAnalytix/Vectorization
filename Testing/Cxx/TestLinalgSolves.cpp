/*
 * Vectorization: High-Performance Quantitative Library
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "terminals/tensor.h"

namespace
{
using vectorization::tensor;

template <typename A> std::vector<size_t> shape(A const& a)
{
    return {a.dimensions().begin(), a.dimensions().end()};
}

template <typename T> class LinalgSolves : public ::testing::Test
{
};
using SolveTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(LinalgSolves, SolveTypes);

// Form B independently from a known X, including both sides and transposes.
template <typename T>
tensor<T> rhs(tensor<T> const& a, tensor<T> const& x, bool left, bool adjoint = false)
{
    tensor<T>  b(shape(x));
    const auto n = a.dimension(0);
    for (size_t i = 0; i < x.dimension(0); ++i)
    {
        for (size_t j = 0; j < x.dimension(1); ++j)
        {
            T sum = 0;
            for (size_t k = 0; k < n; ++k)
            {
                const auto row         = left ? i : k;
                const auto col         = left ? k : j;
                const T    coefficient = adjoint ? a[col * n + row] : a[row * n + col];
                sum += coefficient * (left ? x[k * x.dimension(1) + j] : x[i * x.dimension(1) + k]);
            }
            b[i * x.dimension(1) + j] = sum;
        }
    }
    return b;
}

template <typename A, typename B> void expect_close(A const& actual, B const& expected)
{
    ASSERT_EQ(shape(actual), shape(expected));
    auto values    = actual.to_host_vector();
    auto reference = expected.to_host_vector();
    using T        = typename A::value_type;
    for (size_t i = 0; i < values.size(); ++i)
    {
        const double tolerance = 128 * std::numeric_limits<T>::epsilon() *
                                 std::max(1.0, std::fabs(static_cast<double>(reference[i])));
        EXPECT_NEAR(values[i], reference[i], tolerance) << "element " << i;
    }
}

TYPED_TEST(LinalgSolves, MultipleRhsAndReusedFactorsOnBothSides)
{
    using T = TypeParam;
    tensor<T>                                  a{{0, 2, 1}, {1, 0, 3}, {4, 1, 1}};
    vectorization::linalg::lu_factorization<T> factor(a);
    const auto                                 saved_lu     = factor.LU().to_host_vector();
    const auto                                 saved_pivots = factor.pivots().to_host_vector();
    for (bool left : {true, false})
    {
        tensor<T> x =
            left ? tensor<T>{{1, 2}, {-1, 3}, {2, -2}} : tensor<T>{{1, -1, 2}, {2, 3, -2}};
        const auto b = rhs(a, x, left);
        expect_close(vectorization::linalg::solve(a, b, left), x);
        for (bool adjoint : {false, true})
        {
            const auto bt = rhs(a, x, left, adjoint);
            expect_close(factor.solve(bt, left, adjoint), x);
            expect_close(factor.solve(bt, left, adjoint), x);
        }
    }
    EXPECT_EQ(factor.LU().to_host_vector(), saved_lu);
    EXPECT_EQ(factor.pivots().to_host_vector(), saved_pivots);
}

TYPED_TEST(LinalgSolves, VectorDetectionBroadcastingAndStridedRhs)
{
    using T = TypeParam;
    tensor<T> matrices(std::vector<size_t>{2, 3, 3});
    matrices = T(0);
    for (size_t b = 0; b < 2; ++b)
    {
        for (size_t i = 0; i < 3; ++i)
        {
            matrices[b * 9 + i * 3 + i] = static_cast<T>(b + 1);
        }
    }
    tensor<T> vector{2, 4, 6};
    auto      vectors = vectorization::solve(matrices, vector);
    EXPECT_EQ(shape(vectors), (std::vector<size_t>{2, 3}));
    EXPECT_EQ(vectors.to_host_vector(), (std::vector<T>{2, 4, 6, 1, 2, 3}));
    tensor<T> batched_vectors{{2, 4, 6}, {4, 8, 12}};
    auto      batch = vectorization::solve(matrices, batched_vectors);
    EXPECT_EQ(batch.to_host_vector(), (std::vector<T>{2, 4, 6, 2, 4, 6}));
    tensor<T> storage{{2, 4, 6}};
    auto      column = storage.t();
    auto      matrix = vectorization::solve_ex(matrices, column);
    EXPECT_EQ(shape(matrix.result), (std::vector<size_t>{2, 3, 1}));
    EXPECT_EQ(matrix.result.to_host_vector(), vectors.to_host_vector());
    EXPECT_EQ(shape(matrix.info), (std::vector<size_t>{2}));
    EXPECT_EQ(matrix.info.to_host_vector(), (std::vector<T>{0, 0}));
    EXPECT_ANY_THROW(vectorization::solve(matrices, storage));

    auto      expanded_a = matrices.unsqueeze(1);
    tensor<T> expanded_b(std::vector<size_t>{1, 4, 3, 2});
    expanded_b  = T(4);
    auto result = vectorization::solve_ex(expanded_a, expanded_b);
    EXPECT_EQ(shape(result.result), (std::vector<size_t>{2, 4, 3, 2}));
    // solve_ex status describes A's factorization, before RHS broadcasting.
    EXPECT_EQ(shape(result.info), (std::vector<size_t>{2, 1}));
    for (size_t i = 0; i < result.result.size(); ++i)
    {
        EXPECT_EQ(result.result[i], i < 24 ? T(4) : T(2));
    }
}

TYPED_TEST(LinalgSolves, TriangularSelectedTriangleAndUnitDiagonal)
{
    using T          = TypeParam;
    const T   poison = std::numeric_limits<T>::quiet_NaN();
    tensor<T> x{{1, 2}, {-1, 3}, {2, -2}};
    for (bool upper : {false, true})
    {
        for (bool unit : {false, true})
        {
            tensor<T> a{{2, 1, 3}, {1, 3, 2}, {3, 2, 4}};
            tensor<T> selected(a);
            for (size_t i = 0; i < 3; ++i)
            {
                for (size_t j = 0; j < 3; ++j)
                {
                    if (upper ? j < i : j > i)
                    {
                        a[i * 3 + j]        = poison;
                        selected[i * 3 + j] = T(0);
                    }
                    if (unit && i == j)
                    {
                        a[i * 3 + j]        = poison;
                        selected[i * 3 + j] = T(1);
                    }
                }
            }
            auto b = rhs(selected, x, true);
            expect_close(vectorization::solve_triangular(a, b, upper, true, unit), x);
            tensor<T> xt(x.t());
            auto      br = rhs(selected, xt, false);
            expect_close(vectorization::solve_triangular(a, br, upper, false, unit), xt);
        }
    }
}

TYPED_TEST(LinalgSolves, CholeskyUsesSelectedTriangleAndReusesUpperOrLowerFactors)
{
    using T = TypeParam;
    tensor<T>  a{{4, 2, 0}, {2, 5, 1}, {0, 1, 3}}, x{{1, 2}, {-1, 3}, {2, -2}};
    const auto b = rhs(a, x, true);
    for (bool upper : {false, true})
    {
        tensor<T> input(a);
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                if (upper ? j < i : j > i)
                {
                    input[i * 3 + j] = std::numeric_limits<T>::quiet_NaN();
                }
            }
        }
        vectorization::cholesky_factorization<T> factor(input, upper);
        const auto                               saved = factor.L().to_host_vector();
        expect_close(factor.solve(b), x);
        expect_close(factor.solve(b), x);
        EXPECT_EQ(factor.L().to_host_vector(), saved);
    }
}

TYPED_TEST(LinalgSolves, RejectsMalformedRhsPivotsAndPlacement)
{
    using T = TypeParam;
    tensor<T> a{{2, 0}, {0, 3}}, b{{2}, {3}};
    auto      factors = vectorization::lu_factor(a);
    tensor<T> scalar(std::vector<size_t>{}), short_rhs(1, 1), vector{1, 2};
    for (auto const* invalid : {&scalar, &short_rhs})
    {
        EXPECT_ANY_THROW(vectorization::solve(a, *invalid));
        EXPECT_ANY_THROW(vectorization::lu_solve(factors.LU, factors.pivots, *invalid));
        EXPECT_ANY_THROW(vectorization::cholesky_solve(*invalid, a));
        EXPECT_ANY_THROW(vectorization::solve_triangular(a, *invalid, false));
    }
    EXPECT_ANY_THROW(vectorization::lu_solve(factors.LU, factors.pivots, vector));
    EXPECT_ANY_THROW(vectorization::cholesky_solve(vector, a));
    tensor<T> short_pivots{1};
    EXPECT_ANY_THROW(vectorization::lu_solve(factors.LU, short_pivots, b));
    for (T invalid : {T(0),
             T(-1),
             T(3),
             T(1.5),
             std::numeric_limits<T>::quiet_NaN(),
             std::numeric_limits<T>::infinity()})
    {
        tensor<T> pivots{invalid, T(2)};
        EXPECT_ANY_THROW(vectorization::lu_solve(factors.LU, pivots, b));
    }
    tensor<T> other_index(std::vector<size_t>{2, 1}, memory::device_enum::CPU, 1);
    EXPECT_ANY_THROW(vectorization::solve(a, other_index));
    EXPECT_ANY_THROW(vectorization::lu_solve(factors.LU, factors.pivots, other_index));
    EXPECT_ANY_THROW(vectorization::cholesky_solve(other_index, a));
    EXPECT_ANY_THROW(vectorization::solve_triangular(a, other_index, false));
}

TYPED_TEST(LinalgSolves, EmptyMatricesRhsAndBroadcastBatches)
{
    using T = TypeParam;
    for (auto const& dims :
        {std::vector<size_t>{0, 0}, std::vector<size_t>{0, 2, 2}, std::vector<size_t>{3, 0, 0}})
    {
        tensor<T> a(dims);
        a             = T(0);
        auto b_dims   = dims;
        b_dims.back() = 2;
        tensor<T> b(b_dims);
        auto      factors = vectorization::lu_factor(a);
        EXPECT_EQ(vectorization::solve(a, b).size(), 0u);
        EXPECT_EQ(vectorization::lu_solve(factors.LU, factors.pivots, b).size(), 0u);
        EXPECT_EQ(vectorization::cholesky_solve(b, a).size(), 0u);
        EXPECT_EQ(vectorization::solve_triangular(a, b, false).size(), 0u);
    }
    tensor<T> a{{2, 0}, {0, 3}}, no_columns(2, 0);
    EXPECT_EQ(vectorization::solve_triangular(a, no_columns, false).size(), 0u);
    EXPECT_EQ(vectorization::solve(a, no_columns).size(), 0u);
    tensor<T> empty_batch(std::vector<size_t>{0, 2, 1});
    auto      result = vectorization::solve(a, empty_batch);
    EXPECT_EQ(shape(result), shape(empty_batch));
}

TYPED_TEST(LinalgSolves, SingularStatusAndTinyNonsingularSystems)
{
    using T = TypeParam;
    tensor<T> singular{{0, 1}, {0, 2}}, b{{1}, {2}};
    auto      ex = vectorization::solve_ex(singular, b);
    EXPECT_EQ(ex.info[0], T(1));
    EXPECT_ANY_THROW(vectorization::solve_ex(singular, b, true, true));
    EXPECT_ANY_THROW(vectorization::solve(singular, b));
    tensor<T> tiny{{T(1e-8), 0}, {0, T(2e-8)}}, x{{2}, {3}};
    expect_close(vectorization::solve(tiny, rhs(tiny, x, true)), x);
}

TYPED_TEST(LinalgSolves, LargerSystemsCrossTheBlockedKernelBoundary)
{
    using T            = TypeParam;
    constexpr size_t n = 37;
    tensor<T>        a(n, n), x(n, 5);
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            a[i * n + j] = i == j ? T(4) : T(0.01) * static_cast<T>((i + j) % 5);
        }
        for (size_t j = 0; j < 5; ++j)
        {
            x[i * 5 + j] = static_cast<T>(static_cast<int>((i + j) % 7) - 3);
        }
    }
    // Force a pivot that crosses the native 32-row panel boundary.
    for (size_t j = 0; j < n; ++j)
    {
        std::swap(a[j], a[(n - 1) * n + j]);
    }
    vectorization::lu_factorization<T> factor(a);
    expect_close(factor.solve(rhs(a, x, true)), x);
    expect_close(factor.solve(rhs(a, x, true, true), true, true), x);
}
}  // namespace
