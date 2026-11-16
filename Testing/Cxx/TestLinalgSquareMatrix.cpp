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

template <typename T> class LinalgSquareMatrix : public ::testing::Test
{
};
using SquareTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(LinalgSquareMatrix, SquareTypes);

TYPED_TEST(LinalgSquareMatrix, InverseResidualAndStridedBatchStatus)
{
    using T = TypeParam;
    tensor<T> a{{0, 2, 1}, {1, 0, 3}, {4, 1, 1}};
    auto      inverse = vectorization::linalg::inv(a);
    // Check both A*inv(A) and inv(A)*A independently of matmul.
    for (size_t i = 0; i < 3; ++i)
    {
        for (size_t j = 0; j < 3; ++j)
        {
            T left = 0, right = 0;
            for (size_t k = 0; k < 3; ++k)
            {
                left += a[i * 3 + k] * inverse[k * 3 + j];
                right += inverse[i * 3 + k] * a[k * 3 + j];
            }
            const T expected = i == j ? T(1) : T(0);
            EXPECT_NEAR(left, expected, 32 * std::numeric_limits<T>::epsilon());
            EXPECT_NEAR(right, expected, 32 * std::numeric_limits<T>::epsilon());
        }
    }
    tensor<T>            batches(std::vector<size_t>{2, 2, 2});
    const std::vector<T> values{2, 1, 0, 4, 0, 1, 0, 2};
    std::copy(values.begin(), values.end(), batches.data());
    auto input = batches.permute({0, 2, 1});
    auto ex    = vectorization::linalg::inv_ex(input);
    ASSERT_EQ(ex.info.size(), 2u);
    EXPECT_EQ(ex.info[0], T(0));
    EXPECT_EQ(ex.info[1], T(2));
    EXPECT_NEAR(ex.inverse[0], T(0.5), 8 * std::numeric_limits<T>::epsilon());
    EXPECT_NEAR(ex.inverse[1], T(0), 8 * std::numeric_limits<T>::epsilon());
    EXPECT_NEAR(ex.inverse[2], T(-0.125), 8 * std::numeric_limits<T>::epsilon());
    EXPECT_NEAR(ex.inverse[3], T(0.25), 8 * std::numeric_limits<T>::epsilon());
    EXPECT_ANY_THROW(vectorization::inv(input));
    EXPECT_ANY_THROW(vectorization::inv_ex(input, true));
}

TYPED_TEST(LinalgSquareMatrix, DeterminantPivotParitySingularityAndLogScale)
{
    using T = TypeParam;
    tensor<T> a{{0, 2, 1}, {1, 0, 3}, {4, 1, 1}};
    auto      determinant = vectorization::linalg::det(a);
    EXPECT_EQ(determinant.rank(), 0u);
    EXPECT_NEAR(determinant[0], T(23), 64 * std::numeric_limits<T>::epsilon());
    auto [sign, logabsdet] = vectorization::linalg::slogdet(a);
    EXPECT_EQ(sign.rank(), 0u);
    EXPECT_EQ(sign[0], T(1));
    EXPECT_NEAR(logabsdet[0], std::log(T(23)), 16 * std::numeric_limits<T>::epsilon());
    tensor<T> negative{{0, 2}, {3, 0}};
    EXPECT_EQ(vectorization::det(negative)[0], T(-6));
    EXPECT_EQ(vectorization::slogdet(negative).sign[0], T(-1));
    tensor<T> singular{{0, 1}, {0, 2}};
    EXPECT_EQ(vectorization::det(singular)[0], T(0));
    auto zero = vectorization::slogdet(singular);
    EXPECT_EQ(zero.sign[0], T(0));
    EXPECT_EQ(zero.logabsdet[0], -std::numeric_limits<T>::infinity());
    const T   large = std::sqrt(std::numeric_limits<T>::max()) * T(2);
    tensor<T> scaled{{large, 0}, {0, large}};
    auto      logs = vectorization::slogdet(scaled);
    EXPECT_EQ(logs.sign[0], T(1));
    EXPECT_TRUE(std::isfinite(logs.logabsdet[0]));
    EXPECT_NEAR(logs.logabsdet[0],
        T(2) * std::log(large),
        16 * std::numeric_limits<T>::epsilon() * std::log(large));
}

TYPED_TEST(LinalgSquareMatrix, EmptyMatrixIdentityAndZeroBatches)
{
    using T = TypeParam;
    for (auto const& dims :
        {std::vector<size_t>{0, 0}, std::vector<size_t>{3, 0, 0}, std::vector<size_t>{0, 2, 2}})
    {
        tensor<T> a(dims);
        auto      inverse = vectorization::inv_ex(a, true);
        EXPECT_EQ(inverse.inverse.size(), 0u);
        auto determinant = vectorization::det(a);
        auto logs        = vectorization::slogdet(a);
        EXPECT_EQ(determinant.size(), dims.size() == 2 ? 1u : dims[0]);
        for (size_t b = 0; b < determinant.size(); ++b)
        {
            EXPECT_EQ(determinant[b], T(1));
            EXPECT_EQ(logs.sign[b], T(1));
            EXPECT_EQ(logs.logabsdet[b], T(0));
            EXPECT_EQ(inverse.info[b], T(0));
        }
    }
    tensor<T> rectangular(2, 3), vector{1, 2, 3};
    EXPECT_ANY_THROW(vectorization::inv(rectangular));
    EXPECT_ANY_THROW(vectorization::det(rectangular));
    EXPECT_ANY_THROW(vectorization::slogdet(vector));
}
}  // namespace
