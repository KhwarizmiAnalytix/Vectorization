/*
 * Vectorization: High-Performance Quantitative Library
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "terminals/tensor.h"

namespace
{
using vectorization::bmm;
using vectorization::dot;
using vectorization::matmul;
using vectorization::matmul_out;
using vectorization::tensor;
using vectorization::tensor_view;

template <typename A, typename B, typename = void> struct can_matmul : std::false_type
{
};
template <typename A, typename B>
struct can_matmul<A, B, std::void_t<decltype(matmul(std::declval<A>(), std::declval<B>()))>>
    : std::true_type
{
};

static_assert(can_matmul<tensor<float>&, tensor_view<const float>>::value);
static_assert(!can_matmul<tensor<float>, tensor<double>>::value);
static_assert(!can_matmul<decltype(std::declval<tensor<float>>() + 1.0f), tensor<float>>::value);
static_assert(!vectorization::is_writable_tensor_operand_v<tensor_view<const float>>);
static_assert(!vectorization::is_writable_tensor_operand_v<tensor<float> const&>);

// Structural epilogue matching (alpha * matmul +/- beta * leaf); everything else
// keeps the product-temporary path.
template <typename E> constexpr bool epilogue_v = vectorization::detail::is_gemm_epilogue_v<E>;
using product_f = decltype(matmul(std::declval<tensor<float>&>(), std::declval<tensor<float>&>()));
static_assert(epilogue_v<decltype(2.0f * std::declval<product_f>())>);
static_assert(epilogue_v<decltype(std::declval<product_f>() * 2.0f)>);
static_assert(epilogue_v<decltype(2.0f * (3.0f * std::declval<product_f>()))>);
static_assert(epilogue_v<decltype(std::declval<product_f>() + std::declval<tensor<float>&>())>);
static_assert(epilogue_v<decltype(std::declval<tensor<float>&>() - std::declval<product_f>())>);
static_assert(
    epilogue_v<decltype(std::declval<product_f>() - 2.0f * std::declval<tensor_view<float>>())>);
static_assert(!epilogue_v<product_f>);
static_assert(!epilogue_v<decltype(std::declval<product_f>() + 1.0f)>);
static_assert(!epilogue_v<decltype(std::declval<product_f>() + std::declval<product_f>())>);
static_assert(!epilogue_v<decltype(std::declval<product_f>() * std::declval<tensor<float>&>())>);
static_assert(
    !epilogue_v<decltype(std::declval<product_f>() +
                         (std::declval<tensor<float>&>() + std::declval<tensor<float>&>()))>);

template <typename T> class TensorProducts : public ::testing::Test
{
};
using ProductTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(TensorProducts, ProductTypes);

template <typename A> auto shape_of(A const& a)
{
    auto dims = a.dimensions();
    return std::vector<int64_t>(dims.begin(), dims.end());
}

template <typename A, typename B> auto reference_matrix(A const& a, B const& b)
{
    using T = typename A::value_type;
    std::vector<T> result(static_cast<size_t>(a.size(0) * b.size(1)), T(0));
    for (size_t i = 0; i < static_cast<size_t>(a.size(0)); ++i)
    {
        for (size_t j = 0; j < static_cast<size_t>(b.size(1)); ++j)
        {
            for (size_t k = 0; k < static_cast<size_t>(a.size(1)); ++k)
            {
                result[i * static_cast<size_t>(b.size(1)) + j] += a.at(i, k) * b.at(k, j);
            }
        }
    }
    return result;
}

TYPED_TEST(TensorProducts, RectangularProductsAndVectorRanks)
{
    using T = TypeParam;
    tensor<T> a{{1, 2, 3}, {4, 5, 6}}, b{{7, 8}, {9, 10}, {11, 12}};
    tensor<T> c(matmul(a, b));
    EXPECT_EQ(shape_of(c), (std::vector<int64_t>{2, 2}));
    EXPECT_EQ(c.to_host_vector(), (std::vector<T>{58, 64, 139, 154}));
    tensor<T> v{1, 2, 3};
    tensor<T> mv(matmul(a, v));
    tensor<T> vm(matmul(v, b));
    tensor<T> vv(matmul(v, v));
    EXPECT_EQ(shape_of(mv), (std::vector<int64_t>{2}));
    EXPECT_EQ(shape_of(vm), (std::vector<int64_t>{2}));
    EXPECT_EQ(vv.rank(), 0u);
    EXPECT_EQ(vv.size(), 1u);
    EXPECT_EQ(mv.to_host_vector(), (std::vector<T>{14, 32}));
    EXPECT_EQ(vm.to_host_vector(), (std::vector<T>{58, 64}));
    EXPECT_EQ(vv[0], T(14));
    EXPECT_EQ(dot(v, v), T(14));
    tensor<T> const&     const_a = a;
    tensor<T> const&     const_b = b;
    auto                 av      = a.as_view();
    auto                 bv      = b.as_view();
    tensor_view<const T> ac(av), bc(bv);
    auto                 check = [&](auto const& left, auto const& right)
    {
        tensor<T> value(matmul(left, right));
        EXPECT_EQ(value.to_host_vector(), c.to_host_vector());
    };
    check(const_a, const_b);
    check(a, bv);
    check(a, bc);
    check(av, b);
    check(ac, b);
    check(av, bc);
    check(ac, bv);
    check(ac, bc);
    tensor<T> facade(vectorization::linalg::matmul(a, b));
    EXPECT_EQ(facade.to_host_vector(), c.to_host_vector());
    tensor<T> elementwise(a * a);
    EXPECT_EQ(elementwise[1], T(4));
}

TYPED_TEST(TensorProducts, BroadcastBatchesAndStrictBmm)
{
    using T = TypeParam;
    tensor<T> a(std::vector<size_t>{2, 1, 2, 3});
    tensor<T> b(std::vector<size_t>{1, 4, 3, 2});
    for (size_t i = 0; i < a.size(); ++i)
    {
        a[i] = static_cast<T>(i + 1);
    }
    for (size_t i = 0; i < b.size(); ++i)
    {
        b[i] = static_cast<T>(i % 9 + 1);
    }
    tensor<T> result(matmul(a, b));
    EXPECT_EQ(shape_of(result), (std::vector<int64_t>{2, 4, 2, 2}));
    for (size_t batch_a = 0; batch_a < 2; ++batch_a)
    {
        for (size_t batch_b = 0; batch_b < 4; ++batch_b)
        {
            for (size_t i = 0; i < 2; ++i)
            {
                for (size_t j = 0; j < 2; ++j)
                {
                    T expected = 0;
                    for (size_t k = 0; k < 3; ++k)
                    {
                        expected += a[batch_a * 6 + i * 3 + k] * b[batch_b * 6 + k * 2 + j];
                    }
                    EXPECT_EQ(result[(batch_a * 4 + batch_b) * 4 + i * 2 + j], expected);
                }
            }
        }
    }
    auto      aa = a.reshape({2, 2, 3});
    auto      bb = b.slice(1, 0, 2).reshape({2, 3, 2});
    tensor<T> strict(bmm(aa, bb));
    EXPECT_EQ(shape_of(strict), (std::vector<int64_t>{2, 2, 2}));
    tensor<T> broadcast(matmul(aa, bb));
    EXPECT_EQ(strict.to_host_vector(), broadcast.to_host_vector());
    for (size_t i = 0; i < 4; ++i)
    {
        EXPECT_EQ(strict[i], result[i]);
        EXPECT_EQ(strict[4 + i], result[20 + i]);
    }
    tensor<T> squared(::sqr(bmm(aa, bb)));
    for (size_t i = 0; i < squared.size(); ++i)
    {
        EXPECT_EQ(squared[i], strict[i] * strict[i]);
    }
    EXPECT_ANY_THROW(bmm(a, b));
    EXPECT_ANY_THROW(bmm(aa.slice(0, 0, 1), bb));

    tensor<T> v{1, 2, 3};
    tensor<T> vector_batch(matmul(v, b));
    EXPECT_EQ(shape_of(vector_batch), (std::vector<int64_t>{1, 4, 2}));
    for (size_t batch = 0; batch < 4; ++batch)
    {
        for (size_t j = 0; j < 2; ++j)
        {
            EXPECT_EQ(vector_batch[batch * 2 + j],
                b[batch * 6 + j] + T(2) * b[batch * 6 + 2 + j] + T(3) * b[batch * 6 + 4 + j]);
        }
    }
}

TYPED_TEST(TensorProducts, StridedOperandsAndReusableDestinations)
{
    using T = TypeParam;
    tensor<T> backing(6, 8);
    for (size_t i = 0; i < backing.size(); ++i)
    {
        backing[i] = static_cast<T>(i % 7 + 1);
    }
    auto      a = backing.slice(0, 0, 6, 2).slice(1, 0, 8, 2);
    tensor<T> b(4, 5);
    for (size_t i = 0; i < b.size(); ++i)
    {
        b[i] = static_cast<T>(i % 5 + 1);
    }
    const auto expected = reference_matrix(a, b);
    tensor<T>  packed(3, 5);
    auto*      original = packed.data();
    packed              = matmul(a, b);
    EXPECT_EQ(packed.data(), original);
    EXPECT_EQ(packed.to_host_vector(), expected);
    tensor<T> storage(5, 3);
    auto      transposed = storage.t();
    matmul_out(transposed, a, b);
    EXPECT_EQ(transposed.to_host_vector(), expected);
    tensor<T> spaced(3, 10);
    spaced       = T(-1);
    auto stepped = spaced.slice(1, 0, 10, 2);
    matmul_out(stepped, a, b);
    EXPECT_EQ(stepped.to_host_vector(), expected);
    for (size_t i = 1; i < spaced.size(); i += 2)
    {
        EXPECT_EQ(spaced[i], T(-1));
    }
    tensor<T> bt(b.t());
    tensor<T> at(a.t());
    tensor<T> both_transposed(matmul(at.t(), bt.t()));
    EXPECT_EQ(both_transposed.to_host_vector(), expected);
    tensor_view<const T> readonly(a);
    tensor<T>            const_result(matmul(readonly, b.as_view()));
    EXPECT_EQ(const_result.to_host_vector(), expected);
}

TYPED_TEST(TensorProducts, LargerRectanglesAndTransposedLeadingDimensions)
{
    using T = TypeParam;
    tensor<T> a(67, 59), b(59, 73);
    for (size_t i = 0; i < a.size(); ++i)
    {
        a[i] = static_cast<T>(static_cast<int>(i % 7) - 3);
    }
    for (size_t i = 0; i < b.size(); ++i)
    {
        b[i] = static_cast<T>(static_cast<int>(i % 5) - 2);
    }
    const auto expected = reference_matrix(a, b);
    tensor<T>  at(a.t()), bt(b.t());
    auto       check = [&](auto const& left, auto const& right)
    {
        tensor<T> result(matmul(left, right));
        EXPECT_EQ(result.to_host_vector(), expected);
    };
    check(a, b);
    check(at.t(), b);
    check(a, bt.t());
    check(at.t(), bt.t());
    tensor<T> backing(73, 67);
    auto      output = backing.t();
    matmul_out(output, at.t(), bt.t());
    EXPECT_EQ(output.to_host_vector(), expected);
}

TYPED_TEST(TensorProducts, AliasingAlphaBetaAndNoalias)
{
    using T = TypeParam;
    tensor<T>  a{{1, 2}, {3, 4}}, b{{2, 0}, {1, 2}};
    const auto expected = reference_matrix(a, b);
    tensor<T>  output(2, 2);
    output = T(3);
    matmul_out(output, a, b, T(2), T(4));
    for (size_t i = 0; i < output.size(); ++i)
    {
        EXPECT_EQ(output[i], T(2) * expected[i] + T(12));
    }
    output = std::numeric_limits<T>::quiet_NaN();
    matmul_out(output, a, b, T(2), T(0));
    for (size_t i = 0; i < output.size(); ++i)
    {
        EXPECT_EQ(output[i], T(2) * expected[i]);
    }
    EXPECT_ANY_THROW(matmul_out(a, a, b));
    output.noalias() = matmul(a, b);
    output.noalias() += matmul(a, b);
    output.noalias() -= matmul(a, b);
    EXPECT_EQ(output.to_host_vector(), expected);
#ifndef NDEBUG
    EXPECT_ANY_THROW(a.noalias() = matmul(a, b));
#endif
    a = matmul(a, b);
    EXPECT_EQ(a.to_host_vector(), expected);
    auto       view                = a.t();
    const auto transposed_expected = reference_matrix(view, b);
    view                           = matmul(view, b);
    EXPECT_EQ(view.to_host_vector(), transposed_expected);
    tensor<T>      invalid_storage(4);
    tensor_view<T> invalid(invalid_storage.memory_view(), {2, 2}, {1, 1});
    EXPECT_ANY_THROW(invalid.noalias() = matmul(a, b));
}

TYPED_TEST(TensorProducts, GemmEpiloguesMatchTwoPassReferences)
{
    using T = TypeParam;
    tensor<T> a(3, 4), b(4, 2), d(3, 2);
    for (size_t i = 0; i < a.size(); ++i)
    {
        a[i] = static_cast<T>(static_cast<int>(i % 5) - 2);
    }
    for (size_t i = 0; i < b.size(); ++i)
    {
        b[i] = static_cast<T>(static_cast<int>(i % 3) + 1);
    }
    for (size_t i = 0; i < d.size(); ++i)
    {
        d[i] = static_cast<T>(i + 1);
    }
    const auto p         = reference_matrix(a, b);
    const auto d_values  = d.to_host_vector();
    auto       expect_is = [&](tensor<T> const& actual, auto&& reference)
    {
        for (size_t i = 0; i < actual.size(); ++i)
        {
            EXPECT_EQ(actual[i], reference(i)) << "element " << i;
        }
    };

    tensor<T> c(3, 2);
    auto*     original = c.data();
    c                  = std::numeric_limits<T>::quiet_NaN();
    c                  = T(2) * matmul(a, b);  // beta = 0 never reads c
    expect_is(c, [&](size_t i) { return T(2) * p[i]; });
    c = T(2) * (T(3) * matmul(a, b));
    expect_is(c, [&](size_t i) { return T(6) * p[i]; });
    c = T(1);
    c = matmul(a, b) + c;
    expect_is(c, [&](size_t i) { return p[i] + T(1); });
    c = T(1);
    c = matmul(a, b) * T(2) - c;
    expect_is(c, [&](size_t i) { return T(2) * p[i] - T(1); });
    c = T(1);
    c = T(4) * c - T(3) * matmul(a, b);
    expect_is(c, [&](size_t i) { return T(4) - T(3) * p[i]; });
    c = T(1);
    c += matmul(a, b);
    c -= T(2) * matmul(a, b);
    expect_is(c, [&](size_t i) { return T(1) - p[i]; });
    c.noalias() += T(2) * matmul(a, b);
    expect_is(c, [&](size_t i) { return T(1) + p[i]; });
    EXPECT_EQ(c.data(), original);

    // A distinct d is copied into c, then GEMM accumulates with beta.
    c = std::numeric_limits<T>::quiet_NaN();
    c = matmul(a, b) + T(2) * d;
    expect_is(c, [&](size_t i) { return p[i] + T(2) * d_values[i]; });
    c = d - matmul(a, b);
    expect_is(c, [&](size_t i) { return d_values[i] - p[i]; });
    EXPECT_EQ(d.to_host_vector(), d_values);
    tensor<T> constructed(matmul(a, b) + d);
    expect_is(constructed, [&](size_t i) { return p[i] + d_values[i]; });
    tensor<T> bias{10, 20};
    c = matmul(a, b) + bias;  // broadcast leaf: copy then beta = 1
    expect_is(c, [&](size_t i) { return p[i] + (i % 2 == 0 ? T(10) : T(20)); });

    // Column-major destination takes the transposed GEMM with beta.
    tensor<T> storage(2, 3);
    auto      transposed = storage.t();
    transposed           = d;
    transposed           = matmul(a, b) + transposed;
    tensor<T> transposed_copy(transposed);
    expect_is(transposed_copy, [&](size_t i) { return p[i] + d_values[i]; });

    // The product broadcasting against d is not a single GEMM: general path.
    tensor<T> batched_d(std::vector<size_t>{2, 3, 2});
    batched_d = T(1);
    tensor<T> batched(matmul(a, b) + batched_d);
    EXPECT_EQ(shape_of(batched), (std::vector<int64_t>{2, 3, 2}));
    expect_is(batched, [&](size_t i) { return p[i % p.size()] + T(1); });

    // Destination aliasing a GEMM input: staged, still exact.
    tensor<T>  square{{1, 2}, {3, 4}}, identity{{0, 1}, {1, 0}};
    const auto swapped = reference_matrix(square, identity);
    square             = matmul(square, identity) + square;
    EXPECT_EQ(square.to_host_vector(),
        (std::vector<T>{swapped[0] + 1, swapped[1] + 2, swapped[2] + 3, swapped[3] + 4}));
    tensor<T> other{{1, 1}, {1, 1}};
    square = tensor<T>{{1, 2}, {3, 4}};
    square = matmul(square, identity) + other;
    EXPECT_EQ(square.to_host_vector(),
        (std::vector<T>{swapped[0] + 1, swapped[1] + 1, swapped[2] + 1, swapped[3] + 1}));

    // Partial overlap between d and c is still rejected before any write.
    tensor<T> backing(4, 2);
    backing      = T(0);
    auto top     = backing.slice(0, 0, 3, 1);
    auto shifted = backing.slice(0, 1, 4, 1);
    EXPECT_ANY_THROW(top = matmul(a, b) + shifted);
}

TYPED_TEST(TensorProducts, DeferredCompositionAndProductReductions)
{
    using T         = TypeParam;
    auto expression = []
    {
        tensor<T> a{{1, 2}, {3, 4}}, b{{1, 0}, {0, 1}};
        return ::log(::sqrt(::sqr(matmul(a.t(), b))) + T(1));
    }();
    tensor<T>            result(expression);
    const std::vector<T> input{1, 3, 2, 4};
    for (size_t i = 0; i < result.size(); ++i)
    {
        EXPECT_NEAR(result[i], std::log(input[i] + T(1)), 1e-5);
    }
    tensor<T> a{{1, 2}, {3, 4}}, identity{{1, 0}, {0, 1}}, bias{1, 2};
    auto      lazy = matmul(a, identity);
    a[0]           = T(5);
    tensor<T> saved(lazy);
    EXPECT_EQ(saved[0], T(5));
    a = ::sqr(matmul(a, identity) + bias);
    EXPECT_EQ(a.to_host_vector(), (std::vector<T>{36, 16, 16, 36}));
    tensor<T> exponent(::exp(matmul(identity, identity) + T(1)));
    EXPECT_NEAR(exponent[0], std::exp(T(2)), 1e-5);
    EXPECT_EQ(vectorization::sum(matmul(a, identity)), T(104));
    EXPECT_EQ(vectorization::accumulate(matmul(a, identity)), T(104));
    EXPECT_EQ(vectorization::hmin(matmul(a, identity)), T(16));
    EXPECT_EQ(vectorization::hmax(matmul(a, identity)), T(36));
    EXPECT_EQ(vectorization::mean(matmul(a, identity)), T(26));
    EXPECT_EQ(vectorization::sum(T(2) * matmul(a, identity)), T(208));
    tensor<T> fused(fma(matmul(a, identity), T(2), bias));
    EXPECT_EQ(fused.to_host_vector(), (std::vector<T>{73, 34, 33, 74}));
    EXPECT_EQ(vectorization::sum(fma(matmul(a, identity), T(2), bias)), T(214));
    tensor<T> repeated(matmul(identity, identity) + matmul(identity, identity));
    EXPECT_EQ(repeated.to_host_vector(), (std::vector<T>{2, 0, 0, 2}));
}

TYPED_TEST(TensorProducts, EmptyContractionScalarAndInvalidShapes)
{
    using T = TypeParam;
    tensor<T> a(std::vector<size_t>{2, 0}), b(std::vector<size_t>{0, 3});
    tensor<T> zero(matmul(a, b));
    EXPECT_EQ(zero.to_host_vector(), (std::vector<T>(6, T(0))));
    tensor<T> exponential(::exp(matmul(a, b)));
    EXPECT_EQ(exponential.to_host_vector(), (std::vector<T>(6, T(1))));
    zero = T(3);
    matmul_out(zero, a, b, T(2), T(4));
    EXPECT_EQ(zero.to_host_vector(), (std::vector<T>(6, T(12))));
    tensor<T> empty_a(std::vector<size_t>{0, 3}), empty_b(3, 2);
    tensor<T> empty(matmul(empty_a, empty_b));
    EXPECT_EQ(shape_of(empty), (std::vector<int64_t>{0, 2}));
    tensor<T> empty_vector(size_t(0));
    EXPECT_EQ(dot(empty_vector, empty_vector), T(0));
    tensor<T> scalar(std::vector<size_t>{});
    EXPECT_ANY_THROW(matmul(scalar, b));
    tensor<T> bad_a(2, 3), bad_b(4, 2), wrong_output(3, 3);
    EXPECT_ANY_THROW(matmul(bad_a, bad_b));
    EXPECT_ANY_THROW(dot(bad_a, bad_a));
    EXPECT_ANY_THROW(wrong_output = matmul(bad_a, empty_b));
    tensor<T> batches_a(std::vector<size_t>{2, 2, 3});
    tensor<T> batches_b(std::vector<size_t>{3, 3, 2});
    EXPECT_ANY_THROW(matmul(batches_a, batches_b));
    tensor<T> singleton_a(1, 3), singleton_b(3, 1);
    singleton_a = T(2);
    singleton_b = T(3);
    tensor<T> singleton(matmul(singleton_a, singleton_b));
    EXPECT_EQ(shape_of(singleton), (std::vector<int64_t>{1, 1}));
    EXPECT_EQ(singleton[0], T(18));
}

TYPED_TEST(TensorProducts, ProductMathPreservesNonfiniteResults)
{
    using T = TypeParam;
    tensor<T> a{{-4, 0}, {1000, 1}}, identity{{1, 0}, {0, 1}};
    tensor<T> roots(::sqrt(matmul(a, identity)));
    tensor<T> logarithms(::log(matmul(a, identity)));
    tensor<T> exponentials(::exp(matmul(a, identity)));
    EXPECT_TRUE(std::isnan(roots[0]));
    EXPECT_EQ(roots[1], T(0));
    EXPECT_TRUE(std::isnan(logarithms[0]));
    EXPECT_TRUE(std::isinf(logarithms[1]));
    EXPECT_LT(logarithms[1], T(0));
    EXPECT_TRUE(std::isinf(exponentials[2]));
    EXPECT_EQ(exponentials[1], T(1));
}
}  // namespace
