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

#include "VectorizationTest.h"
#include "terminals/tensor.h"

#include <type_traits>
#include <vector>

namespace
{
using vectorization::overlap_status;
using vectorization::tensor;
using vectorization::tensor_view;

static_assert(std::is_same_v<decltype(std::declval<tensor<float>&>().t()), tensor_view<float>>);
static_assert(
    std::is_same_v<decltype(std::declval<tensor<float> const&>().t()), tensor_view<const float>>);
static_assert(
    std::is_same_v<decltype(std::declval<tensor_view<const float>&>().data()), const float*>);
static_assert(!std::is_assignable_v<tensor_view<const float>&, tensor_view<const float> const&>);
static_assert(!std::is_assignable_v<tensor_view<const float>&, float>);
static_assert(!std::is_convertible_v<tensor_view<float>, tensor<float>>);
static_assert(std::is_convertible_v<tensor<float>&, tensor_view<float>>);
static_assert(std::is_convertible_v<tensor<float> const&, tensor_view<const float>>);
static_assert(!std::is_convertible_v<tensor<float> const&, tensor_view<float>>);
static_assert(std::is_trivially_copyable_v<vectorization::strided_ref<float>>);
static_assert(std::is_trivially_copyable_v<vectorization::raw_lowered_t<tensor<float>>>);
static_assert(std::is_trivially_copyable_v<vectorization::strided_ref<const float>>);

tensor_view<float> retained_slice()
{
    tensor<float> source{{1, 2, 3}, {4, 5, 6}};
    return source.t().slice(0, 1, 3);
}
}  // namespace

VECTORIZATIONTEST(Math, TensorViewOwnership)
{
    auto view = retained_slice();
    EXPECT_EQ(view.to_host_vector(), (std::vector<float>{2, 5, 3, 6}));
    auto shallow = view;
    EXPECT_EQ(shallow.data(), view.data());
    shallow[0] = 20;
    EXPECT_EQ(view[0], 20);

    tensor<float> packed(view);
    EXPECT_TRUE(packed.is_contiguous());
    EXPECT_EQ(packed.stride(1), 1);
    EXPECT_NE(packed.data(), view.data());
    EXPECT_EQ(packed.to_host_vector(), view.to_host_vector());

    tensor<float> dst{{0, 0}, {0, 0}};
    auto          transposed = dst.t();
    transposed               = view;
    EXPECT_EQ(dst.to_host_vector(), (std::vector<float>{20, 3, 5, 6}));
    transposed = 7;
    EXPECT_EQ(dst.to_host_vector(), (std::vector<float>{7, 7, 7, 7}));
    auto  first     = dst.slice(0, 0, 1);
    auto  second    = dst.slice(0, 1, 2);
    auto* first_ptr = first.data();
    second          = 9;
    first           = std::move(second);
    EXPECT_EQ(first.data(), first_ptr);
    EXPECT_EQ(first[0], 9);
    first.rebind(view.slice(0, 0, 1));
    EXPECT_EQ(first.data(), view.data());

    tensor_view<const float> readonly(view);
    tensor<float>            from_const(readonly);
    EXPECT_EQ(from_const.to_host_vector(), view.to_host_vector());
    tensor<float> evaluated(readonly + 1.0f);
    EXPECT_EQ(evaluated[0], 21);
    END_TEST();
}

VECTORIZATIONTEST(Math, TensorExpressionLifetime)
{
    auto expr = []
    {
        tensor<float> a{{1, 2}, {3, 4}};
        tensor<float> b{{10, 20}, {30, 40}};
        return fma(a.t(), 2.0f, b) + tensor<float>{1, 2}.expand({2, 2});
    }();
    tensor<float> result(expr);
    EXPECT_EQ(result.to_host_vector(), (std::vector<float>{13, 28, 35, 50}));
    EXPECT_EQ(vectorization::accumulate(expr), 126);
    auto raw = vectorization::lower_raw(expr);
    static_assert(std::is_trivially_copyable_v<decltype(raw)>);
    END_TEST();
}

VECTORIZATIONTEST(Math, TensorNoaliasAndOverlap)
{
    tensor<float> a{1, 2, 3, 4, 5, 6};
    auto          dst = a.slice(0, 1, 6);
    auto          src = a.slice(0, 0, 5);
    EXPECT_ANY_THROW(dst = src);
    EXPECT_ANY_THROW(dst = src + 1.0f);
#ifndef NDEBUG
    EXPECT_ANY_THROW(dst.noalias() = src);
#endif
    dst = src.clone();
    EXPECT_EQ(a.to_host_vector(), (std::vector<float>{1, 1, 2, 3, 4, 5}));

    tensor<float> b{10, 20, 30, 40, 50, 60};
    a.noalias() = b + 2.0f;
    a.noalias() += b;
    a.noalias() -= b;
    EXPECT_EQ(a[0], 12);
    EXPECT_EQ(a[5], 62);
    EXPECT_ANY_THROW(a.noalias() = b.slice(0, 0, 3));
    auto expanded = a.slice(0, 0, 1).expand({6});
    EXPECT_ANY_THROW(expanded.noalias() = b);
    EXPECT_ANY_THROW(expanded = 0.0f);

    auto even = a.slice(0, 0, 6, 2);
    auto odd  = a.slice(0, 1, 6, 2);
    EXPECT_EQ(vectorization::classify_overlap(even.to_strided_ref(), odd.to_strided_ref()),
        overlap_status::none);
    even = odd;
    EXPECT_EQ(a.to_host_vector(), (std::vector<float>{22, 22, 42, 42, 62, 62}));

    tensor<float> square{{1, 2}, {3, 4}};
    EXPECT_ANY_THROW(square.as_view() = square.t());
    square.as_view() = square.t().clone();
    EXPECT_EQ(square.to_host_vector(), (std::vector<float>{1, 3, 2, 4}));
    END_TEST();
}

VECTORIZATIONTEST(Math, TensorOverlapLattice)
{
    // Exhaustively compare nested two-axis layouts with explicit address sets.
    // This includes intersections requiring negative inner coefficients.
    float buffer[128]{};
    for (int inner = 1; inner <= 3; ++inner)
    {
        for (int period = 2; period <= 5; ++period)
        {
            for (int an = 1; an <= period; ++an)
            {
                for (int bn = 1; bn <= period; ++bn)
                {
                    for (int offset = 0; offset <= inner * period * 2; ++offset)
                    {
                        auto make_ref = [&](int n, int off)
                        {
                            vectorization::strided_ref<float> ref;
                            ref.view_ = memory::data_view<float>::borrow(
                                buffer + off, 64, memory::device_enum::CPU);
                            ref.rank       = 2;
                            ref.sizes[0]   = 2;
                            ref.sizes[1]   = n;
                            ref.strides[0] = static_cast<int64_t>(inner) * period;
                            ref.strides[1] = inner;
                            ref.numel      = size_t{2} * static_cast<size_t>(n);
                            return ref;
                        };
                        auto a      = make_ref(an, 0);
                        auto b      = make_ref(bn, offset);
                        bool shares = false;
                        for (size_t i = 0; i < a.numel; ++i)
                        {
                            for (size_t j = 0; j < b.numel; ++j)
                            {
                                shares = shares || vectorization::strided_offset(a, i) ==
                                                       static_cast<size_t>(offset) +
                                                           vectorization::strided_offset(b, j);
                            }
                        }
                        const auto status = vectorization::classify_overlap(a, b);
                        // Dropping a singleton axis can make the stride sets differ.
                        if (status != overlap_status::too_hard)
                        {
                            EXPECT_EQ(status != overlap_status::none, shares);
                        }
                    }
                }
            }
        }
    }

    tensor<float>      storage(16u);
    tensor_view<float> repeated(storage.memory_view(), {2, 2}, {1, 1});
    tensor<float>      values{{1, 2}, {3, 4}};
    EXPECT_ANY_THROW(repeated = values);
    EXPECT_ANY_THROW(repeated.noalias() = values);
    EXPECT_ANY_THROW(repeated = 0.0f);
    EXPECT_ANY_THROW((tensor_view<float>(storage.memory_view(), {2, 2}, {-1, 1})));
    EXPECT_ANY_THROW((tensor_view<float>(storage.memory_view(), {2, 2}, {INT64_MAX, 1})));

    // Sufficient injectivity proof deliberately rejects this exotic, injective layout.
    tensor_view<float> unproven(storage.memory_view(), {3, 2}, {2, 3});
    tensor<float>      six(typename tensor<float>::dimensions_type{3, 2});
    EXPECT_ANY_THROW(unproven.noalias() = six);

    // Different stride sets and intersecting ranges require the explicit assertion.
    storage = 1.0f;
    tensor_view<float> dest(storage.memory_view(), {2, 2}, {6, 2});
    tensor_view<float> source(storage.memory_view().subview(1, 15), {2, 2}, {6, 3});
    EXPECT_EQ(vectorization::classify_overlap(dest.to_strided_ref(), source.to_strided_ref()),
        overlap_status::too_hard);
    EXPECT_ANY_THROW(dest = source);
    EXPECT_NO_THROW(dest.noalias() = source);
    END_TEST();
}

VECTORIZATIONTEST(Math, TensorAdoptedStorageLifetime)
{
    int released = 0;
    {
        auto result = [&]
        {
            auto adopted = memory::retained_ptr<float>::adopt(new float[4]{1, 2, 3, 4},
                4,
                memory::execution_context::cpu(),
                [&](float* ptr, size_t, memory::execution_context const&)
                {
                    ++released;
                    delete[] ptr;
                });
            tensor_view<float> input(std::move(adopted), {2, 2});
            return input.t() + 1.0f;
        }();
        EXPECT_EQ(released, 0);
        tensor<float> evaluated(result);
        EXPECT_EQ(evaluated.to_host_vector(), (std::vector<float>{2, 4, 3, 5}));
    }
    EXPECT_EQ(released, 1);
    auto packed = []
    {
        tensor<float> input{{1, 2, 3}, {4, 5, 6}};
        return input.t().reshape({6});
    }();
    EXPECT_EQ(packed.to_host_vector(), (std::vector<float>{1, 4, 2, 5, 3, 6}));
    tensor_view<float> temporary = tensor<float>{7, 8};
    EXPECT_EQ(temporary[1], 8);
    END_TEST();
}
