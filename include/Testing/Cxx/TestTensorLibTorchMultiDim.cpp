/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * N-D tensor correctness tests against LibTorch.
 * Verifies that elementwise kernels produce correct results for 3-D {3,8,9}
 * and 4-D {2,3,8,9} shapes (trailing dimension is non-multiple of SIMD width).
 * Covers: +, -, *, exp, sqrt, log, tanh, neg.
 */

#include "VectorizationTest.h"

#if VECTORIZATION_HAS_LIBTORCH

#include "TestTensorLibTorchHelpers.h"

namespace
{
using namespace libtorch_test;

template <typename T>
void run_nd_suite(const std::vector<int64_t>& dims, unsigned seed_base)
{
    using tensor_t       = vectorization::tensor<T>;
    using dims_t         = typename tensor_t::dimensions_type;
    constexpr double tol = std::is_same_v<T, float> ? 1e-5 : 1e-13;

    std::size_t n = 1;
    for (auto d : dims)
        n *= static_cast<std::size_t>(d);

    dims_t tensor_dims;
    tensor_dims.reserve(dims.size());
    for (auto d : dims)
        tensor_dims.push_back(static_cast<std::size_t>(d));

    auto ax = rand_vec<T>(n, T(-2), T(2), seed_base);
    auto ay = rand_vec<T>(n, T(-2), T(2), seed_base + 1);
    auto az = rand_vec<T>(n, T(0.5), T(3), seed_base + 2);  // positive for log/sqrt

    tensor_t xa(ax.data(), tensor_dims);
    tensor_t xb(ay.data(), tensor_dims);
    tensor_t xc(az.data(), tensor_dims);

    auto ta = to_torch_nd(ax.data(), dims);
    auto tb = to_torch_nd(ay.data(), dims);
    auto tc = to_torch_nd(az.data(), dims);

    {
        tensor_t r(tensor_dims);
        r = xa + xb;
        EXPECT_LT(max_diff(r, ta + tb), tol) << dims[0] << "D add";
    }
    {
        tensor_t r(tensor_dims);
        r = xa - xb;
        EXPECT_LT(max_diff(r, ta - tb), tol) << dims[0] << "D sub";
    }
    {
        tensor_t r(tensor_dims);
        r = xa * xb;
        EXPECT_LT(max_diff(r, ta * tb), tol) << dims[0] << "D mul";
    }
    {
        tensor_t r(tensor_dims);
        r = ::exp(xc);
        EXPECT_LT(max_diff(r, torch::exp(tc)), tol * 10) << dims[0] << "D exp";
    }
    {
        tensor_t r(tensor_dims);
        r = ::sqrt(xc);
        EXPECT_LT(max_diff(r, torch::sqrt(tc)), tol) << dims[0] << "D sqrt";
    }
    {
        tensor_t r(tensor_dims);
        r = ::log(xc);
        EXPECT_LT(max_diff(r, torch::log(tc)), tol * 10) << dims[0] << "D log";
    }
    {
        tensor_t r(tensor_dims);
        r = ::tanh(xa);
        EXPECT_LT(max_diff(r, torch::tanh(ta)), tol * 10) << dims[0] << "D tanh";
    }
    {
        tensor_t r(tensor_dims);
        r = -xa;
        EXPECT_LT(max_diff(r, -ta), tol) << dims[0] << "D neg";
    }
}

template <typename T>
void test_libtorch_nd()
{
    run_nd_suite<T>({3, 8, 9}, 15);     // 3-D: depth × rows × cols
    run_nd_suite<T>({2, 3, 8, 9}, 18);  // 4-D: batch × depth × rows × cols
}

}  // namespace

VECTORIZATIONTEST(LibTorch, MultiDimFloat)
{
    test_libtorch_nd<float>();
    END_TEST();
}

VECTORIZATIONTEST(LibTorch, MultiDimDouble)
{
    test_libtorch_nd<double>();
    END_TEST();
}

#else

VECTORIZATIONTEST(LibTorch, MultiDimFloat)
{
    END_TEST();
}
VECTORIZATIONTEST(LibTorch, MultiDimDouble)
{
    END_TEST();
}

#endif  // VECTORIZATION_HAS_LIBTORCH
