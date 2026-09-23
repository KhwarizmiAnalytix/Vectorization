/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * LibTorch cross-checks that mirror TestTensor.cpp:
 *
 *   TensorShape      — rank, dimension(), stride() match torch metadata
 *   TensorAccess     — element values after construction match torch
 *   TensorViews      — t(), permute(), view(), reshape(), slice() match torch
 *   TensorReductions — trace() matches torch::trace()
 *   TensorClone      — clone() produces the same values and is independent
 *
 * Non-contiguous views (t, permute, slice) are compared via logical indices
 * using at() on the XSigma side and .contiguous() on the torch side, so that
 * stride-based element access is exercised rather than raw flat memory.
 */

#include "VectorizationTest.h"

#if VECTORIZATION_HAS_LIBTORCH

#include "TestTensorLibTorchHelpers.h"

namespace
{
using namespace libtorch_test;

// ── strided comparison helpers ────────────────────────────────────────────────

// Compare a 2-D tensor element-wise via xs.at(dims_t{i,j}) vs the contiguous
// C-order layout of th.  Uses the N-D at() overload which multiplies by strides,
// so it works correctly for non-contiguous views (transpose, slice, etc.).
template <typename T>
double max_diff_2d(const vectorization::tensor<T>& xs, const torch::Tensor& th)
{
    using dims_t           = typename vectorization::tensor<T>::dimensions_type;
    auto              th_c = th.cpu().to(TorchDtype<T>::value).contiguous();
    const T*          p    = reinterpret_cast<const T*>(th_c.data_ptr());
    const std::size_t rows = xs.dimension(0), cols = xs.dimension(1);
    double            d = 0.0;
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j)
            d = std::max(d, static_cast<double>(std::fabs(xs.at(dims_t{i, j}) - p[i * cols + j])));
    return d;
}

// Compare a 3-D tensor element-wise via xs.at(dims_t{i,j,k}).
template <typename T>
double max_diff_3d(const vectorization::tensor<T>& xs, const torch::Tensor& th)
{
    using dims_t           = typename vectorization::tensor<T>::dimensions_type;
    auto              th_c = th.cpu().to(TorchDtype<T>::value).contiguous();
    const T*          p    = reinterpret_cast<const T*>(th_c.data_ptr());
    const std::size_t d0 = xs.dimension(0), d1 = xs.dimension(1), d2 = xs.dimension(2);
    double            d = 0.0;
    for (std::size_t i = 0; i < d0; ++i)
        for (std::size_t j = 0; j < d1; ++j)
            for (std::size_t k = 0; k < d2; ++k)
            {
                T xs_v = xs.at(dims_t{i, j, k});
                T th_v = p[i * d1 * d2 + j * d2 + k];
                d      = std::max(d, static_cast<double>(std::fabs(xs_v - th_v)));
            }
    return d;
}

// Compare a 1-D strided view via xs.at(dims_t{i}) vs the contiguous copy of th.
// Uses the N-D at() overload so stride-based indexing is applied (needed for
// slices with step > 1).
template <typename T>
double max_diff_1d(const vectorization::tensor<T>& xs, const torch::Tensor& th)
{
    using dims_t  = typename vectorization::tensor<T>::dimensions_type;
    auto     th_c = th.cpu().to(TorchDtype<T>::value).contiguous();
    const T* p    = reinterpret_cast<const T*>(th_c.data_ptr());
    double   d    = 0.0;
    for (std::size_t i = 0; i < xs.dimension(0); ++i)
        d = std::max(d, static_cast<double>(std::fabs(xs.at(dims_t{i}) - p[i])));
    return d;
}

// ── test suites ───────────────────────────────────────────────────────────────

// 1. Shape: rank, dimension(), stride() match torch metadata exactly.
template <typename T>
void test_shape_vs_libtorch()
{
    using dims_t = typename vectorization::tensor<T>::dimensions_type;

    // 1-D
    {
        constexpr std::size_t    n    = 8;
        auto                     data = rand_vec<T>(n, T(-1), T(1), 40);
        vectorization::tensor<T> xs(data.data(), n);
        auto                     th = to_torch(data.data(), n);

        EXPECT_EQ((int64_t)xs.rank(), th.dim()) << "1D rank";
        EXPECT_EQ((int64_t)xs.dimension(0), th.size(0)) << "1D size(0)";
        EXPECT_EQ((int64_t)xs.stride(0), th.stride(0)) << "1D stride(0)";
    }
    // 2-D
    {
        constexpr std::size_t    rows = 4, cols = 5;
        auto                     data = rand_vec<T>(rows * cols, T(-1), T(1), 41);
        vectorization::tensor<T> xs(data.data(), rows, cols);
        auto                     th = to_torch_2d(data.data(), rows, cols);

        EXPECT_EQ((int64_t)xs.rank(), th.dim()) << "2D rank";
        EXPECT_EQ((int64_t)xs.dimension(0), th.size(0)) << "2D rows";
        EXPECT_EQ((int64_t)xs.dimension(1), th.size(1)) << "2D cols";
        EXPECT_EQ((int64_t)xs.stride(0), th.stride(0)) << "2D stride(0)";
        EXPECT_EQ((int64_t)xs.stride(1), th.stride(1)) << "2D stride(1)";
    }
    // 3-D
    {
        const dims_t               dims       = {2, 3, 4};
        const std::vector<int64_t> torch_dims = {2, 3, 4};
        auto                       data       = rand_vec<T>(2 * 3 * 4, T(-1), T(1), 42);
        vectorization::tensor<T>   xs(data.data(), dims);
        auto                       th = to_torch_nd(data.data(), torch_dims);

        EXPECT_EQ((int64_t)xs.rank(), th.dim()) << "3D rank";
        for (int i = 0; i < 3; ++i)
        {
            EXPECT_EQ((int64_t)xs.dimension(i), th.size(i)) << "3D dim " << i;
            EXPECT_EQ((int64_t)xs.stride(i), th.stride(i)) << "3D stride " << i;
        }
    }
}

// 2. Element access: at() / operator[] values match torch after construction.
template <typename T>
void test_element_access_vs_libtorch()
{
    using tensor_t       = vectorization::tensor<T>;
    using dims_t         = typename tensor_t::dimensions_type;
    constexpr double tol = std::is_same_v<T, float> ? 1e-6 : 1e-14;

    // 1-D flat comparison
    {
        constexpr std::size_t n    = 16;
        auto                  data = rand_vec<T>(n, T(-5), T(5), 43);
        tensor_t              xs(data.data(), n);
        EXPECT_LT(max_diff(xs, to_torch(data.data(), n)), tol) << "1D values";
    }
    // 2-D: verify at(i,j) matches torch[i][j]
    {
        constexpr std::size_t rows = 3, cols = 4;
        auto                  data = rand_vec<T>(rows * cols, T(-5), T(5), 44);
        tensor_t              xs(data.data(), rows, cols);
        EXPECT_LT(max_diff_2d(xs, to_torch_2d(data.data(), rows, cols)), tol) << "2D values";
    }
    // N-D: at(dims_t{i,j,k}) matches torch element
    {
        const dims_t               dims       = {2, 3, 4};
        const std::vector<int64_t> torch_dims = {2, 3, 4};
        auto                       data       = rand_vec<T>(2 * 3 * 4, T(-5), T(5), 45);
        tensor_t                   xs(data.data(), dims);
        EXPECT_LT(max_diff_3d(xs, to_torch_nd(data.data(), torch_dims)), tol) << "3D values";
    }
}

// 3. Views: t(), permute(), view(), reshape(), slice() — shape, strides, and
//    element values must match the corresponding torch view operations.
template <typename T>
void test_views_vs_libtorch()
{
    using tensor_t       = vectorization::tensor<T>;
    using dims_t         = typename tensor_t::dimensions_type;
    constexpr double tol = std::is_same_v<T, float> ? 1e-6 : 1e-14;

    // --- t(): 2-D transpose ---
    {
        constexpr std::size_t rows = 3, cols = 5;
        auto                  data = rand_vec<T>(rows * cols, T(-4), T(4), 46);
        tensor_t              m(data.data(), rows, cols);
        auto                  th = to_torch_2d(data.data(), rows, cols);

        auto tv = m.t();
        auto tt = th.t();

        EXPECT_EQ((int64_t)tv.dimension(0), tt.size(0)) << "t() rows";
        EXPECT_EQ((int64_t)tv.dimension(1), tt.size(1)) << "t() cols";
        EXPECT_EQ((int64_t)tv.stride(0), tt.stride(0)) << "t() stride(0)";
        EXPECT_EQ((int64_t)tv.stride(1), tt.stride(1)) << "t() stride(1)";
        EXPECT_FALSE(tv.is_contiguous()) << "t() non-contiguous";
        EXPECT_EQ(tv.data(), m.data()) << "t() shares data";
        EXPECT_LT(max_diff_2d(tv, tt), tol) << "t() element values";
    }

    // --- permute(): {2,3,4} → order {2,0,1} → shape {4,2,3} ---
    {
        const dims_t               src_dims       = {2, 3, 4};
        const std::vector<int64_t> torch_src_dims = {2, 3, 4};
        auto                       data           = rand_vec<T>(2 * 3 * 4, T(-4), T(4), 47);
        tensor_t                   tmp(data.data(), src_dims);
        auto                       th = to_torch_nd(data.data(), torch_src_dims);

        auto pv = tmp.permute(dims_t{2, 0, 1});
        auto tp = th.permute({2, 0, 1});

        for (int i = 0; i < 3; ++i)
        {
            EXPECT_EQ((int64_t)pv.dimension(i), tp.size(i)) << "permute dim " << i;
            EXPECT_EQ((int64_t)pv.stride(i), tp.stride(i)) << "permute stride " << i;
        }
        EXPECT_EQ(pv.data(), tmp.data()) << "permute shares data";
        EXPECT_LT(max_diff_3d(pv, tp), tol) << "permute element values";
    }

    // --- view(): 1-D → 3×4 ---
    {
        constexpr std::size_t n    = 12;
        auto                  data = rand_vec<T>(n, T(-4), T(4), 48);
        tensor_t              v(data.data(), n);
        auto                  th = to_torch(data.data(), n);

        auto vw = v.view(dims_t{3, 4});
        auto tw = th.view({3, 4});

        EXPECT_EQ((int64_t)vw.dimension(0), tw.size(0)) << "view dim 0";
        EXPECT_EQ((int64_t)vw.dimension(1), tw.size(1)) << "view dim 1";
        EXPECT_EQ(vw.data(), v.data()) << "view shares data";
        EXPECT_LT(max_diff_2d(vw, tw), tol) << "view element values";
    }

    // --- reshape(): 1-D → 2×6 ---
    {
        constexpr std::size_t n    = 12;
        auto                  data = rand_vec<T>(n, T(-4), T(4), 49);
        tensor_t              v(data.data(), n);
        auto                  th = to_torch(data.data(), n);

        auto rs = v.reshape(dims_t{2, 6});
        auto tr = th.reshape({2, 6});

        EXPECT_EQ((int64_t)rs.dimension(0), tr.size(0)) << "reshape dim 0";
        EXPECT_EQ((int64_t)rs.dimension(1), tr.size(1)) << "reshape dim 1";
        EXPECT_EQ(rs.data(), v.data()) << "reshape shares data";
        EXPECT_LT(max_diff_2d(rs, tr), tol) << "reshape element values";
    }

    // --- slice(): 1-D with step=3, [1:8:3] → {1, 4, 7} ---
    {
        constexpr std::size_t n    = 10;
        auto                  data = rand_vec<T>(n, T(-4), T(4), 50);
        tensor_t              v(data.data(), n);
        auto                  th = to_torch(data.data(), n);

        auto xs_sl = v.slice(0, 1, 8, 3);
        auto th_sl = th.slice(0, 1, 8, 3);

        EXPECT_EQ((int64_t)xs_sl.dimension(0), th_sl.size(0)) << "1D slice size";
        EXPECT_EQ((int64_t)xs_sl.stride(0), th_sl.stride(0)) << "1D slice stride";
        EXPECT_LT(max_diff_1d(xs_sl, th_sl), tol) << "1D slice values";
    }

    // --- slice(): 2-D along dim=1, columns [1:5:2] ---
    {
        constexpr std::size_t rows = 3, cols = 6;
        auto                  data = rand_vec<T>(rows * cols, T(-4), T(4), 51);
        tensor_t              m(data.data(), rows, cols);
        auto                  th = to_torch_2d(data.data(), rows, cols);

        auto xs_sl = m.slice(1, 1, 5, 2);
        auto th_sl = th.slice(1, 1, 5, 2);

        EXPECT_EQ((int64_t)xs_sl.dimension(0), th_sl.size(0)) << "2D slice rows";
        EXPECT_EQ((int64_t)xs_sl.dimension(1), th_sl.size(1)) << "2D slice cols";
        EXPECT_EQ((int64_t)xs_sl.stride(1), th_sl.stride(1)) << "2D slice stride(1)";
        EXPECT_LT(max_diff_2d(xs_sl, th_sl), tol) << "2D slice values";
    }
}

// 4. Reductions: trace() must match torch::trace().
template <typename T>
void test_reductions_vs_libtorch()
{
    using tensor_t       = vectorization::tensor<T>;
    constexpr double tol = std::is_same_v<T, float> ? 1e-5 : 1e-13;

    // 3×3
    {
        auto     data = rand_vec<T>(9, T(-4), T(4), 52);
        tensor_t m(data.data(), 3u, 3u);
        auto     th = to_torch_2d(data.data(), 3, 3);
        EXPECT_NEAR(static_cast<double>(m.trace()), torch::trace(th).template item<double>(), tol)
            << "trace 3×3";
    }
    // 5×5
    {
        auto     data = rand_vec<T>(25, T(-4), T(4), 53);
        tensor_t m(data.data(), 5u, 5u);
        auto     th = to_torch_2d(data.data(), 5, 5);
        EXPECT_NEAR(static_cast<double>(m.trace()), torch::trace(th).template item<double>(), tol)
            << "trace 5×5";
    }
}

// 5. Clone: deep copy produces matching values and is independent of the source.
template <typename T>
void test_clone_vs_libtorch()
{
    using tensor_t            = vectorization::tensor<T>;
    constexpr double      tol = std::is_same_v<T, float> ? 1e-6 : 1e-14;
    constexpr std::size_t n   = 12;

    auto     data = rand_vec<T>(n, T(-4), T(4), 54);
    tensor_t src(data.data(), n);
    auto     th_src = to_torch(data.data(), n);

    auto dst    = src.clone();
    auto th_dst = th_src.clone();

    // Clone values must match the source
    EXPECT_LT(max_diff(dst, th_src), tol) << "clone values match source";
    EXPECT_LT(max_diff(dst, th_dst), tol) << "clone values match torch clone";

    // Modifying the clone must not affect the source (deep-copy independence)
    const T original_val = dst[0];
    dst[0]               = original_val + T(1000);
    EXPECT_NEAR(static_cast<double>(src[0]), static_cast<double>(data[0]), tol)
        << "clone is independent";
}

}  // namespace

// ── test entry points ─────────────────────────────────────────────────────────

VECTORIZATIONTEST(LibTorch, TensorShapeFloat)
{
    test_shape_vs_libtorch<float>();
    END_TEST();
}
VECTORIZATIONTEST(LibTorch, TensorShapeDouble)
{
    test_shape_vs_libtorch<double>();
    END_TEST();
}
VECTORIZATIONTEST(LibTorch, TensorAccessFloat)
{
    test_element_access_vs_libtorch<float>();
    END_TEST();
}
VECTORIZATIONTEST(LibTorch, TensorAccessDouble)
{
    test_element_access_vs_libtorch<double>();
    END_TEST();
}
VECTORIZATIONTEST(LibTorch, TensorViewsFloat)
{
    test_views_vs_libtorch<float>();
    END_TEST();
}
VECTORIZATIONTEST(LibTorch, TensorViewsDouble)
{
    test_views_vs_libtorch<double>();
    END_TEST();
}
VECTORIZATIONTEST(LibTorch, TensorReductionsFloat)
{
    test_reductions_vs_libtorch<float>();
    END_TEST();
}
VECTORIZATIONTEST(LibTorch, TensorReductionsDouble)
{
    test_reductions_vs_libtorch<double>();
    END_TEST();
}
VECTORIZATIONTEST(LibTorch, TensorCloneFloat)
{
    test_clone_vs_libtorch<float>();
    END_TEST();
}
VECTORIZATIONTEST(LibTorch, TensorCloneDouble)
{
    test_clone_vs_libtorch<double>();
    END_TEST();
}

#else

VECTORIZATIONTEST(LibTorch, TensorShapeFloat)
{
    END_TEST();
}
VECTORIZATIONTEST(LibTorch, TensorShapeDouble)
{
    END_TEST();
}
VECTORIZATIONTEST(LibTorch, TensorAccessFloat)
{
    END_TEST();
}
VECTORIZATIONTEST(LibTorch, TensorAccessDouble)
{
    END_TEST();
}
VECTORIZATIONTEST(LibTorch, TensorViewsFloat)
{
    END_TEST();
}
VECTORIZATIONTEST(LibTorch, TensorViewsDouble)
{
    END_TEST();
}
VECTORIZATIONTEST(LibTorch, TensorReductionsFloat)
{
    END_TEST();
}
VECTORIZATIONTEST(LibTorch, TensorReductionsDouble)
{
    END_TEST();
}
VECTORIZATIONTEST(LibTorch, TensorCloneFloat)
{
    END_TEST();
}
VECTORIZATIONTEST(LibTorch, TensorCloneDouble)
{
    END_TEST();
}

#endif  // VECTORIZATION_HAS_LIBTORCH
