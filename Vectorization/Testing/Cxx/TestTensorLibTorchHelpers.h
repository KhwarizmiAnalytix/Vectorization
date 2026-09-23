/*
 * XSigma: High-Performance Quantitative Library
 *
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 *
 * Shared helpers for all TestTensorLibTorch*.cpp translation units.
 * Include this header inside the #if VECTORIZATION_HAS_LIBTORCH block of each test file.
 */

#pragma once

#if VECTORIZATION_HAS_LIBTORCH

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

// LibTorch — must come before any header that pulls <cassert> on MSVC
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996 4100)
#endif
#include <torch/torch.h>
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif

#include "terminals/tensor.h"

namespace libtorch_test
{

template <typename T>
struct TorchDtype
{
    static constexpr auto value = torch::kFloat32;
};
template <>
struct TorchDtype<double>
{
    static constexpr auto value = torch::kFloat64;
};

template <typename T>
std::vector<T> rand_vec(std::size_t n, T lo, T hi, unsigned seed = 42)
{
    std::mt19937                      gen(seed);
    std::uniform_real_distribution<T> dist(lo, hi);
    std::vector<T>                    v(n);
    for (auto& x : v)
        x = dist(gen);
    return v;
}

// Returns the max absolute element-wise difference between an XSigma tensor and a torch::Tensor.
template <typename T>
double max_diff(const vectorization::tensor<T>& xs, const torch::Tensor& th)
{
    auto     th_cpu = th.cpu().to(TorchDtype<T>::value).contiguous();
    const T* ptr    = reinterpret_cast<const T*>(th_cpu.data_ptr());
    double   d      = 0.0;
    for (std::size_t i = 0; i < xs.size(); ++i)
        d = std::max(d, static_cast<double>(std::fabs(xs.data()[i] - ptr[i])));
    return d;
}

template <typename T>
torch::Tensor to_torch(const T* data, std::size_t n)
{
    return torch::from_blob(
               const_cast<T*>(data),
               {static_cast<int64_t>(n)},
               torch::TensorOptions().dtype(TorchDtype<T>::value).device(torch::kCPU))
        .clone();
}

template <typename T>
torch::Tensor to_torch_2d(const T* data, std::size_t rows, std::size_t cols)
{
    return torch::from_blob(
               const_cast<T*>(data),
               {static_cast<int64_t>(rows), static_cast<int64_t>(cols)},
               torch::TensorOptions().dtype(TorchDtype<T>::value).device(torch::kCPU))
        .clone();
}

template <typename T>
torch::Tensor to_torch_nd(const T* data, const std::vector<int64_t>& dims)
{
    return torch::from_blob(
               const_cast<T*>(data),
               dims,
               torch::TensorOptions().dtype(TorchDtype<T>::value).device(torch::kCPU))
        .clone();
}

}  // namespace libtorch_test

#endif  // VECTORIZATION_HAS_LIBTORCH
