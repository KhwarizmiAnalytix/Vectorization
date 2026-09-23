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

#pragma once

#include <cassert>
#include <cstddef>

#ifndef VECTORIZATION_FORCE_INLINE
#if defined(_MSC_VER)
#define VECTORIZATION_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
#define VECTORIZATION_FORCE_INLINE __attribute__((always_inline)) inline
#else
#define VECTORIZATION_FORCE_INLINE inline
#endif
#endif

#if defined(_MSC_VER)
#define VECTORIZATION_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
#define VECTORIZATION_NOINLINE __attribute__((noinline))
#else
#define VECTORIZATION_NOINLINE
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define VECTORIZATION_VECTORCALL __vectorcall
#else
#define VECTORIZATION_VECTORCALL
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#define VECTORIZATION_CUDA_FUNCTION_TYPE __host__ __device__
#else
#define VECTORIZATION_CUDA_FUNCTION_TYPE
#endif

// Host-only entry points (CUDA kernel launchers, CPU SIMD loops).  Must not be
// __device__: the device instantiation would either call host APIs illegally or
// fall through to scalar stores against GPU pointers.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define VECTORIZATION_HOST_ONLY __host__
#else
#define VECTORIZATION_HOST_ONLY
#endif
//------------------------------------------------------------------------
// VECTORIZATION_SIMD_RETURN_TYPE — static methods on simd<> that do not
// return a value (stores, prefetch, transpose, etc.).
// VECTORIZATION_SIMD_METHOD — prefix for simd<> ops that return the same
// type as the underlying intrinsic (simd_t, mask_t, simd_half_t, …).
// The CUDA_FUNCTION_TYPE qualifier makes them __host__ __device__ when
// compiling with NVCC/HIPCC so the methods are callable from both host
// and device contexts (needed when the simd<> type is instantiated inside
// a .cu translation unit even if the SIMD path is inactive on the device).
#ifdef NDEBUG
#define VECTORIZATION_SIMD_RETURN_TYPE \
    VECTORIZATION_FORCE_INLINE VECTORIZATION_CUDA_FUNCTION_TYPE static void VECTORIZATION_VECTORCALL
#else
#define VECTORIZATION_SIMD_RETURN_TYPE VECTORIZATION_CUDA_FUNCTION_TYPE static void
#endif

#ifdef NDEBUG
#define VECTORIZATION_SIMD_METHOD VECTORIZATION_FORCE_INLINE VECTORIZATION_CUDA_FUNCTION_TYPE static
#else
#define VECTORIZATION_SIMD_METHOD VECTORIZATION_CUDA_FUNCTION_TYPE static
#endif

// True when compiling the device-side pass (GPU kernel body).
// Safe to use inside __host__ __device__ functions to select device-only paths.
// Used in #if VECTORIZATION_ON_GPU_DEVICE elsewhere, so it must stay a
// preprocessor macro rather than an enum.
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
// NOLINTNEXTLINE(modernize-macro-to-enum)
#define VECTORIZATION_ON_GPU_DEVICE 1
#else
// NOLINTNEXTLINE(modernize-macro-to-enum)
#define VECTORIZATION_ON_GPU_DEVICE 0
#endif

#ifdef NDEBUG
#define VECTORIZATION_FUNCTION_ATTRIBUTE VECTORIZATION_FORCE_INLINE VECTORIZATION_CUDA_FUNCTION_TYPE
#else
#define VECTORIZATION_FUNCTION_ATTRIBUTE VECTORIZATION_CUDA_FUNCTION_TYPE
#endif

#ifdef NDEBUG
#define VECTORIZATION_HOST_FUNCTION_ATTRIBUTE VECTORIZATION_FORCE_INLINE VECTORIZATION_HOST_ONLY
#else
#define VECTORIZATION_HOST_FUNCTION_ATTRIBUTE VECTORIZATION_HOST_ONLY
#endif

#if __cplusplus >= 201703L
#define VECTORIZATION_NODISCARD [[nodiscard]]
#define VECTORIZATION_UNUSED [[maybe_unused]]
#else
#define VECTORIZATION_NODISCARD
#define VECTORIZATION_UNUSED
#endif  // __cplusplus >= 201703L

#if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
#define VECTORIZATION_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#define VECTORIZATION_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define VECTORIZATION_LIKELY(expr) (expr)
#define VECTORIZATION_UNLIKELY(expr) (expr)
#endif

#if defined(_MSC_VER)
#if (_MSC_VER < 1900)
// Visual studio until 2015 is not supporting standard 'alignas' keyword
#ifdef alignas
// This check can be removed when verified that for all other versions alignas
// works as requested
#error "VECTORIZATION error: alignas already defined"
#else
#define alignas(alignment) __declspec(align(alignment))
#endif
#endif

#ifdef alignas
#define VECTORIZATION_ALIGN(alignment) alignas(alignment)
#else
#define VECTORIZATION_ALIGN(alignment) __declspec(align(alignment))
#endif
#elif defined(__GNUC__)
#define VECTORIZATION_ALIGN(alignment) __attribute__((aligned(alignment)))
#elif defined(__ICC) || defined(__INTEL_COMPILER)
#define VECTORIZATION_ALIGN(alignment) __attribute__((aligned(alignment)))
#endif

#ifdef VECTORIZATION_MOBILE
inline constexpr std::size_t VECTORIZATION_ALIGNMENT = 16;
#else
inline constexpr std::size_t VECTORIZATION_ALIGNMENT = 64;
#endif

// Manual-unroll factor for the vectorized hot loops in expressions_evaluator.h; see
// VECTORIZATION_PACKET_SIZE in the top-level CMakeLists.txt for the full rationale.
// Falls back to 1 (no manual unroll — current single-register-per-step behavior) for any
// translation unit built outside the CMake target that doesn't get the -D from the build system.
#ifndef VECTORIZATION_PACKET_SIZE
#define VECTORIZATION_PACKET_SIZE 1
#endif
static_assert(VECTORIZATION_PACKET_SIZE >= 1, "VECTORIZATION_PACKET_SIZE must be >= 1");

// Logging uses fmt-style placeholders ({}) in format strings. Include logger.h first
// (unique to Logging) so LOGGING_LOG is always defined. Include Logging's exception
// header explicitly. VECTORIZATION_LOGF / VECTORIZATION_CHECK / VECTORIZATION_THROW
// are host-only — do not use them in __device__ code.
#include <logging/logging.h>

#define VECTORIZATION_LOGF(verbosity_name, format_string, ...) \
    LOGGING_LOG(verbosity_name, format_string, ##__VA_ARGS__)
#define VECTORIZATION_LOG_DEBUG(format_string, ...) \
    LOGGING_LOG_INFO_DEBUG(format_string, ##__VA_ARGS__)
#define VECTORIZATION_WARNING(format_string, ...) LOGGING_LOG_WARNING(format_string, ##__VA_ARGS__)
#define VECTORIZATION_ERROR(format_string, ...) LOGGING_LOG_ERROR(format_string, ##__VA_ARGS__)
#define VECTORIZATION_FATAL(format_string, ...) LOGGING_LOG_FATAL(format_string, ##__VA_ARGS__)

#define VECTORIZATION_CHECK(cond, ...) LOGGING_CHECK(cond, ##__VA_ARGS__)
// For functions tagged VECTORIZATION_FUNCTION_ATTRIBUTE / VECTORIZATION_CUDA_FUNCTION_TYPE
// (__host__ __device__) whose check is only meaningful on the host (e.g. validating
// an initializer_list before a host-side copy) -- expands to nothing under the
// CUDA/HIP device compiler instead of trying to make VECTORIZATION_CHECK itself
// device-safe. See LOGGING_CHECK_IF_NOT_ON_CUDA for the full rationale (mirrors
// PyTorch's TORCH_CHECK_IF_NOT_ON_CUDA). Do not use in functions that are
// host-only in practice -- use VECTORIZATION_CHECK there so the check still fires.
#define VECTORIZATION_CHECK_IF_NOT_ON_CUDA(cond, ...) \
    LOGGING_CHECK_IF_NOT_ON_CUDA(cond, ##__VA_ARGS__)
#ifndef NDEBUG
#define VECTORIZATION_CHECK_DEBUG(cond, ...) LOGGING_CHECK_DEBUG(cond, ##__VA_ARGS__)
#else
#define VECTORIZATION_CHECK_DEBUG(cond, ...) ((void)0)
#endif
// Debug-only counterpart to VECTORIZATION_CHECK_IF_NOT_ON_CUDA -- same rule: use in
// functions tagged VECTORIZATION_FUNCTION_ATTRIBUTE / VECTORIZATION_CUDA_FUNCTION_TYPE
// (__host__ __device__), not in functions that are host-only in practice.
#ifndef NDEBUG
#define VECTORIZATION_CHECK_DEBUG_IF_NOT_ON_CUDA(cond, ...) \
    LOGGING_CHECK_DEBUG_IF_NOT_ON_CUDA(cond, ##__VA_ARGS__)
#else
#define VECTORIZATION_CHECK_DEBUG_IF_NOT_ON_CUDA(cond, ...) ((void)0)
#endif
#define VECTORIZATION_THROW(format_str, ...) LOGGING_THROW(format_str, ##__VA_ARGS__)
#define VECTORIZATION_NOT_IMPLEMENTED(format_str, ...) \
    LOGGING_NOT_IMPLEMENTED(format_str, ##__VA_ARGS__)

// Memory integration — allocator, data_ptr, data_view, and device_enum. Aliased
// into namespace vectorization so terminal headers can use them unqualified.
#include "allocator.h"
#include "common/data_ptr.h"
#include "common/data_view.h"
#include "common/device.h"
namespace vectorization
{
using memory::allocator;
using memory::data_ptr;
using memory::data_view;
using memory::device_enum;
}  // namespace vectorization
