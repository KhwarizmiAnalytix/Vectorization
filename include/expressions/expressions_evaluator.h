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

#ifndef __VECTORIZATION_EXPRESSIONS_INCLUDES_INSIDE__
Do_not_include_expression_evaluator_directly_use_expression_it;
#endif

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#if VECTORIZATION_ENABLE_PARALLEL
#include "include/tools/parallel_tools.h"
#endif
#include "common/vectorization_type_traits.h"
#include "expressions/expression_interface_loader.h"
#include "expressions/iteration_plan.h"
#include "expressions/reduce_op.h"

#if VECTORIZATION_HAS_CUDA || VECTORIZATION_HAS_HIP
#include "expressions/expressions_evaluator_gpu.h"
#endif

#if VECTORIZATION_HAS_METAL
#include "expressions/expressions_evaluator_metal.h"
#endif

namespace vectorization
{
#if !(VECTORIZATION_HAS_CUDA || VECTORIZATION_HAS_HIP)
// No GPU backend selected: expressions_evaluator_gpu.h (which defines gpu_stream_t for
// CUDA/HIP builds) is not included above. Still declare the alias so the stream
// parameter on expressions_evaluator::run/fill and the tensor-level *_async API compile
// uniformly regardless of backend; it is inert (never dereferenced) in this configuration.
using gpu_stream_t = void*;
#endif

namespace detail
{
// Applies f(integral_constant<size_t, I>) for I in [0, N) via a compile-time fold. Used to
// manually unroll the hot loops below by VECTORIZATION_PACKET_SIZE independent SIMD registers:
// N independent temp/accumulator chains let the compiler interleave them, hiding the latency of
// FMA/div/transcendental ops (and, for reductions, breaking the serial dependency chain through
// a single accumulator) instead of the compiler having to discover that on its own.
template <std::size_t N, typename F, std::size_t... Is>
VECTORIZATION_FUNCTION_ATTRIBUTE void unroll_impl(F&& f, std::index_sequence<Is...>) noexcept
{
    (f(std::integral_constant<std::size_t, Is>{}), ...);
}

template <std::size_t N, typename F>
VECTORIZATION_FUNCTION_ATTRIBUTE void unroll(F&& f) noexcept
{
    unroll_impl<N>(std::forward<F>(f), std::make_index_sequence<N>{});
}
}  // namespace detail

struct expressions_evaluator
{
private:
    // Shared by run()/fill() once their GPU/Metal dispatch branches (if compiled in) have
    // returned without handling rhs: only a CPU destination should still reach this point.
    // Catches the case where this TU wasn't compiled by nvcc/hipcc (or VECTORIZATION_HAS_METAL
    // is off), so the dispatch branch above was compiled out entirely and a device destination
    // would otherwise fall through silently into the CPU path, dereferencing a device pointer
    // as host memory.
    template <typename T>
    VECTORIZATION_HOST_FUNCTION_ATTRIBUTE static void check_cpu_reachable(
        T const& rhs, const char* caller)
    {
        VECTORIZATION_CHECK(
            rhs.device() == device_enum::CPU,
            "{}: destination device() is not CPU, and no GPU/Metal dispatch handled it in this "
            "translation unit (CUDA/HIP require compiling with nvcc/hipcc; Metal requires "
            "VECTORIZATION_HAS_METAL)",
            caller);
    }

    // Matches the equivalent source-operand check in store_operand() (expression_interface.h):
    // every destination path below -- CPU SIMD store as well as the GPU/Metal kernels -- assumes
    // a packed buffer.
    template <typename T>
    VECTORIZATION_HOST_FUNCTION_ATTRIBUTE static void check_contiguous(
        T const& rhs, const char* caller)
    {
        VECTORIZATION_CHECK(
            rhs.is_contiguous(),
            "{}: destination must be contiguous; call contiguous() first",
            caller);
    }

    template <typename T>
    VECTORIZATION_HOST_FUNCTION_ATTRIBUTE static void check_result_shape(
        expression_shape const& shape, T const& rhs)
    {
        auto const dims  = rhs.dimensions();
        bool       equal = static_cast<size_t>(shape.rank) == dims.size();
        for (int d = 0; equal && d < shape.rank; ++d)
        {
            equal = shape.sizes[d] == dims[static_cast<size_t>(d)];
        }
        if (!equal)
        {
            VECTORIZATION_CHECK(
                false,
                "expression result shape {} does not match destination shape {}",
                shape_to_string(shape.rank, shape.sizes),
                shape_to_string(static_cast<int>(dims.size()), dims.data()));
        }
    }

    // SIMD chunk kernel for contiguous sources/destination. Runs the expression
    // over the flat range [begin, end) of `data[]` using unaligned stores.
    // Called from run() directly (single-threaded) and via parallel_for (threaded).
    // VectorLength is simd<value_t>::size — passed as a template arg because this
    // helper doesn't have access to the destination tensor type.
    template <typename E, typename value_t, size_t VectorLength>
    VECTORIZATION_HOST_FUNCTION_ATTRIBUTE static void run_chunk_contiguous(
        E const& expr, value_t* data, size_t begin, size_t end) noexcept
    {
        size_t i = begin;
#if VECTORIZATION_VECTORIZED
        constexpr size_t      length = VectorLength;
        constexpr std::size_t packet = VECTORIZATION_PACKET_SIZE;
        constexpr std::size_t block  = length * packet;

        if constexpr (packet > 1)
        {
            const size_t loop_peel = begin + ((end - begin) / block) * block;
            for (; i < loop_peel; i += block)
            {
                detail::unroll<packet>(
                    [&](auto k)
                    {
                        constexpr size_t off = decltype(k)::value * length;
                        const auto       temp =
                            vectorization::expression_loader<E, true, false>::evaluate(
                                expr, i + off);
                        simd<value_t>::storeu(temp, &data[i + off]);
                    });
            }
        }
        {
            const size_t loop_peel2 = begin + ((end - begin) / length) * length;
            for (; i < loop_peel2; i += length)
            {
                const auto temp =
                    vectorization::expression_loader<E, true, false>::evaluate(expr, i);
                simd<value_t>::storeu(temp, &data[i]);
            }
        }
#endif
        for (; i < end; ++i)
            data[i] = vectorization::expression_loader<E, false, false>::evaluate(expr, i);
    }

public:
    //================================================================================================
    // `stream` is forwarded to run_gpu when rhs is a CUDA/HIP tensor, launching the fused
    // expression kernel on that stream instead of the default stream (nullptr = default
    // stream, i.e. the previous, always-synchronous-with-everything-else behavior). Ignored
    // for CPU destinations.
    template <typename E, typename T>
    VECTORIZATION_HOST_FUNCTION_ATTRIBUTE static void run(
        E const& expr, T& rhs, gpu_stream_t stream = nullptr)
    {
        // Operands broadcast against each other; the result must have exactly the
        // destination's shape, which is never broadcast (design §5, planner step 2).
        const expression_shape shape = expression_result_shape(expr);
        check_result_shape(shape, rhs);
        // Every leaf packed with the destination's element count: flat indexing is
        // exact and no planning is needed (design §5, contiguous fast path).
        const bool flat = detail::leaves_flat(expr, rhs.size());

#if (VECTORIZATION_HAS_CUDA && defined(__CUDACC__)) || (VECTORIZATION_HAS_HIP && defined(__HIPCC__))
        if (rhs.device() == device_enum::CUDA || rhs.device() == device_enum::HIP)
        {
            using value_t = typename vectorization::scalar_type<T, T>::value;
            // The kernel writes out[tid]; strided destinations need the P4 offset calculator.
            check_contiguous(rhs, "expressions_evaluator::run (CUDA/HIP)");
            if (!flat)
            {
                // Strided or broadcast sources: rewrite the leaves into the result's
                // index space so the kernel's flat index is exact.
                const auto lowered = flat_lowered(expr, shape);
                vectorization::run_gpu(
                    lowered, static_cast<value_t*>(rhs.begin()), rhs.size(), stream);
                return;
            }
            // `expr` is already `E const&` here (this overload only ever deduces E without
            // cv/ref — the E&& forwarding overload below strips those before recursing into
            // this one), so no remove_cvref_t cast is needed. Passing `expr` directly (rather
            // than through a `static_cast<remove_cvref_t<E> const&>`) avoids an nvcc codegen
            // bug where the alias-template-sugared cast type leaks into the mangled name of
            // the gpu_eval_kernel<E, T> instantiation inside run_gpu, producing an
            // unparsable host stub (`remove_cvref_t<...>` used as a declarator).
            vectorization::run_gpu(expr, static_cast<value_t*>(rhs.begin()), rhs.size(), stream);
            return;
        }
#endif
#if VECTORIZATION_HAS_METAL
        // No __CUDACC__/__HIPCC__-style device-compiler-pass check needed here: unlike
        // run_gpu (a __global__ kernel only defined when nvcc/hipcc processes this TU),
        // run_metal is ordinary host C++, always defined when VECTORIZATION_HAS_METAL=1.
        // if constexpr (not runtime if): run() is instantiated for every T reachable
        // from any call site regardless of which device a given tensor uses at runtime,
        // and run_metal is float-only (MSL has no double) — for T=tensor<double> this
        // block must not even be instantiated, since a double tensor can never actually
        // have device()==METAL (memory::allocator<double>::allocate already throws for
        // device_enum::METAL, so construction fails long before evaluation).
        if constexpr (std::is_same_v<typename vectorization::scalar_type<T, T>::value, float>)
        {
            if (rhs.device() == device_enum::METAL)
            {
                // The fused MSL kernel indexes every buffer as in[tid].
                check_contiguous(rhs, "expressions_evaluator::run (Metal)");
                VECTORIZATION_CHECK(
                    flat,
                    "expressions_evaluator::run (Metal): strided or broadcast operands are not "
                    "supported on Metal yet; call contiguous() on them first");
                vectorization::run_metal(expr, rhs);
                return;
            }
        }
#endif
        check_cpu_reachable(rhs, "expressions_evaluator::run");
        (void)stream;

        // Strided destination or sources, or broadcasting: planner + loop2d.
        if (!flat || !rhs.is_contiguous())
        {
            run_planned(expr, rhs.to_strided_ref(), shape);
            return;
        }

        auto*  data = rhs.data();
        size_t n    = rhs.size();
        size_t i    = 0;

#if VECTORIZATION_VECTORIZED
        using value_t  = typename vectorization::scalar_type<T, T>::value;
        using raw_dest = vectorization::remove_cvref_t<T>;
        static_assert(
            vectorization::is_base_expression<raw_dest>::value,
            "expressions_evaluator::run (vectorized) requires a tensor destination");

        // length == simd<value_t>::size: one SIMD register per loop step.
        // block == length * VECTORIZATION_PACKET_SIZE: the unrolled step below processes
        // this many elements per iteration when VECTORIZATION_PACKET_SIZE > 1.
        constexpr auto        length = T::length();
        constexpr std::size_t packet = VECTORIZATION_PACKET_SIZE;
        constexpr std::size_t block  = length * packet;

        // Scalar peel to the first SIMD-aligned destination element.
        const size_t cap = rhs.align_start();
        for (; i < cap; ++i)
            data[i] = vectorization::expression_loader<E, false, false>::evaluate(expr, i);

        if (i >= n)
            return;

        // Aligned destination body: for large tensors split across threads; each
        // chunk uses unaligned stores (loadu/storeu on aligned data is free on
        // current x86 and Arm — C6). For small tensors stay single-threaded.
        static constexpr int64_t kGrainSize = 32768;  // elements, as in PyTorch

        if (static_cast<int64_t>(n - i) >= kGrainSize)
        {
#if VECTORIZATION_ENABLE_PARALLEL
            // Parallel path: every chunk is run_chunk_contiguous over [begin, end).
            // The scalar peel [0, cap) was already handled above, so chunks start at i.
            const size_t aligned_begin = i;
            const size_t n_remain      = n - aligned_begin;
            auto _chunk_fn = [&](size_t begin, size_t end)
            {
                run_chunk_contiguous<E, value_t, length>(
                    expr,
                    data,
                    aligned_begin + begin,
                    aligned_begin + end);
            };
            parallel_tools::parallel_for(
                size_t{0}, n_remain, static_cast<size_t>(kGrainSize), _chunk_fn);
            i = n;  // scalar tail below will be a no-op
#endif
        }
        else
        {
            // Single-threaded path with aligned stores in the destination-aligned body.
            const size_t loop_peel = rhs.align_end();
            if constexpr (packet > 1)
            {
                const size_t block_end = i + ((loop_peel - i) / block) * block;
                for (; i < block_end; i += block)
                {
                    detail::unroll<packet>(
                        [&](auto k)
                        {
                            constexpr size_t off = decltype(k)::value * length;
                            const auto       temp =
                                vectorization::expression_loader<E, true, false>::evaluate(
                                    expr, i + off);
                            simd<value_t>::store(temp, &data[i + off]);
                        });
                }
            }
            for (; i < loop_peel; i += length)
            {
                const auto temp0 =
                    vectorization::expression_loader<E, true, false>::evaluate(expr, i);
                simd<value_t>::store(temp0, &data[i]);
            }

            // Unaligned tail (remainder after aligned body).
            const size_t loop_peel2 = i + ((n - i) / length) * length;
            if constexpr (packet > 1)
            {
                const size_t block_end2 = i + ((loop_peel2 - i) / block) * block;
                for (; i < block_end2; i += block)
                {
                    detail::unroll<packet>(
                        [&](auto k)
                        {
                            constexpr size_t off = decltype(k)::value * length;
                            const auto       temp =
                                vectorization::expression_loader<E, true, false>::evaluate(
                                    expr, i + off);
                            simd<value_t>::storeu(temp, &data[i + off]);
                        });
                }
            }
            for (; i < loop_peel2; i += length)
            {
                const auto temp = vectorization::expression_loader<E, true, false>::evaluate(expr, i);
                simd<value_t>::storeu(temp, &data[i]);
            }
        }
#endif

        for (; i < n; ++i)
            data[i] = vectorization::expression_loader<E, false, false>::evaluate(expr, i);
    }

    //================================================================================================
    template <typename E, typename T>
    VECTORIZATION_HOST_FUNCTION_ATTRIBUTE static void run(
        E&& expr, T& rhs, gpu_stream_t stream = nullptr)
    {
        // Forwarding overload: avoid duplicating the hot loop implementation.
        run(static_cast<vectorization::remove_cvref_t<E> const&>(expr), rhs, stream);
    }

    //================================================================================================
    template <typename T>
    VECTORIZATION_HOST_FUNCTION_ATTRIBUTE static void scatter(
        T const& from, T& to, size_t k, size_t index)
    {
        size_t loop_peel = 0;
        auto*  data      = to.begin() + k;
        size_t n         = from.size();

#if VECTORIZATION_VECTORIZED
        using value_t = typename vectorization::scalar_type<T, T>::value;

        constexpr auto length = T::length();
        loop_peel             = length * (n / length);

        for (size_t i = 0, size = loop_peel; i < size; i += length)
        {
            const auto temp = simd<value_t>::loadu(&from.data()[i]);
            simd<value_t>::scatter(temp, static_cast<int>(index), &data[i * index]);
        }
#endif

        for (auto i = loop_peel; i < n; i++)
            data[i * index] = from.data()[i];
    }

    //================================================================================================
    // `stream` is forwarded to fill_gpu when rhs is a CUDA/HIP tensor (see run() above).
    template <typename S, typename T>
    VECTORIZATION_HOST_FUNCTION_ATTRIBUTE static void fill(
        S value, T& rhs, gpu_stream_t stream = nullptr)
    {
        // See the matching check in run() above: every destination path below assumes a
        // packed buffer.
        VECTORIZATION_CHECK(
            rhs.is_contiguous(), "expressions_evaluator::fill: destination must be contiguous");

#if (VECTORIZATION_HAS_CUDA && defined(__CUDACC__)) || (VECTORIZATION_HAS_HIP && defined(__HIPCC__))
        if (rhs.device() == device_enum::CUDA || rhs.device() == device_enum::HIP)
        {
            using value_t = typename vectorization::scalar_type<T, T>::value;
            vectorization::fill_gpu(
                static_cast<value_t*>(rhs.begin()),
                static_cast<value_t>(value),
                rhs.size(),
                stream);
            return;
        }
#endif
#if VECTORIZATION_HAS_METAL
        // See the matching if constexpr in run() above: fill_metal is float-only.
        if constexpr (std::is_same_v<typename vectorization::scalar_type<T, T>::value, float>)
        {
            if (rhs.device() == device_enum::METAL)
            {
                vectorization::fill_metal(rhs, static_cast<float>(value));
                return;
            }
        }
#endif
        check_cpu_reachable(rhs, "expressions_evaluator::fill");
        (void)stream;

        auto*  data = rhs.begin();
        size_t i    = 0;

#if VECTORIZATION_VECTORIZED
        using value_t = typename vectorization::scalar_type<T, T>::value;

        constexpr auto length = T::length();

        const auto temp = simd<value_t>::set(static_cast<value_t>(value));

        const size_t cap = rhs.align_start();
        for (; i < cap; ++i)
            data[i] = value;

        const size_t loop_peel = rhs.align_end();
        for (; i < loop_peel; i += length)
            simd<value_t>::store(temp, &data[i]);

        const size_t loop_peel2 = i + ((rhs.size() - i) / length) * length;
        for (; i < loop_peel2; i += length)
            simd<value_t>::storeu(temp, &data[i]);
#endif
        for (; i < rhs.size(); ++i)
            data[i] = value;
    }
};
}  // namespace vectorization
//================================================================================================

namespace vectorization
{
namespace detail
{
// Walk tensor / data_view leaves for device placement. Scalar-only trees stay CPU.
// Host-only: reductions never run as __device__ code.
template <typename E>
VECTORIZATION_HOST_FUNCTION_ATTRIBUTE void accumulate_expression_device(
    E const& expr, device_enum& kind, int& index, gpu_stream_t& stream, bool& seen)
{
    using expr_t = vectorization::remove_cvref_t<E>;
    if constexpr (is_pure_expression<expr_t>::value)
    {
        if constexpr (VECTORIZATION_EXPR_HAS_MHS(expr))
        {
            accumulate_expression_device(expr.lhs(), kind, index, stream, seen);
            accumulate_expression_device(expr.mhs(), kind, index, stream, seen);
            accumulate_expression_device(expr.rhs(), kind, index, stream, seen);
        }
        else if constexpr (VECTORIZATION_EXPR_HAS_LHS(expr))
        {
            accumulate_expression_device(expr.lhs(), kind, index, stream, seen);
            accumulate_expression_device(expr.rhs(), kind, index, stream, seen);
        }
        else if constexpr (VECTORIZATION_EXPR_HAS_RHS(expr))
        {
            accumulate_expression_device(expr.rhs(), kind, index, stream, seen);
        }
    }
    else if constexpr (is_base_expression<expr_t>::value)
    {
        if (!seen)
        {
            kind   = expr.device();
            index  = expr.device_index();
            stream = static_cast<gpu_stream_t>(expr.stream());
            seen   = true;
        }
        else
        {
            VECTORIZATION_CHECK(
                expr.device() == kind && expr.device_index() == index,
                "expression mixes devices or device indices");
        }
    }
}

// True when the GPU reduce kernels can read every operand at flat index i:
// a packed tensor, or an expression whose leaves are all packed with the
// result's element count. Validates the expression's shape either way.
template <typename E>
VECTORIZATION_HOST_FUNCTION_ATTRIBUTE bool reduce_operands_flat(E const& expr)
{
    if constexpr (is_pure_expression<E>::value)
    {
        return leaves_flat(expr, expression_result_shape(expr).numel());
    }
    else if constexpr (is_tensor<E>::value)
    {
        return expr.is_contiguous();
    }
    else
    {
        return true;
    }
}

template <typename E, typename value_t, reduce_op Op>
VECTORIZATION_HOST_FUNCTION_ATTRIBUTE bool try_reduce_gpu(E const& expr, value_t& out)
{
    device_enum  kind   = device_enum::CPU;
    int          index  = 0;
    gpu_stream_t stream = nullptr;
    bool         seen   = false;
    accumulate_expression_device(expr, kind, index, stream, seen);
    if (kind == device_enum::CPU)
    {
        return false;
    }
    const bool flat = reduce_operands_flat(expr);

#if (VECTORIZATION_HAS_CUDA && defined(__CUDACC__)) || (VECTORIZATION_HAS_HIP && defined(__HIPCC__))
    if (kind == device_enum::CUDA || kind == device_enum::HIP)
    {
        if constexpr (is_pure_expression<E>::value)
        {
            if (!flat)
            {
                // Strided or broadcast leaves: rebase them so the kernel's flat index is exact.
                const auto shape   = expression_result_shape(expr);
                const auto lowered = flat_lowered(expr, shape);
                out = reduce_gpu<E, value_t, Op>(lowered, shape.numel(), kind, index, stream);
                return true;
            }
        }
        else
        {
            VECTORIZATION_CHECK(
                flat, "accumulate/hmin/hmax: strided GPU tensor; call contiguous() first");
        }
        out = reduce_gpu<E, value_t, Op>(expr, expr.size(), kind, index, stream);
        return true;
    }
#endif
#if VECTORIZATION_HAS_METAL
    if constexpr (std::is_same_v<value_t, float>)
    {
        if (kind == device_enum::METAL)
        {
            VECTORIZATION_CHECK(
                flat,
                "accumulate/hmin/hmax (Metal): strided or broadcast operands are not supported "
                "on Metal yet; call contiguous() on them first");
            out = reduce_metal<E, Op>(expr, expr.size());
            return true;
        }
    }
#endif
    (void)flat;
    VECTORIZATION_CHECK(
        kind == device_enum::CPU,
        "accumulate/hmin/hmax: GPU expression but this translation unit has no GPU reduce "
        "path (CUDA/HIP require compiling with nvcc/hipcc; Metal requires "
        "VECTORIZATION_HAS_METAL)");
    return false;
}

// CPU reductions over strided tensors or strided / broadcast expressions:
// folds `combine` over the logical elements in C order. Returns false when the
// operands are packed and the SIMD paths of the caller apply.
template <typename E, typename value_t, typename F>
VECTORIZATION_HOST_FUNCTION_ATTRIBUTE bool reduce_strided(E const& expr, value_t& acc, F&& combine)
{
    if constexpr (is_tensor<E>::value)
    {
        if (expr.is_contiguous())
        {
            return false;
        }
        for (size_t i = 0, n = expr.size(); i < n; ++i)
        {
            acc = combine(acc, expr[i]);
        }
        return true;
    }
    else if constexpr (is_pure_expression<E>::value)
    {
        const auto shape = expression_result_shape(expr);
        if (leaves_flat(expr, shape.numel()))
        {
            return false;
        }
        const auto lowered = flat_lowered(expr, shape);
        for (size_t i = 0, n = shape.numel(); i < n; ++i)
        {
            acc = combine(acc, expression_loader<E, false, false>::evaluate(lowered, i));
        }
        return true;
    }
    else
    {
        return false;
    }
}
}  // namespace detail

//================================================================================================
// accumulate: returns value_t (not hardcoded double) to avoid implicit widening
// for float expressions. Not noexcept: the GPU path can VECTORIZATION_CHECK /
// allocate.
template <typename EXPR>
VECTORIZATION_HOST_FUNCTION_ATTRIBUTE auto accumulate(EXPR&& expression)
{
    using E       = vectorization::remove_cvref_t<EXPR>;
    using value_t = typename vectorization::scalar_type<E, E>::value;

    value_t gpu_result{};
    if (detail::try_reduce_gpu<E, value_t, reduce_op::sum>(
            static_cast<E const&>(expression), gpu_result))
    {
        return gpu_result;
    }

    value_t sum = 0;
    if (detail::reduce_strided(
            static_cast<E const&>(expression), sum, [](value_t a, value_t b) { return a + b; }))
    {
        return sum;
    }
    size_t i = 0;
    size_t n = expression.size();

#if VECTORIZATION_VECTORIZED
    constexpr auto        length = E::length();
    constexpr std::size_t packet = VECTORIZATION_PACKET_SIZE;
    constexpr std::size_t block  = length * packet;
    using simd_t                 = typename simd<value_t>::simd_t;

    if constexpr (vectorization::is_base_expression<E>::value)
    {
        const size_t cap = expression.align_start();
        for (; i < cap; ++i)
            sum += vectorization::expression_loader<E, false, false>::evaluate(
                static_cast<E const&>(expression), i);

        if (i < n)
        {
            simd_t sum_packet[packet];
            detail::unroll<packet>(
                [&](auto k)
                { sum_packet[decltype(k)::value] = simd<value_t>::set(static_cast<value_t>(0.)); });

            const size_t loop_peel = expression.align_end();
            if constexpr (packet > 1)
            {
                const size_t block_end = i + ((loop_peel - i) / block) * block;
                for (; i < block_end; i += block)
                {
                    vectorization::expression_loader<E, true, true>::prefetch(
                        static_cast<E const&>(expression), i + block);
                    detail::unroll<packet>(
                        [&](auto k)
                        {
                            constexpr size_t kk = decltype(k)::value;
                            const auto&      temp =
                                vectorization::expression_loader<E, true, true>::evaluate(
                                    static_cast<E const&>(expression), i + kk * length);
                            sum_packet[kk] = simd<value_t>::add(sum_packet[kk], temp);
                        });
                }
            }
            for (; i < loop_peel; i += length)
            {
                vectorization::expression_loader<E, true, true>::prefetch(
                    static_cast<E const&>(expression), i);
                const auto& temp = vectorization::expression_loader<E, true, true>::evaluate(
                    static_cast<E const&>(expression), i);
                sum_packet[0] = simd<value_t>::add(sum_packet[0], temp);
            }

            const size_t loop_peel2 = i + ((n - i) / length) * length;
            if constexpr (packet > 1)
            {
                const size_t block_end2 = i + ((loop_peel2 - i) / block) * block;
                for (; i < block_end2; i += block)
                {
                    detail::unroll<packet>(
                        [&](auto k)
                        {
                            constexpr size_t kk = decltype(k)::value;
                            vectorization::expression_loader<E, true, false>::prefetch(
                                static_cast<E const&>(expression), i + kk * length);
                        });
                    detail::unroll<packet>(
                        [&](auto k)
                        {
                            constexpr size_t kk = decltype(k)::value;
                            const auto&      temp =
                                vectorization::expression_loader<E, true, false>::evaluate(
                                    static_cast<E const&>(expression), i + kk * length);
                            sum_packet[kk] = simd<value_t>::add(sum_packet[kk], temp);
                        });
                }
            }
            for (; i < loop_peel2; i += length)
            {
                vectorization::expression_loader<E, true, false>::prefetch(
                    static_cast<E const&>(expression), i);
                const auto& temp = vectorization::expression_loader<E, true, false>::evaluate(
                    static_cast<E const&>(expression), i);
                sum_packet[0] = simd<value_t>::add(sum_packet[0], temp);
            }

            simd_t combined = sum_packet[0];
            for (std::size_t k = 1; k < packet; ++k)
                combined = simd<value_t>::add(combined, sum_packet[k]);

            sum += simd<value_t>::accumulate(combined);
        }
    }
    else
    {
        const size_t loop_peel = length * (n / length);
        if (loop_peel > 0)
        {
            simd_t sum_packet[packet];
            detail::unroll<packet>(
                [&](auto k)
                { sum_packet[decltype(k)::value] = simd<value_t>::set(static_cast<value_t>(0.)); });

            if constexpr (packet > 1)
            {
                const size_t block_end = i + ((loop_peel - i) / block) * block;
                for (; i < block_end; i += block)
                {
                    detail::unroll<packet>(
                        [&](auto k)
                        {
                            constexpr size_t kk = decltype(k)::value;
                            vectorization::expression_loader<E, true, false>::prefetch(
                                static_cast<E const&>(expression), i + kk * length);
                        });
                    detail::unroll<packet>(
                        [&](auto k)
                        {
                            constexpr size_t kk = decltype(k)::value;
                            const auto&      temp =
                                vectorization::expression_loader<E, true, false>::evaluate(
                                    static_cast<E const&>(expression), i + kk * length);
                            sum_packet[kk] = simd<value_t>::add(sum_packet[kk], temp);
                        });
                }
            }
            for (; i < loop_peel; i += length)
            {
                vectorization::expression_loader<E, true, false>::prefetch(
                    static_cast<E const&>(expression), i);
                const auto& temp = vectorization::expression_loader<E, true, false>::evaluate(
                    static_cast<E const&>(expression), i);
                sum_packet[0] = simd<value_t>::add(sum_packet[0], temp);
            }

            simd_t combined = sum_packet[0];
            for (std::size_t k = 1; k < packet; ++k)
                combined = simd<value_t>::add(combined, sum_packet[k]);

            sum += simd<value_t>::accumulate(combined);
        }
    }
#endif
    for (; i < n; ++i)
        sum += vectorization::expression_loader<E, false, false>::evaluate(
            static_cast<E const&>(expression), i);

    return sum;
}

//================================================================================================
template <typename EXPR>
VECTORIZATION_HOST_FUNCTION_ATTRIBUTE auto hmin(EXPR&& expression)
{
    using E       = vectorization::remove_cvref_t<EXPR>;
    using value_t = typename vectorization::scalar_type<E, E>::value;

    value_t gpu_result{};
    if (detail::try_reduce_gpu<E, value_t, reduce_op::min>(
            static_cast<E const&>(expression), gpu_result))
    {
        return gpu_result;
    }

    value_t ret = std::numeric_limits<value_t>::max();
    if (detail::reduce_strided(
            static_cast<E const&>(expression),
            ret,
            [](value_t a, value_t b) { return std::fmin(a, b); }))
    {
        return ret;
    }
    size_t i = 0;
    size_t n = expression.size();

#if VECTORIZATION_VECTORIZED
    constexpr auto        length = E::length();
    constexpr std::size_t packet = VECTORIZATION_PACKET_SIZE;
    constexpr std::size_t block  = length * packet;
    using simd_t                 = typename simd<value_t>::simd_t;

    if constexpr (vectorization::is_base_expression<E>::value)
    {
        const size_t cap = expression.align_start();
        for (; i < cap; ++i)
            ret = std::fmin(
                ret,
                vectorization::expression_loader<E, false, false>::evaluate(
                    static_cast<E const&>(expression), i));

        if (i < n)
        {
            simd_t min_packet[packet];
            detail::unroll<packet>(
                [&](auto k)
                {
                    min_packet[decltype(k)::value] =
                        simd<value_t>::set(std::numeric_limits<value_t>::max());
                });

            const size_t loop_peel = expression.align_end();
            if constexpr (packet > 1)
            {
                const size_t block_end = i + ((loop_peel - i) / block) * block;
                for (; i < block_end; i += block)
                {
                    detail::unroll<packet>(
                        [&](auto k)
                        {
                            constexpr size_t kk = decltype(k)::value;
                            vectorization::expression_loader<E, true, true>::prefetch(
                                static_cast<E const&>(expression), i + kk * length);
                        });
                    detail::unroll<packet>(
                        [&](auto k)
                        {
                            constexpr size_t kk = decltype(k)::value;
                            const auto&      temp =
                                vectorization::expression_loader<E, true, true>::evaluate(
                                    static_cast<E const&>(expression), i + kk * length);
                            min_packet[kk] = simd<value_t>::min(min_packet[kk], temp);
                        });
                }
            }
            for (; i < loop_peel; i += length)
            {
                vectorization::expression_loader<E, true, true>::prefetch(
                    static_cast<E const&>(expression), i);
                const auto& temp = vectorization::expression_loader<E, true, true>::evaluate(
                    static_cast<E const&>(expression), i);
                min_packet[0] = simd<value_t>::min(min_packet[0], temp);
            }

            const size_t loop_peel2 = i + ((n - i) / length) * length;
            if constexpr (packet > 1)
            {
                const size_t block_end2 = i + ((loop_peel2 - i) / block) * block;
                for (; i < block_end2; i += block)
                {
                    detail::unroll<packet>(
                        [&](auto k)
                        {
                            constexpr size_t kk = decltype(k)::value;
                            vectorization::expression_loader<E, true, false>::prefetch(
                                static_cast<E const&>(expression), i + kk * length);
                        });
                    detail::unroll<packet>(
                        [&](auto k)
                        {
                            constexpr size_t kk = decltype(k)::value;
                            const auto&      temp =
                                vectorization::expression_loader<E, true, false>::evaluate(
                                    static_cast<E const&>(expression), i + kk * length);
                            min_packet[kk] = simd<value_t>::min(min_packet[kk], temp);
                        });
                }
            }
            for (; i < loop_peel2; i += length)
            {
                vectorization::expression_loader<E, true, false>::prefetch(
                    static_cast<E const&>(expression), i);
                const auto& temp = vectorization::expression_loader<E, true, false>::evaluate(
                    static_cast<E const&>(expression), i);
                min_packet[0] = simd<value_t>::min(min_packet[0], temp);
            }

            simd_t combined = min_packet[0];
            for (std::size_t k = 1; k < packet; ++k)
                combined = simd<value_t>::min(combined, min_packet[k]);

            ret = std::fmin(ret, simd<value_t>::hmin(combined));
        }
    }
    else
    {
        const size_t loop_peel = length * (n / length);
        if (loop_peel > 0)
        {
            simd_t min_packet[packet];
            detail::unroll<packet>(
                [&](auto k)
                {
                    min_packet[decltype(k)::value] =
                        simd<value_t>::set(std::numeric_limits<value_t>::max());
                });

            if constexpr (packet > 1)
            {
                const size_t block_end = i + ((loop_peel - i) / block) * block;
                for (; i < block_end; i += block)
                {
                    detail::unroll<packet>(
                        [&](auto k)
                        {
                            constexpr size_t kk = decltype(k)::value;
                            vectorization::expression_loader<E, true, false>::prefetch(
                                static_cast<E const&>(expression), i + kk * length);
                        });
                    detail::unroll<packet>(
                        [&](auto k)
                        {
                            constexpr size_t kk = decltype(k)::value;
                            const auto&      temp =
                                vectorization::expression_loader<E, true, false>::evaluate(
                                    static_cast<E const&>(expression), i + kk * length);
                            min_packet[kk] = simd<value_t>::min(min_packet[kk], temp);
                        });
                }
            }
            for (; i < loop_peel; i += length)
            {
                vectorization::expression_loader<E, true, false>::prefetch(
                    static_cast<E const&>(expression), i);
                const auto& temp = vectorization::expression_loader<E, true, false>::evaluate(
                    static_cast<E const&>(expression), i);
                min_packet[0] = simd<value_t>::min(min_packet[0], temp);
            }

            simd_t combined = min_packet[0];
            for (std::size_t k = 1; k < packet; ++k)
                combined = simd<value_t>::min(combined, min_packet[k]);

            ret = std::fmin(ret, simd<value_t>::hmin(combined));
        }
    }
#endif
    for (; i < n; ++i)
        ret = std::fmin(
            ret,
            vectorization::expression_loader<E, false, false>::evaluate(
                static_cast<E const&>(expression), i));

    return ret;
}

//================================================================================================
template <typename EXPR>
VECTORIZATION_HOST_FUNCTION_ATTRIBUTE auto hmax(EXPR&& expression)
{
    using E       = vectorization::remove_cvref_t<EXPR>;
    using value_t = typename vectorization::scalar_type<E, E>::value;

    value_t gpu_result{};
    if (detail::try_reduce_gpu<E, value_t, reduce_op::max>(
            static_cast<E const&>(expression), gpu_result))
    {
        return gpu_result;
    }

    // Use -max(), not min(): for floats std::numeric_limits<float>::min() is the
    // smallest *positive* value (~1.2e-38), not the most negative one.
    value_t ret = -std::numeric_limits<value_t>::max();
    if (detail::reduce_strided(
            static_cast<E const&>(expression),
            ret,
            [](value_t a, value_t b) { return std::fmax(a, b); }))
    {
        return ret;
    }
    size_t i = 0;
    size_t n = expression.size();

#if VECTORIZATION_VECTORIZED
    constexpr auto        length = E::length();
    constexpr std::size_t packet = VECTORIZATION_PACKET_SIZE;
    constexpr std::size_t block  = length * packet;
    using simd_t                 = typename simd<value_t>::simd_t;

    if constexpr (vectorization::is_base_expression<E>::value)
    {
        const size_t cap = expression.align_start();
        for (; i < cap; ++i)
            ret = std::fmax(
                ret,
                vectorization::expression_loader<E, false, false>::evaluate(
                    static_cast<E const&>(expression), i));

        if (i < n)
        {
            simd_t max_packet[packet];
            detail::unroll<packet>(
                [&](auto k)
                {
                    max_packet[decltype(k)::value] =
                        simd<value_t>::set(-std::numeric_limits<value_t>::max());
                });

            const size_t loop_peel = expression.align_end();
            if constexpr (packet > 1)
            {
                const size_t block_end = i + ((loop_peel - i) / block) * block;
                for (; i < block_end; i += block)
                {
                    detail::unroll<packet>(
                        [&](auto k)
                        {
                            constexpr size_t kk = decltype(k)::value;
                            vectorization::expression_loader<E, true, true>::prefetch(
                                static_cast<E const&>(expression), i + kk * length);
                        });
                    detail::unroll<packet>(
                        [&](auto k)
                        {
                            constexpr size_t kk = decltype(k)::value;
                            const auto&      temp =
                                vectorization::expression_loader<E, true, true>::evaluate(
                                    static_cast<E const&>(expression), i + kk * length);
                            max_packet[kk] = simd<value_t>::max(max_packet[kk], temp);
                        });
                }
            }
            for (; i < loop_peel; i += length)
            {
                vectorization::expression_loader<E, true, true>::prefetch(
                    static_cast<E const&>(expression), i);
                const auto& temp = vectorization::expression_loader<E, true, true>::evaluate(
                    static_cast<E const&>(expression), i);
                max_packet[0] = simd<value_t>::max(max_packet[0], temp);
            }

            const size_t loop_peel2 = i + ((n - i) / length) * length;
            if constexpr (packet > 1)
            {
                const size_t block_end2 = i + ((loop_peel2 - i) / block) * block;
                for (; i < block_end2; i += block)
                {
                    detail::unroll<packet>(
                        [&](auto k)
                        {
                            constexpr size_t kk = decltype(k)::value;
                            vectorization::expression_loader<E, true, false>::prefetch(
                                static_cast<E const&>(expression), i + kk * length);
                        });
                    detail::unroll<packet>(
                        [&](auto k)
                        {
                            constexpr size_t kk = decltype(k)::value;
                            const auto&      temp =
                                vectorization::expression_loader<E, true, false>::evaluate(
                                    static_cast<E const&>(expression), i + kk * length);
                            max_packet[kk] = simd<value_t>::max(max_packet[kk], temp);
                        });
                }
            }
            for (; i < loop_peel2; i += length)
            {
                vectorization::expression_loader<E, true, false>::prefetch(
                    static_cast<E const&>(expression), i);
                const auto& temp = vectorization::expression_loader<E, true, false>::evaluate(
                    static_cast<E const&>(expression), i);
                max_packet[0] = simd<value_t>::max(max_packet[0], temp);
            }

            simd_t combined = max_packet[0];
            for (std::size_t k = 1; k < packet; ++k)
                combined = simd<value_t>::max(combined, max_packet[k]);

            ret = std::fmax(ret, simd<value_t>::hmax(combined));
        }
    }
    else
    {
        const size_t loop_peel = length * (n / length);
        if (loop_peel > 0)
        {
            simd_t max_packet[packet];
            detail::unroll<packet>(
                [&](auto k)
                {
                    max_packet[decltype(k)::value] =
                        simd<value_t>::set(-std::numeric_limits<value_t>::max());
                });

            if constexpr (packet > 1)
            {
                const size_t block_end = i + ((loop_peel - i) / block) * block;
                for (; i < block_end; i += block)
                {
                    detail::unroll<packet>(
                        [&](auto k)
                        {
                            constexpr size_t kk = decltype(k)::value;
                            vectorization::expression_loader<E, true, false>::prefetch(
                                static_cast<E const&>(expression), i + kk * length);
                        });
                    detail::unroll<packet>(
                        [&](auto k)
                        {
                            constexpr size_t kk = decltype(k)::value;
                            const auto&      temp =
                                vectorization::expression_loader<E, true, false>::evaluate(
                                    static_cast<E const&>(expression), i + kk * length);
                            max_packet[kk] = simd<value_t>::max(max_packet[kk], temp);
                        });
                }
            }
            for (; i < loop_peel; i += length)
            {
                vectorization::expression_loader<E, true, false>::prefetch(
                    static_cast<E const&>(expression), i);
                const auto& temp = vectorization::expression_loader<E, true, false>::evaluate(
                    static_cast<E const&>(expression), i);
                max_packet[0] = simd<value_t>::max(max_packet[0], temp);
            }

            simd_t combined = max_packet[0];
            for (std::size_t k = 1; k < packet; ++k)
                combined = simd<value_t>::max(combined, max_packet[k]);

            ret = std::fmax(ret, simd<value_t>::hmax(combined));
        }
    }
#endif
    for (; i < n; ++i)
        ret = std::fmax(
            ret,
            vectorization::expression_loader<E, false, false>::evaluate(
                static_cast<E const&>(expression), i));

    return ret;
}
}  // namespace vectorization
