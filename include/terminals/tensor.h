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

#if defined(_MSC_VER) && !defined(VECTORIZATION_DISPLAY_WIN32_WARNINGS)
#pragma warning(push)
#pragma warning(disable : 4267)
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

#if VECTORIZATION_HAS_PROFILER
#include <profiler.h>
#endif

#include "backend/simd.h"
#include "expressions/expressions.h"
#include "sizes_and_strides.h"
#include "stream_guard.h"

namespace vectorization
{

template <typename T>
inline constexpr bool is_almost_zero(T x, T epsilon = std::numeric_limits<T>::epsilon()) noexcept
{
    return (std::fabs(x) < epsilon);
}

template <typename E>
VECTORIZATION_HOST_FUNCTION_ATTRIBUTE void record_expression_streams(
    E const& expr, gpu_stream_t stream)
{
    using expr_t = vectorization::remove_cvref_t<E>;
    if constexpr (is_pure_expression<expr_t>::value)
    {
        if constexpr (VECTORIZATION_EXPR_HAS_MHS(expr))
        {
            record_expression_streams(expr.lhs(), stream);
            record_expression_streams(expr.mhs(), stream);
            record_expression_streams(expr.rhs(), stream);
        }
        else if constexpr (VECTORIZATION_EXPR_HAS_LHS(expr))
        {
            record_expression_streams(expr.lhs(), stream);
            record_expression_streams(expr.rhs(), stream);
        }
        else
        {
            record_expression_streams(expr.rhs(), stream);
        }
    }
    else if constexpr (VECTORIZATION_EXPR_HAS_RECORD_STREAM(expr, stream))
    {
        expr.record_stream(stream);
    }
}

// Device (not stream) placement inferred from an expression's tensor operands. The
// execution/allocation stream is deliberately not part of this: like PyTorch, it always
// comes from the ambient stream_guard for `index` (see init_from_expression), never from
// an operand tensor's own carried stream.
struct expression_placement
{
    device_enum kind  = device_enum::CPU;
    int         index = 0;
};

template <typename E>
VECTORIZATION_HOST_FUNCTION_ATTRIBUTE void accumulate_expression_placement(
    E const& expr, expression_placement& out, bool& seen)
{
    using expr_t = vectorization::remove_cvref_t<E>;
    if constexpr (is_pure_expression<expr_t>::value)
    {
        if constexpr (VECTORIZATION_EXPR_HAS_MHS(expr))
        {
            accumulate_expression_placement(expr.lhs(), out, seen);
            accumulate_expression_placement(expr.mhs(), out, seen);
            accumulate_expression_placement(expr.rhs(), out, seen);
        }
        else if constexpr (VECTORIZATION_EXPR_HAS_LHS(expr))
        {
            accumulate_expression_placement(expr.lhs(), out, seen);
            accumulate_expression_placement(expr.rhs(), out, seen);
        }
        else if constexpr (VECTORIZATION_EXPR_HAS_RHS(expr))
        {
            accumulate_expression_placement(expr.rhs(), out, seen);
        }
    }
    else if constexpr (is_base_expression<expr_t>::value)
    {
        if (!seen)
        {
            out.kind  = expr.device();
            out.index = expr.device_index();
            seen      = true;
        }
        else
        {
            VECTORIZATION_CHECK(
                expr.device() == out.kind && expr.device_index() == out.index,
                "expression mixes devices or device indices");
        }
    }
}

template <typename E>
VECTORIZATION_HOST_FUNCTION_ATTRIBUTE expression_placement infer_expression_placement(E const& expr)
{
    expression_placement out;
    bool                 seen = false;
    accumulate_expression_placement(expr, out, seen);
    return out;
}

// ---------------------------------------------------------------------------
// tensor<T> — unified N-dimensional container
//
// Rank-1 behaves like the former vector<T>.
// Rank-2 behaves like the former matrix<T>.
// Higher ranks generalise to arbitrary N-D storage.
//
// tensor * tensor always produces matrix_multiplication_expression so that
// matrix algebra is preserved for rank-2 operands.  Element-wise multiply
// is available via mul(a, b) or fma().
// ---------------------------------------------------------------------------
template <typename value_t>
class tensor
{
public:
    using value_type                          = value_t;
    using size_type                           = std::size_t;
    static constexpr size_type alignment      = VECTORIZATION_ALIGNMENT;
    static constexpr size_type scalar_size    = sizeof(value_type);
    static constexpr size_type alignment_size = alignment / scalar_size;
    static constexpr size_type alignment_mask = alignment_size - 1;
    using dimensions_type                     = std::vector<size_type>;
    using evaluator                           = expressions_evaluator;
    // simd<T>::simd_t does not exist when SIMD is disabled (simd<T> is an
    // empty struct). Alias the scalar type so tensor<T>::simd_t stays valid.
#if VECTORIZATION_VECTORIZED
    using simd_t = typename simd<value_t>::simd_t;
#else
    using simd_t = value_t;
#endif
    // SIMD-type alignment constants — precomputed at class scope so first_aligned()
    // references simple members rather than local constexpr variables. This avoids a
    // Clang 20 CFG-optimizer crash that fires when the constexpr locals are inlined
    // three levels deep (first_aligned → recompute_cpu_simd_alignment_state → caller).
    // Guarded by VECTORIZATION_VECTORIZED because alignof(simd_t) would otherwise
    // be evaluated at class-instantiation time even in a non-taken if constexpr branch.
#if VECTORIZATION_VECTORIZED
    static constexpr size_type simd_align      = alignof(simd_t);
    static constexpr size_type simd_align_size = simd_align / scalar_size;
    static constexpr size_type simd_align_mask = simd_align_size - 1;
#else
    static constexpr size_type simd_align      = scalar_size;
    static constexpr size_type simd_align_size = 1;
    static constexpr size_type simd_align_mask = 0;
#endif
    using owner_t                = data_ptr<value_t>;
    using view_t                 = data_view<value_t>;
    using allocator_t            = allocator<value_t>;
    using iterator               = value_t*;
    using const_iterator         = const value_t*;
    using reverse_iterator       = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    // NOINLINE: called only at construction time (not on the hot SIMD compute path).
    // Force-inlining + reinterpret_cast in the body triggers a Clang 20 optimizer/code-gen crash
    // when inlined two levels deep (here → recompute_cpu_simd_alignment_state → ctor): the
    // accumulation of ptrtoint casts across many constructor call-sites causes ComputeValueVTs
    // infinite recursion in the instruction selector. noinline breaks the inlining chain.
    VECTORIZATION_NOINLINE static size_type first_aligned(const value_t* array, size_type size)
    {
        if constexpr (alignment_size <= 1)
        {
            return 0;
        }

        // Use the SIMD type's alignment requirement, not the tensor allocator's alignment.
        // VECTORIZATION_ALIGNMENT (e.g. 64) can exceed alignof(simd_t) (e.g. 32 for AVX2), so
        // owned allocations are always SIMD-aligned, but view tensors wrapping external buffers
        // (e.g. std::vector data, which is only 16-byte aligned) may not be. Computing the
        // prologue length against alignof(simd_t) keeps vmovaps stores correctly aligned for both.
        // simd_align / simd_align_size / simd_align_mask are class-level static constexpr to avoid
        // a Clang 20 CFG-optimizer crash triggered by local constexpr vars in always_inline code.
        static_assert(
            simd_align_size >= 1 && (simd_align % scalar_size) == 0,
            "SIMD alignment must be a positive multiple of scalar_size");

        if (reinterpret_cast<std::uintptr_t>(array) & (scalar_size - 1))
        {
            return size;
        }

        size_type first =
            (simd_align_size -
             (reinterpret_cast<std::uintptr_t>(array) / scalar_size & simd_align_mask)) &
            simd_align_mask;
        return (first < size) ? first : size;
    }

    VECTORIZATION_FORCE_INLINE static size_type last_aligned(
        size_type aligned_start, size_type size, size_type packet_size)
    {
        return aligned_start + ((size - aligned_start) / packet_size) * packet_size;
    }

    // SIMD stride — identical for every rank; used by expression_loader.
    // One simd<value_t> register per step (no manual unroll).
    VECTORIZATION_FUNCTION_ATTRIBUTE static constexpr size_t length() noexcept
    {
#if VECTORIZATION_VECTORIZED
        return simd<value_t>::size;
#else
        return 1;
#endif
    }

    /// First element index where CPU SIMD lanes are memory-aligned (scalar prologue is \c [0, align_start) ).
    VECTORIZATION_FUNCTION_ATTRIBUTE std::size_t align_start() const noexcept
    {
        return align_start_;
    }

    /// Exclusive end index: for \c i in <tt>[align_start, align_end)</tt> at stride \ref length(), use \c load / \c store.
    VECTORIZATION_FUNCTION_ATTRIBUTE std::size_t align_end() const noexcept { return align_end_; }

    // -----------------------------------------------------------------------
    // Constructors
    // -----------------------------------------------------------------------

    VECTORIZATION_FUNCTION_ATTRIBUTE tensor() noexcept { recompute_cpu_simd_alignment_state(); }

    // --- 1-D constructors (vector-like) ------------------------------------

    // Not noexcept: owner_(n, type) allocates via memory::allocator<T>, which can throw
    // (e.g. std::bad_alloc, or std::invalid_argument for an unsupported type/device
    // combination such as tensor<double> on device_enum::METAL — MSL has no double type).
    // A noexcept constructor that throws internally calls std::terminate() instead of
    // letting the exception propagate, which would make that rejection uncatchable.
    VECTORIZATION_CUDA_FUNCTION_TYPE explicit tensor(
        size_type    n,
        device_enum  type         = device_enum::CPU,
        int          device_index = 0,
        gpu_stream_t stream       = nullptr)
        : owner_(n, type, device_index, static_cast<typename owner_t::stream_t>(stream)),
          view_(owner_.view())
    {
        sizes_and_strides_.size_at_unchecked(0)   = static_cast<int64_t>(n);
        sizes_and_strides_.stride_at_unchecked(0) = 1;
        recompute_cpu_simd_alignment_state();
    }

    // 1D view constructor — wraps an existing contiguous buffer without owning it.
    // Copy-construct the result to take an owned clone.
    VECTORIZATION_FUNCTION_ATTRIBUTE tensor(
        value_t*     ptr,
        size_type    n,
        device_enum  type         = device_enum::CPU,
        int          device_index = 0,
        gpu_stream_t stream       = nullptr) noexcept
        : view_(view_t::borrow(
              ptr, n, type, device_index, static_cast<typename view_t::stream_t>(stream)))
    {
        sizes_and_strides_.size_at_unchecked(0)   = static_cast<int64_t>(n);
        sizes_and_strides_.stride_at_unchecked(0) = 1;
        recompute_cpu_simd_alignment_state();
    }

    // 2D view constructor — wraps an existing contiguous buffer as a rows×cols
    // matrix without owning it. Copy-construct to take an owned clone.
    VECTORIZATION_FUNCTION_ATTRIBUTE tensor(
        value_t*     ptr,
        size_type    rows,
        size_type    cols,
        device_enum  type         = device_enum::CPU,
        int          device_index = 0,
        gpu_stream_t stream       = nullptr) noexcept
        : view_(view_t::borrow(
              ptr, rows * cols, type, device_index, static_cast<typename view_t::stream_t>(stream)))
    {
        sizes_and_strides_.resize(2);
        sizes_and_strides_.size_at_unchecked(0)   = static_cast<int64_t>(rows);
        sizes_and_strides_.size_at_unchecked(1)   = static_cast<int64_t>(cols);
        sizes_and_strides_.stride_at_unchecked(0) = static_cast<int64_t>(cols);
        sizes_and_strides_.stride_at_unchecked(1) = 1;
        recompute_cpu_simd_alignment_state();
    }

    VECTORIZATION_CUDA_FUNCTION_TYPE tensor(
        value_t start, value_t end, size_type n, device_enum type = device_enum::CPU)
        : tensor(n, type)
    {
        if (n == 0)
        {
            return;
        }
        auto const fill = [start, end, n](value_t* dst)
        {
            if (n == 1)
            {
                dst[0] = start;
                return;
            }
            const auto dx = (end - start) / static_cast<value_t>(n - 1);
            for (size_t i = 0; i < n; ++i)
            {
                dst[i] = static_cast<value_t>(i) * dx + start;
            }
        };
        if (device() == device_enum::CPU)
        {
            fill(data());
            return;
        }
        owner_t staging(n, device_enum::CPU);
        fill(staging.data());
        copy_from_host(staging.data(), n);
    }

    VECTORIZATION_CUDA_FUNCTION_TYPE tensor(
        std::initializer_list<value_t> list, device_enum type = device_enum::CPU)
        : tensor(list.size(), type)
    {
        if (list.size() != 0)
        {
            copy_from_host(list.begin(), list.size());
        }
    }

    // --- 2-D constructors (matrix-like) ------------------------------------

    // Not noexcept — see the 1-D sized constructor above for why.
    VECTORIZATION_CUDA_FUNCTION_TYPE tensor(
        size_type    rows,
        size_type    cols,
        device_enum  type         = device_enum::CPU,
        int          device_index = 0,
        gpu_stream_t stream       = nullptr)
        : owner_(rows * cols, type, device_index, static_cast<typename owner_t::stream_t>(stream)),
          view_(owner_.view())
    {
        sizes_and_strides_.resize(2);
        sizes_and_strides_.size_at_unchecked(0)   = static_cast<int64_t>(rows);
        sizes_and_strides_.size_at_unchecked(1)   = static_cast<int64_t>(cols);
        sizes_and_strides_.stride_at_unchecked(0) = static_cast<int64_t>(cols);
        sizes_and_strides_.stride_at_unchecked(1) = 1;
        recompute_cpu_simd_alignment_state();
    }

    VECTORIZATION_CUDA_FUNCTION_TYPE tensor(
        std::initializer_list<std::initializer_list<value_t>> list,
        device_enum                                           type = device_enum::CPU)
        : tensor(
              static_cast<size_type>(list.size()),
              list.size() == 0 ? size_type(0) : static_cast<size_type>(list.begin() -> size()),
              type)
    {
        if (empty())
        {
            return;
        }
        auto const cols = static_cast<size_t>(dimension(1));
        size_t     i    = 0;
        if (device() == device_enum::CPU)
        {
            value_t* dst = data();
            for (auto const& row : list)
            {
                // Must be VECTORIZATION_CHECK_IF_NOT_ON_CUDA, not a plain
                // VECTORIZATION_CHECK: this constructor is tagged __host__ __device__,
                // and nvcc compiles a device pass for every such function that is
                // instantiated -- whether or not device code ever calls it. A plain
                // VECTORIZATION_CHECK throws, and "device code does not support
                // exception handling" is a hard error in that pass, so the whole TU
                // fails to compile. IF_NOT_ON_CUDA expands to nothing only in that
                // device pass; it is *not* VECTORIZATION_CHECK_DEBUG, which would also
                // drop the check from release host builds, where a jagged row would
                // then run std::copy past the end of the allocation.
                VECTORIZATION_CHECK_IF_NOT_ON_CUDA(
                    row.size() == cols, "2-D initializer_list has jagged rows");
                std::copy(row.begin(), row.end(), dst + i * cols);
                ++i;
            }
            return;
        }
        owner_t  packed(size(), device_enum::CPU);
        value_t* dst = packed.data();
        for (auto const& row : list)
        {
            VECTORIZATION_CHECK_IF_NOT_ON_CUDA(
                row.size() == cols, "2-D initializer_list has jagged rows");
            std::copy(row.begin(), row.end(), dst + i * cols);
            ++i;
        }
        copy_from_host(packed.data(), size());
    }

    // --- N-D constructors (general tensor) ---------------------------------

    VECTORIZATION_FUNCTION_ATTRIBUTE tensor(
        const dimensions_type& dims,
        device_enum            type         = device_enum::CPU,
        int                    device_index = 0,
        gpu_stream_t           stream       = nullptr)
        : owner_(
              compute_total(dims),
              type,
              device_index,
              static_cast<typename owner_t::stream_t>(stream)),
          view_(owner_.view())
    {
        set_shape(dims);
        recompute_cpu_simd_alignment_state();
    }

    VECTORIZATION_FUNCTION_ATTRIBUTE tensor(
        dimensions_type&& dims,
        device_enum       type         = device_enum::CPU,
        int               device_index = 0,
        gpu_stream_t      stream       = nullptr)
        : owner_(
              compute_total(dims),
              type,
              device_index,
              static_cast<typename owner_t::stream_t>(stream)),
          view_(owner_.view())
    {
        set_shape(dims);
        recompute_cpu_simd_alignment_state();
    }

    // Wrap external memory as a non-owning view. Copy-construct to take an
    // owned clone. The wrapped buffer must outlive this tensor (and any
    // views derived from it) unless it is copied first.
    VECTORIZATION_FUNCTION_ATTRIBUTE tensor(
        value_t*               data,
        const dimensions_type& dims,
        device_enum            type         = device_enum::CPU,
        int                    device_index = 0,
        gpu_stream_t           stream       = nullptr)
        : view_(view_t::borrow(
              data,
              compute_total(dims),
              type,
              device_index,
              static_cast<typename view_t::stream_t>(stream)))
    {
        set_shape(dims);
        recompute_cpu_simd_alignment_state();
    }

    // -----------------------------------------------------------------------
    // Copy / move — copy always deep-clones into a new data_ptr (same contract
    // as memory::data_ptr). Wrap / t() / slice results borrow; copying them
    // materializes an owned buffer. Move transfers owner_ when present.
    // -----------------------------------------------------------------------

    VECTORIZATION_FUNCTION_ATTRIBUTE tensor(tensor const& rhs)
        : sizes_and_strides_(rhs.sizes_and_strides_), owner_(rhs.view_), view_(owner_.view())
    {
        recompute_cpu_simd_alignment_state();
    }

    VECTORIZATION_FUNCTION_ATTRIBUTE tensor(tensor&& rhs) noexcept
        : sizes_and_strides_(std::move(rhs.sizes_and_strides_)),
          owner_(std::move(rhs.owner_)),
          view_(rhs.view_),
          align_start_(rhs.align_start_),
          align_end_(rhs.align_end_),
          numel_(rhs.numel_),
          contiguous_(rhs.contiguous_)
    {
        rhs.view_        = view_t{};
        rhs.align_start_ = rhs.align_end_ = 0;
        rhs.numel_                        = 0;
        rhs.contiguous_                   = true;
    }

    VECTORIZATION_FUNCTION_ATTRIBUTE tensor& operator=(tensor const& rhs)
    {
        if (this != &rhs)
        {
            sizes_and_strides_ = rhs.sizes_and_strides_;
            owner_             = owner_t(rhs.view_);
            view_              = owner_.view();
            recompute_cpu_simd_alignment_state();
        }
        return *this;
    }

    VECTORIZATION_FUNCTION_ATTRIBUTE tensor& operator=(tensor&& rhs)
    {
        if (this != &rhs)
        {
            sizes_and_strides_ = std::move(rhs.sizes_and_strides_);
            owner_             = std::move(rhs.owner_);
            view_              = rhs.view_;
            align_start_       = rhs.align_start_;
            align_end_         = rhs.align_end_;
            numel_             = rhs.numel_;
            contiguous_        = rhs.contiguous_;
            rhs.view_          = view_t{};
            rhs.align_start_ = rhs.align_end_ = 0;
            rhs.numel_                        = 0;
            rhs.contiguous_                   = true;
        }
        return *this;
    }

    // Returns a new tensor that is a deep, contiguous copy of *this on the
    // same device/index/stream. Non-contiguous sources are gathered in C-order.
    VECTORIZATION_HOST_FUNCTION_ATTRIBUTE tensor clone() const
    {
        const size_t total = size();
        tensor       result(total, device(), device_index(), stream());
        if (total != 0)
        {
            if (is_contiguous())
            {
                allocator_t::copy(
                    data(),
                    total,
                    result.data(),
                    device(),
                    device(),
                    device_index(),
                    device_index(),
                    static_cast<typename allocator_t::stream_t>(stream()));
            }
            else if (device() == device_enum::CPU)
            {
                gather_logical(data(), result.data());
            }
            else
            {
                owner_t packed(total, device_enum::CPU);
                copy_logical_to_host(packed.data());
                result.copy_from_host(packed.data(), total);
            }
        }
        result.stamp_contiguous_shape(*this);
        return result;
    }

    // Packed C-order tensor with the same logical values. Already-contiguous
    // sources return a borrow of this buffer (same as view()); otherwise clone().
    VECTORIZATION_HOST_FUNCTION_ATTRIBUTE tensor contiguous() const
    {
        if (is_contiguous())
        {
            return tensor(
                data(),
                view_.size(),
                sizes_and_strides_,
                device(),
                view_.device_index(),
                view_.stream());
        }
        return clone();
    }

    VECTORIZATION_FUNCTION_ATTRIBUTE ~tensor() = default;

    // -----------------------------------------------------------------------
    // Expression constructors / assignment
    // -----------------------------------------------------------------------

    template <
        typename E,
        std::enable_if_t<vectorization::is_pure_expression<E>::value, bool> = true>
    VECTORIZATION_HOST_FUNCTION_ATTRIBUTE tensor(E const& expr)
    {
#if VECTORIZATION_HAS_PROFILER
        PROFILER_RECORD_USER_SCOPE("vectorization::tensor::construct");
#endif
        init_from_expression(expr);
    }

    template <
        typename E,
        std::enable_if_t<vectorization::is_pure_expression<E>::value, bool> = true>
    VECTORIZATION_HOST_FUNCTION_ATTRIBUTE tensor(E&& expr)  // NOLINT
    {
#if VECTORIZATION_HAS_PROFILER
        PROFILER_RECORD_USER_SCOPE("vectorization::tensor::construct");
#endif
        init_from_expression(static_cast<E const&>(expr));
    }

    template <
        typename E,
        std::enable_if_t<vectorization::is_pure_expression<E>::value, bool> = true>
    VECTORIZATION_HOST_FUNCTION_ATTRIBUTE tensor& operator=(E const& expr)
    {
#if VECTORIZATION_HAS_PROFILER
        PROFILER_RECORD_USER_SCOPE("vectorization::tensor::assign");
#endif
        // Ambient stream set by a stream_guard on the calling thread, if any -- nullptr
        // (default stream) otherwise, matching the previous always-default-stream behavior.
        auto const stream = resolve_ambient_stream(expr, device_index());
        evaluator::template run<E, tensor>(expr, *this, stream);
        record_stream_if_redirected(stream);
        return *this;
    }

    template <
        typename E,
        std::enable_if_t<vectorization::is_pure_expression<E>::value, bool> = true>
    VECTORIZATION_HOST_FUNCTION_ATTRIBUTE tensor& operator=(E&& expr)
    {
#if VECTORIZATION_HAS_PROFILER
        PROFILER_RECORD_USER_SCOPE("vectorization::tensor::assign");
#endif
        auto const& cexpr  = static_cast<E const&>(expr);
        auto const  stream = resolve_ambient_stream(cexpr, device_index());
        evaluator::template run<E, tensor>(cexpr, *this, stream);
        record_stream_if_redirected(stream);
        return *this;
    }

    // Not noexcept: fill() can throw (e.g. destination device()/contiguity check
    // failure) -- same rationale as the constructor comment above.
    template <typename T2, std::enable_if_t<std::is_fundamental<T2>::value, bool> = true>
    VECTORIZATION_HOST_FUNCTION_ATTRIBUTE tensor& operator=(T2 value)
    {
#if VECTORIZATION_HAS_PROFILER
        PROFILER_RECORD_USER_SCOPE("vectorization::tensor::assign_scalar");
#endif
        auto const stream = current_stream(device_index());
        evaluator::template fill<value_t, tensor>(static_cast<value_t>(value), *this, stream);
        record_stream_if_redirected(stream);
        return *this;
    }

    // Evaluate `expr` into *this on the given CUDA/HIP stream instead of the default
    // stream, so independent GPU tensor ops (built on different streams) can overlap.
    // `stream` is ignored for CPU tensors. The caller owns the stream and must
    // synchronize it (or rely on a subsequent synchronous call such as
    // to_host_vector(), which implicitly waits for all outstanding device work) before
    // reading the result.
    template <
        typename E,
        std::enable_if_t<vectorization::is_pure_expression<E>::value, bool> = true>
    VECTORIZATION_HOST_FUNCTION_ATTRIBUTE tensor& assign_async(E const& expr, gpu_stream_t stream)
    {
        record_expression_streams(expr, stream);
        evaluator::template run<E, tensor>(expr, *this, stream);
        record_stream(stream);
        return *this;
    }

    // Same as operator=(scalar), but on the given CUDA/HIP stream. See assign_async().
    template <typename T2, std::enable_if_t<std::is_fundamental<T2>::value, bool> = true>
    VECTORIZATION_HOST_FUNCTION_ATTRIBUTE tensor& fill_async(T2 value, gpu_stream_t stream)
    {
        evaluator::template fill<value_t, tensor>(static_cast<value_t>(value), *this, stream);
        record_stream(stream);
        return *this;
    }

    // -----------------------------------------------------------------------
    // Raw data / size
    // -----------------------------------------------------------------------

    VECTORIZATION_FUNCTION_ATTRIBUTE value_t* data() const noexcept { return view_.data(); }
    VECTORIZATION_FUNCTION_ATTRIBUTE size_t   size() const noexcept { return numel_; }
    VECTORIZATION_FUNCTION_ATTRIBUTE bool     empty() const noexcept { return numel_ == 0; }

    VECTORIZATION_FUNCTION_ATTRIBUTE device_enum device() const noexcept { return view_.device(); }

    VECTORIZATION_FUNCTION_ATTRIBUTE int device_index() const noexcept
    {
        return view_.device_index();
    }

    VECTORIZATION_FUNCTION_ATTRIBUTE gpu_stream_t stream() const noexcept
    {
        return static_cast<gpu_stream_t>(view_.stream());
    }

    // Mark this tensor's storage as in-use on @p stream so the caching allocator
    // will not recycle the block until that stream completes (PyTorch recordStream).
    VECTORIZATION_HOST_FUNCTION_ATTRIBUTE void record_stream(gpu_stream_t stream) const
    {
        view_.record_stream(static_cast<typename view_t::stream_t>(stream));
    }

    // -----------------------------------------------------------------------
    // Host ↔ device transfer helpers
    // -----------------------------------------------------------------------

    // Upload count elements from a host pointer into this tensor.
    // CPU destinations write in place; GPU destinations copy on this tensor's stream.
    void copy_from_host(const value_t* ptr, size_type count)
    {
        VECTORIZATION_CHECK_DEBUG(is_contiguous(), "copy_from_host requires a contiguous tensor");
        VECTORIZATION_CHECK_DEBUG(count == size(), "copy_from_host: element count mismatch");
        if (ptr == nullptr || count == 0)
        {
            return;
        }
        if (device() == device_enum::CPU)
        {
            std::copy_n(ptr, static_cast<size_t>(count), data());
            return;
        }
        allocator_t::copy(
            ptr,
            count,
            data(),
            device_enum::CPU,
            device(),
            0,
            view_.device_index(),
            static_cast<typename allocator_t::stream_t>(stream()));
    }

    // Convenience overload — uploads from a std::vector.
    void copy_from_host(const std::vector<value_t>& src) { copy_from_host(src.data(), src.size()); }

    // Same as copy_from_host(ptr, count), but issued on the given CUDA/HIP stream
    // (cudaMemcpyAsync / hipMemcpyAsync) instead of blocking the calling thread. `ptr` must
    // stay valid and unmodified until the stream completes; a subsequent synchronous call
    // such as to_host_vector() (which implicitly waits for all outstanding device work) is
    // a safe join point. CPU destinations ignore `stream` and write in place.
    void copy_from_host(const value_t* ptr, size_type count, gpu_stream_t stream)
    {
        VECTORIZATION_CHECK_DEBUG(is_contiguous(), "copy_from_host requires a contiguous tensor");
        VECTORIZATION_CHECK_DEBUG(count == size(), "copy_from_host: element count mismatch");
        if (ptr == nullptr || count == 0)
        {
            return;
        }
        if (device() == device_enum::CPU)
        {
            std::copy_n(ptr, static_cast<size_t>(count), data());
            return;
        }
        allocator_t::copy(
            ptr,
            count,
            data(),
            device_enum::CPU,
            device(),
            0,
            view_.device_index(),
            static_cast<typename allocator_t::stream_t>(stream));
        record_stream(stream);
    }

    // Convenience overload — uploads from a std::vector on the given stream.
    void copy_from_host(const std::vector<value_t>& src, gpu_stream_t stream)
    {
        copy_from_host(src.data(), src.size(), stream);
    }

    // Owned contiguous CPU tensor with the same logical values (C-order).
    // Already on CPU and packed: borrow this buffer (no copy).
    VECTORIZATION_HOST_FUNCTION_ATTRIBUTE tensor to_cpu() const
    {
        if (device() == device_enum::CPU && is_contiguous())
        {
            return tensor(
                data(),
                view_.size(),
                sizes_and_strides_,
                device(),
                view_.device_index(),
                view_.stream());
        }
        tensor result(size(), device_enum::CPU);
        copy_logical_to_host(result.data());
        result.stamp_contiguous_shape(*this);
        return result;
    }

    // Download all elements to a host std::vector.  Blocks until complete.
    // Allocates the vector; prefer copy_to_host(ptr) when the buffer exists.
    std::vector<value_t> to_host_vector() const
    {
        const size_t         total = size();
        std::vector<value_t> h(total);
        copy_logical_to_host(h.data());
        return h;
    }

    // Download logical elements into a caller-owned host buffer. Blocks until
    // complete. Does not allocate. dst must have room for size() elements.
    // Non-contiguous sources are gathered in C-order (same as to_host_vector).
    void copy_to_host(value_t* dst) const
    {
        copy_logical_to_host(dst);
    }

    void copy_to_host(value_t* dst, size_type count) const
    {
        VECTORIZATION_CHECK_DEBUG(count == size(), "copy_to_host: element count mismatch");
        copy_logical_to_host(dst);
    }

    // Not noexcept: VECTORIZATION_CHECK_DEBUG throws in debug (MSVC C4297 /WX).
    VECTORIZATION_FUNCTION_ATTRIBUTE iterator begin()
    {
        VECTORIZATION_CHECK_DEBUG_IF_NOT_ON_CUDA(
            is_contiguous(), "begin/end require a contiguous tensor");
        return data();
    }
    VECTORIZATION_FUNCTION_ATTRIBUTE iterator end()
    {
        VECTORIZATION_CHECK_DEBUG_IF_NOT_ON_CUDA(
            is_contiguous(), "begin/end require a contiguous tensor");
        return data() + size();
    }
    VECTORIZATION_FUNCTION_ATTRIBUTE const_iterator begin() const
    {
        VECTORIZATION_CHECK_DEBUG_IF_NOT_ON_CUDA(
            is_contiguous(), "begin/end require a contiguous tensor");
        return data();
    }
    VECTORIZATION_FUNCTION_ATTRIBUTE const_iterator end() const
    {
        VECTORIZATION_CHECK_DEBUG_IF_NOT_ON_CUDA(
            is_contiguous(), "begin/end require a contiguous tensor");
        return data() + size();
    }

    VECTORIZATION_FUNCTION_ATTRIBUTE reverse_iterator rbegin() { return reverse_iterator(end()); }
    VECTORIZATION_FUNCTION_ATTRIBUTE reverse_iterator rend() { return reverse_iterator(begin()); }
    VECTORIZATION_FUNCTION_ATTRIBUTE const_reverse_iterator crbegin() const
    {
        return const_reverse_iterator(end());
    }
    VECTORIZATION_FUNCTION_ATTRIBUTE const_reverse_iterator crend() const
    {
        return const_reverse_iterator(begin());
    }

    // -----------------------------------------------------------------------
    // Shape / rank
    // -----------------------------------------------------------------------

    VECTORIZATION_FUNCTION_ATTRIBUTE size_t rank() const noexcept
    {
        return sizes_and_strides_.size();
    }

    VECTORIZATION_FUNCTION_ATTRIBUTE span<const int64_t> dimensions() const noexcept
    {
        return sizes_and_strides_.sizes_arrayref();
    }

    VECTORIZATION_FUNCTION_ATTRIBUTE size_t dimension(size_t n) const noexcept
    {
        return static_cast<size_t>(sizes_and_strides_.size_at_unchecked(n));
    }

    // Per-dimension size and stride
    VECTORIZATION_FUNCTION_ATTRIBUTE int64_t size(size_t dim) const noexcept
    {
        return sizes_and_strides_.size_at_unchecked(dim);
    }
    VECTORIZATION_FUNCTION_ATTRIBUTE int64_t stride(size_t dim) const noexcept
    {
        return sizes_and_strides_.stride_at_unchecked(dim);
    }
    VECTORIZATION_FUNCTION_ATTRIBUTE span<const int64_t> strides() const noexcept
    {
        return sizes_and_strides_.strides_arrayref();
    }

    VECTORIZATION_FUNCTION_ATTRIBUTE bool is_aligned() const noexcept { return view_.is_aligned(); }

    // C-order contiguity: cached at construction / shape change.
    VECTORIZATION_FUNCTION_ATTRIBUTE bool is_contiguous() const noexcept { return contiguous_; }

    // -----------------------------------------------------------------------
    // Element access
    // -----------------------------------------------------------------------

    // Flat indexed access (1-D semantic, element by element)
    VECTORIZATION_FUNCTION_ATTRIBUTE const value_t& operator[](size_type i) const noexcept
    {
        return data()[logical_offset(i)];
    }
    VECTORIZATION_FUNCTION_ATTRIBUTE value_t& operator[](size_type i) noexcept
    {
        return data()[logical_offset(i)];
    }

    VECTORIZATION_FUNCTION_ATTRIBUTE value_t at(size_type i) const noexcept
    {
        return data()[logical_offset(i)];
    }
    VECTORIZATION_FUNCTION_ATTRIBUTE value_t& at(size_type i) noexcept
    {
        return data()[logical_offset(i)];
    }

    VECTORIZATION_FUNCTION_ATTRIBUTE value_t at(size_type i, size_type j) const
    {
        VECTORIZATION_CHECK_DEBUG_IF_NOT_ON_CUDA(rank() >= 2, "at(i,j) requires rank >= 2");
        VECTORIZATION_CHECK_DEBUG_IF_NOT_ON_CUDA(i < dimension(0), "row index out of range");
        VECTORIZATION_CHECK_DEBUG_IF_NOT_ON_CUDA(j < dimension(1), "column index out of range");
        return data()[i * static_cast<size_t>(stride(0)) + j * static_cast<size_t>(stride(1))];
    }
    VECTORIZATION_FUNCTION_ATTRIBUTE value_t& at(size_type i, size_type j)
    {
        VECTORIZATION_CHECK_DEBUG_IF_NOT_ON_CUDA(rank() >= 2, "at(i,j) requires rank >= 2");
        VECTORIZATION_CHECK_DEBUG_IF_NOT_ON_CUDA(i < dimension(0), "row index out of range");
        VECTORIZATION_CHECK_DEBUG_IF_NOT_ON_CUDA(j < dimension(1), "column index out of range");
        return data()[i * static_cast<size_t>(stride(0)) + j * static_cast<size_t>(stride(1))];
    }

    // N-D element by multi-index
    VECTORIZATION_FUNCTION_ATTRIBUTE value_t at(const dimensions_type& indices) const
    {
        return data()[linearized_index(indices)];
    }
    VECTORIZATION_FUNCTION_ATTRIBUTE value_t& at(const dimensions_type& indices)
    {
        return data()[linearized_index(indices)];
    }

    // -----------------------------------------------------------------------
    // Views — metadata-only transforms (no data copy)
    //
    // Returned tensors borrow this buffer (empty owner_). The source must
    // outlive the view. Copy-construct the result to take an owned clone.
    // -----------------------------------------------------------------------

    VECTORIZATION_FUNCTION_ATTRIBUTE tensor t() const
    {
        VECTORIZATION_CHECK_DEBUG_IF_NOT_ON_CUDA(rank() == 2, "t() requires a rank-2 tensor");
        sizes_and_strides sas = sizes_and_strides_;
        std::swap(sas.size_at_unchecked(0), sas.size_at_unchecked(1));
        std::swap(sas.stride_at_unchecked(0), sas.stride_at_unchecked(1));
        return tensor(data(), view_.size(), sas, device(), view_.device_index(), view_.stream());
    }

    VECTORIZATION_FUNCTION_ATTRIBUTE tensor permute(const dimensions_type& order) const
    {
        const size_t n = rank();
        VECTORIZATION_CHECK_DEBUG_IF_NOT_ON_CUDA(
            order.size() == n, "permute: order length must equal rank");
        sizes_and_strides sas;
        sas.resize(n);
        for (size_t i = 0; i < n; ++i)
        {
            auto const src = order[i];
            VECTORIZATION_CHECK_DEBUG_IF_NOT_ON_CUDA(src < n, "permute: axis out of range");
            for (size_t j = 0; j < i; ++j)
            {
                VECTORIZATION_CHECK_DEBUG_IF_NOT_ON_CUDA(order[j] != src, "permute: repeated axis");
            }
            sas.size_at_unchecked(i)   = sizes_and_strides_.size_at_unchecked(src);
            sas.stride_at_unchecked(i) = sizes_and_strides_.stride_at_unchecked(src);
        }
        return tensor(data(), view_.size(), sas, device(), view_.device_index(), view_.stream());
    }

    VECTORIZATION_FUNCTION_ATTRIBUTE tensor view(const dimensions_type& new_dims) const
    {
        VECTORIZATION_CHECK_DEBUG_IF_NOT_ON_CUDA(
            is_contiguous(), "view() requires a contiguous tensor");
        VECTORIZATION_CHECK_DEBUG_IF_NOT_ON_CUDA(
            compute_total(new_dims) == size(), "view: element count must not change");
        sizes_and_strides sas;
        make_contiguous_sas(sas, new_dims);
        return tensor(data(), view_.size(), sas, device(), view_.device_index(), view_.stream());
    }

    VECTORIZATION_FUNCTION_ATTRIBUTE tensor reshape(const dimensions_type& new_dims) const
    {
        VECTORIZATION_CHECK_DEBUG_IF_NOT_ON_CUDA(
            is_contiguous(), "reshape: call contiguous() first for non-contiguous tensors");
        VECTORIZATION_CHECK_DEBUG_IF_NOT_ON_CUDA(
            compute_total(new_dims) == size(), "reshape: element count must not change");
        sizes_and_strides sas;
        make_contiguous_sas(sas, new_dims);
        return tensor(data(), view_.size(), sas, device(), view_.device_index(), view_.stream());
    }

    // Slice along one dimension. stop == -1 (the default) means "to end".
    // Negative start is wrapped like Python (start += dim_size). step == 0 is invalid.
    VECTORIZATION_FUNCTION_ATTRIBUTE tensor
    slice(size_t dim, int64_t start, int64_t stop = -1, int64_t step = 1) const
    {
        VECTORIZATION_CHECK_DEBUG_IF_NOT_ON_CUDA(dim < rank(), "slice: dim out of range");
        VECTORIZATION_CHECK_DEBUG_IF_NOT_ON_CUDA(step != 0, "slice: step must not be zero");
        const int64_t dim_size = sizes_and_strides_.size_at_unchecked(dim);
        if (start < 0)
        {
            start += dim_size;
        }
        if (start < 0)
        {
            start = 0;
        }
        if (stop < 0)
        {
            stop = dim_size;
        }
        stop                         = std::min(stop, dim_size);
        start                        = std::min(start, dim_size);
        const int64_t     span       = stop - start;
        const int64_t     new_size   = (span > 0 && step > 0) || (span < 0 && step < 0)
                                           ? (std::abs(span) + std::abs(step) - 1) / std::abs(step)
                                           : 0;
        sizes_and_strides sas        = sizes_and_strides_;
        sas.size_at_unchecked(dim)   = new_size;
        sas.stride_at_unchecked(dim) = sizes_and_strides_.stride_at_unchecked(dim) * step;
        const size_t offset          = static_cast<size_t>(start) *
                              static_cast<size_t>(sizes_and_strides_.stride_at_unchecked(dim));
        value_t*     new_ptr      = data() + offset;
        const size_t storage_size = view_.size() > offset ? view_.size() - offset : 0;
        return tensor(new_ptr, storage_size, sas, device(), view_.device_index(), view_.stream());
    }

#if 0
    // -----------------------------------------------------------------------
    // Matrix operations (2-D)
    // -----------------------------------------------------------------------

    // Called by matrix_multiplication_expression::evaluate() with *this as output
    VECTORIZATION_FUNCTION_ATTRIBUTE void matrix_multiplication(
        bool          trA,
        bool          trB,
        tensor const& A,
        tensor const& B) noexcept
    {
        const size_t colsA = A.rank() >= 2 ? A.dimension(1) : 1;
        const size_t colsB = B.rank() >= 2 ? B.dimension(1) : 1;
        const size_t ldA   = trA ? 1 : colsA;
        const size_t ldB   = trB ? 1 : colsB;

        vectorization::matrix_multiplication(
            trA, trB,
            dimension(0), dimension(1),
            trA ? A.dimension(0) : colsA,
            A.data(), ldA,
            B.data(), ldB,
            data(), dimension(1));
    }

    // Called by matrix_transpose_expression::evaluate()
    VECTORIZATION_FUNCTION_ATTRIBUTE void matrix_transpose(tensor const& A)
    {
        clone(A);
        vectorization::matrix_transpose(A.dimension(0), A.dimension(1), data());
        if (sizes_and_strides_.size() >= 2)
        {
            // Swap shape dims 0 and 1, then recompute strides.
            const int64_t tmp = sizes_and_strides_.size_at_unchecked(0);
            sizes_and_strides_.size_at_unchecked(0) = sizes_and_strides_.size_at_unchecked(1);
            sizes_and_strides_.size_at_unchecked(1) = tmp;
            dimensions_type dims(sizes_and_strides_.sizes_begin(), sizes_and_strides_.sizes_end());
            set_shape(dims);
        }
    }

    friend auto transpose(tensor const& rhs)
    {
        return matrix_transpose_expression<tensor>(rhs);
    }
#endif

    // -----------------------------------------------------------------------
    // Comparison / predicates
    // -----------------------------------------------------------------------

    bool operator==(tensor const& rhs) const
    {
        if (rank() != rhs.rank())
        {
            return false;
        }
        for (size_t i = 0; i < rank(); ++i)
        {
            if (size(i) != rhs.size(i))
            {
                return false;
            }
        }
        if (device() == device_enum::CPU && rhs.device() == device_enum::CPU)
        {
            return cpu_values_equal(rhs);
        }
        if (device() == device_enum::CPU)
        {
            return cpu_values_equal(rhs.to_cpu());
        }
        if (rhs.device() == device_enum::CPU)
        {
            return to_cpu().cpu_values_equal(rhs);
        }
        return to_cpu().cpu_values_equal(rhs.to_cpu());
    }

    bool operator!=(tensor const& rhs) const { return !(*this == rhs); }

    bool is_zero() const
    {
        return all_logical([](value_t v) { return is_almost_zero(v); });
    }

    bool non_negative() const
    {
        return all_logical([](value_t v) { return v >= -std::numeric_limits<value_t>::epsilon(); });
    }

    bool positive() const
    {
        return all_logical([](value_t v) { return v >= std::numeric_limits<value_t>::epsilon(); });
    }

    bool symmetric() const
    {
        if (rank() < 2 || dimension(0) != dimension(1))
        {
            return false;
        }
        if (device() != device_enum::CPU)
        {
            return to_cpu().symmetric();
        }
        const size_t r = dimension(0);
        for (size_t i = 0; i < r; ++i)
        {
            for (size_t j = 0; j < i; ++j)
            {
                if (!is_almost_zero(at(i, j) - at(j, i)))
                {
                    return false;
                }
            }
        }
        return true;
    }

    bool identity() const
    {
        if (rank() < 2 || dimension(0) != dimension(1))
        {
            return false;
        }
        if (device() != device_enum::CPU)
        {
            return to_cpu().identity();
        }
        const size_t r = dimension(0);
        const size_t c = dimension(1);
        for (size_t i = 0; i < r; ++i)
        {
            for (size_t j = 0; j < c; ++j)
            {
                if (at(i, j) != static_cast<value_t>(i == j))
                {
                    return false;
                }
            }
        }
        return true;
    }

    bool is_correlation() const
    {
        if (rank() < 2 || dimension(0) != dimension(1))
        {
            return false;
        }
        if (device() != device_enum::CPU)
        {
            return to_cpu().is_correlation();
        }
        const size_t r = dimension(0);
        for (size_t i = 0; i < r; ++i)
        {
            if (!is_almost_zero(at(i, i) - value_t(1)))
            {
                return false;
            }
            for (size_t j = 0; j < i; ++j)
            {
                const value_t v = at(i, j);
                if (std::fabs(v) > value_t(1) || !is_almost_zero(v - at(j, i)))
                {
                    return false;
                }
            }
        }
        return true;
    }

    value_t trace() const
    {
        VECTORIZATION_CHECK_DEBUG(
            rank() >= 2 && dimension(0) == dimension(1), "trace requires square rank-2 tensor");
        if (device() != device_enum::CPU)
        {
            return to_cpu().trace();
        }
        const size_t c = dimension(1);
        value_t      t = 0;
        for (size_t i = 0; i < c; ++i)
        {
            t += at(i, i);
        }
        return t;
    }

    // -----------------------------------------------------------------------
    // Formatting
    // -----------------------------------------------------------------------

    std::string to_string() const
    {
        if (empty())
            return "[]";
        if (device() != device_enum::CPU)
        {
            return to_cpu().to_string();
        }

        auto const at_flat = [&](size_t i) -> value_t { return data()[logical_offset(i)]; };

        const size_t n     = size();
        size_t       width = 0;
        {
            std::ostringstream tmp;
            for (size_t i = 0; i < n; ++i)
            {
                tmp.str(std::string{});
                tmp << at_flat(i);
                width = std::max(width, tmp.str().size());
            }
        }

        std::ostringstream s;
        if (rank() <= 1)
        {
            s << "[";
            for (size_t i = 0; i < n; ++i)
            {
                if (i)
                    s << ",\n ";
                if (width)
                    s.width(static_cast<std::streamsize>(width));
                s << at_flat(i);
            }
            s << "]";
        }
        else
        {
            const size_t r = dimension(0);
            const size_t c = dimension(1);
            s << "[";
            for (size_t i = 0; i < r; ++i)
            {
                if (i)
                    s << ";\n ";
                if (width)
                    s.width(static_cast<std::streamsize>(width));
                s << at(i, 0);
                for (size_t j = 1; j < c; ++j)
                {
                    s << ", ";
                    if (width)
                        s.width(static_cast<std::streamsize>(width));
                    s << at(i, j);
                }
            }
            s << "]";
        }
        return s.str();
    }

private:
    // View constructor — non-owning tensor over an existing buffer with
    // explicit shape/strides. Used by t() / permute / view / reshape / slice.
    tensor(
        value_t*                  ptr,
        size_t                    storage_size,
        sizes_and_strides         sas,
        device_enum               type         = device_enum::CPU,
        int                       device_index = 0,
        typename view_t::stream_t stream       = nullptr)
        : sizes_and_strides_(std::move(sas)),
          view_(view_t::borrow(ptr, storage_size, type, device_index, stream))
    {
        recompute_cpu_simd_alignment_state();
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    // Populate sizes_and_strides_ with the given shape and derived C-order strides.
    void set_shape(const dimensions_type& dims) { make_contiguous_sas(sizes_and_strides_, dims); }

    static size_type compute_total(const dimensions_type& dims)
    {
        size_type total = 1;
        for (auto d : dims)
        {
            VECTORIZATION_CHECK(
                d == 0 || total <= std::numeric_limits<size_type>::max() / d,
                "tensor dimensions overflow size_t");
            total *= d;
        }
        return total;
    }

    static int64_t checked_dimension_to_storage(size_type dim)
    {
        VECTORIZATION_CHECK(
            dim <= static_cast<size_type>(std::numeric_limits<int64_t>::max()),
            "tensor dimension {} exceeds int64_t storage limit",
            dim);
        return static_cast<int64_t>(dim);
    }

    // Build a sizes_and_strides with the given shape and C-order strides.
    static void make_contiguous_sas(sizes_and_strides& sas, const dimensions_type& dims)
    {
        const size_t n = dims.size();
        sas.resize(n);
        if (n == 0)
            return;
        sas.stride_at_unchecked(n - 1) = 1;
        for (size_t i = n - 1; i-- > 0;)
        {
            const int64_t next_dim    = checked_dimension_to_storage(dims[i + 1]);
            const int64_t next_stride = sas.stride_at_unchecked(i + 1);
            VECTORIZATION_CHECK(
                next_dim == 0 || next_stride <= std::numeric_limits<int64_t>::max() / next_dim,
                "tensor stride exceeds int64_t storage limit");
            sas.stride_at_unchecked(i) = next_stride * next_dim;
        }
        for (size_t i = 0; i < n; ++i)
            sas.size_at_unchecked(i) = checked_dimension_to_storage(dims[i]);
    }

    VECTORIZATION_FUNCTION_ATTRIBUTE size_t linearized_index(const dimensions_type& indices) const
    {
        const size_t n = sizes_and_strides_.size();
        const size_t m = indices.size();
        VECTORIZATION_CHECK_DEBUG_IF_NOT_ON_CUDA(m <= n, "number of indices exceeds tensor rank");

        size_t ret = 0;
        for (size_t i = 0; i < m; ++i)
            ret += static_cast<size_t>(sizes_and_strides_.stride_at_unchecked(i)) *
                   static_cast<size_t>(indices[i]);
        return ret;
    }

    // Resolves the ambient stream for device_index (see stream_guard.h) and, only when
    // non-null, records expr's GPU tensor operands as touched on it so the caching
    // allocator won't recycle their storage until that stream's work completes (same
    // bookkeeping assign_async does, just sourced from the ambient guard instead of a
    // caller-supplied stream). Shared by operator=/init_from_expression so this contract
    // can't drift between call sites.
    template <typename E>
    VECTORIZATION_HOST_FUNCTION_ATTRIBUTE static gpu_stream_t resolve_ambient_stream(
        E const& expr, int device_index)
    {
        auto const stream = current_stream(device_index);
        if (stream != nullptr)
        {
            record_expression_streams(expr, stream);
        }
        return stream;
    }

    // Companion to resolve_ambient_stream(): records *this as touched on `stream` after
    // an evaluation that used it. No-op for the default stream.
    VECTORIZATION_HOST_FUNCTION_ATTRIBUTE void record_stream_if_redirected(
        gpu_stream_t stream) const
    {
        if (stream != nullptr)
        {
            record_stream(stream);
        }
    }

    template <typename E>
    void init_from_expression(E const& expr)
    {
        auto const p = infer_expression_placement(expr);
        // Allocate directly on the stream that will evaluate the expression (the ambient
        // stream_guard stream for the inferred device, or nullptr/default) rather than
        // whatever stream the first operand happened to carry -- otherwise a freshly
        // constructed tensor could be allocated on one stream and evaluated on another.
        auto const stream = resolve_ambient_stream(expr, p.index);
        owner_ =
            owner_t(expr.size(), p.kind, p.index, static_cast<typename owner_t::stream_t>(stream));
        view_                                     = owner_.view();
        sizes_and_strides_.size_at_unchecked(0)   = static_cast<int64_t>(expr.size());
        sizes_and_strides_.stride_at_unchecked(0) = 1;
        recompute_cpu_simd_alignment_state();
        evaluator::template run<E, tensor>(expr, *this, stream);
    }

    void stamp_contiguous_shape(tensor const& src)
    {
        const size_t n = src.rank();
        sizes_and_strides_.resize(n);
        if (n == 0)
        {
            recompute_cpu_simd_alignment_state();
            return;
        }
        for (size_t i = 0; i < n; ++i)
        {
            sizes_and_strides_.size_at_unchecked(i) = src.sizes_and_strides_.size_at_unchecked(i);
        }
        sizes_and_strides_.stride_at_unchecked(n - 1) = 1;
        for (int i = static_cast<int>(n) - 2; i >= 0; --i)
        {
            sizes_and_strides_.stride_at_unchecked(i) =
                sizes_and_strides_.stride_at_unchecked(i + 1) *
                sizes_and_strides_.size_at_unchecked(i + 1);
        }
        recompute_cpu_simd_alignment_state();
    }

    VECTORIZATION_FUNCTION_ATTRIBUTE size_t logical_offset(size_t flat) const noexcept
    {
        if (contiguous_)
        {
            return flat;
        }
        const size_t n   = rank();
        size_t       off = 0;
        size_t       rem = flat;
        for (int k = static_cast<int>(n) - 1; k >= 0; --k)
        {
            const size_t dk = static_cast<size_t>(sizes_and_strides_.size_at_unchecked(k));
            if (dk == 0)
            {
                return 0;
            }
            const size_t idx = rem % dk;
            rem /= dk;
            off += idx * static_cast<size_t>(sizes_and_strides_.stride_at_unchecked(k));
        }
        return off;
    }

    void copy_logical_to_host(value_t* dst) const
    {
        const size_t total = size();
        if (dst == nullptr || total == 0)
        {
            return;
        }
        if (device() == device_enum::CPU)
        {
            if (is_contiguous())
            {
                std::copy_n(data(), total, dst);
            }
            else
            {
                gather_logical(data(), dst);
            }
            return;
        }
        if (is_contiguous())
        {
            allocator_t::copy(
                data(), total, dst, device(), device_enum::CPU, view_.device_index(), 0);
            return;
        }
        owner_t window(view_.size(), device_enum::CPU);
        if (view_.size() != 0)
        {
            allocator_t::copy(
                data(),
                view_.size(),
                window.data(),
                device(),
                device_enum::CPU,
                view_.device_index(),
                0);
        }
        gather_logical(window.data(), dst);
    }

    void gather_logical(value_t const* storage, value_t* dst) const
    {
        const size_t total = numel_;
        if (total == 0)
        {
            return;
        }
        if (contiguous_)
        {
            std::copy_n(storage, total, dst);
            return;
        }
        const size_t     n        = rank();
        constexpr size_t kMaxRank = 16;
        VECTORIZATION_CHECK(n <= kMaxRank, "gather_logical: rank exceeds 16");
        size_t coord[kMaxRank] = {};
        size_t off             = 0;
        for (size_t i = 0; i < total; ++i)
        {
            dst[i] = storage[off];
            for (int k = static_cast<int>(n) - 1; k >= 0; --k)
            {
                ++coord[k];
                off += static_cast<size_t>(sizes_and_strides_.stride_at_unchecked(k));
                if (coord[k] < static_cast<size_t>(sizes_and_strides_.size_at_unchecked(k)))
                {
                    break;
                }
                off -= static_cast<size_t>(coord[k]) *
                       static_cast<size_t>(sizes_and_strides_.stride_at_unchecked(k));
                coord[k] = 0;
            }
        }
    }

    bool cpu_values_equal(tensor const& rhs) const
    {
        const size_t n = size();
        for (size_t i = 0; i < n; ++i)
        {
            if (data()[logical_offset(i)] != rhs.data()[rhs.logical_offset(i)])
            {
                return false;
            }
        }
        return true;
    }

    template <typename Pred>
    bool all_logical(Pred pred) const
    {
        const size_t n = size();
        if (device() != device_enum::CPU)
        {
            return to_cpu().all_logical(pred);
        }
        for (size_t i = 0; i < n; ++i)
        {
            if (!pred(data()[logical_offset(i)]))
            {
                return false;
            }
        }
        return true;
    }

    VECTORIZATION_FUNCTION_ATTRIBUTE void recompute_cpu_simd_alignment_state() noexcept
    {
        const size_t n = rank();
        if (n == 0)
        {
            numel_       = 0;
            contiguous_  = true;
            align_start_ = 0;
            align_end_   = 0;
            return;
        }
        size_t total = 1;
        for (size_t i = 0; i < n; ++i)
        {
            total *= static_cast<size_t>(sizes_and_strides_.size_at_unchecked(i));
        }
        numel_ = total;

        // Matches PyTorch's TensorImpl::compute_contiguous(): a size-1 dimension's stride is
        // never read by any valid index (its only index is 0), so it must not gate
        // contiguity -- only dimensions with size > 1 are checked against the running
        // expected packed stride. permute()/t() copy strides verbatim rather than
        // recomputing them, so a singleton axis can end up next to a different neighbor
        // than the one its stride was originally canonical for, while the tensor is still
        // genuinely flat-indexable (data[i] for i in [0, numel_) valid). An empty tensor
        // (numel_ == 0) is unconditionally contiguous, same as the rank-0 case above.
        if (numel_ == 0)
        {
            contiguous_ = true;
        }
        else
        {
            contiguous_             = true;
            int64_t expected_stride = 1;
            for (int i = static_cast<int>(n) - 1; i >= 0; --i)
            {
                const int64_t size_i = sizes_and_strides_.size_at_unchecked(i);
                if (size_i == 1)
                {
                    continue;
                }
                if (sizes_and_strides_.stride_at_unchecked(i) != expected_stride)
                {
                    contiguous_ = false;
                    break;
                }
                expected_stride *= size_i;
            }
        }

#if !VECTORIZATION_VECTORIZED || VECTORIZATION_ON_GPU_DEVICE
        align_start_ = 0;
        align_end_   = 0;
#else
        if (view_.device() != device_enum::CPU || numel_ == 0 || !contiguous_)
        {
            align_start_ = 0;
            align_end_   = 0;
            return;
        }
        value_t const* base = view_.data();
        align_start_        = first_aligned(base, numel_);
        align_end_          = last_aligned(align_start_, numel_, length());
#endif
    }

    sizes_and_strides sizes_and_strides_;
    owner_t           owner_{};
    view_t            view_{};
    std::size_t       align_start_{0};
    /// Exclusive end index for the aligned SIMD body (matches \c expressions_evaluator::run peeling).
    std::size_t align_end_{0};
    std::size_t numel_{0};
    bool        contiguous_{true};
};

// ---------------------------------------------------------------------------
// Stream output
// ---------------------------------------------------------------------------
template <typename T>
std::ostream& operator<<(std::ostream& s, const tensor<T>& t)
{
    s << t.to_string();
    return s;
}

}  // namespace vectorization

#if defined(_MSC_VER) && !defined(VECTORIZATION_DISPLAY_WIN32_WARNINGS)
#pragma warning(pop)
#endif
