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

#ifndef VECTORIZATION_EXPRESSIONS_INCLUDES_INSIDE
Do_not_include_expression_reductions_directly_use_expressions_h;
#endif

// Included after expressions_evaluator.h: shares device placement, shape
// validation and strided lowering with the original horizontal reductions.

namespace vectorization
{
namespace detail
{
#if VECTORIZATION_VECTORIZED
template <typename T, reduce_op Op>
VECTORIZATION_HOST_FUNCTION_ATTRIBUTE typename simd<T>::simd_t reduce_packet_transform(
    typename simd<T>::simd_t x)
{
    using S         = simd<T>;
    const auto zero = S::set(T(0));
    const auto one  = S::set(T(1));
    if constexpr (Op == reduce_op::all || Op == reduce_op::any || Op == reduce_op::count_nonzero)
    {
        return S::if_else(S::neq(x, zero), one, zero);
    }
    else if constexpr (Op == reduce_op::count_valid)
    {
        return S::if_else(S::eq(x, x), one, zero);
    }
    else if constexpr (Op == reduce_op::nansum)
    {
        return S::if_else(S::eq(x, x), x, zero);
    }
    else
    {
        return x;
    }
}

template <typename T, reduce_op Op>
VECTORIZATION_HOST_FUNCTION_ATTRIBUTE typename simd<T>::simd_t reduce_packet_combine(
    typename simd<T>::simd_t a, typename simd<T>::simd_t b)
{
    if constexpr (Op == reduce_op::product)
    {
        return simd<T>::mul(a, b);
    }
    else if constexpr (Op == reduce_op::all)
    {
        return simd<T>::min(a, b);
    }
    else if constexpr (Op == reduce_op::any)
    {
        return simd<T>::max(a, b);
    }
    else if constexpr (Op == reduce_op::min)
    {
        // The new value goes first: x86 min/max return the second operand when
        // either is NaN, so a NaN input is skipped, as std::fmin does. NEON and
        // SVE min/max propagate NaN instead (unchanged from the previous loops).
        return simd<T>::min(b, a);
    }
    else if constexpr (Op == reduce_op::max)
    {
        return simd<T>::max(b, a);
    }
    else
    {
        return simd<T>::add(a, b);
    }
}
#endif

// Scalar combine for the CPU head, tail and strided rows. Min and max skip NaN
// inputs (std::fmin / std::fmax), as hmin / hmax always have.
template <typename acc_t, reduce_op Op>
VECTORIZATION_HOST_FUNCTION_ATTRIBUTE acc_t reduce_scalar_combine(acc_t a, acc_t b)
{
    if constexpr (Op == reduce_op::min)
    {
        return std::fmin(a, b);
    }
    else if constexpr (Op == reduce_op::max)
    {
        return std::fmax(a, b);
    }
    else
    {
        return reduce_combine<acc_t, Op>(a, b);
    }
}

#if VECTORIZATION_VECTORIZED
// Independent SIMD accumulators carried across calls, so a reduction keeps
// several dependency chains in flight over one packed span or over many rows
// of a strided one. flush() folds them into the scalar result.
//
// Between calls the accumulators are kept as plain lanes and reloaded with
// unaligned loads: run() works on register locals only. A simd_t member would
// be an over-aligned stack object, and clang-cl was observed addressing such an
// object with aligned stores in a frame it had not realigned (access violation).
template <typename T, reduce_op Op> struct simd_reducer
{
    using S                        = simd<T>;
    using simd_t                   = typename S::simd_t;
    using acc_t                    = reduce_accumulator_t<T, Op>;
    static constexpr size_t length = S::size;
    static constexpr size_t packet = VECTORIZATION_PACKET_SIZE < 4 ? 4 : VECTORIZATION_PACKET_SIZE;
    static constexpr size_t block  = length * packet;
    // Count lanes are floating point: flush well before 2^24 so each stays exact.
    static constexpr size_t kCountFlushVectors = 4096;

    std::array<T, length * packet> lanes;
    size_t                         pending{0};  // vectors folded in since the last flush

    simd_reducer() { reset(); }

    void reset()
    {
        lanes.fill(static_cast<T>(reduce_identity<acc_t, Op>()));
        pending = 0;
    }

    template <bool A, typename F>
    static void update(std::array<simd_t, packet>& acc, F const& flat, size_t k, size_t offset)
    {
        auto x = expression_loader<F, true, A>::evaluate(flat, offset);
        // Comparison expressions produce predicates, including on ISAs whose
        // predicate and numeric register types happen to coincide.
        using scalar_result = decltype(expression_loader<F, false, false>::evaluate(flat, 0));
        if constexpr (std::is_same_v<scalar_result, bool>)
        {
            acc[k] = reduce_packet_combine<T, Op>(
                acc[k], reduce_packet_transform<T, Op>(S::if_else(x, S::set(T(1)), S::set(T(0)))));
        }
        else
        {
            acc[k] = reduce_packet_combine<T, Op>(acc[k], reduce_packet_transform<T, Op>(x));
        }
    }

    // Folds elements [i, end) of `flat` into the accumulators; end - i must be
    // a multiple of the SIMD width. A selects aligned loads.
    template <bool A, typename F> void run(F const& flat, size_t i, size_t end, acc_t& result)
    {
        VECTORIZATION_MATH_CALL_FRAME(F);
        while (i < end)
        {
            size_t stop = end;
            if constexpr (reduce_is_count<Op>)
            {
                const size_t room = (kCountFlushVectors - pending) * length;
                stop              = end - i > room ? i + room : end;
                pending += (stop - i) / length;
            }
            std::array<simd_t, packet> acc;
            unroll<packet>(
                [&](auto k)
                {
                    constexpr size_t kk = decltype(k)::value;
                    acc[kk]             = S::loadu(lanes.data() + kk * length);
                });
            for (; stop - i >= block; i += block)
            {
                unroll<packet>([&](auto k)
                    { update<A>(acc, flat, decltype(k)::value, i + decltype(k)::value * length); });
            }
            for (; i < stop; i += length)
            {
                update<A>(acc, flat, 0, i);
            }
            unroll<packet>(
                [&](auto k)
                {
                    constexpr size_t kk = decltype(k)::value;
                    S::storeu(acc[kk], lanes.data() + kk * length);
                });
            if constexpr (reduce_is_count<Op>)
            {
                if (pending >= kCountFlushVectors)
                {
                    flush(result);
                }
            }
        }
    }

    // Runs `rows` rows of `vector_n` elements each (a multiple of the SIMD width)
    // with the accumulators held in registers from the first row to the last.
    // `tree` must already point at the first row; tail(tree, acc) folds the
    // row's scalar remainder into a register-local scalar; advance(tree) moves
    // the leaves to the next row. Not for counts, whose float lanes must be
    // flushed periodically (use run() per row).
    template <typename F, typename Tail, typename Advance>
    void run_rows(F const& first_row,
        int64_t            rows,
        size_t             vector_n,
        Tail&&             tail,
        Advance&&          advance,
        acc_t&             result)
    {
        static_assert(!reduce_is_count<Op>, "count reductions flush; use run() per row");
        VECTORIZATION_MATH_CALL_FRAME(F);
        // A local copy of the (pointer-only) tree keeps the leaf pointers in
        // registers; advanced through a reference they would be stored and
        // reloaded at every row, delaying each row's first loads.
        F                          tree = first_row;
        std::array<simd_t, packet> acc;
        unroll<packet>(
            [&](auto k)
            {
                constexpr size_t kk = decltype(k)::value;
                acc[kk]             = S::loadu(lanes.data() + kk * length);
            });
        acc_t tail_acc = result;
        for (int64_t row = 0; row < rows; ++row)
        {
            size_t i = 0;
            for (; vector_n - i >= block; i += block)
            {
                unroll<packet>(
                    [&](auto k)
                    {
                        update<false>(
                            acc, tree, decltype(k)::value, i + decltype(k)::value * length);
                    });
            }
            // At most packet - 1 whole registers remain; each goes to its own
            // accumulator so the remainder does not serialize on one of them.
            unroll<packet - 1>(
                [&](auto k)
                {
                    if (i < vector_n)
                    {
                        update<false>(acc, tree, decltype(k)::value, i);
                        i += length;
                    }
                });
            tail(tree, tail_acc);
            if (row + 1 < rows)
            {
                advance(tree);
            }
        }
        result = tail_acc;
        unroll<packet>(
            [&](auto k)
            {
                constexpr size_t kk = decltype(k)::value;
                S::storeu(acc[kk], lanes.data() + kk * length);
            });
    }

    void flush(acc_t& result)
    {
        if constexpr (reduce_is_count<Op> || Op == reduce_op::product)
        {
            // Convert each lane to the accumulator type before merging; never sum
            // a large count in float.
            for (size_t lane = 0; lane < length * packet; ++lane)
            {
                result = reduce_combine<acc_t, Op>(result, static_cast<acc_t>(lanes[lane]));
            }
        }
        else
        {
            auto combined = S::loadu(lanes.data());
            for (size_t k = 1; k < packet; ++k)
            {
                combined =
                    reduce_packet_combine<T, Op>(combined, S::loadu(lanes.data() + k * length));
            }
            if constexpr (Op == reduce_op::all || Op == reduce_op::min)
            {
                result = reduce_scalar_combine<acc_t, Op>(result, S::hmin(combined));
            }
            else if constexpr (Op == reduce_op::any || Op == reduce_op::max)
            {
                result = reduce_scalar_combine<acc_t, Op>(result, S::hmax(combined));
            }
            else
            {
                result += static_cast<T>(S::accumulate(combined));
            }
        }
        reset();
    }
};
#endif

// Reduces `expr` (tensor leaves are strided_ref / retained_ref / data_view) of
// result shape `shape` through the iteration planner: dimensions are merged and
// ordered by the sources' strides, and the space is walked one row at a time
// with a dimension counter, so no index is divided. Rows whose sources all have
// inner stride 1 run the packed SIMD body; rows with a broadcast (stride 0)
// source run SIMD with splatted loads; other rows are scalar. Accumulators are
// carried from row to row.
template <reduce_op Op, typename T, typename E>
VECTORIZATION_HOST_FUNCTION_ATTRIBUTE void reduce_planned(
    E const& expr, expression_shape const& shape, reduce_accumulator_t<T, Op>& result)
{
    using acc_t = reduce_accumulator_t<T, Op>;
    iteration_plan plan;
    plan.rank       = shape.rank;
    plan.numel      = static_cast<int64_t>(shape.numel());
    plan.n_operands = 1;  // operand 0: no destination, stride 0 so the sources set the order
    for (int d = 0; d < shape.rank; ++d)
    {
        plan.sizes[d] = shape.sizes[d];
    }
    plan_leaves(expr, shape, plan);
    if (plan.numel == 0)
    {
        return;
    }
    sort_merge_classify(plan);

    const int     inner   = plan.rank - 1;
    const int64_t inner_n = plan.row_size();
    const int64_t rows    = plan.rows();
    bool          unit    = true;  // every source has inner stride 1
    bool          splat   = true;  // every source has inner stride 0 or 1
    for (int op = 1; op < plan.n_operands; ++op)
    {
        const int64_t s = plan.strides[op][inner];
        unit            = unit && s == 1;
        splat           = splat && (s == 0 || s == 1);
    }

    auto rowed = lower_rows(expr);

    std::array<int64_t, kMaxRank>     coord{};
    std::array<int64_t, kMaxOperands> offsets{};
    // Folds elements [begin, inner_n) of the bound row into `acc`, which the
    // callers keep in a local so the running value stays in a register.
    const auto scalar_row = [inner_n](auto const& tree, int64_t begin, acc_t& acc)
    {
        using L = remove_cvref_t<decltype(tree)>;
        acc_t a = acc;
        for (int64_t k = begin; k < inner_n; ++k)
        {
            const T x = expression_loader<L, false, false>::evaluate(tree, static_cast<size_t>(k));
            a         = reduce_scalar_combine<acc_t, Op>(a, reduce_transform<T, Op>(x));
        }
        acc = a;
    };

    // Row stepping. A single outer dimension (the common case after merging)
    // advances each leaf pointer by its outer stride; deeper plans step the
    // dimension counter and rebind every leaf.
    std::array<int64_t, kMaxOperands> outer_step{};
    const bool                        two_d = plan.rank == 2;
    if (two_d)
    {
        for (int op = 0; op < plan.n_operands; ++op)
        {
            outer_step[op] = plan.strides[op][0];
        }
    }
    const auto make_advance = [&](auto&& bind_leaves)
    {
        return [&, bind_leaves](auto& tree)
        {
            int op = 1;
            if (two_d)
            {
                advance_row(tree, outer_step.data(), op);
            }
            else
            {
                next_row(plan, coord, offsets);
                bind_leaves(tree, plan, offsets.data(), op);
            }
        };
    };
    const auto bind_rows = [](auto& t, auto const& p, int64_t const* o, int& op)
    { bind_row(t, p, o, op); };
    const auto bind_flat = [](auto& t, auto const& p, int64_t const* o, int& op)
    { bind_flat_row(t, p, o, op); };

#if VECTORIZATION_VECTORIZED
    using reducer_t             = simd_reducer<T, Op>;
    constexpr int64_t length    = static_cast<int64_t>(reducer_t::length);
    const size_t      vector_n  = static_cast<size_t>((inner_n / length) * length);
    const bool        simd_rows = splat && inner_n >= length;
    reducer_t         reducer;
    // One SIMD pass over every row of `tree`; bind_leaves points its leaves at a row.
    const auto simd_pass = [&](auto& tree, auto const& bind_leaves)
    {
        int op = 1;
        bind_leaves(tree, plan, offsets.data(), op);
        const auto advance = make_advance(bind_leaves);
        const auto tail    = [&](auto const& t, acc_t& acc)
        { scalar_row(t, static_cast<int64_t>(vector_n), acc); };
        if constexpr (reduce_is_count<Op>)
        {
            for (int64_t row = 0; row < rows; ++row)
            {
                reducer.template run<false>(tree, 0, vector_n, result);
                tail(tree, result);
                if (row + 1 < rows)
                {
                    advance(tree);
                }
            }
        }
        else
        {
            reducer.run_rows(tree, rows, vector_n, tail, advance, result);
        }
        reducer.flush(result);
    };
    if (simd_rows && unit)
    {
        auto flat = lower_flat(expr);
        simd_pass(flat, bind_flat);
        return;
    }
    if (simd_rows)
    {
        simd_pass(rowed, bind_rows);
        return;
    }
#else
    (void)splat;
    (void)unit;
    (void)bind_flat;
#endif
    int op = 1;
    bind_row(rowed, plan, offsets.data(), op);
    const auto advance = make_advance(bind_rows);
    acc_t      acc     = result;
    for (int64_t row = 0; row < rows; ++row)
    {
        scalar_row(rowed, 0, acc);
        if (row + 1 < rows)
        {
            advance(rowed);
        }
    }
    result = acc;
}

// CPU reduction over strided tensors and strided / broadcast expressions.
// Returns false when every operand is packed and the flat SIMD path applies.
template <reduce_op Op, typename T, typename E>
VECTORIZATION_HOST_FUNCTION_ATTRIBUTE bool reduce_strided_planned(
    E const& expression, reduce_accumulator_t<T, Op>& result)
{
    using acc_t = reduce_accumulator_t<T, Op>;
    if constexpr (is_tensor<E>::value)
    {
        if (expression.is_contiguous())
        {
            return false;
        }
        if (expression.rank() > static_cast<size_t>(kMaxRank))
        {
            // Beyond a strided_ref's rank: fold the logical elements one by one.
            acc_t acc = result;
            for (size_t i = 0, n = expression.size(); i < n; ++i)
            {
                acc = reduce_scalar_combine<acc_t, Op>(acc, reduce_transform<T, Op>(expression[i]));
            }
            result = acc;
            return true;
        }
        const auto       leaf = expression.to_strided_ref();
        expression_shape shape;
        shape.rank = leaf.rank;
        for (int d = 0; d < leaf.rank; ++d)
        {
            shape.sizes[d] = leaf.sizes[d];
        }
        reduce_planned<Op, T>(leaf, shape, result);
        return true;
    }
    else if constexpr (is_pure_expression<E>::value)
    {
        const auto shape = expression_result_shape(expression);
        if (leaves_flat(expression, shape.numel()))
        {
            return false;
        }
        reduce_planned<Op, T>(expression, shape, result);
        return true;
    }
    else
    {
        return false;
    }
}

template <reduce_op Op, typename E>
VECTORIZATION_HOST_FUNCTION_ATTRIBUTE auto reduce_expression_impl(E const& expression)
{
    using T     = typename scalar_type<E, E>::value;
    using acc_t = reduce_accumulator_t<T, Op>;
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
        "reductions support float and double expressions");
    acc_t result = reduce_identity<acc_t, Op>();
    if (try_reduce_gpu<E, T, Op>(expression, result))
    {
        return result;
    }
    if (reduce_strided_planned<Op, T>(expression, result))
    {
        return result;
    }

    // Packed operands: reduce over pointer-only leaves, as the elementwise
    // evaluator does.
    const auto flat          = lower_flat(expression);
    using F                  = remove_cvref_t<decltype(flat)>;
    const size_t n           = expression.size();
    size_t       i           = 0;
    auto         scalar_step = [&](size_t k)
    {
        const T x = expression_loader<F, false, false>::evaluate(flat, k);
        result    = reduce_scalar_combine<acc_t, Op>(result, reduce_transform<T, Op>(x));
    };
#if VECTORIZATION_VECTORIZED
    using S                             = simd<T>;
    constexpr size_t         length     = S::size;
    constexpr std::uintptr_t simd_align = alignof(typename S::simd_t);

    // Alignment start and end: a scalar head up to the first SIMD-aligned element of
    // the first operand, then a SIMD body of whole registers, then the scalar tail.
    // When every operand shares that alignment the body uses aligned loads.
    void const* const anchor  = first_flat_leaf(flat);
    const bool        aligned = anchor != nullptr && leaves_co_aligned(flat, anchor, simd_align);
    const size_t      head =
        aligned ? align_head(static_cast<T const*>(anchor), n, simd_align) : size_t{0};
    for (; i < head; ++i)
    {
        scalar_step(i);
    }
    const size_t vector_end = i + ((n - i) / length) * length;
    if (i < vector_end)
    {
        simd_reducer<T, Op> reducer;
        if (aligned)
        {
            reducer.template run<true>(flat, i, vector_end, result);
        }
        else
        {
            reducer.template run<false>(flat, i, vector_end, result);
        }
        reducer.flush(result);
        i = vector_end;
    }
#endif
    for (; i < n; ++i)
    {
        scalar_step(i);
    }
    return result;
}

template <reduce_op Op, typename E>
VECTORIZATION_HOST_FUNCTION_ATTRIBUTE auto reduce_expression(E const& expression)
{
    if constexpr (is_product_expression<remove_cvref_t<E>>::value)
    {
        return reduce_expression<Op>(expression.eval());
    }
    else if constexpr (contains_product<remove_cvref_t<E>>::value)
    {
        (void)expression_result_shape(expression);
        return reduce_expression<Op>(lower_products(expression));
    }
    else
    {
        return reduce_expression_impl<Op>(expression);
    }
}
}  // namespace detail

// Whole-expression reductions; no intermediate tensor is constructed.
template <typename E> VECTORIZATION_HOST_FUNCTION_ATTRIBUTE auto sum(E const& expression)
{
    return detail::reduce_expression<reduce_op::sum>(expression);
}

// accumulate is sum under its original name. hmin / hmax return max() / -max()
// for an empty input. On x86 CPU backends they skip NaN elements, as std::fmin /
// std::fmax do; on NEON / SVE a NaN in the vector body propagates, as before.
// Not noexcept: the GPU path can VECTORIZATION_CHECK / allocate.
template <typename E> VECTORIZATION_HOST_FUNCTION_ATTRIBUTE auto accumulate(E const& expression)
{
    return detail::reduce_expression<reduce_op::sum>(expression);
}

template <typename E> VECTORIZATION_HOST_FUNCTION_ATTRIBUTE auto hmin(E const& expression)
{
    return detail::reduce_expression<reduce_op::min>(expression);
}

template <typename E> VECTORIZATION_HOST_FUNCTION_ATTRIBUTE auto hmax(E const& expression)
{
    return detail::reduce_expression<reduce_op::max>(expression);
}

template <typename E> VECTORIZATION_HOST_FUNCTION_ATTRIBUTE auto prod(E const& expression)
{
    return detail::reduce_expression<reduce_op::product>(expression);
}

template <typename E> VECTORIZATION_HOST_FUNCTION_ATTRIBUTE auto mean(E const& expression)
{
    using T       = typename scalar_type<E, E>::value;
    const T total = sum(expression);  // also validates shapes and device placement for empty inputs
    return expression.size() == 0 ? std::numeric_limits<T>::quiet_NaN()
                                  : total / static_cast<T>(expression.size());
}

template <typename E> VECTORIZATION_HOST_FUNCTION_ATTRIBUTE bool all(E const& expression)
{
    return detail::reduce_expression<reduce_op::all>(expression) != 0;
}

template <typename E> VECTORIZATION_HOST_FUNCTION_ATTRIBUTE bool any(E const& expression)
{
    return detail::reduce_expression<reduce_op::any>(expression) != 0;
}

template <typename E>
VECTORIZATION_HOST_FUNCTION_ATTRIBUTE std::uint64_t count_nonzero(E const& expression)
{
    return detail::reduce_expression<reduce_op::count_nonzero>(expression);
}

template <typename E> VECTORIZATION_HOST_FUNCTION_ATTRIBUTE auto nansum(E const& expression)
{
    return detail::reduce_expression<reduce_op::nansum>(expression);
}

template <typename E> VECTORIZATION_HOST_FUNCTION_ATTRIBUTE auto nanmean(E const& expression)
{
    using T          = typename scalar_type<E, E>::value;
    const T    total = nansum(expression);
    const auto count = detail::reduce_expression<reduce_op::count_valid>(expression);
    return count == 0 ? std::numeric_limits<T>::quiet_NaN() : total / static_cast<T>(count);
}
}  // namespace vectorization
