/*
 * Vectorization: High-Performance Quantitative Library
 * SPDX-License-Identifier: GPL-3.0-or-later OR Commercial
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

#include "expressions/expression_interface.h"
#include "expressions/overlap.h"
#include "include/matrix_operation/multiplication.h"

namespace vectorization
{
namespace detail
{
template <typename T, bool = is_tensor<remove_cvref_t<T>>::value>
struct tensor_operand : std::false_type
{
    using value_type = void;
};

template <typename T> struct tensor_operand<T, true> : std::true_type
{
    using value_type = typename remove_cvref_t<T>::value_type;
};
}  // namespace detail

template <typename A> using operand_value_t = typename detail::tensor_operand<A>::value_type;

template <typename A, typename... Rest>
inline constexpr bool are_tensor_operands_v =
    detail::tensor_operand<A>::value && (detail::tensor_operand<Rest>::value && ...) &&
    (std::is_same_v<operand_value_t<A>, operand_value_t<Rest>> && ...);

template <typename C, bool = is_tensor<remove_cvref_t<C>>::value>
struct is_writable_tensor_operand : std::false_type
{
};

template <typename C>
struct is_writable_tensor_operand<C, true>
    : std::bool_constant<!std::is_const_v<std::remove_reference_t<C>> &&
                         !std::is_const_v<typename remove_cvref_t<C>::element_type>>
{
};

template <typename C>
inline constexpr bool is_writable_tensor_operand_v = is_writable_tensor_operand<C>::value;

namespace detail
{
struct product_shape
{
    expression_shape output;
    expression_shape batch;
    int64_t          rows{1};
    int64_t          columns{1};
    int64_t          depth{0};
    bool             left_vector{false};
    bool             right_vector{false};
};

template <typename A, typename B>
product_shape make_product_shape(A const& a, B const& b, bool strict_batch)
{
    VECTORIZATION_CHECK(a.rank > 0 && b.rank > 0, "matmul requires non-scalar operands");
    VECTORIZATION_CHECK(a.device() == b.device() && a.device_index() == b.device_index(),
        "matmul operands must have the same device and device index");
    if (strict_batch)
    {
        VECTORIZATION_CHECK(a.rank == 3 && b.rank == 3 && a.sizes[0] == b.sizes[0],
            "bmm requires rank-3 operands with equal batch sizes");
    }
    product_shape p;
    p.left_vector  = a.rank == 1;
    p.right_vector = b.rank == 1;
    p.rows         = p.left_vector ? 1 : a.sizes[a.rank - 2];
    p.columns      = p.right_vector ? 1 : b.sizes[b.rank - 1];
    p.depth        = a.sizes[a.rank - 1];
    VECTORIZATION_CHECK(p.depth == b.sizes[p.right_vector ? 0 : b.rank - 2],
        "matmul contraction dimensions do not match");
    VECTORIZATION_CHECK(broadcast_into(p.batch, std::max(a.rank - 2, 0), a.sizes.data()) &&
                            broadcast_into(p.batch, std::max(b.rank - 2, 0), b.sizes.data()),
        "matmul batch dimensions cannot broadcast");
    p.output = p.batch;
    if (!p.left_vector)
    {
        p.output.sizes[p.output.rank++] = p.rows;
    }
    if (!p.right_vector)
    {
        p.output.sizes[p.output.rank++] = p.columns;
    }
    size_t count = 1;
    for (int d = 0; d < p.output.rank; ++d)
    {
        const auto size = static_cast<size_t>(p.output.sizes[d]);
        VECTORIZATION_CHECK(size == 0 || count <= static_cast<size_t>(INT64_MAX) / size,
            "matmul output element count overflows int64_t");
        count *= size;
    }
    return p;
}

template <typename C> void check_product_output(C const& c, expression_shape const& shape)
{
    VECTORIZATION_CHECK(c.rank() == static_cast<size_t>(shape.rank),
        "matmul output rank does not match destination");
    for (int d = 0; d < shape.rank; ++d)
    {
        VECTORIZATION_CHECK(c.size(static_cast<size_t>(d)) == shape.sizes[d],
            "matmul output shape does not match destination");
    }
    VECTORIZATION_CHECK(c.is_non_overlapping(), "matmul destination has internal overlap");
}

template <typename R>
size_t product_batch_offset(
    R const& ref, expression_shape const& batch, size_t index, int matrix_rank = 2)
{
    const int batch_rank = std::max(ref.rank - matrix_rank, 0);
    size_t    offset     = 0;
    for (int d = batch.rank - 1; d >= 0; --d)
    {
        const auto size       = static_cast<size_t>(batch.sizes[d]);
        const auto coordinate = index % size;
        index /= size;
        const int source_dim = d - (batch.rank - batch_rank);
        if (source_dim >= 0 && ref.sizes[source_dim] != 1)
        {
            offset += coordinate * static_cast<size_t>(ref.strides[source_dim]);
        }
    }
    return offset;
}

// The current upstream API has GEMM (with alpha/beta) only. Vectors use promoted
// GEMM until U2 supplies GEMV/DOT; batches dispatch one GEMM per matrix without
// host staging.
#if LINALG_HAS_MKL
inline constexpr auto product_cpu_backend = ::linalg::linalg_backend::MKL;
#elif LINALG_HAS_BLAS
inline constexpr auto product_cpu_backend = ::linalg::linalg_backend::BLAS_LAPACKE;
#else
inline constexpr auto product_cpu_backend = ::linalg::linalg_backend::NATIVE;
#endif

template <typename T> struct product_matrix
{
    T const*                 data{nullptr};
    bool                     transpose{false};
    int64_t                  ld{1};
    std::optional<tensor<T>> packed;
};

template <typename T, typename R>
auto prepare_product_matrix(R const& ref,
    size_t                           offset,
    int64_t                          rows,
    int64_t                          columns,
    int64_t                          row_stride,
    int64_t                          column_stride)
{
    product_matrix<T> matrix;
    matrix.data = ref.data() + offset;
    if (column_stride == 1 && (rows == 1 || row_stride >= columns))
    {
        matrix.ld = std::max(row_stride, columns);
    }
    else if (row_stride == 1 && (columns == 1 || column_stride >= rows))
    {
        matrix.transpose = true;
        matrix.ld        = std::max(column_stride, rows);
    }
    else
    {
        matrix.packed.emplace(static_cast<size_t>(rows), static_cast<size_t>(columns));
        for (int64_t i = 0; i < rows; ++i)
        {
            for (int64_t j = 0; j < columns; ++j)
            {
                matrix.packed->data()[i * columns + j] =
                    matrix.data[i * row_stride + j * column_stride];
            }
        }
        matrix.data = matrix.packed->data();
        matrix.ld   = columns;
    }
    return matrix;
}

template <typename T>
void dispatch_product(product_matrix<T> const& a,
    product_matrix<T> const&                   b,
    T*                                         c,
    int64_t                                    m,
    int64_t                                    n,
    int64_t                                    k,
    int64_t                                    ldc,
    bool                                       transpose_output,
    T                                          alpha,
    T                                          beta)
{
    using kernel = ::linalg::multiplication<T, product_cpu_backend>;
    const auto required =
        kernel::workspace_size(transpose_output ? n : m, transpose_output ? m : n, k);
    // Memory owns aligned allocations. Cache scratch per host thread and dtype.
    thread_local memory::data_ptr<T> workspace;
    const size_t elements = required / sizeof(T) + (required % sizeof(T) != 0 ? 1 : 0);
    if (workspace.size() < elements)
    {
        workspace = memory::data_ptr<T>(elements, device_enum::CPU);
    }
    if (transpose_output)
    {
        kernel::apply(!b.transpose,
            !a.transpose,
            n,
            m,
            k,
            alpha,
            b.data,
            b.ld,
            a.data,
            a.ld,
            beta,
            c,
            ldc,
            workspace.data(),
            required);
    }
    else
    {
        kernel::apply(a.transpose,
            b.transpose,
            m,
            n,
            k,
            alpha,
            a.data,
            a.ld,
            b.data,
            b.ld,
            beta,
            c,
            ldc,
            workspace.data(),
            required);
    }
}
}  // namespace detail

// Host-only lazy node: retained descriptors preserve both temporary and lvalue
// allocations. Product nodes are materialized before raw/device lowering.
template <typename A, typename B> class product_expression
{
public:
    using value_type = typename A::value_type;
    static_assert(std::is_same_v<value_type, float> || std::is_same_v<value_type, double>,
        "matrix products support float and double");

    product_expression(A const& a, B const& b, bool strict_batch = false)
        : a_(store_operand(a)), b_(store_operand(b)),
          shape_(detail::make_product_shape(a_, b_, strict_batch))
    {
    }

    product_expression(product_expression const&)            = default;
    product_expression(product_expression&&) noexcept        = default;
    product_expression& operator=(product_expression const&) = delete;
    product_expression& operator=(product_expression&&)      = delete;

    auto const&             lhs() const { return a_; }
    auto const&             rhs() const { return b_; }
    auto const&             result_shape() const { return shape_.output; }
    size_t                  size() const noexcept { return shape_.output.numel(); }
    static constexpr size_t length() { return A::length(); }

    template <typename D> bool overlaps(D const& destination) const
    {
        return classify_overlap(destination, a_) != overlap_status::none ||
               classify_overlap(destination, b_) != overlap_status::none;
    }

    template <typename D> void check_noalias(D const& destination) const
    {
#ifndef NDEBUG
        VECTORIZATION_CHECK(!overlaps(destination), "noalias matrix product overlaps its inputs");
#else
        (void)destination;
#endif
    }

    auto eval() const
    {
        using result_type = tensor<value_type>;
        typename result_type::dimensions_type dims;
        for (int d = 0; d < shape_.output.rank; ++d)
        {
            dims.push_back(static_cast<size_t>(shape_.output.sizes[d]));
        }
        VECTORIZATION_CHECK(a_.device() == device_enum::CPU,
            "matrix product GPU adapters are not available in this build");
        result_type result(dims, device_enum::CPU, a_.device_index());
        evaluate_into(result);
        return result;
    }

    template <typename C>
    void evaluate_into(C& c,
        bool              noalias        = false,
        bool              reject_overlap = false,
        value_type        alpha          = value_type(1),
        value_type        beta           = value_type(0)) const
    {
        static_assert(std::is_same_v<typename C::value_type, value_type>,
            "matrix product destination element type must match its inputs");
        detail::check_product_output(c, shape_.output);
        VECTORIZATION_CHECK(a_.device() == device_enum::CPU && c.device() == device_enum::CPU &&
                                a_.device_index() == c.device_index(),
            "matrix products currently require CPU operands and destination on the same device "
            "index");
        const auto out = c.to_strided_ref();
        if (noalias)
        {
            check_noalias(out);
        }
        const bool overlap = !noalias && overlaps(out);
        VECTORIZATION_CHECK(!reject_overlap || !overlap, "matmul_out overlaps its inputs");
        if (c.size() == 0)
        {
            return;
        }
        if (alpha == value_type(0) || shape_.depth == 0)
        {
            if (beta == value_type(0))
            {
                c = value_type(0);
            }
            else
            {
                c.assign(c * beta);
            }
            return;
        }
        const int     batch_rank = shape_.batch.rank;
        const int64_t row_stride = shape_.left_vector ? shape_.columns : out.strides[batch_rank];
        const int64_t column_stride =
            shape_.right_vector ? 1 : out.strides[batch_rank + (shape_.left_vector ? 0 : 1)];
        const bool row_major =
            column_stride == 1 && (shape_.rows == 1 || row_stride >= shape_.columns);
        const bool column_major =
            row_stride == 1 && (shape_.columns == 1 || column_stride >= shape_.rows);
        if (overlap || (!row_major && !column_major))
        {
            // GEMM cannot write c in place: stage the product, then apply the
            // epilogue elementwise. c is not read when beta is zero.
            auto result = eval();
            if (beta != value_type(0))
            {
                c.assign(result * alpha + c * beta);
            }
            else if (alpha != value_type(1))
            {
                c.assign(result * alpha);
            }
            else
            {
                c.assign(result.to_strided_ref());
            }
            return;
        }
        const auto a_row             = shape_.left_vector ? shape_.depth : a_.strides[a_.rank - 2];
        const auto a_col             = a_.strides[a_.rank - 1];
        const auto b_row             = b_.strides[shape_.right_vector ? 0 : b_.rank - 2];
        const auto b_col             = shape_.right_vector ? 1 : b_.strides[b_.rank - 1];
        const int output_matrix_rank = (shape_.left_vector ? 0 : 1) + (shape_.right_vector ? 0 : 1);
        for (size_t batch = 0; batch < shape_.batch.numel(); ++batch)
        {
            auto  a           = detail::prepare_product_matrix<value_type>(a_,
                detail::product_batch_offset(a_, shape_.batch, batch),
                shape_.rows,
                shape_.depth,
                a_row,
                a_col);
            auto  b           = detail::prepare_product_matrix<value_type>(b_,
                detail::product_batch_offset(b_, shape_.batch, batch),
                shape_.depth,
                shape_.columns,
                b_row,
                b_col);
            auto* destination = c.data() + detail::product_batch_offset(
                                               out, shape_.batch, batch, output_matrix_rank);
            detail::dispatch_product(a,
                b,
                destination,
                shape_.rows,
                shape_.columns,
                shape_.depth,
                row_major ? std::max(row_stride, shape_.columns)
                          : std::max(column_stride, shape_.rows),
                !row_major,
                alpha,
                beta);
        }
    }

private:
    stored_operand_t<A>   a_;
    stored_operand_t<B>   b_;
    detail::product_shape shape_;
};

template <typename A, typename B, std::enable_if_t<are_tensor_operands_v<A, B>, bool> = true>
auto matmul(A&& a, B&& b)
{
    return product_expression<remove_cvref_t<A>, remove_cvref_t<B>>(a, b);
}

template <typename A, typename B, std::enable_if_t<are_tensor_operands_v<A, B>, bool> = true>
auto bmm(A&& a, B&& b)
{
    return product_expression<remove_cvref_t<A>, remove_cvref_t<B>>(a, b, true);
}

template <typename C,
    typename A,
    typename B,
    std::enable_if_t<are_tensor_operands_v<C, A, B> && is_writable_tensor_operand_v<C>, bool> =
        true>
void matmul_out(C&&    c,
    A const&           a,
    B const&           b,
    operand_value_t<A> alpha = operand_value_t<A>(1),
    operand_value_t<A> beta  = operand_value_t<A>(0))
{
    matmul(a, b).evaluate_into(c, false, true, alpha, beta);
}

template <typename A, typename B, std::enable_if_t<are_tensor_operands_v<A, B>, bool> = true>
auto dot(A const& a, B const& b)
{
    VECTORIZATION_CHECK(a.rank() == 1 && b.rank() == 1, "dot requires two vectors");
    auto result = matmul(a, b).eval();
    return result[0];
}

namespace linalg
{
using vectorization::bmm;
using vectorization::dot;
using vectorization::matmul;
using vectorization::matmul_out;
}  // namespace linalg

struct add_evaluator;
struct sub_evaluator;
struct mul_evaluator;
struct madd_evaluator;
struct msub_evaluator;

namespace detail
{
// Product epilogues (design plan section 11): assignments of the form
// c = alpha * matmul(a, b) +/- beta * d, with d a tensor leaf, lower to one
// GEMM with alpha/beta instead of a product temporary plus an elementwise pass.
// Like Eigen's scalar * product recognition, matching is purely structural;
// whether c can be written directly is decided by the overlap classifier at
// run time. Terms match on the *stored* node types, so lvalue and rvalue
// operands and nested scalar factors (2 * (3 * p)) are handled alike.
enum class gemm_term_kind : uint8_t
{
    none,
    product,
    leaf
};

template <typename E> struct gemm_term
{
    static constexpr gemm_term_kind kind = gemm_term_kind::none;
};

template <typename A, typename B> struct gemm_term<product_expression<A, B>>
{
    static constexpr gemm_term_kind kind = gemm_term_kind::product;
    template <typename V> static V  scale(product_expression<A, B> const&) { return V(1); }
    static auto const&              node(product_expression<A, B> const& e) { return e; }
};

template <typename T> struct gemm_term<retained_ref<T>>
{
    static constexpr gemm_term_kind kind = gemm_term_kind::leaf;
    template <typename V> static V  scale(retained_ref<T> const&) { return V(1); }
    static auto const&              node(retained_ref<T> const& e) { return e; }
};

template <typename L, typename R> struct gemm_term<binary_expression<L, R, mul_evaluator>>
{
    using expression_type            = binary_expression<L, R, mul_evaluator>;
    using stored_lhs                 = remove_cvref_t<stored_operand_t<remove_cvref_t<L>>>;
    using stored_rhs                 = remove_cvref_t<stored_operand_t<remove_cvref_t<R>>>;
    static constexpr bool scalar_lhs = std::is_arithmetic_v<stored_lhs>;
    static constexpr bool scalar_rhs = std::is_arithmetic_v<stored_rhs>;
    using inner = std::conditional_t<scalar_lhs, gemm_term<stored_rhs>, gemm_term<stored_lhs>>;
    static constexpr gemm_term_kind kind =
        scalar_lhs != scalar_rhs ? inner::kind : gemm_term_kind::none;

    template <typename V> static V scale(expression_type const& e)
    {
        if constexpr (scalar_lhs)
        {
            return static_cast<V>(e.lhs()) * inner::template scale<V>(e.rhs());
        }
        else
        {
            return inner::template scale<V>(e.lhs()) * static_cast<V>(e.rhs());
        }
    }
    static auto const& node(expression_type const& e)
    {
        if constexpr (scalar_lhs)
        {
            return inner::node(e.rhs());
        }
        else
        {
            return inner::node(e.lhs());
        }
    }
};

template <typename Ev> inline constexpr int gemm_sum_sign                 = 0;
template <> inline constexpr int            gemm_sum_sign<add_evaluator>  = 1;
template <> inline constexpr int            gemm_sum_sign<madd_evaluator> = 1;
template <> inline constexpr int            gemm_sum_sign<sub_evaluator>  = -1;
template <> inline constexpr int            gemm_sum_sign<msub_evaluator> = -1;

// A matched epilogue: product(), alpha(), and for sums leaf() and beta().
template <typename E> struct gemm_epilogue
{
    static constexpr bool value = false;
};

template <typename L, typename R, typename Ev> struct gemm_epilogue<binary_expression<L, R, Ev>>
{
    using expression_type = binary_expression<L, R, Ev>;
    using stored_lhs      = remove_cvref_t<stored_operand_t<remove_cvref_t<L>>>;
    using stored_rhs      = remove_cvref_t<stored_operand_t<remove_cvref_t<R>>>;
    using scaled          = gemm_term<expression_type>;
    using lhs_term        = gemm_term<stored_lhs>;
    using rhs_term        = gemm_term<stored_rhs>;

    static constexpr int  sign      = gemm_sum_sign<Ev>;
    static constexpr bool is_scaled = scaled::kind == gemm_term_kind::product;
    static constexpr bool product_on_left =
        lhs_term::kind == gemm_term_kind::product && rhs_term::kind == gemm_term_kind::leaf;
    static constexpr bool product_on_right =
        lhs_term::kind == gemm_term_kind::leaf && rhs_term::kind == gemm_term_kind::product;
    static constexpr bool is_sum   = sign != 0 && (product_on_left || product_on_right);
    static constexpr bool value    = is_scaled || is_sum;
    static constexpr bool has_leaf = is_sum;

    using product_term = std::
        conditional_t<is_scaled, scaled, std::conditional_t<product_on_left, lhs_term, rhs_term>>;
    using leaf_term = std::conditional_t<product_on_left, rhs_term, lhs_term>;

    static auto const& product(expression_type const& e)
    {
        if constexpr (is_scaled)
        {
            return scaled::node(e);
        }
        else if constexpr (product_on_left)
        {
            return lhs_term::node(e.lhs());
        }
        else
        {
            return rhs_term::node(e.rhs());
        }
    }
    template <typename V> static V alpha(expression_type const& e)
    {
        if constexpr (is_scaled)
        {
            return scaled::template scale<V>(e);
        }
        else if constexpr (product_on_left)
        {
            return lhs_term::template scale<V>(e.lhs());
        }
        else
        {
            return static_cast<V>(sign) * rhs_term::template scale<V>(e.rhs());
        }
    }
    static auto const& leaf(expression_type const& e)
    {
        if constexpr (product_on_left)
        {
            return rhs_term::node(e.rhs());
        }
        else
        {
            return lhs_term::node(e.lhs());
        }
    }
    template <typename V> static V beta(expression_type const& e)
    {
        if constexpr (product_on_left)
        {
            return static_cast<V>(sign) * rhs_term::template scale<V>(e.rhs());
        }
        else
        {
            return lhs_term::template scale<V>(e.lhs());
        }
    }
};

template <typename E>
inline constexpr bool is_gemm_epilogue_v = gemm_epilogue<remove_cvref_t<E>>::value;

// Lowers a matched epilogue into c. Returns false, before any write, when the
// shapes or aliasing need the general path (product temporary, then one
// elementwise pass). d fully aliasing c becomes beta * c with no copy; a
// distinct d is copied into c first, so c must not overlap a, b or d.
template <typename C, typename E> bool lower_gemm_epilogue(C& c, E const& expr, bool noalias)
{
    using match         = gemm_epilogue<remove_cvref_t<E>>;
    auto const& product = match::product(expr);
    using value_type    = typename remove_cvref_t<decltype(product)>::value_type;
    const auto& shape   = product.result_shape();
    if (c.rank() != static_cast<size_t>(shape.rank))
    {
        return false;  // product broadcasts against d: not a single GEMM
    }
    for (int d = 0; d < shape.rank; ++d)
    {
        if (c.size(static_cast<size_t>(d)) != shape.sizes[d])
        {
            return false;
        }
    }
    const auto alpha = match::template alpha<value_type>(expr);
    if constexpr (!match::has_leaf)
    {
        product.evaluate_into(c, noalias, false, alpha, value_type(0));
        return true;
    }
    else
    {
        auto const& leaf   = match::leaf(expr);
        const auto  beta   = match::template beta<value_type>(expr);
        const auto  out    = c.to_strided_ref();
        const auto  status = classify_overlap(out, leaf);
        if (status == overlap_status::full)
        {
            product.evaluate_into(c, noalias, false, alpha, beta);
            return true;
        }
        if (status != overlap_status::none || (!noalias && product.overlaps(out)))
        {
            return false;
        }
        if (noalias)
        {
            product.check_noalias(out);
        }
        // Assignment does not broadcast; expand d (e.g. a bias row) to c's shape
        // with stride-0 axes, as tensor_view::expand does.
        auto source =
            static_cast<strided_ref<std::remove_pointer_t<decltype(leaf.data())>> const&>(leaf);
        const int offset = shape.rank - source.rank;
        if (offset < 0)
        {
            return false;
        }
        bool reshaped = offset != 0;
        for (int d = shape.rank - 1; d >= 0; --d)
        {
            const int     from   = d - offset;
            const int64_t size   = from >= 0 ? source.sizes[from] : 1;
            const int64_t stride = from >= 0 ? source.strides[from] : 0;
            if (size != shape.sizes[d] && size != 1)
            {
                return false;
            }
            reshaped          = reshaped || size != shape.sizes[d];
            source.sizes[d]   = shape.sizes[d];
            source.strides[d] = size == shape.sizes[d] ? stride : 0;
        }
        if (reshaped)
        {
            source.rank        = shape.rank;
            source.numel       = shape.numel();
            source.contiguous  = false;
            source.align_start = 0;
            source.align_end   = 0;
        }
        c.assign(source, true);
        product.evaluate_into(c, true, false, alpha, beta);
        return true;
    }
}

template <typename E> struct product_lowering;
template <typename E> auto lower_products(E const& expression)
{
    return product_lowering<remove_cvref_t<E>>::apply(expression);
}

template <typename E> struct product_lowering
{
    static auto apply(E const& expression) { return store_operand(expression); }
};

template <typename A, typename B> struct product_lowering<product_expression<A, B>>
{
    static auto apply(product_expression<A, B> const& expression)
    {
        auto result = expression.eval();
        return store_operand(result);
    }
};

template <typename L, typename Ev> struct product_lowering<unary_expression<L, Ev>>
{
    static auto apply(unary_expression<L, Ev> const& expression)
    {
        auto rhs = lower_products(expression.rhs());
        return unary_expression<decltype(rhs), Ev>(std::move(rhs));
    }
};

template <typename L, typename R, typename Ev> struct product_lowering<binary_expression<L, R, Ev>>
{
    static auto apply(binary_expression<L, R, Ev> const& expression)
    {
        auto lhs = lower_products(expression.lhs());
        auto rhs = lower_products(expression.rhs());
        return binary_expression<decltype(lhs), decltype(rhs), Ev>(std::move(lhs), std::move(rhs));
    }
};

template <typename L, typename M, typename R, typename Ev>
struct product_lowering<trinary_expression<L, M, R, Ev>>
{
    static auto apply(trinary_expression<L, M, R, Ev> const& expression)
    {
        auto lhs = lower_products(expression.lhs());
        auto mhs = lower_products(expression.mhs());
        auto rhs = lower_products(expression.rhs());
        return trinary_expression<decltype(lhs), decltype(mhs), decltype(rhs), Ev>(
            std::move(lhs), std::move(mhs), std::move(rhs));
    }
};
}  // namespace detail
}  // namespace vectorization
