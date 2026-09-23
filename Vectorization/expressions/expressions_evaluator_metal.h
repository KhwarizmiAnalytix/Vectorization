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

// Metal expression evaluation — the counterpart to expressions_evaluator_gpu.h for
// CUDA/HIP, but architecturally different: nvcc/hipcc compile the same templated C++
// header for host AND device, so run_gpu<E,T> can instantiate one fused kernel per
// expression-tree type E. Metal's shader compiler (MSL) is a separate toolchain with no
// such unified pass — .metal source never sees tensor<T>/the expression templates at
// all (see backend/gpu/metal/kernels.metal).
//
// Fusion: this file walks the expression tree on the host and emits one MSL kernel
// whose body is the whole tree (`out[tid] = y[tid] + a[tid] + 5.0 * d[tid]`). That
// source is compiled once via newLibraryWithSource: and cached, then dispatched as a
// single kernel — the same fused model as run_gpu, without per-node temps. Element-wise
// in-place (`a = a + b`) is safe because each thread reads and writes only index tid.
// There is no unfused per-node lowering: an unsupported op or a tree that exceeds the
// Metal buffer limit throws.
//
// Supported fused ops: the full expression-template set except cdf/inv_cdf (MSL has
// no erf/erfinv). Arithmetic, comparisons, if_else, min/max/pow/hypot/copysign, and
// the metal_stdlib unaries all emit into one kernel. Scalar fill uses dispatch_fill.
// kernels.metal still holds fill/reduce and the named kernels used by
// metal_backend::dispatch() tests.
//
// float-only: MSL has no double type on any Apple GPU. memory::allocator<double> already
// throws at allocation time for device_enum::METAL (see Library/Memory/allocator.h), so
// a tensor<double> with device()==METAL can never exist to reach this code — no
// additional guard is needed here.
//
// Compiled only when VECTORIZATION_HAS_METAL=1.

#if VECTORIZATION_HAS_METAL

#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "allocator.h"
#include "backend/gpu/metal/metal_dispatch.h"
#include "common/data_view.h"
#include "common/device.h"
#include "expressions/expression_interface.h"
#include "expressions/reduce_op.h"

namespace vectorization
{
namespace metal_detail
{
using metal_alloc_t = memory::allocator<float>;

// Maps an expression-functor evaluator type (add_evaluator, sin_evaluator, ...) to the
// MSL function/operator name used in the fused kernel body.
//
// This is deliberately NOT a compile-time rejection (e.g. static_assert in the primary
// template) for unsupported ops: expressions_evaluator::run<E,T> is instantiated for
// every expression type E ever used with T=tensor<float> ANYWHERE in the codebase,
// regardless of which device a given tensor actually uses at runtime (the device()
// check inside run() is a runtime `if`, not `if constexpr`) — e.g. plain CPU code doing
// `out = cdf(a);` on tensor<float> would force-instantiate metal_kernel_traits<
// cdf_evaluator> even though no METAL tensor is involved. A hard compile-time error
// here would therefore break unrelated CPU-only code using an op outside the Metal
// set. metal_expr_fusable is false for those ops, and run_metal throws
// std::runtime_error only if genuinely reached at runtime with a METAL-device tensor.
template <typename EVALUATOR>
struct metal_kernel_traits
{
    static constexpr bool        is_supported = false;
    static constexpr const char* name         = nullptr;
};

#define VECTORIZATION_METAL_KERNEL(EVAL, NAME)            \
    template <>                                           \
    struct metal_kernel_traits<EVAL>                      \
    {                                                     \
        static constexpr bool        is_supported = true; \
        static constexpr const char* name         = NAME; \
    };

VECTORIZATION_METAL_KERNEL(add_evaluator, "+")
VECTORIZATION_METAL_KERNEL(sub_evaluator, "-")
VECTORIZATION_METAL_KERNEL(mul_evaluator, "*")
VECTORIZATION_METAL_KERNEL(div_evaluator, "/")
VECTORIZATION_METAL_KERNEL(madd_evaluator, "+")
VECTORIZATION_METAL_KERNEL(msub_evaluator, "-")
VECTORIZATION_METAL_KERNEL(mmul_evaluator, "*")
VECTORIZATION_METAL_KERNEL(mdiv_evaluator, "/")
VECTORIZATION_METAL_KERNEL(min_evaluator, "min")
VECTORIZATION_METAL_KERNEL(max_evaluator, "max")
VECTORIZATION_METAL_KERNEL(pow_evaluator, "pow")
VECTORIZATION_METAL_KERNEL(hypot_evaluator, "hypot")
VECTORIZATION_METAL_KERNEL(copysign_evaluator, "copysign")
VECTORIZATION_METAL_KERNEL(cmpgt_evaluator, ">")
VECTORIZATION_METAL_KERNEL(cmplt_evaluator, "<")
VECTORIZATION_METAL_KERNEL(cmpge_evaluator, ">=")
VECTORIZATION_METAL_KERNEL(cmple_evaluator, "<=")
VECTORIZATION_METAL_KERNEL(cmpeq_evaluator, "==")
VECTORIZATION_METAL_KERNEL(cmpne_evaluator, "!=")
#if VECTORIZATION_VECTORIZED
VECTORIZATION_METAL_KERNEL(land_evaluator, "&&")
VECTORIZATION_METAL_KERNEL(lor_evaluator, "||")
VECTORIZATION_METAL_KERNEL(lxor_evaluator, "!=")
#endif
VECTORIZATION_METAL_KERNEL(fma_evaluator, "fma")
VECTORIZATION_METAL_KERNEL(if_else_evaluator, "select")
VECTORIZATION_METAL_KERNEL(fabs_evaluator, "fabs")
VECTORIZATION_METAL_KERNEL(floor_evaluator, "floor")
VECTORIZATION_METAL_KERNEL(ceil_evaluator, "ceil")
VECTORIZATION_METAL_KERNEL(trunc_evaluator, "trunc")
VECTORIZATION_METAL_KERNEL(sqrt_evaluator, "sqrt")
VECTORIZATION_METAL_KERNEL(sqr_evaluator, "sqr")
VECTORIZATION_METAL_KERNEL(exp_evaluator, "exp")
VECTORIZATION_METAL_KERNEL(expm1_evaluator, "expm1")
VECTORIZATION_METAL_KERNEL(exp2_evaluator, "exp2")
VECTORIZATION_METAL_KERNEL(log_evaluator, "log")
VECTORIZATION_METAL_KERNEL(log1p_evaluator, "log1p")
VECTORIZATION_METAL_KERNEL(log2_evaluator, "log2")
VECTORIZATION_METAL_KERNEL(log10_evaluator, "log10")
VECTORIZATION_METAL_KERNEL(sin_evaluator, "sin")
VECTORIZATION_METAL_KERNEL(cos_evaluator, "cos")
VECTORIZATION_METAL_KERNEL(tan_evaluator, "tan")
VECTORIZATION_METAL_KERNEL(asin_evaluator, "asin")
VECTORIZATION_METAL_KERNEL(acos_evaluator, "acos")
VECTORIZATION_METAL_KERNEL(atan_evaluator, "atan")
VECTORIZATION_METAL_KERNEL(sinh_evaluator, "sinh")
VECTORIZATION_METAL_KERNEL(cosh_evaluator, "cosh")
VECTORIZATION_METAL_KERNEL(tanh_evaluator, "tanh")
VECTORIZATION_METAL_KERNEL(asinh_evaluator, "asinh")
VECTORIZATION_METAL_KERNEL(acosh_evaluator, "acosh")
VECTORIZATION_METAL_KERNEL(atanh_evaluator, "atanh")
VECTORIZATION_METAL_KERNEL(cbrt_evaluator, "cbrt")
VECTORIZATION_METAL_KERNEL(invsqrt_evaluator, "rsqrt")
VECTORIZATION_METAL_KERNEL(neg_evaluator, "neg")
VECTORIZATION_METAL_KERNEL(lnot_evaluator, "lnot")

#undef VECTORIZATION_METAL_KERNEL

template <typename T, typename = void>
struct metal_expr_fusable : std::false_type
{
};

template <typename V>
struct metal_expr_fusable<memory::data_view<V>>
    : std::bool_constant<std::is_same_v<std::remove_cv_t<V>, float>>
{
};

template <typename V>
struct metal_expr_fusable<tensor<V>> : std::bool_constant<std::is_same_v<V, float>>
{
};

template <typename S>
struct metal_expr_fusable<S, std::enable_if_t<std::is_fundamental<S>::value>> : std::true_type
{
};

template <typename L, typename Ev>
struct metal_expr_fusable<unary_expression<L, Ev>>
    : std::bool_constant<
          metal_kernel_traits<Ev>::is_supported &&
          metal_expr_fusable<stored_operand_t<vectorization::remove_cvref_t<L>>>::value>
{
};

template <typename L, typename R, typename Ev>
struct metal_expr_fusable<binary_expression<L, R, Ev>>
    : std::bool_constant<
          metal_kernel_traits<Ev>::is_supported &&
          metal_expr_fusable<stored_operand_t<vectorization::remove_cvref_t<L>>>::value &&
          metal_expr_fusable<stored_operand_t<vectorization::remove_cvref_t<R>>>::value>
{
};

template <typename L, typename M, typename R, typename Ev>
struct metal_expr_fusable<trinary_expression<L, M, R, Ev>>
    : std::bool_constant<
          metal_kernel_traits<Ev>::is_supported &&
          metal_expr_fusable<stored_operand_t<vectorization::remove_cvref_t<L>>>::value &&
          metal_expr_fusable<stored_operand_t<vectorization::remove_cvref_t<M>>>::value &&
          metal_expr_fusable<stored_operand_t<vectorization::remove_cvref_t<R>>>::value>
{
};

// Host-side plan for one fused MSL kernel. Pointers/scalars are runtime;
// `body` is the MSL expression (`in0[tid]+c0*in1[tid]`).
struct metal_fuse_state
{
    static constexpr int kMaxSlots = 29;
    void const*          buffers[kMaxSlots]{};
    float                scalars[kMaxSlots]{};
    int                  n_buffers{0};
    int                  n_scalars{0};
    std::string          body;
    bool                 ok{true};
};

inline void metal_fuse_fail(metal_fuse_state& st)
{
    st.ok = false;
}

template <typename Value>
void metal_fuse_emit(memory::data_view<Value> const& t, metal_fuse_state& st)
{
    if (!st.ok || st.n_buffers >= metal_fuse_state::kMaxSlots)
    {
        metal_fuse_fail(st);
        return;
    }
    VECTORIZATION_CHECK_DEBUG(
        t.device() == memory::device_enum::METAL,
        "Metal expression mixes a non-METAL tensor operand");
    st.body += "in";
    st.body += std::to_string(st.n_buffers);
    st.body += "[tid]";
    st.buffers[st.n_buffers++] = t.data();
}

template <typename Value>
void metal_fuse_emit(tensor<Value> const& t, metal_fuse_state& st)
{
    if (!st.ok || st.n_buffers >= metal_fuse_state::kMaxSlots)
    {
        metal_fuse_fail(st);
        return;
    }
    VECTORIZATION_CHECK_DEBUG(
        t.device() == memory::device_enum::METAL,
        "Metal expression mixes a non-METAL tensor operand");
    st.body += "in";
    st.body += std::to_string(st.n_buffers);
    st.body += "[tid]";
    st.buffers[st.n_buffers++] = t.data();
}

template <typename S, std::enable_if_t<std::is_fundamental<S>::value, bool> = true>
void metal_fuse_emit(S value, metal_fuse_state& st)
{
    if (!st.ok || st.n_scalars >= metal_fuse_state::kMaxSlots)
    {
        metal_fuse_fail(st);
        return;
    }
    st.body += "c";
    st.body += std::to_string(st.n_scalars);
    st.scalars[st.n_scalars++] = static_cast<float>(value);
}

template <typename LHS, typename EVALUATOR>
void metal_fuse_emit(unary_expression<LHS, EVALUATOR> const& e, metal_fuse_state& st);
template <typename LHS, typename RHS, typename EVALUATOR>
void metal_fuse_emit(binary_expression<LHS, RHS, EVALUATOR> const& e, metal_fuse_state& st);
template <typename LHS, typename MHS, typename RHS, typename EVALUATOR>
void metal_fuse_emit(trinary_expression<LHS, MHS, RHS, EVALUATOR> const& e, metal_fuse_state& st);

template <typename E>
std::string metal_fuse_snapshot(E const& e, metal_fuse_state& st)
{
    std::size_t const mark = st.body.size();
    metal_fuse_emit(e, st);
    std::string piece = st.body.substr(mark);
    st.body.resize(mark);
    return piece;
}

template <typename LHS, typename EVALUATOR>
void metal_fuse_emit(unary_expression<LHS, EVALUATOR> const& e, metal_fuse_state& st)
{
    if constexpr (!metal_kernel_traits<EVALUATOR>::is_supported)
    {
        metal_fuse_fail(st);
        return;
    }
    if constexpr (std::is_same_v<EVALUATOR, neg_evaluator>)
    {
        st.body += "(-(";
        metal_fuse_emit(e.rhs(), st);
        st.body += "))";
    }
    else if constexpr (std::is_same_v<EVALUATOR, lnot_evaluator>)
    {
        st.body += "float(!bool(";
        metal_fuse_emit(e.rhs(), st);
        st.body += "))";
    }
    else if constexpr (std::is_same_v<EVALUATOR, sqr_evaluator>)
    {
        std::string const x = metal_fuse_snapshot(e.rhs(), st);
        st.body += '(';
        st.body += x;
        st.body += '*';
        st.body += x;
        st.body += ')';
    }
    else if constexpr (std::is_same_v<EVALUATOR, expm1_evaluator>)
    {
        st.body += "(exp(";
        metal_fuse_emit(e.rhs(), st);
        st.body += ")-1.0)";
    }
    else if constexpr (std::is_same_v<EVALUATOR, log1p_evaluator>)
    {
        st.body += "log(1.0+(";
        metal_fuse_emit(e.rhs(), st);
        st.body += "))";
    }
    else if constexpr (std::is_same_v<EVALUATOR, cbrt_evaluator>)
    {
        std::string const x = metal_fuse_snapshot(e.rhs(), st);
        st.body += "copysign(pow(fabs(";
        st.body += x;
        st.body += "),0.3333333333333333),";
        st.body += x;
        st.body += ')';
    }
    else
    {
        st.body += metal_kernel_traits<EVALUATOR>::name;
        st.body += "(";
        metal_fuse_emit(e.rhs(), st);
        st.body += ")";
    }
}

template <typename LHS, typename RHS, typename EVALUATOR>
void metal_fuse_emit(binary_expression<LHS, RHS, EVALUATOR> const& e, metal_fuse_state& st)
{
    if constexpr (!metal_kernel_traits<EVALUATOR>::is_supported)
    {
        metal_fuse_fail(st);
        return;
    }
    constexpr bool arith =
        std::is_same_v<EVALUATOR, add_evaluator> || std::is_same_v<EVALUATOR, madd_evaluator> ||
        std::is_same_v<EVALUATOR, sub_evaluator> || std::is_same_v<EVALUATOR, msub_evaluator> ||
        std::is_same_v<EVALUATOR, mul_evaluator> || std::is_same_v<EVALUATOR, mmul_evaluator> ||
        std::is_same_v<EVALUATOR, div_evaluator> || std::is_same_v<EVALUATOR, mdiv_evaluator>;
    constexpr bool cmp =
        std::is_same_v<EVALUATOR, cmpgt_evaluator> || std::is_same_v<EVALUATOR, cmplt_evaluator> ||
        std::is_same_v<EVALUATOR, cmpge_evaluator> || std::is_same_v<EVALUATOR, cmple_evaluator> ||
        std::is_same_v<EVALUATOR, cmpeq_evaluator> || std::is_same_v<EVALUATOR, cmpne_evaluator>;
#if VECTORIZATION_VECTORIZED
    constexpr bool logic = std::is_same_v<EVALUATOR, land_evaluator> ||
                           std::is_same_v<EVALUATOR, lor_evaluator> ||
                           std::is_same_v<EVALUATOR, lxor_evaluator>;
#else
    constexpr bool logic = false;
#endif
    if constexpr (arith || cmp)
    {
        if constexpr (cmp)
        {
            st.body += "float(";
        }
        else
        {
            st.body += '(';
        }
        metal_fuse_emit(e.lhs(), st);
        st.body += metal_kernel_traits<EVALUATOR>::name;
        metal_fuse_emit(e.rhs(), st);
        st.body += ')';
    }
    else if constexpr (logic)
    {
        st.body += "float(bool(";
        metal_fuse_emit(e.lhs(), st);
        st.body += ')';
        st.body += metal_kernel_traits<EVALUATOR>::name;
        st.body += "bool(";
        metal_fuse_emit(e.rhs(), st);
        st.body += "))";
    }
    else if constexpr (std::is_same_v<EVALUATOR, hypot_evaluator>)
    {
        std::string const a = metal_fuse_snapshot(e.lhs(), st);
        std::string const b = metal_fuse_snapshot(e.rhs(), st);
        st.body += "sqrt((";
        st.body += a;
        st.body += '*';
        st.body += a;
        st.body += ")+(";
        st.body += b;
        st.body += '*';
        st.body += b;
        st.body += "))";
    }
    else
    {
        st.body += metal_kernel_traits<EVALUATOR>::name;
        st.body += '(';
        metal_fuse_emit(e.lhs(), st);
        st.body += ',';
        metal_fuse_emit(e.rhs(), st);
        st.body += ')';
    }
}

template <typename LHS, typename MHS, typename RHS, typename EVALUATOR>
void metal_fuse_emit(trinary_expression<LHS, MHS, RHS, EVALUATOR> const& e, metal_fuse_state& st)
{
    if constexpr (!metal_kernel_traits<EVALUATOR>::is_supported)
    {
        metal_fuse_fail(st);
        return;
    }
    if constexpr (std::is_same_v<EVALUATOR, fma_evaluator>)
    {
        st.body += "fma(";
        metal_fuse_emit(e.lhs(), st);
        st.body += ',';
        metal_fuse_emit(e.mhs(), st);
        st.body += ',';
        metal_fuse_emit(e.rhs(), st);
        st.body += ')';
    }
    else if constexpr (std::is_same_v<EVALUATOR, if_else_evaluator>)
    {
        // MSL select(falseVal, trueVal, cond) — if_else(mask, true, false).
        st.body += "select(";
        metal_fuse_emit(e.rhs(), st);
        st.body += ',';
        metal_fuse_emit(e.mhs(), st);
        st.body += ",bool(";
        metal_fuse_emit(e.lhs(), st);
        st.body += "))";
    }
    else
    {
        metal_fuse_fail(st);
    }
}

inline std::string metal_fuse_source(metal_fuse_state const& st)
{
    std::string src = "#include <metal_stdlib>\nusing namespace metal;\nkernel void fused_float(\n";
    int         idx = 0;
    for (int i = 0; i < st.n_buffers; ++i)
    {
        src += "  device const float* in";
        src += std::to_string(i);
        src += " [[buffer(";
        src += std::to_string(idx++);
        src += ")]],\n";
    }
    for (int j = 0; j < st.n_scalars; ++j)
    {
        src += "  constant float& c";
        src += std::to_string(j);
        src += " [[buffer(";
        src += std::to_string(idx++);
        src += ")]],\n";
    }
    src += "  device float* out [[buffer(";
    src += std::to_string(idx++);
    src += ")]],\n  constant uint& n [[buffer(";
    src += std::to_string(idx);
    src += ")]],\n  uint tid [[thread_position_in_grid]])\n{\n  if (tid < n)\n    out[tid] = ";
    src += st.body;
    src += ";\n}\n";
    return src;
}

inline bool metal_fuse_fits(metal_fuse_state const& st)
{
    return st.ok && (st.n_buffers + st.n_scalars + 2) <= 31;
}

}  // namespace metal_detail

// ---------------------------------------------------------------------------
// Root entry points, called from expressions_evaluator.h — mirrors run_gpu/fill_gpu's
// signatures (expressions_evaluator_gpu.h) so the two backends plug into run()/fill()
// the same way.
// ---------------------------------------------------------------------------

template <typename E, typename T>
void run_metal(E const& expr, T& rhs)
{
    using RE            = vectorization::remove_cvref_t<E>;
    const std::size_t n = rhs.size();

    if constexpr (vectorization::is_base_expression<RE>::value)
    {
        if (expr.data() != rhs.data())
        {
            metal_detail::metal_alloc_t::copy(
                expr.data(), n, rhs.data(), memory::device_enum::METAL, memory::device_enum::METAL);
        }
        return;
    }

    static_assert(
        vectorization::is_pure_expression<RE>::value,
        "run_metal: expression is neither a tensor leaf nor a unary/binary/trinary node");

    if constexpr (metal_detail::metal_expr_fusable<RE>::value)
    {
        metal_detail::metal_fuse_state st;
        metal_detail::metal_fuse_emit(expr, st);
        if (!metal_detail::metal_fuse_fits(st))
        {
            throw std::runtime_error("Metal fused kernel exceeds the device buffer limit");
        }
        std::string const src = metal_detail::metal_fuse_source(st);
        metal_backend::dispatch_fused(
            src, st.buffers, st.n_buffers, st.scalars, st.n_scalars, rhs.data(), n);
    }
    else
    {
        throw std::runtime_error(
            "Metal backend: operator has no fused Metal kernel (cdf/inv_cdf are "
            "unsupported; MSL has no erf/erfinv)");
    }
}

template <typename T>
void fill_metal(T& rhs, float value)
{
    metal_backend::dispatch_fill(rhs.data(), value, rhs.size());
}

template <reduce_op Op>
float reduce_metal_buffer(void const* data, std::size_t n)
{
    if constexpr (Op == reduce_op::sum)
    {
        return metal_backend::reduce_sum(data, n);
    }
    else if constexpr (Op == reduce_op::min)
    {
        return metal_backend::reduce_min(data, n);
    }
    else
    {
        return metal_backend::reduce_max(data, n);
    }
}

// Leaf tensors reduce in place. Fused trees materialize through the same JIT
// kernel as run_metal, then reduce the temp buffer (MSL cannot instantiate the
// C++ expression tree the way nvcc/hipcc can).
template <typename E, reduce_op Op>
float reduce_metal(E const& expr, std::size_t n)
{
    using RE = vectorization::remove_cvref_t<E>;
    if (n == 0)
    {
        return reduce_identity<float, Op>();
    }

    if constexpr (vectorization::is_base_expression<RE>::value)
    {
        return reduce_metal_buffer<Op>(expr.data(), n);
    }
    else
    {
        static_assert(
            vectorization::is_pure_expression<RE>::value,
            "reduce_metal: expression is neither a tensor leaf nor a unary/binary/trinary node");

        if constexpr (metal_detail::metal_expr_fusable<RE>::value)
        {
            metal_detail::metal_fuse_state st;
            metal_detail::metal_fuse_emit(expr, st);
            if (!metal_detail::metal_fuse_fits(st))
            {
                throw std::runtime_error("Metal fused kernel exceeds the device buffer limit");
            }
            std::string const src = metal_detail::metal_fuse_source(st);
            float* tmp = metal_detail::metal_alloc_t::allocate(n, memory::device_enum::METAL);
            metal_backend::dispatch_fused(
                src, st.buffers, st.n_buffers, st.scalars, st.n_scalars, tmp, n);
            float const result = reduce_metal_buffer<Op>(tmp, n);
            metal_detail::metal_alloc_t::free(tmp, memory::device_enum::METAL);
            return result;
        }
        else
        {
            throw std::runtime_error(
                "Metal backend: operator has no fused Metal kernel (cdf/inv_cdf are "
                "unsupported; MSL has no erf/erfinv)");
        }
    }
}

}  // namespace vectorization

#endif  // VECTORIZATION_HAS_METAL
