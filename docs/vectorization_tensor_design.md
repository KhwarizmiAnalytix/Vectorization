# Tensor Design

Scope: `include/terminals/tensor.h`, `include/expressions/`, and the parts of
Memory and LinearAlgebra the tensor builds on, as of October 2026. This is a
**proposed** design, not the current contract. It covers storage ownership,
the `tensor` / `tensor_view` split, broadcasting and strided evaluation,
aliasing, negative strides, transfers, and the LinearAlgebra adapter, plus a
phased implementation order.

Current contiguity rules are in
[vectorization_tensor_contiguity.md](vectorization_tensor_contiguity.md).
Backend dispatch is in [vectorization_backends.md](vectorization_backends.md).

---

## Contents

1. [Problems with the current tensor](#1-problems-with-the-current-tensor)
2. [Decisions](#2-decisions)
3. [Layers: storage, geometry, kernel descriptor](#3-layers-storage-geometry-kernel-descriptor)
4. [`tensor` and `tensor_view`](#4-tensor-and-tensor_view)
5. [Expressions and the iteration planner](#5-expressions-and-the-iteration-planner)
6. [Aliasing and overlap](#6-aliasing-and-overlap)
7. [Negative strides](#7-negative-strides)
8. [View operations](#8-view-operations)
9. [Copies and transfers](#9-copies-and-transfers)
10. [LinearAlgebra adapter](#10-linearalgebra-adapter)
11. [Upstream API changes](#11-upstream-api-changes)
12. [Implementation phases](#12-implementation-phases)
13. [Verification](#13-verification)
14. [Key files](#14-key-files)

---

## 1. Problems with the current tensor

Confirmed by source inspection and Release-build probes:

| Behavior | Where | Effect |
|---|---|---|
| Expression construction always builds a 1-D result | `tensor::init_from_expression()` | `tensor c = a + b` on `2×3` operands has shape `{6}` |
| `binary_expression::size()` reports one operand's size | `expression_interface.h` | Lengths 8 and 4 are accepted; result length is 4 |
| The evaluator checks only that size against the destination | `expressions_evaluator::run()` | Shapes `{2,3}` and `{3,2}` are accepted |
| `view`, `reshape`, `permute`, `slice` checks are Debug-only | `tensor.h` | `view({6,6})` on six elements is accepted in Release |
| The overlap helper is never called | `expression_aliases_ptr()` | Shifted overlapping slice assignment corrupts the source |
| Views borrow storage | `t()`, `slice()`, `permute()`, `contiguous()`, `to_cpu()` | A view returned from a function can dangle |
| Expression leaves borrow `data_view` | `store_operand()` | `auto e = tensor(n) + a` dangles |
| Expression operands must be contiguous | `store_operand()` | A transpose must be materialized before arithmetic |
| Copying a slice copies `offset + size` elements | copy constructor | A 4-element slice of 1,024 elements copies 1,024 |
| GPU `clone()` of any slice is staged through host memory | `tensor::clone()` | Device → host → device for an interior slice |
| Negative-step slices with the default `stop` are empty | `slice()` | `stop = -1` means "to end", so a reverse slice can't reach index 0 |
| Matrix operations are under `#if 0`; the class comment says `*` is matrix multiplication | `tensor.h` | `*` is elementwise; there is no tensor-level GEMM |

---

## 2. Decisions

| Question | Decision | Reference point |
|---|---|---|
| Copy semantics | Separate `tensor<T>` (owning, deep copy) and `tensor_view<T>` (shallow copy, assignment writes through) | Eigen `Matrix` vs `Block` / `Ref`; not PyTorch's shared handles |
| Partial overlap between destination and source | Raise an error | PyTorch `assert_no_partial_overlap`; not Eigen's silent temporary |
| Negative slice steps | Supported fully | NumPy; PyTorch forbids them |
| Memory / LinearAlgebra API changes | Designed and landed in their own repositories first, then the submodule is bumped | — |

`*` stays elementwise, which matches PyTorch and Eigen `Array`. Matrix
multiplication is only available through the explicit `matmul` family.

---

## 3. Layers: storage, geometry, kernel descriptor

```cpp
// Shared state of tensor<T> and tensor_view<T>.
template <typename T>
struct tensor_impl
{
    memory::retained_ptr<std::remove_const_t<T>> storage_;  // whole allocation
    sizes_and_strides                            sas_;      // inline up to 5 dims
    int64_t                                      offset_{0};  // elements from base()
    std::size_t                                  numel_{0};
    uint8_t flags_{};  // contiguous | dense_non_overlapping | has_zero_stride
};

// What kernels receive: trivially copyable, GPU-memcpy-safe, never owns.
template <typename T, int MaxRank = 8>
struct strided_ref
{
    T*      ptr;
    int     rank;
    int64_t sizes[MaxRank];
    int64_t strides[MaxRank];
};
```

- **Storage is always `memory::retained_ptr`.**
  - Sized allocations still go through `data_ptr`, then are promoted with
    `retained_ptr(data_ptr&&)`, which does not reallocate.
  - External buffers use `retained_ptr::adopt`: a real deleter transfers
    ownership; an explicit no-op deleter is a deliberate unsafe borrow.
  - Every view keeps its storage alive. The cost is one atomic increment per
    view, negligible next to any data-touching operation.
  - This matches PyTorch's `TensorImpl` holding `intrusive_ptr<StorageImpl>`.
    Eigen `Block` / `Map` hold raw pointers, which only works because they
    are expression types that are not meant to be stored.
- **The offset is a field**, not recomputed by pointer subtraction, so copies,
  overlap checks and GPU transfers work from `(base, offset)` directly.
- **Kernels only see `strided_ref`.** Reference-counted handles never reach a
  device.

---

## 4. `tensor` and `tensor_view`

| | `tensor<T>` | `tensor_view<T>` | `tensor_view<const T>` |
|---|---|---|---|
| Invariant | Owns, packed C-order, offset 0 | Any strides (including negative and zero); retains storage | Same, read-only |
| Copy constructor | Deep copy | Shallow (shares storage) | Shallow |
| `operator=` | Deep copy / evaluate | Writes through | Deleted |
| Converts to | Either view type, implicitly | `tensor<T>`, explicitly (packed copy) | `tensor<T>`, explicitly |

- **Return types.**
  - Views: `slice`, `t`, `permute`, `expand`, `view`, `flip`, `squeeze`,
    `unsqueeze`.
  - Tensors: `clone`, `matmul`, construction from an expression.
  - `reshape` and `contiguous` return a view: of the existing storage when
    possible, otherwise of a new packed copy.
- **Parameters.** Inputs take `tensor_view<const T> const&`; outputs take
  `tensor_view<T>`, like Eigen's `Ref<const MatrixXd>`.
- **Rebinding.** Because `operator=` writes through, re-pointing a view uses
  `v.rebind(other)`. Assigning elements of a `std::vector<tensor_view<T>>`
  writes data, the same caveat as Eigen `Ref`.
- **Fast path by type.** When every expression leaf is a `tensor<T>`,
  contiguity is known at compile time; the planner reduces to a shape
  comparison.

---

## 5. Expressions and the iteration planner

### Capture

- An lvalue tensor or view is stored as a `tensor_impl const*`.
- An rvalue (for example `a.t() + b`, or `tensor(n) + a`) is moved into the
  node and keeps its storage alive. Eigen does not do this; dangling `auto`
  expressions are a known Eigen pitfall.

### Lowering

At evaluation, the tree is converted at compile time (`lower_t<E>`) into a
tree whose leaves are `strided_ref`. The lowered tree is what CUDA/HIP kernels
take by value and what the Metal emitter walks.

### Planner

Modelled on PyTorch's TensorIterator; it runs once per evaluation and costs
O(leaves × rank).

1. Collect each leaf's shape, strides, offset, device and storage identity.
2. Broadcast shapes and validate them in every build type. Construction
   infers the result shape; assignment requires it to equal the
   destination's.
3. Compute each operand's strides in output index order, with stride 0 for
   broadcast dimensions.
4. Reverse any dimension whose stride is negative for every operand
   ([section 7](#7-negative-strides)).
5. Sort dimensions by output stride, then merge dimensions that are adjacent
   in memory for every operand.
6. Classify destination/source overlap ([section 6](#6-aliasing-and-overlap)).
7. Pick a loop tier.

| Tier | Condition after merging | Loop |
|---|---|---|
| 0 | 1-D, every leaf has stride 1 | Existing SIMD loop, unchanged |
| 1 | Innermost stride is 1 for every operand (outer broadcast, e.g. `x[m,n] + b[n]`) | Outer loop over rows; existing SIMD inner loop on re-based pointers |
| 2 | Some leaf has innermost stride 0 (column broadcast) | Tier 1 with a splat load for that leaf |
| 3 | Anything else | Scalar loop on CPU; offset-calculator kernel on GPU |

- Tier 2's per-leaf mode is a loop-invariant branch. Benchmark it first; add
  a compile-time specialization for one broadcast operand (as PyTorch does)
  only if needed.
- Tier 3 stays scalar on CPU to limit code size.
- A non-contiguous destination becomes legal. A destination containing a
  stride-0 dimension is always rejected.
- SIMD alignment (`align_start` / `align_end`) moves from the tensor into the
  plan, computed on the normalized destination layout.

### GPU and Metal

- CUDA/HIP keep the contiguous one-thread-per-element kernel as the fast
  path. The strided kernel maps each thread index to operand offsets with
  precomputed fast integer division, like PyTorch's `OffsetCalculator` /
  `IntDivider`.
- The Metal emitter adds a strided variant. The pipeline cache key includes
  the tier and rank; strides are passed in a buffer.

---

## 6. Aliasing and overlap

An address-range test alone gives false positives: interleaved even/odd
column slices overlap in range but share no element. Each destination/source
pair is therefore classified like PyTorch's `get_overlap_status`:

| Status | Detection | Elementwise ops | `matmul_out` and other non-elementwise ops |
|---|---|---|---|
| `none` | Different storage, disjoint address ranges, or the lattice test proves no shared element | Run | Run |
| `full` | Same pointer, sizes and strides | Run in place (`a = a + 1`) | Error |
| `partial` | The lattice test finds a shared element | Error | Error |
| `too_hard` | Strides are not nested, so the test cannot decide | Error | Error |

- **Lattice test.** For two views with equal strides whose absolute values
  each divide the next, decompose the offset difference greedily across
  dimensions. The views share an element exactly when the difference
  decomposes and each coefficient is smaller than its dimension size. This
  covers shifted slices, interleaved slices and negative-step patterns.
- **Deviation from PyTorch.** PyTorch lets `too_hard` cases run silently. This
  design errors on them.
- **Escape hatch.** The error message names `dst = src.clone();`.
- **Internal overlap.** A destination that overlaps itself (stride 0 from
  `expand`) is always an error.

---

## 7. Negative strides

Supported following NumPy.

- **Slice API.** The current `stop = -1` meaning "to end" collides with
  Python's -1 index. The signature becomes
  `slice(dim, std::optional<int64_t> start, std::optional<int64_t> stop,
  int64_t step = 1)`, normalized with CPython's `PySlice_AdjustIndices` rules:
  - for a negative step, `start` defaults to `size - 1` and `stop` defaults to
    "before 0";
  - `step == 0` is an error;
  - `offset += start * stride`, then `stride *= step`.
- **`flip(dim)`** is a free view (slice with step -1), not a copy.
- **Signed arithmetic.** These currently cast strides to `size_t` and rely on
  unsigned wrap-around; they move to `ptrdiff_t`: `logical_offset`,
  `at(i, j)`, `linearized_index`, `gather_logical`, and the subview size in
  `slice`.
- **Address range.** A view touches
  `[offset + Σ min(0, (nᵢ−1)·sᵢ), offset + Σ max(0, (nᵢ−1)·sᵢ)]`. Both ends are
  checked against the allocation; overlap checks and transfers use the same
  range.
- **Planner normalization.** A dimension whose stride is negative for every
  operand is traversed in reverse: the pointer moves to the other end and the
  strides are negated, as NumPy's `nditer` does. A reversed packed view then
  runs on Tier 0 or 1. Mixed directions on the innermost dimension fall to
  Tier 3; a SIMD reverse-load tier can follow if Scalars exposes a
  lane-reverse operation.
- **Flags.** A reversed packed view is dense and non-overlapping, but not
  contiguous.
- **BLAS.** Leading dimensions must be positive, so negative-stride operands
  are packed, and a negative-stride output is computed into a temporary and
  copied.

---

## 8. View operations

- **`view(shape)`**: metadata only. Uses NumPy's `_attempt_nocopy_reshape`
  rule, which handles negative strides (PyTorch's `computeStride` does not
  need to). Errors when no view exists; never copies.
- **`reshape(shape)`**: `view` when possible, otherwise a packed copy viewed
  with the new shape.
- **`contiguous()`**: a view of the same storage when already packed (safe
  now that views retain storage), otherwise a packed copy.
- **New**: `expand` / `broadcast_to` (stride-0 views, never valid
  destinations), `squeeze`, `unsqueeze`, `flatten`, `flip`.
- **Validation**: every argument check uses `VECTORIZATION_CHECK`, not the
  Debug-only variant. These operations only touch metadata, so the cost is not
  measurable.

---

## 9. Copies and transfers

`clone()`, `contiguous()` and conversion from a view all evaluate an
**identity expression through the planner**. One mechanism gives:

- CPU gathers on the best tier;
- a device-side strided GPU kernel, with no host staging;
- copying only logical elements (a 4-element slice copies 4 elements).

Cross-device copies:

- Contiguous source: one `(base, offset)` copy.
- Strided source: pack on the source device, then transfer.
- Asynchronous variants return a `copy_token` that keeps both endpoints alive.

Stream handling separates two concerns:

- `record_stream` only prevents the caching allocator from reusing a block.
- Ordering one stream after another needs an event. The tensor exposes
  `tensor_view::wait_stream(stream)` on top of Memory's event API
  ([section 11](#11-upstream-api-changes)).

---

## 10. LinearAlgebra adapter

```cpp
template <typename T>
tensor<T> matmul(tensor_view<const T> const& a, tensor_view<const T> const& b);

template <typename T>
void matmul_out(
    tensor_view<T> c, tensor_view<const T> const& a, tensor_view<const T> const& b,
    T alpha = T(1), T beta = T(0));

template <typename T>
T dot(tensor_view<const T> const& a, tensor_view<const T> const& b);

template <typename T>
tensor<T> bmm(tensor_view<const T> const& a, tensor_view<const T> const& b);
```

Shapes follow PyTorch's `matmul` rules for vectors, matrices and batches.
Inner dimensions are validated in every build type.

### Stride mapping

`linalg::multiplication::apply` is row-major with explicit leading dimensions:

| Operand layout | `transpose` | `ld` |
|---|---|---|
| `stride(1) == 1` | `false` | `stride(0)` |
| `stride(0) == 1` | `true` | `stride(1)` |
| Otherwise, or any negative stride | Pack with an identity copy | `cols` |

- **Output.** `stride(1) == 1`: write directly. `stride(0) == 1`: compute
  Cᵀ = Bᵀ·Aᵀ by swapping the operands and flipping the flags. Otherwise compute
  into a temporary and copy.
- **Size-1 dimensions.** BLAS rejects a leading dimension smaller than the row
  length even where the stride is irrelevant, so `ld` is clamped to at least
  the row length.
- **Aliasing.** Any overlap between `c` and `a` or `b`, including `full`, is
  an error ([section 6](#6-aliasing-and-overlap)). `a = matmul(a, b)` is fine:
  `matmul` returns a new tensor.

### Backend selection

| Device | `linalg_backend` |
|---|---|
| CPU | `MKL` if `LINALG_HAS_MKL`, else `BLAS_LAPACKE` if available, else `NATIVE` |
| CUDA | `CUDA_BLAS` |
| HIP | `ROCM_BLAS` |
| Metal | `METAL_BLAS` (float only; `double` is a compile-time error) |

### Workspace and lifetime

- Host workspace: per-thread, grow-only, sized by `workspace_size()`.
- GPU workspace: per stream, from Memory's `gpu_workspace.h`.
- On a non-default stream, `a`, `b`, `c` and the workspace are recorded on
  that stream.

### Factorizations

Reusable factorization objects, holding their factors (and pivots) and
offering `.solve(b)`: `lu_factorization<T>`, `cholesky<T>`, `qr<T>`, and SVD
where the backend supports it. Eigen's `PartialPivLU` / `LLT` /
`HouseholderQR` and PyTorch's `lu_factor` / `lu_solve` are the models.

### Cleanup

The `#if 0` matrix block in `tensor.h` and the class comment claiming
`tensor * tensor` is matrix multiplication are removed.

---

## 11. Upstream API changes

Memory and LinearAlgebra are separate repositories (submodules under
`ThirdParty/`). These changes are designed, tested and landed there first;
Vectorization then bumps the submodule.

### Memory (needed by phase P4)

- Offset-aware copy, synchronous and asynchronous:
  `copy(src_base, src_offset, dst_base, dst_offset, count, ctx)`. It replaces
  the interior-slice rejection in `data_ptr(data_view)`; the caching allocator
  keeps looking blocks up by base address.
- `copy_async` overloads on `retained_ptr` returning a `copy_token` that keeps
  both endpoints alive.
- Stream events (`record_event` / `wait_event`) for cross-stream ordering,
  separate from `record_stream`.
- `record_stream` is a no-op for adopted memory with an invalid allocation id,
  instead of failing the allocator lookup.

### LinearAlgebra (needed by phase P5)

- `alpha` / `beta` on `multiplication::apply`; the existing signature stays as
  the `alpha = 1, beta = 0` overload.
- Rectangular strided-batched GEMM: `rows`, `columns`, `depth`, a batch stride
  per operand (0 for broadcasting), positive leading dimensions, a stream and
  a workspace. The square `batched_matrix_multiplication(dim, count, …)` stays
  as a wrapper.

---

## 12. Implementation phases

| Phase | Repository | Scope | Done when |
|---|---|---|---|
| P0 | Vectorization | Result shape when leaf shapes are equal; validate every leaf's shape; Release checks on view operations; overlap classifier raising errors; copy constructor copies logical elements only | Every probe in [section 1](#1-problems-with-the-current-tensor) has a passing Release test |
| P1 | Vectorization | `retained_ptr` storage, explicit offset, views retain storage | Lifetime tests (view outlives owner, deferred expression) clean under AddressSanitizer |
| P2 | Vectorization | `tensor` / `tensor_view` split, rvalue capture into expressions | View assignment writes through; existing tests pass unchanged |
| P3 | Vectorization | CPU planner tiers 0–2, broadcasting, negative-stride normalization, NumPy reshape rule, identity-copy `clone` | Tier 0 benchmark within noise of today; strided, broadcast and negative-step tests pass |
| U1 | Memory | [Section 11](#memory-needed-by-phase-p4) APIs | Landed upstream with tests; submodule bumped |
| U2 | LinearAlgebra | [Section 11](#linearalgebra-needed-by-phase-p5) APIs | Landed upstream with tests; submodule bumped |
| P4 | Vectorization | CUDA/HIP offset-calculator kernel, Metal strided variant, offset-aware transfers | Device clone of an interior slice passes without host memory (Metal locally; CUDA/HIP in CI) |
| P5 | Vectorization | `matmul`, `matmul_out`, `dot`, `bmm`, factorization objects | Rectangular, transposed-view, negative-stride and aliased-output GEMM tests pass |
| P6 | Vectorization | Performance tuning | See [section 13](#13-verification) |

U1 and U2 run in parallel with P1–P3; only P4 and P5 depend on them. P0 is
independent of everything else and fixes incorrect results, so it goes first.

---

## 13. Verification

Correctness tests, in addition to numerical values:

- Result shape of expression construction, including broadcasting.
- Invalid shapes, axes, reshape sizes and slice steps rejected in Release.
- View lifetime: a view returned from a function, and deferred expressions
  holding temporaries.
- Overlap: exact aliasing allowed; shifted, interleaved and `too_hard` cases
  rejected.
- Negative steps on every dimension, `flip`, and reverse slices reaching
  index 0.
- Strided and broadcast arithmetic on every tier.
- Rectangular and batched GEMM with transposed and negative-stride views.
- Stream transitions: allocation reuse after `record_stream`, and ordering
  through events.

Performance measurements:

- Contiguous vs strided elementwise throughput, per tier.
- Allocation counts and peak memory for complete expression chains.
- Small and large matrices, including packing overhead.
- The existing LibTorch comparison, which preallocates outputs, plus a
  separate run that includes allocation and ownership costs.

---

## 14. Key files

- `include/terminals/tensor.h` — `tensor<T>`, views, `clone()`,
  `contiguous()`, `init_from_expression()`, the `#if 0` matrix block.
- `include/terminals/sizes_and_strides.h` — inline sizes/strides storage.
- `include/expressions/expression_interface.h` — `store_operand()`,
  expression nodes, `buffers_overlap()` / `expression_aliases_ptr()`.
- `include/expressions/expressions_evaluator.h` — CPU SIMD loop (Tier 0).
- `include/expressions/expressions_evaluator_gpu.h` — CUDA/HIP kernels.
- `include/expressions/expressions_evaluator_metal.h` — Metal emitter.
- `include/expressions/expressions_builder.h` — operators, including
  elementwise `*`.
- `ThirdParty/Memory/include/common/retained_ptr.h`, `data_ptr.h`,
  `data_view.h`, `copy_token.h` — storage and transfer primitives.
- `ThirdParty/Memory/include/gpu/cuda_caching_allocator.h`,
  `gpu_workspace.h` — GPU allocation and workspace.
- `ThirdParty/LinearAlgebra/include/matrix_operation/multiplication.h`,
  `multiplication_batched.h` — GEMM entry points.
- `ThirdParty/LinearAlgebra/include/common/backend.h` — `linalg_backend`.
