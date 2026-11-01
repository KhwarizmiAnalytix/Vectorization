# Vectorization Design Plan

Scope: `include/terminals/tensor.h`, `include/expressions/`, the CPU, CUDA,
HIP and Metal evaluators, and the parts of Memory, Scalars and LinearAlgebra
they build on, as of October 31, 2026. This is the **single proposed plan**
for the tensor and evaluator redesign. It is not the current contract.

It covers storage ownership, the `tensor` / `tensor_view` split,
broadcasting and strided evaluation, aliasing and `noalias()`, slice steps,
transfers, reductions, the LinearAlgebra adapter, backend launch
unification, allocation, clang-tidy conformance, phased implementation with
per-phase performance gates, and the benchmark protocol.

Current contracts stay in their own documents:

- Contiguity rules: [vectorization_tensor_contiguity.md](vectorization_tensor_contiguity.md).
- Backend evaluator contracts: [vectorization_backends.md](vectorization_backends.md).
- CPU ISA selection: [vectorization.md](vectorization.md).

Code comments cite this plan as "design §N". Sections 5–8 keep the numbers
used by those comments.

---

## Contents

1. [Direction and status](#1-direction-and-status)
2. [Decisions](#2-decisions)
3. [Layers: storage, geometry, kernel descriptor](#3-layers-storage-geometry-kernel-descriptor)
4. [`tensor` and `tensor_view`](#4-tensor-and-tensor_view)
5. [Expressions and the iteration planner](#5-expressions-and-the-iteration-planner)
6. [Aliasing and overlap](#6-aliasing-and-overlap)
7. [Slice steps](#7-slice-steps)
8. [View operations](#8-view-operations)
9. [Copies and transfers](#9-copies-and-transfers)
10. [Reductions](#10-reductions)
11. [LinearAlgebra adapter](#11-linearalgebra-adapter)
12. [Upstream API changes](#12-upstream-api-changes)
13. [Backend launch unification](#13-backend-launch-unification)
14. [Allocation and workspace](#14-allocation-and-workspace)
15. [clang-tidy conformance](#15-clang-tidy-conformance)
16. [Implementation phases](#16-implementation-phases)
17. [Verification and benchmark protocol](#17-verification-and-benchmark-protocol)
18. [Key files](#18-key-files)
19. [Sources merged into this plan](#19-sources-merged-into-this-plan)

---

## 1. Direction and status

### Direction

Combine Eigen-style expression fusion with PyTorch-style storage, strides,
broadcasting and backend dispatch. The main performance opportunities are
avoiding allocations, intermediate memory passes, unnecessary packing, and
CPU/GPU synchronization. SIMD and threading support those goals.

Use a cheap contiguous path for packed tensors and a general iteration
planner for views and broadcasting. Keep numerical backend calls behind a
thin adapter that resolves shape, layout, aliasing and workspace.

| Area | PyTorch | Eigen | This library |
|---|---|---|---|
| Ownership | Shared tensor/storage handles | Owning matrices, borrowing `Block` / `Map` / `Ref` | Owning `tensor`, storage-retaining `tensor_view`; deep copy is explicit and documented |
| Elementwise execution | Eager operators; `torch.compile` can fuse larger regions | Lazy expression templates | One CPU traversal or one GPU kernel per expression |
| Layout handling | General strides and broadcasting (`TensorIterator`) | Matrix/array expressions; separate Tensor module | Contiguous fast path plus a general stride planner |
| Device selection | Runtime dispatch keys | Compile-time `Device` in `TensorDevice` | Compile-time; one CPU ISA and at most one GPU backend per binary |
| Small matrices | General runtime dispatch | Fixed sizes allow inline storage and full unrolling | Runtime small-matrix kernels first; static shapes out of scope |
| Large linear algebra | Vendor BLAS/LAPACK | Native kernels or external BLAS/LAPACK | LinearAlgebra and its backends |

Eigen introduces temporaries when aliasing or subexpression reuse makes them
worthwhile. The same reasoning applies around products here; not every
expression must fuse.

### Implementation status

Confirmed by source inspection on October 31, 2026. "Done" items still need
their correctness and performance gates verified ([section 16](#16-implementation-phases)).

| Item | Status | Where |
|---|---|---|
| Expression result takes the broadcast shape; leaf shapes validated in every build | Done | `expression_shape.h`, `tensor::init_from_expression()` |
| CPU planner: sort, merge, tiers, `loop2d` | Done | `iteration_plan.h` |
| CPU `parallel_for` above a 32,768-element grain | Done; serial SIMD fallback fixed in `c4fde41`, not yet benchmarked | `iteration_plan.h`, `expressions_evaluator.h` |
| Strided expression leaves (`strided_ref`) — transposes need no materialization | Done | `strided_ref.h`, `store_operand()` |
| Stride-0 destination rejected | Done | `make_iteration_plan()` |
| `slice(dim, optional start, optional stop, step ≥ 1)` | Done | `tensor.h` |
| `expand`, `broadcast_to`, `squeeze`, `unsqueeze`, `flatten`; Release checks on view operations | Done | `tensor.h` |
| Copy constructor keeps the source layout | Done in `c7f436f` | `tensor.h` |
| CUDA/HIP strided sources | Partial: leaves are rebased, destination must be contiguous | `rebase_leaves()` |
| `retained_ptr` storage; views retain storage | Not started: storage is `data_ptr` + `data_view`, views borrow | `tensor.h` |
| Expression leaves keep rvalue operands alive | Not started: leaves borrow through `strided_ref` | `store_operand()` |
| `tensor` / `tensor_view` split, `noalias()` | Not started | — |
| Destination/source overlap classifier | Not started: `expression_aliases_ptr()` is never called | `expression_interface.h` |
| `flip`, tiled transpose | Not started | — |
| Planner-owned prefetch; removal of the aligned-source variant | Not started: per-leaf prefetch and both alignment variants remain | `expressions_evaluator.h` |
| Axis reductions; strided reductions | Not started | `reduce_op.h` |
| GPU `clone()` of a strided slice without host staging | Not started: stages through host | `tensor::clone()` |
| CUDA/HIP offset calculator, 32-bit indexing, vector loads | Not started: one thread per element, scalar loads | `expressions_evaluator_gpu.h` |
| GPU reduction without per-pass allocation | Not started | `reduce_gpu_partials()` |
| Metal pipeline cache, batched command buffers, per-allocation `MTLBuffer` | Not started: MSL rebuilt per call, commit-and-wait, per-call buffer wrapping | `expressions_evaluator_metal.h`, Scalars `metal_dispatch.mm` |
| Unified launch contract across backends | Not started | [section 13](#13-backend-launch-unification) |
| Tensor-level `matmul`, `dot`, factorizations | Not started: matrix block under `#if 0` | `tensor.h` |
| Memory U1 and LinearAlgebra U2 APIs | Not started | [section 12](#12-upstream-api-changes) |
| clang-tidy coverage of `include/` and `Testing/` | Not started: CI analyzes one source file; 65 findings in a two-file baseline | [section 15](#15-clang-tidy-conformance) |

---

## 2. Decisions

| Question | Decision | Reference point |
|---|---|---|
| Copy semantics | Separate `tensor<T>` (owning, deep copy) and `tensor_view<T>` (shallow copy, assignment writes through) | Eigen `Matrix` vs `Block` / `Ref`; not PyTorch's shared handles |
| Partial overlap between destination and source | Raise an error | PyTorch `assert_no_partial_overlap`; not Eigen's silent temporary |
| Opting out of the overlap check | `dst.noalias() = expr`: the caller asserts no overlap. It skips the O(rank) check in Release and lets `too_hard` layouts run. It does not decide whether a product needs a temporary; the classifier does | Eigen `noalias()`; PyTorch has only `out=` and the internal `check_mem_overlap(false)` |
| Negative slice steps | Forbidden: `step < 1` is an error, strides are never negative, `flip` returns a copy | PyTorch; NumPy supports them |
| `*` on tensors | Elementwise; matrix multiplication only through the `matmul` family | PyTorch, Eigen `Array` |
| Backend selection | Compile-time; no runtime-polymorphic launch interface | Eigen `TensorDevice`; deviates from PyTorch's runtime dispatcher |
| GPU synchronization | Asynchronous launch on every GPU backend; the host waits only on a read or an explicit event | CUDA streams; PyTorch MPS commit-and-continue |
| Memory / LinearAlgebra / Scalars API changes | Designed and landed in their own repositories first, then the submodule is bumped | — |
| Static analysis | All first-party C++ is clang-tidy clean under `.clang-tidy`, warnings as errors, in every configuration CI builds; suppressions name the check and a reason | PyTorch gates on clang-tidy through lintrunner with named `NOLINTNEXTLINE`; Eigen does not gate on clang-tidy |

Value semantics are intentional: `tensor b = a` copies elements, unlike a
PyTorch handle copy. Code that only needs to borrow takes a view.

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
  - Sized allocations go through `data_ptr` and are promoted with
    `retained_ptr(data_ptr&&)`, which does not reallocate.
  - External buffers use `retained_ptr::adopt`: a real deleter transfers
    ownership; an explicit no-op deleter is a deliberate unsafe borrow.
  - Every view keeps its storage alive, at the cost of one atomic increment
    per view.
  - PyTorch's `TensorImpl` holds `intrusive_ptr<StorageImpl>` the same way.
    Eigen `Block` / `Map` hold raw pointers, which only works because they
    are expression types not meant to be stored.
- **The offset is a field**, not recomputed by pointer subtraction, so
  copies, overlap checks and GPU transfers work from `(base, offset)`.
- **Kernels only see `strided_ref`.** Reference-counted handles never reach
  a device. Worker threads also use `strided_ref` and raw pointers, so they
  never contend on a shared control block.
- **One allocation per CPU tensor** (Memory U1): for sized CPU allocations
  the control block sits in the same allocation as the data, as
  `std::make_shared` does. GPU and adopted storage keep a separate control
  block ([section 14](#14-allocation-and-workspace)).

---

## 4. `tensor` and `tensor_view`

| | `tensor<T>` | `tensor_view<T>` | `tensor_view<const T>` |
|---|---|---|---|
| Invariant | Owns, packed C-order, offset 0 | Any non-negative strides (including zero); retains storage | Same, read-only |
| Copy constructor | Deep copy | Shallow (shares storage) | Shallow |
| `operator=` | Deep copy / evaluate | Writes through | Deleted |
| Converts to | Either view type, implicitly | `tensor<T>`, explicitly (packed copy) | `tensor<T>`, explicitly |

- **Return types.**
  - Views: `slice`, `t`, `permute`, `expand`, `view`, `squeeze`,
    `unsqueeze`.
  - Tensors: `clone`, `flip`, construction from an expression.
  - Lazy products: `matmul`, `bmm` ([section 11](#11-linearalgebra-adapter)).
  - `reshape` and `contiguous` return a view: of the existing storage when
    possible, otherwise of a new packed copy.
- **Parameters.** Inputs take `tensor_view<const T> const&`; outputs take
  `tensor_view<T>`, like Eigen's `Ref<const MatrixXd>`.
- **Rebinding.** Because `operator=` writes through, re-pointing a view uses
  `v.rebind(other)`. Assigning elements of a `std::vector<tensor_view<T>>`
  writes data, the same caveat as Eigen `Ref`.
- **`noalias()`.** `tensor<T>` and `tensor_view<T>` have `noalias()`, which
  returns a proxy offering only `=`, `+=` and `-=` ([section 6](#6-aliasing-and-overlap)).
  It is deleted on `tensor_view<const T>`.
- **Fast path by type.** When every expression leaf is a `tensor<T>`,
  contiguity is known at compile time and planning reduces to a shape
  comparison.

---

## 5. Expressions and the iteration planner

### Capture

- An lvalue tensor or view is stored as a `tensor_impl const*`.
- An rvalue (`a.t() + b`, `tensor(n) + a`) is moved into the node and keeps
  its storage alive. Eigen does not do this; dangling `auto` expressions are
  a known Eigen pitfall.

### Lowering

At evaluation the tree is converted at compile time (`lower_t<E>`) into a
tree whose leaves are `strided_ref`, or `T*` for tier-0 leaves on GPU. The
lowered tree is what CUDA/HIP kernels take by value and what the Metal
emitter walks.

### Planner

Modelled on PyTorch's `TensorIterator`; runs once per evaluation at
O(leaves × rank) cost.

1. Collect each leaf's shape, strides, offset, device and storage identity.
2. Broadcast shapes and validate them in every build type. Construction
   infers the result shape; assignment requires it to equal the
   destination's.
3. Compute each operand's strides in output index order, with stride 0 for
   broadcast dimensions.
4. Sort dimensions by output stride, breaking ties by the stride of the
   largest source operand, then merge dimensions that are adjacent in memory
   for every operand. Strides are never negative ([section 7](#7-slice-steps)),
   so no dimension needs reversing.
5. Classify destination/source overlap ([section 6](#6-aliasing-and-overlap)),
   unless the assignment goes through `noalias()` in a Release build.
6. Pick a loop tier.

| Tier | Condition after merging | Loop |
|---|---|---|
| 0 | 1-D, every leaf has stride 1 | SIMD loop via `loop2d` with `outer_n == 1` |
| 1 | Innermost stride is 1 for every operand (outer broadcast, `x[m,n] + b[n]`) | `loop2d` over rows; SIMD inner loop on re-based pointers |
| 2 | Some leaf has innermost stride 0 (column broadcast) | `loop2d` with a splatted load for the broadcast leaf |
| 3 | Anything else | Tiled transpose or lane-reversing `flip` when the shape permits; otherwise scalar on CPU, offset-calculator kernel on GPU |

#### Contiguous fast path

Planning costs more than the arithmetic for small tensors. An O(leaves)
check runs first:

1. Every leaf has the `contiguous` flag from `tensor_impl::flags_`.
2. Every leaf has the destination's shape.

When both hold, evaluation goes straight to tier 0 with no sort, merge or
lattice test. The overlap check is ordered cheapest first:

1. Different storage → `none`.
2. Same base pointer, sizes and strides → `full`.
3. Only then the lattice test.

Two distinct `tensor<T>` objects can never partially overlap, so step 1 or 2
always decides for them. A freshly constructed `tensor<T>` destination skips
the check. The target is Eigen-like cost for trees of packed tensors, not
`TensorIterator`'s microsecond-level setup.

#### Iteration plan and threading

```cpp
// Produced once per evaluation; consumed by one or more threads.
struct iteration_plan
{
    int     rank;                        // after merging, usually 1–3
    int64_t sizes[MaxRank];              // shared by every operand
    int64_t strides[MaxOperands][MaxRank];
    int     tier;
};

// Runs the plan over [begin, end) of the linearized iteration space.
template <typename Lowered>
void run_range(Lowered const& expr, iteration_plan const& plan,
               int64_t begin, int64_t end);
```

- **Threshold.** 32,768 elements (PyTorch's grain size) is the starting
  point. Tune it per backend from bytes processed and per-element operation
  cost; a transcendental-heavy expression pays off at fewer elements than a
  copy.
- **Splitting.** Split within long rows as well as between rows, so a layout
  with a few long rows can still use every worker. The current
  implementation splits by rows only.
- **Chunk edges.** Round to multiples of
  `simd::size × VECTORIZATION_PACKET_SIZE` so each chunk keeps the aligned
  body loop; only the first and last chunks peel.
- **Oversubscription.** Coordinate tensor workers with BLAS threads so a
  `parallel_for` around a threaded GEMM does not oversubscribe cores.
- **Serial fallback.** With parallel support disabled, large contiguous
  expressions must still take the explicit SIMD loop.
- **Reference points.** PyTorch `TensorIterator::for_each` with
  `at::parallel_for`; Eigen `TensorExecutor` with `ThreadPoolDevice`.

#### `loop2d` inner kernel

Tiers 0–2 share one kernel shape, following PyTorch's `loop2d`:

```cpp
// Inner dimension: unit stride (or stride 0 for a broadcast leaf).
// Outer dimension: one pointer increment per operand per row.
template <typename Lowered>
void loop2d(Lowered const& expr, char** ptrs, int64_t const* outer_strides,
            int64_t inner_n, int64_t outer_n);
```

- Tier 0 is `outer_n == 1`; tier 1 differs only in having `outer_n > 1`.
- Short rows (`x[m, 3] + b[3]`) pay one pointer increment per row, not a
  peel, unrolled body and tail per row.
- Unit-stride loads and stores are unaligned, because the row start moves
  when the row stride is not a multiple of the SIMD width.
- Rank above 2 steps an outer counter (PyTorch's `DimCounter`). No CPU index
  is computed by division.

#### Tier 2 specialization

A runtime per-leaf flag inside the hot loop is often not hoisted when several
leaves and packet unrolling are involved. Build a compile-time
specialization for "exactly one leaf has inner stride 0" (the bias-add case),
as PyTorch's scalar-operand `vectorized_loop` does. Other combinations use
`loop2d` with the broadcast value splatted once per row.

#### Tiled transpose and `flip` kernels

`clone`, `contiguous`, conversion from a view, `flip` and GEMM packing
([section 11](#11-linearalgebra-adapter)) all go through the planner. A
purely scalar tier 3 would make `a.t().contiguous()`, the most common tier-3
case, the slowest copy in the library.

- **Tiled transpose.** When exactly one 2-D pair of dimensions is swapped
  between destination and source, copy in 8×8 or 16×16 tiles transposed in
  registers. Other tier-3 layouts stay scalar.
- **`flip`.** Load contiguous registers, reverse lanes with a permute, store.
  A scalar back-to-front loop is not acceptable.

#### Prefetch and alignment

- **Prefetch.** The plan owns the prefetch distance. Sequential unit-stride
  loops issue no prefetch or prefetch 256–512 bytes ahead, chosen by
  benchmark. Tier 1 prefetches the next row's start when the row stride is
  large. The per-leaf, one-register-ahead prefetch is removed; the hardware
  prefetcher already covers that distance.
- **Alignment.** Remove the aligned-source template variant; `loadu` on
  aligned data costs the same as `load` on current x86 and Arm cores. Keep
  the destination peel, which avoids cache-line-splitting stores on AVX-512.
  This roughly halves the instantiations per expression type.

#### Streaming stores (P6, optional)

When the destination overlaps no source and is larger than about half the
last-level cache, non-temporal stores avoid reading each destination line
before writing it, typically 1.2–1.5× on write-heavy work. They can hurt when
the next operation immediately reads the result, so they stay behind a flag
and are enabled only where measured to win. Neither Eigen nor PyTorch does
this; glibc and MKL `memcpy` do.

### GPU and Metal

#### CUDA / HIP — offset calculator and indexing width

- **One size array.** After broadcasting every operand has the same sizes;
  only strides differ. Store sizes once (as fast divisors) and strides per
  operand, like PyTorch's `OffsetCalculator`. A `strided_ref` per leaf with
  `MaxRank = 8` is 144 bytes, so ten leaves would be about 1.4 KB of kernel
  parameters.
- **Lowering by tier.** Tier-0 leaves lower to `T*`.
- **32-bit indexing.** An element count below 2³¹ is not sufficient: a
  small tensor with large strides can reach offsets beyond that range. Use
  32-bit indices only when the element count **and** the maximum reachable
  offset of every operand, including the destination, fit, computed with
  overflow-checked arithmetic in the units the offset calculator uses.
  Otherwise rebase and split the launch, or fall back to 64-bit indexing.
  PyTorch's `can_use_32bit_indexing` checks both. `IntDivider` only speeds
  up 32-bit division; 64-bit division is very slow on GPUs.
- **Vector loads.** When every leaf is 16-byte aligned and contiguous, each
  thread processes 4 elements with 16-byte loads and stores, like PyTorch's
  `vectorized_elementwise_kernel`. Other cases use a scalar unrolled kernel.
- **Strided destinations.** The offset calculator removes today's
  requirement that GPU destinations be contiguous.

#### Metal — pipeline cache, synchronization and buffers

- **Cache key.** The pipeline is held in a function-local static per
  expression type `E`, tier and rank. MSL source is generated only on the
  first call, so later calls build no strings and take no global lock.
  Scalars already travel in a buffer and do not change the source.
- **Batched command buffers.** Evaluations are encoded into a shared command
  buffer and committed without waiting. The host waits only on a read or a
  stream event, as PyTorch MPS's `MPSStream` does with commit-and-continue.
- **One `MTLBuffer` per allocation.** Because storage is a `retained_ptr`,
  the `MTLBuffer` is created once at allocation and kept with it. A view is
  passed with `setBuffer:offset:` using `tensor_impl::offset_`. This removes
  per-call wrapping and makes interior slices work, which
  `newBufferWithBytesNoCopy` cannot because it requires page alignment.
- **Ownership.** Metal dispatch code lives in Scalars
  (`backend/gpu/metal/`), so buffer and command-buffer changes land there
  first.

---

## 6. Aliasing and overlap

An address-range test alone gives false positives: interleaved even/odd
column slices overlap in range but share no element. Each destination/source
pair is classified like PyTorch's `get_overlap_status`:

| Status | Detection | Elementwise ops | `matmul_out` and other non-elementwise ops |
|---|---|---|---|
| `none` | Different storage, disjoint address ranges, or the lattice test proves no shared element | Run | Run |
| `full` | Same pointer, sizes and strides | Run in place (`a = a + 1`) | Error |
| `partial` | The lattice test finds a shared element | Error | Error |
| `too_hard` | Strides are not nested, so the test cannot decide | Error | Error |

- **Lattice test.** For two views with equal strides that each divide the
  next, decompose the offset difference greedily across dimensions. The
  views share an element exactly when it decomposes and each coefficient is
  smaller than its dimension size. This covers shifted and interleaved
  slices.
- **Deviation from PyTorch.** PyTorch lets `too_hard` cases run silently;
  this design errors on them.
- **Escape hatches.** For `partial` the error names `dst = src.clone();`.
  For `too_hard` it also names `dst.noalias() = …`.
- **Internal overlap.** A destination that overlaps itself (stride 0 from
  `expand`) is always an error, with or without `noalias()`. This check is
  already implemented.

### `noalias()`

```cpp
c.noalias() = a + b;          // skip the overlap classifier
c.noalias() = matmul(a, b);   // GEMM straight into c, check skipped in Release
c.noalias() += matmul(a, b);  // beta = 1
c.noalias() -= matmul(a, b);  // alpha = -1, beta = 1
```

- **Contract.** The caller asserts that the destination shares no element
  with any source. Shape, device and internal-overlap checks still run in
  every build type.
- **Checking.** Debug runs the classifier and errors on a proven overlap
  (`partial`, or `full` for a non-elementwise operation); `too_hard` is
  accepted. Release skips the classifier; a false assertion is undefined
  behavior, as in Eigen.
- **Proxy.** `noalias()` returns a non-copyable
  `noalias_proxy<tensor_view<T>>` holding a reference to the destination.
  `+=` and `-=` on an elementwise expression evaluate `dst = dst ± expr`,
  which is `full` overlap and safe. The proxy never resizes; the result shape
  must equal the destination's.
- **What it buys.** For elementwise work it saves the O(leaves × rank) check
  and lets `too_hard` layouts run. For products, `c = matmul(a, b)` already
  writes directly when the classifier reports `none`, so `noalias()` only
  allows `too_hard` layouts and skips the check in Release. SIMD loops issue
  explicit loads and stores, so code generation does not change.
- **Reference points.** Eigen's `noalias()` has the same spelling but never
  checks; the Debug check is this design's deviation. PyTorch has no
  user-facing equivalent: `out=` still checks, and
  `TensorIteratorConfig::check_mem_overlap(false)` is internal.

---

## 7. Slice steps

Negative steps are forbidden, following PyTorch. Every stride is
non-negative, which keeps the planner, the overlap test and the BLAS mapping
free of reversed dimensions.

- **Slice API** (implemented):
  `slice(dim, std::optional<int64_t> start, std::optional<int64_t> stop,
  int64_t step = 1)`.
  - `step < 1` is an error in every build type; the message names `flip`.
  - Negative `start` / `stop` count from the end, then clamp to `[0, size]`.
  - `offset += start * stride`, then `stride *= step`.
- **`flip(dims)`** returns a new `tensor<T>`, like PyTorch's `flip`, through
  the planner's lane-reversing kernel ([section 5](#tiled-transpose-and-flip-kernels)).
  Negative strides stay internal to that kernel and never appear in a view.
- **Arithmetic.** `logical_offset`, `at(i, j)`, `linearized_index` and
  `gather_logical` cast strides to `size_t`, which is correct under the
  non-negative invariant. Every view constructor checks the invariant with
  `VECTORIZATION_CHECK`.
- **Address range.** A non-empty view touches
  `[offset, offset + Σ (nᵢ−1)·sᵢ]`. Both ends are checked against the
  allocation; overlap checks, transfers and GPU index-width checks use the
  same range.
- **Cost.** Reversed traversal pays for a `flip` copy where NumPy would
  return a free view.

---

## 8. View operations

- **`view(shape)`**: metadata only, using PyTorch's `computeStride` rule.
  Errors when no view exists; never copies.
- **`reshape(shape)`**: `view` when possible, otherwise a packed copy viewed
  with the new shape.
- **`contiguous()`**: a view of the same storage when already packed (safe
  once views retain storage), otherwise a packed copy.
- **`expand` / `broadcast_to`** (stride-0 views, never valid destinations),
  **`squeeze`**, **`unsqueeze`**, **`flatten`**: implemented. **`flip`** as a
  copy ([section 7](#7-slice-steps)): not yet.
- **Validation**: every argument check uses `VECTORIZATION_CHECK`, not the
  Debug-only variant. These operations touch only metadata.

---

## 9. Copies and transfers

`clone()`, `contiguous()` and conversion from a view evaluate an **identity
expression through the planner**. One mechanism gives:

- CPU gathers on the best tier, including tiled transpose and `flip`;
- a device-side strided GPU kernel with no host staging;
- copying only logical elements (a 4-element slice copies 4 elements).

Cross-device copies:

- Contiguous source: one `(base, offset)` copy.
- Strided source: pack on the source device, then transfer.
- Asynchronous variants return a `copy_token` that keeps both endpoints
  alive.

Stream handling separates three concerns:

- **Ownership**: storage and temporaries stay alive until queued work that
  uses them completes.
- **Reuse protection**: `record_stream` only stops the caching allocator
  from reusing a block early.
- **Ordering**: one stream waits for another through an event, exposed as
  `tensor_view::wait_stream(stream)` on Memory's event API.

---

## 10. Reductions

`accumulate`, `hmin` and `hmax` require contiguous input today, and axis
reductions (`sum(dim)`) do not exist. The planner serves elementwise
operations and reductions alike, as `TensorIterator` does: a reduction plan
marks the reduced dimensions in the same `iteration_plan`.

### Semantics first

Before tuning, define:

- **Accumulator precision** per input type (for example, `float` inputs
  accumulate in `float` or `double`; the choice is documented and tested).
- **Empty input**: identity for `sum`; error or documented identity for
  `min` / `max`.
- **Determinism**: parallel summation changes floating-point addition order.
  State whether results are bitwise reproducible for a fixed thread count,
  and test against that statement.

### CPU (P3)

A reduction over the inner dimension uses per-register accumulators; one
over an outer dimension uses vertical SIMD accumulation across rows.
Threading follows the elementwise grain policy.

### GPU (P4)

A fixed grid (multiprocessor count × k) with grid-stride accumulation per
thread and warp shuffles inside a block, then one final launch. Partials
come from the per-stream workspace and are reused, never allocated per pass.

**Scalar results stay on the device.** Returning `T` from `dot` or
`accumulate` forces a device-to-host synchronization. `dot_out` and
`accumulate_out` write into a 0-d `tensor<T>`, so the caller decides when to
synchronize.

---

## 11. LinearAlgebra adapter

```cpp
template <typename T>
product_expression<T> matmul(tensor_view<const T> const& a, tensor_view<const T> const& b);

template <typename T>
void matmul_out(
    tensor_view<T> c, tensor_view<const T> const& a, tensor_view<const T> const& b,
    T alpha = T(1), T beta = T(0));

template <typename T>
T dot(tensor_view<const T> const& a, tensor_view<const T> const& b);

template <typename T>
product_expression<T> bmm(tensor_view<const T> const& a, tensor_view<const T> const& b);
```

Shapes follow PyTorch's `matmul` rules for vectors, matrices and batches.
Inner dimensions are validated in every build type.

### Assignment and `noalias()`

`matmul` and `bmm` return a `product_expression` that captures its operands
like any expression leaf and computes nothing until assigned. This follows
Eigen's lazy `Product`; PyTorch computes eagerly and offers `out=`. Eigen
needs `noalias()` to avoid the temporary because it cannot see storage
identity; this design can, so the classifier decides.

| Statement | Lowering |
|---|---|
| `tensor<T> c(matmul(a, b))` | Allocate `c`, then `matmul_out(c, a, b)` |
| `c = matmul(a, b)`, classifier reports `none` | `matmul_out(c, a, b, 1, 0)` |
| `c = matmul(a, b)`, any overlap | Packed temporary, then copy into `c` |
| `c.noalias() = matmul(a, b)` | `matmul_out(c, a, b, 1, 0)`, check skipped in Release |
| `c.noalias() += matmul(a, b)` | `matmul_out(c, a, b, 1, 1)` |
| `c.noalias() -= matmul(a, b)` | `matmul_out(c, a, b, -1, 1)` |

The temporary in the overlapping case makes `a = matmul(a, b)` correct
without `noalias()`. The O(rank) check is negligible next to an O(n³)
product.

### Product epilogues

| Expression assigned to `c` | Lowering |
|---|---|
| `matmul(a, b) + c` | GEMM with β = 1; no copy |
| `matmul(a, b) + d` | Copy `d` into `c`, then GEMM with β = 1 |
| `s * matmul(a, b)` | GEMM with α = s |
| `c += s * matmul(a, b)` | GEMM with α = s, β = 1 |

- **Temporary elimination, not fusion.** `matmul(a, b) + d` removes the
  product temporary, but copying `d` is still an extra memory pass.
- **Aliasing before any write.** Classify `c` against `a`, `b` and `d`
  before copying: the copy must not overwrite a GEMM input. `full` overlap
  with `d` takes the no-copy β = 1 row; any other overlap falls back to a
  temporary.
- **True fused epilogues** (bias, activation) are a separate P6 study using
  backend support such as cuBLASLt epilogues, subject to dtype, layout and
  batching constraints.
- Any other tree containing a product evaluates the product into a temporary
  leaf. Eigen recognizes `scalar * product` the same way.

### Routing by size and shape

| Case | Route | Reference point |
|---|---|---|
| Matrix × vector | GEMV | PyTorch `matmul` |
| Vector · vector | DOT | PyTorch `matmul` |
| Tiny matrices (n ≲ 16–20, tuned) | Native unrolled kernel, or MKL JIT / direct call | Eigen `EIGEN_GEMM_TO_COEFFBASED_THRESHOLD` |
| `bmm` on CPU | `gemm_batch_strided` where available; otherwise split the batch across threads | MKL |
| Everything else | Backend GEMM | — |

Small fixed shapes (Eigen's stack-allocated fixed-size matrices) are out of
scope and benchmarked separately.

### Stride mapping

`linalg::multiplication::apply` is row-major with explicit leading
dimensions. Compatible views are passed directly; only unsupported layouts
are packed.

| Operand layout | `transpose` | `ld` |
|---|---|---|
| `stride(1) == 1` | `false` | `stride(0)` |
| `stride(0) == 1` | `true` | `stride(1)` |
| Otherwise | Pack with an identity copy | `cols` |

- **Output.** `stride(1) == 1`: write directly. `stride(0) == 1`: compute
  Cᵀ = Bᵀ·Aᵀ by swapping operands and flipping flags. Otherwise compute
  into a temporary and copy.
- **Size-1 dimensions.** BLAS rejects a leading dimension smaller than the
  row length even where the stride is irrelevant, so `ld` is clamped to at
  least the row length.
- **Aliasing.** Any overlap between `c` and `a` or `b`, including `full`, is
  an error for `matmul_out`; under `noalias()` the check runs in Debug only.

### Backend selection

| Device | `linalg_backend` |
|---|---|
| CPU | `MKL` if `LINALG_HAS_MKL`, else `BLAS_LAPACKE` if available, else `NATIVE` |
| CUDA | `CUDA_BLAS` |
| HIP | `ROCM_BLAS` |
| Metal | `METAL_BLAS` (float only; `double` is a compile-time error) |

### Workspace and lifetime

- Host workspace: per thread, grow-only, sized by `workspace_size()`. The
  current LinearAlgebra API already accepts caller-owned workspace.
- GPU workspace: per stream, from Memory's `gpu_workspace.h`.
- On a non-default stream, `a`, `b`, `c` and the workspace are recorded on
  that stream.

**Allocation contract.** A warmed-up multiplication with a reusable output,
compatible layouts and sufficient workspace performs no tensor-layer
allocation. Packing and workspace growth are measured separately from the
steady-state call, and backend-internal allocation separately from
tensor-layer allocation.

### Factorizations

Reusable factorization objects hold their factors and pivots and offer
`.solve(b)`: `lu_factorization<T>`, `cholesky<T>`, `qr<T>`, and SVD where
the backend supports it. Models: Eigen `PartialPivLU` / `LLT` /
`HouseholderQR`, PyTorch `lu_factor` / `lu_solve`. Factorization and later
solves are measured separately.

### Cleanup

Remove the `#if 0` matrix block in `tensor.h` and the class comment claiming
`tensor * tensor` is matrix multiplication.

---

## 12. Upstream API changes

Memory, Scalars and LinearAlgebra are separate repositories (submodules
under `ThirdParty/`). Changes are designed, tested and landed there first;
Vectorization then bumps the submodule.

### Memory (U1)

- **One allocation per CPU tensor**: control block and data in one aligned
  allocation for sized CPU storage; separate control block for GPU and
  adopted storage ([section 14](#14-allocation-and-workspace)).
- **Uninitialized allocation** for numerical storage, so kernels that write
  every element do not zero the payload first.
- **Offset-aware copy**, synchronous and asynchronous:
  `copy(src_base, src_offset, dst_base, dst_offset, count, ctx)`. It
  replaces the interior-slice rejection in `data_ptr(data_view)`; the
  caching allocator keeps looking up blocks by base address.
- **`copy_async`** on `retained_ptr` returning a `copy_token` that keeps both
  endpoints alive.
- **Stream events** (`record_event` / `wait_event`) for cross-stream
  ordering, separate from `record_stream`.
- **`record_stream`** becomes a no-op for adopted memory with an invalid
  allocation id instead of failing the allocator lookup.

### LinearAlgebra (U2)

- `alpha` / `beta` on `multiplication::apply`; the existing signature stays
  as the `alpha = 1, beta = 0` overload.
- GEMV and DOT entry points with the same leading-dimension rules as GEMM.
- Rectangular strided-batched GEMM: `rows`, `columns`, `depth`, a batch
  stride per operand (0 for broadcasting), positive leading dimensions, a
  stream and a workspace. The square
  `batched_matrix_multiplication(dim, count, …)` stays as a wrapper.

### Scalars (U3)

- Metal: per-allocation `MTLBuffer` with `setBuffer:offset:`, shared
  command buffer with commit-and-continue, and a stream handle for it
  ([section 13](#13-backend-launch-unification)).
- `simd<T>` lane-reverse permute and register-tile transpose operations for
  `flip` and the tiled transpose on every CPU ISA, so Vectorization needs no
  raw intrinsics ([section 15](#rules-for-new-code)).

---

## 13. Backend launch unification

The public API is already one surface (`tensor<T>`, `a + b`,
`operator=`); the launch signatures behind `expressions_evaluator::run` /
`fill` still diverge ([vectorization_backends.md §3](vectorization_backends.md#3-launch-contracts)).
Unification lives behind the evaluator; the public `tensor` API does not
change.

### No runtime backend objects

Do not add a runtime-polymorphic `IComputeBackend::launch`. Expression
fusion is a compile-time tree, and a vtable on the CPU hot path is the wrong
model. This follows Eigen's compile-time `Device` parameter rather than
PyTorch's runtime dispatcher, which pays for its flexibility with per-call
dispatch cost this library does not need: Memory already fixes the GPU
backend at configure time.

### Target launch contract

Every backend implements the same functions over the lowered tree and the
iteration plan; `run()` only chooses CPU or GPU:

```cpp
template <typename Lowered, typename T>
void eval_expr(Lowered const& expr, strided_ref<T> out, iteration_plan const& plan,
               gpu_stream_t stream);

template <typename T>
void eval_fill(strided_ref<T> out, T value, gpu_stream_t stream);

template <typename Lowered, typename Op>
void eval_reduce(Lowered const& expr, iteration_plan const& plan, Op op,
                 strided_ref<typename Op::value_type> out, gpu_stream_t stream);
```

- **CPU**: the `loop2d` planner path; `stream` unused.
- **CUDA/HIP**: `gpu_eval_kernel` with the offset calculator; nvcc/hipcc keep
  compiling the C++ tree.
- **Metal**: JIT MSL internally, called with the same signature, so the
  dispatcher no longer special-cases a `tensor&` argument.
- `metal_backend::dispatch(name, …)` stays test-only. Metal's named fill and
  reduce kernels remain its implementation of `eval_fill` / `eval_reduce`.

### Semantics made identical

| Topic | Unified behavior |
|---|---|
| Streams | A real `gpu_stream_t` on every GPU backend; on Metal it names a command queue / shared command buffer. Silently ignoring `stream` is the current bug |
| Sync | Asynchronous launch everywhere; the host waits on a read or an event ([section 5](#metal--pipeline-cache-synchronization-and-buffers)) |
| Types | Metal float-only remains an allocator constraint; no second Metal `double` evaluator |
| Ops | One functor set. Metal `cdf` / `inv_cdf` get an MSL approximation or fail through `VECTORIZATION_CHECK`, not a backend-specific `std::runtime_error` |
| Reductions | `accumulate` / `hmin` / `hmax` branch on `device()` the way `run` does |
| Errors | `VECTORIZATION_CHECK` at the evaluator layer; driver and JIT failures only at the CUDA/HIP/Objective-C boundary |

### Not unified

- CPU `simd<T>` ISAs — already one interface.
- GPU `simd<T>` with `size == 1` — CUDA/HIP device math only. A fake Metal
  `simd<float>` would unify nothing: MSL is a separate toolchain that never
  sees the C++ tree. The seam is evaluator launch, not packet math.

---

## 14. Allocation and workspace

### Targets by storage kind

| Storage | Target | Reference point |
|---|---|---|
| CPU owning | One aligned allocation holding control block and payload; inline shape metadata for common ranks | `std::make_shared` |
| CUDA/HIP | Pooled host-side control blocks plus cached device storage; no new device allocation or synchronization during steady-state reuse. A host reference count cannot sit beside a device payload | PyTorch CUDA caching allocator |
| Metal | Like CPU (unified memory), with the `MTLBuffer` created once per allocation | PyTorch MPS allocator |
| Adopted external | Explicit ownership/deleter rules; separate control block | `retained_ptr::adopt` |

### Contracts

1. Uninitialized allocation for storage that a kernel fully overwrites.
2. Assignment into a correctly sized destination reuses its storage.
3. Common shapes, strides, views and iteration plans allocate no metadata on
   the heap; only ranks above the inline capacity spill.
4. CPU scratch arenas and per-stream GPU workspaces serve packing and
   reduction partials, and are released or reset only when their users
   finish.
5. Queued GPU operands and temporaries stay valid until completion
   ([section 9](#9-copies-and-transfers)).

---

## 15. clang-tidy conformance

All first-party C++ under `include/`, `src/` and `Testing/` is clang-tidy
clean under the repository `.clang-tidy`, with warnings treated as errors.
This covers existing code and everything each phase in
[section 16](#16-implementation-phases) adds. `ThirdParty/` is out of scope;
findings there are fixed upstream.

### Current configuration

- **Checks** (`.clang-tidy`): `bugprone-*`, `performance-*`,
  `portability-*`, `modernize-use-nullptr`, `modernize-use-override`,
  `modernize-use-equals-default`, `readability-braces-around-statements`.
  Disabled: `bugprone-easily-swappable-parameters`,
  `bugprone-narrowing-conversions`, `portability-avoid-pragma-once`.
- **CMake**: `VECTORIZATION_ENABLE_CLANGTIDY` (`setup.py` token
  `clangtidy`) attaches clang-tidy to the `Vectorization` target through
  `xsigma_target_clang_tidy()` with `-warnings-as-errors=*`, header filter
  `include|src|Cmake|Testing`, and `ThirdParty/` excluded on LLVM 19 and
  later. `VECTORIZATION_ENABLE_FIX` applies fixes in place.
- **CI**: the `static-analysis / clang-tidy` job on Ubuntu, `avx2`, GPU
  backend `none`, `--testing=OFF`, with clang-tidy from apt.
- **lintrunner**: the `CLANGTIDY` linter covers `include/**` and
  `Testing/**`.

### Coverage gaps

| Gap | Effect | Fix |
|---|---|---|
| The `Vectorization` target compiles one file, `src/terminals/sizes_and_strides.cpp`, and CI configures with `--testing=OFF` | `tensor.h`, `expressions/` and the planner are never analyzed in CI. Header-only templates are analyzed only where a checked translation unit instantiates them | Attach `xsigma_target_clang_tidy()` to `VectorizationCxxTests` as well, and turn testing on in the clang-tidy CI job, so the tests' instantiations cover the headers |
| lintrunner `CLANGTIDY` omits `src/**` | Local lint misses the compiled sources | Add `src/**/*.cpp` and `src/**/*.h` to `include_patterns` |
| CI analyzes only `avx2` with GPU backend `none` | NEON, `VECTORIZATION_HAS_METAL` and CUDA/HIP branches are never analyzed | Add a macOS `neon` + `metal` clang-tidy job. Analyze CUDA/HIP host code where a toolkit is available on the runner; document device-only code as not analyzed |
| clang-tidy version not pinned (Ubuntu 24.04 apt ships 18; the baseline below used 22) | Different findings locally and in CI; no `--exclude-header-filter` before 19 | Pin one version, 19 or later, in CI and document it in the README |
| Suppressions | A bare `// NOLINT` on the expression constructor in `tensor.h`; `NOLINTNEXTLINE` entries naming checks this configuration does not enable (`cppcoreguidelines-no-malloc`, `modernize-macro-to-enum`, `*c-array*`) | Name the check and give a reason; remove suppressions of disabled checks unless kept on purpose for the shared XSigma configuration |

### Baseline

Measured October 31, 2026 with clang-tidy 22.1.2 (Homebrew LLVM), `neon`,
GPU backend `none`, Release, on `TestTensor.cpp` and `TestTensorShapes.cpp`
with the repository header filter: **65 unique findings**. Other test files
and configurations have not been measured.

| Check | Count | Where |
|---|---|---|
| `readability-braces-around-statements` | 49 | `expressions_evaluator.h` (21), `tensor.h` (11), `TestTensor.cpp` (6), `expressions_functors.h` (4), `expression_interface.h` (3), `strided_ref.h` (2), `expression_interface_loader.h` (2) |
| `performance-unnecessary-copy-initialization` | 5 | `TestTensor.cpp` |
| `bugprone-use-after-move` | 3 | `TestTensor.cpp` (moved-from state checks) |
| `bugprone-reserved-identifier` | 3 | Include guards `__expressions_builder_h__` and `__VECTORIZATION_EXPRESSIONS_INCLUDES_INSIDE__` |
| `bugprone-casting-through-void` | 2 | `buffers_overlap()` in `expression_interface.h` |
| `performance-noexcept-move-constructor` | 1 | `tensor` move assignment |
| `bugprone-exception-escape` | 1 | `~stream_guard` |
| `performance-enum-size` | 1 | `reduce_op` |

The backlog is cleared in P0, before the coverage fixes turn these findings
into CI failures.

### Rules for new code

Design elements in this plan that run into enabled checks:

| Design element | Check | Rule |
|---|---|---|
| `tensor` / `tensor_view` moves ([section 4](#4-tensor-and-tensor_view)) | `performance-noexcept-move-constructor` | Move constructor and move assignment are `noexcept`; `retained_ptr` moves must be `noexcept` too (Memory U1) |
| `tensor::operator=` (deep copy), `tensor_view::operator=` (write-through) | `bugprone-unhandled-self-assignment` | Explicit self-assignment check; for a view, self-assignment is `full` overlap and returns immediately |
| Forwarding-reference constructors (`tensor(E&&)`, `product_expression`, `store_operand`) | `bugprone-forwarding-reference-overload` | Constrain the template with a trait, as `tensor(E&&)` already does, then remove its bare `NOLINT` |
| Rvalue capture into expression nodes ([section 5](#capture)) | `bugprone-use-after-move`, `performance-move-const-arg` | Move each operand once into the node; never `std::move` a `const` operand |
| `overlap_status`, loop tier, `reduce_op` | `performance-enum-size` | `enum class … : std::uint8_t` |
| `noexcept` loops (`loop_row`, `loop2d`), destructors (`stream_guard`, `copy_token`, `noalias_proxy`) | `bugprone-exception-escape` | No `VECTORIZATION_CHECK` inside; validate in the planner before the loop; destructors never throw |
| 32-bit GPU indexing and offset arithmetic ([section 5](#cuda--hip--offset-calculator-and-indexing-width)) | `bugprone-implicit-widening-of-multiplication-result`, `bugprone-too-small-loop-variable`, `bugprone-misplaced-widening-cast` | Widen before multiplying (`static_cast<int64_t>(i) * stride`); the loop counter type matches its bound after the index-width decision. `bugprone-narrowing-conversions` is disabled, so the span check, not clang-tidy, guards narrowing to 32 bits |
| Overlap classifier address comparisons ([section 6](#6-aliasing-and-overlap)) | `bugprone-casting-through-void` | `reinterpret_cast<std::uintptr_t>(ptr)`, not a cast through `void*` |
| Metal pipeline cache ([section 5](#metal--pipeline-cache-synchronization-and-buffers)) | `bugprone-dynamic-static-initializers` | Keep the cache a function-local static, never a namespace-scope variable in a header |
| `tensor_view<const T>` storage | `portability-std-allocator-const` | Allocate as `std::remove_const_t<T>`, as `tensor_impl` does ([section 3](#3-layers-storage-geometry-kernel-descriptor)) |
| Input views ([section 4](#4-tensor-and-tensor_view)) | `performance-unnecessary-value-param` | Inputs take `tensor_view<const T> const&`; outputs are taken by value only because they are written through |
| New headers and macros | `bugprone-reserved-identifier`, `bugprone-macro-parentheses` | `#pragma once`; `VECTORIZATION_*` macro names with parenthesized arguments |
| Every `if` / `else` / `for` / `while` body | `readability-braces-around-statements` | Always braced |

Lane reversal for `flip` and the tiled transpose use `simd<T>` operations
from Scalars rather than raw intrinsics in Vectorization. Missing permute
and transpose operations are added to Scalars (U3), where ISA-specific code
already lives.

### Suppression policy

- Use `// NOLINTNEXTLINE(check-name)` with a reason; no bare `NOLINT`, and
  no `NOLINTBEGIN` / `NOLINTEND` around whole files or classes.
- Disabling a check repository-wide is a change to `.clang-tidy` with a
  comment giving the reason, not a scattering of local suppressions.
- Run `VECTORIZATION_ENABLE_FIX` only on a clean working tree and review the
  diff; fix mode edits sources in place.

### Running it

```sh
cd Scripts
python3 setup.py config.build.clangtidy
```

Or analyze single files from an existing build directory with
`clang-tidy -p <build-dir> <file>`, or through `lintrunner` with the
`CLANGTIDY` linter.

---

## 16. Implementation phases

Each phase has a correctness gate and a performance gate. Performance
thresholds are hypotheses until confirmed against a measured baseline before
the phase starts. Every correctness gate also requires zero clang-tidy
findings ([section 15](#15-clang-tidy-conformance)) in each configuration the
phase changes.

| Phase | Repository | Scope | Status | Correctness gate | Performance gate |
|---|---|---|---|---|---|
| P0 | Vectorization | Result shape from broadcasting; every leaf's shape validated; Release checks on view operations; overlap classifier raising errors; copy constructor keeps source strides; clang-tidy baseline cleared and coverage gaps closed ([section 15](#15-clang-tidy-conformance)) | Partly done: classifier and clang-tidy work missing | Every [section 1](#implementation-status) item marked Done has a passing Release test; the clang-tidy CI job analyzes the test target and passes | Release benchmarks at today's speed; shape checks outside the inner loop |
| P1 | Vectorization | `retained_ptr` storage, explicit offset, views retain storage, rvalue capture into expressions | Not started | Lifetime tests (view outlives owner, deferred expression) clean under AddressSanitizer | CPU tensor construction and destruction: one heap allocation after U1 |
| P2 | Vectorization | `tensor` / `tensor_view` split; `noalias()` proxy for elementwise assignment | Not started | View assignment writes through; `noalias()` overlap caught in Debug; existing tests pass unchanged | Expression building with `tensor<T>` leaves adds no atomic operations |
| P3 | Vectorization | CPU planner tiers 0–2 with `run_range`; contiguous fast path; `loop2d`; threading policy (bytes and op cost, intra-row splits, BLAS coordination); tiled transpose; `flip`; tier-2 specialization; planner-owned prefetch; aligned-source variant removed; CPU reductions | Partly done: planner, `loop2d`, row-split threading | Strided, broadcast, `flip` and reduction tests pass; serial SIMD fallback taken with parallel support disabled | Tier 0 ≥ 90% of STREAM triad bandwidth on one thread and scales with threads; tier 1 ≥ 80% for rows ≥ 256 elements; latency at n ≤ 64 within 10% of today; `t().contiguous()` ≥ 50% of tier-0 copy bandwidth |
| U1 | Memory | [Section 12](#memory-u1) | Not started | Landed upstream with tests; submodule bumped | Offset-aware copy at device-copy bandwidth |
| U2 | LinearAlgebra | [Section 12](#linearalgebra-u2) | Not started | Landed upstream with tests; submodule bumped | GEMV, DOT and batched GEMM at backend-library speed |
| U3 | Scalars | [Section 12](#scalars-u3) | Not started | Landed upstream with tests; submodule bumped | Metal interior-slice dispatch with no per-call buffer wrapping |
| P4 | Vectorization | CUDA/HIP offset calculator with span-checked 32-bit indexing and vector loads; strided GPU destinations; Metal pipeline cache, batched command buffers, per-allocation `MTLBuffer`; GPU reductions; offset-aware transfers; unified launch contract ([section 13](#13-backend-launch-unification)) | Not started | Device clone of an interior slice without host memory (Metal locally; CUDA/HIP in CI); large-stride small tensors take the wide-index path; same launch signature on every backend; macOS `neon` + `metal` clang-tidy job passes | Tier 0 and strided tiers in GB/s on each GPU backend; Metal per-call host overhead after the first call in µs and free of string building |
| P5 | Vectorization | `matmul_out`, `dot` / `dot_out`, factorization objects; lazy `matmul` / `bmm` with classifier-based lowering; epilogues; size routing | Not started | Rectangular, transposed-view, strided and aliased-output GEMM tests pass | Warmed-up `c = matmul(a, b)` with reusable output, compatible layouts and sufficient workspace performs no tensor-layer allocation; small-matrix GEMM within 2× of Eigen fixed-size products |
| P6 | Vectorization | Optional, measured: streaming stores, true fused epilogues (cuBLASLt), tiny-matrix kernel tuning | Not started | Behind flags; no regression on existing tiers | Streaming stores 1.2–1.5× on write-heavy tier-0 work where they win; no regression on read-after-write chains |

Dependencies: P0 fixes incorrect results and goes first. U1, U2 and U3 run
in parallel with P1–P3. P1's allocation gate needs U1; P4 needs U1 and U3;
P5 needs U2. Only P6 is a tuning phase; every other phase meets its own
performance gate.

---

## 17. Verification and benchmark protocol

### Correctness tests

In addition to numerical values:

- Result shape of expression construction, including broadcasting.
- Invalid shapes, axes, reshape sizes and slice steps rejected in Release.
- View lifetime: a view returned from a function; deferred expressions
  holding temporaries.
- Overlap: exact aliasing allowed; shifted, interleaved and `too_hard` cases
  rejected.
- `noalias()`: proven overlap rejected in Debug; `too_hard` layouts run;
  internal overlap rejected in every build type; `+=` / `-=` match the
  non-`noalias()` result.
- Zero and negative slice steps rejected in Release; negative `start` /
  `stop`; `flip` on every dimension returns a reversed copy and leaves the
  source untouched.
- Strided and broadcast arithmetic on every tier, including non-contiguous
  destinations.
- Serial and parallel builds give identical results around the threading
  threshold.
- GPU index-width selection for small tensors with large strides.
- Rectangular and batched GEMM with transposed and strided views.
- `c = matmul(a, b)` on disjoint operands writes directly; on overlapping
  operands uses a temporary.
- `c.noalias() = / += / -= matmul(a, b)` against a reference, with an
  allocation counter confirming no temporary.
- Epilogues against two-pass references, including `d` aliasing `c`, `a`
  or `b`.
- Stream transitions: allocation reuse after `record_stream`; ordering
  through events.
- Reductions on contiguous and strided inputs over each axis, on CPU and
  GPU, including empty inputs and the documented determinism policy.
- Backend parity: every backend produces the same results and error types
  for the shared functor set.

### Static analysis

- clang-tidy with warnings as errors on the library and test targets, in
  every CI configuration ([section 15](#15-clang-tidy-conformance)).
- Every new suppression names its check and gives a reason; reviews reject
  bare `NOLINT`.

### Measurements

- **Bandwidth.** GB/s per tier against a STREAM triad baseline on the same
  machine, on CPU and each GPU backend. Record the byte-count convention;
  distinguish algorithmic traffic from measured hardware traffic.
- **Latency.** Nanoseconds per call at n = 1, 16, 64 and 256, including
  planning and overlap checks.
- **Thread scaling.** Tier 0 and tier 1 from one thread to all cores; sizes
  just below and above the threading threshold.
- **Layouts.** Contiguous, row broadcast, column broadcast, transpose copy,
  `flip`, general strided.
- **Allocation.** Construction/destruction latency and counts; cached versus
  new device allocations; peak live, retained and reserved bytes;
  steady-state reuse. Low ranks reported separately from ranks that spill.
- **Products.** Square and rectangular GEMM from n = 2 to 4,096;
  matrix × vector; batched small matrices; `c = matmul(a, b)` with and
  without `noalias()`; epilogues; packing and workspace growth separate from
  the steady-state call; factorization separate from solve.
- **Metal.** First-call and steady-state host overhead; chained expressions
  with and without a host read between them.

### Protocol

- Separate allocation-inclusive runs from preallocated-output runs, and
  first-call setup (compilation, workspace growth) from warmed-up steady
  state.
- Match thread counts, dtype, precision policy, layouts, numerical backend
  and CPU ISA across comparisons. Coordinate BLAS and tensor thread counts.
- Report latency for small tensors, bandwidth for elementwise work and
  throughput for products.
- Time GPU kernels with device timers; time complete operations with
  synchronization at the measurement boundary; report host enqueue overhead
  separately.
- Cover empty and scalar tensors, small, cache-resident and larger-than-cache
  arrays; square, rectangular, transposed and batched products.
- Compare against Eigen and LibTorch. Eigen dynamic and fixed-size cases are
  separate categories; eager and compiled PyTorch are separate baselines.
- Record hardware, compiler and build flags, dependency versions, thread
  settings and run-to-run variability.

Attribute remaining cost to allocation, planning, memory traffic, packing,
synchronization or the numerical kernel.

---

## 18. Key files

- `include/terminals/tensor.h` — `tensor<T>`, views, `clone()`,
  `contiguous()`, `init_from_expression()`, the `#if 0` matrix block.
- `include/terminals/sizes_and_strides.h` — inline sizes/strides storage.
- `include/expressions/strided_ref.h` — kernel-side leaf descriptor.
- `include/expressions/expression_interface.h` — `store_operand()`,
  expression nodes, `buffers_overlap()` / `expression_aliases_ptr()`.
- `include/expressions/expression_shape.h` — broadcast shape inference,
  `rebase_leaves()`.
- `include/expressions/iteration_plan.h` — CPU planner, tiers, `loop2d`,
  threaded row split.
- `include/expressions/expressions_evaluator.h` — `run()` / `fill()`
  dispatch, contiguous SIMD loop, prefetch and alignment variants.
- `include/expressions/expressions_evaluator_gpu.h` — CUDA/HIP kernels and
  reductions.
- `include/expressions/expressions_evaluator_metal.h` — MSL emitter.
- `include/expressions/expressions_builder.h` — operators, including
  elementwise `*`.
- `include/expressions/reduce_op.h` — reduction operations.
- `ThirdParty/Scalars/include/backend/gpu/metal/metal_dispatch.mm` — Metal
  buffers and dispatch.
- `ThirdParty/Memory/include/common/retained_ptr.h`, `data_ptr.h`,
  `data_view.h` — storage primitives.
- `ThirdParty/Memory/include/gpu/cuda_caching_allocator.h`,
  `gpu_workspace.h` — GPU allocation and workspace.
- `ThirdParty/LinearAlgebra/include/matrix_operation/multiplication.h`,
  `multiplication_batched.h` — GEMM entry points.
- `ThirdParty/LinearAlgebra/include/common/backend.h` — `linalg_backend`.
- `.clang-tidy`, `Cmake/tools/clang_tidy.cmake`, `.lintrunner.toml`,
  `.github/workflows/ci.yml` (`static-analysis` job) — clang-tidy checks,
  CMake hook, local lint and CI.

---

## 19. Sources merged into this plan

| Former document | Disposition |
|---|---|
| `vectorization_tensor_design.md` | Renamed to this file; sections 1–12 and the phase table are its base |
| `speed_tensor_plan.md` | Removed; already merged into the design, no remaining differences |
| `tensor_performance_study_plan.md` | Removed; its findings are folded in as listed below |
| `vectorization_backends.md` §5–6 (Unification, Recommended order) | Moved to [section 13](#13-backend-launch-unification); the current contract stays in that file |

Changes made while merging, where sources disagreed or were refined:

- The problem list is replaced by an implementation status table
  ([section 1](#implementation-status)).
- GPU 32-bit indexing checks the maximum reachable offset of every operand,
  not only the element count ([section 5](#cuda--hip--offset-calculator-and-indexing-width)).
- Control-block coallocation applies to CPU storage only; GPU storage uses
  pooled host metadata ([section 14](#14-allocation-and-workspace)).
- `matmul(a, b) + d` is described as temporary elimination with an extra
  copy pass, aliasing is classified before any write, and `+ c` needs no
  copy ([section 11](#product-epilogues)).
- The P5 "allocates nothing" gate is qualified to warmed-up calls with
  reusable output, compatible layouts and sufficient workspace.
- The threading threshold depends on bytes and operation cost, rows can be
  split internally, and BLAS threads are coordinated
  ([section 5](#iteration-plan-and-threading)).
- Streaming stores can hurt read-after-write chains and stay opt-in.
- Reductions define accumulator precision, empty-input behavior and a
  determinism policy before tuning ([section 10](#10-reductions)).
- The backend document's open choice between asynchronous and synchronous
  GPU execution is resolved to asynchronous, matching the Metal batching
  plan ([section 2](#2-decisions)).
- P6 is kept only for optional, measured tuning; all other phases carry
  their own performance gate.
- Scalars is added as an upstream (U3) because Metal dispatch now lives
  there.

Added after the merge:

- clang-tidy conformance ([section 15](#15-clang-tidy-conformance)), with a
  measured baseline, coverage fixes in P0, and a clang-tidy requirement in
  every phase's correctness gate.
