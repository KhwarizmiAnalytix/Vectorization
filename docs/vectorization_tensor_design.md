# Tensor Design

Scope: `include/terminals/tensor.h`, `include/expressions/`, and the parts of
Memory and LinearAlgebra the tensor builds on, as of October 2026. This is a
**proposed** design, not the current contract. It covers storage ownership,
the `tensor` / `tensor_view` split, broadcasting and strided evaluation,
aliasing and `noalias()`, slice steps, transfers, reductions, and the
LinearAlgebra adapter, plus a phased implementation order with per-phase
performance gates.

The companion speed plan at
[speed_tensor_plan.md](speed_tensor_plan.md) is now fully merged here; the
separate document is retained for change history only.

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
7. [Slice steps](#7-slice-steps)
8. [View operations](#8-view-operations)
9. [Copies and transfers](#9-copies-and-transfers)
10. [Reductions](#10-reductions)
11. [LinearAlgebra adapter](#11-linearalgebra-adapter)
12. [Upstream API changes](#12-upstream-api-changes)
13. [Implementation phases](#13-implementation-phases)
14. [Verification](#14-verification)
15. [Key files](#15-key-files)

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
| GPU `clone()` of any slice is staged through host memory | `tensor::clone()` | Device → host → device for an interior slice |
| Negative slice steps are accepted, and with the default `stop` the result is empty | `slice()` | `stop = -1` means "to end", so a reverse slice can't reach index 0; negative strides then reach code that casts strides to `size_t` |
| Matrix operations are under `#if 0`; the class comment says `*` is matrix multiplication | `tensor.h` | `*` is elementwise; there is no tensor-level GEMM |
| CPU evaluation is single-threaded | `expressions_evaluator::run()` | Large elementwise work can't use all of a multi-core server's memory bandwidth |
| Prefetch is issued one SIMD register ahead, per leaf, every iteration | `expression_loader::prefetch()` | The hardware prefetcher already covers that distance on sequential streams, so the instructions do no useful work |
| Hot loops are instantiated for both aligned and unaligned source loads | `run()`, `accumulate()` | Doubles instantiations; `loadu` on aligned data costs the same as `load` on current x86 and Arm cores |
| CUDA/HIP kernel is one thread per element with scalar loads | `gpu_eval_kernel` | No vector loads, so memory-bound kernels use bandwidth less efficiently |
| GPU reduction allocates and frees a partials buffer every pass | `reduce_gpu_partials()` | Several launches plus allocator traffic per reduction |
| Metal regenerates the MSL source string on every evaluation | `run_metal()`, `metal_fuse_source()` | String building every call, then a copy and hash of the whole source under a global mutex |
| Every Metal dispatch commits and waits | `synchronous_dispatch()` | A chain of expressions costs one CPU–GPU round-trip per operation |
| Metal wraps operands with `newBufferWithBytesNoCopy` on each call | `make_buffer()` | Per-call wrapping, and the call requires page alignment, so interior slices cannot be passed |
| Promoting `data_ptr` to `retained_ptr` builds a separate control block | `retained_ptr(data_ptr&&)` | Two heap allocations per tensor |

---

## 2. Decisions

| Question | Decision | Reference point |
|---|---|---|
| Copy semantics | Separate `tensor<T>` (owning, deep copy) and `tensor_view<T>` (shallow copy, assignment writes through) | Eigen `Matrix` vs `Block` / `Ref`; not PyTorch's shared handles |
| Partial overlap between destination and source | Raise an error | PyTorch `assert_no_partial_overlap`; not Eigen's silent temporary |
| Opting out of the overlap check | `dst.noalias() = expr`: the caller asserts there is no overlap; for products, `c = matmul(a, b)` already writes directly when the classifier reports `none`; `noalias()` skips the O(rank) check and allows `too_hard` layouts to run without a temporary | Eigen `noalias()`; PyTorch only has `out=` and an internal `check_mem_overlap(false)` |
| Negative slice steps | Forbidden: `step < 1` is an error and strides are never negative; `flip` returns a copy | PyTorch; NumPy supports them |
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
  device. On GPU, `strided_ref` is lowered to 32-bit indexing when the element
  count is below 2³¹ ([section 5 GPU](#gpu-and-metal)).
- **One allocation per tensor** (Memory upstream, [section 12](#12-upstream-api-changes)):
  for sized allocations the control block is placed in the same allocation as
  the data, as `std::make_shared` does. Adopted external buffers retain the
  separate control block.

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
4. Sort dimensions by output stride, then merge dimensions that are adjacent
   in memory for every operand. Strides are never negative
   ([section 7](#7-slice-steps)), so no dimension needs reversing.
5. Classify destination/source overlap ([section 6](#6-aliasing-and-overlap)),
   unless the assignment goes through `noalias()` in a Release build.
6. Pick a loop tier.

| Tier | Condition after merging | Loop |
|---|---|---|
| 0 | 1-D, every leaf has stride 1 | SIMD loop via `loop2d` with `outer_n == 1` |
| 1 | Innermost stride is 1 for every operand (outer broadcast, e.g. `x[m,n] + b[n]`) | `loop2d` over rows; SIMD inner loop on re-based pointers |
| 2 | Some leaf has innermost stride 0 (column broadcast) | `loop2d` with splatted load for the broadcast leaf; compile-time specialization for one broadcast operand |
| 3 | Anything else | Tiled transpose or lane-reversing `flip` when shape permits; scalar loop otherwise on CPU; offset-calculator kernel on GPU |

#### Iteration plan and threading

The planner's result is a splittable range consumed by one or more threads:

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

- **Threading.** Above a grain size (start at 32,768 elements, as PyTorch
  does, and tune per backend), the range is split with a `parallel_for`.
- **Chunk edges.** Chunk edges are rounded to multiples of
  `simd::size × VECTORIZATION_PACKET_SIZE` so each chunk retains the aligned
  body loop; only the first and last chunks peel.
- **Thread-local data.** Worker threads use `strided_ref` and raw pointers,
  never `tensor_view`. This avoids contended reference-count updates on a
  shared control block.
- **Reference points.** PyTorch: `TensorIterator::for_each` with
  `at::parallel_for`. Eigen: `TensorExecutor` with `ThreadPoolDevice`.

#### Contiguous fast path

Planning costs O(leaves × rank) plus a sort — more than the arithmetic for
small tensors. Before planning, an O(leaves) check is run:

1. Every leaf has the `contiguous` flag from `tensor_impl::flags_`.
2. Every leaf has the destination's shape.

When both hold, evaluation goes straight to tier 0 with no sort, merge or
lattice test. The overlap check is ordered cheapest first:

1. Different storage → `none`.
2. Same base pointer, sizes and strides → `full`.
3. Only then the lattice test.

Two distinct `tensor<T>` objects each own their storage and can never
partially overlap, so for them step 1 or 2 always decides. A freshly
constructed `tensor<T>` destination skips the check entirely.

The target is Eigen-like cost for trees of packed tensors, not
TensorIterator's microsecond-level setup.

#### `loop2d` inner kernel

Tiers 0–2 share one kernel shape, following PyTorch's `loop2d`:

```cpp
// Inner dimension: unit stride (or stride 0 for a broadcast leaf).
// Outer dimension: one pointer increment per operand per row.
template <typename Lowered>
void loop2d(Lowered const& expr, char** ptrs, int64_t const* outer_strides,
            int64_t inner_n, int64_t outer_n);
```

- Tier 0 is `outer_n == 1`. Tier 1 differs only in having `outer_n > 1`.
- Rows of a few elements (`x[m, 3] + b[3]`) pay one pointer increment per row,
  not a full peel, unrolled body and tail per row.
- Unit-stride loads and stores in the 2-D kernel are unaligned, because the
  row start moves when the row stride is not a multiple of the SIMD width.
- Rank above 2 is handled by stepping an outer counter (PyTorch's
  `DimCounter`). No index is ever computed by division on the CPU.

#### Tiled transpose and `flip` kernels

`clone`, `contiguous`, conversion from a view, `flip`, and the GEMM packing
step ([section 11](#11-linearalgebra-adapter)) all go through the planner.
A purely scalar tier 3 would make `a.t().contiguous()`, the most common
tier-3 case, the slowest copy in the library.

- **Tiled transpose.** When exactly one 2-D pair of dimensions is swapped
  between destination and source, copy in 8×8 or 16×16 tiles, transposed in
  registers. Other tier-3 layouts stay scalar.
- **Inner dimension choice.** Planner step 4 sorts by output stride, then
  breaks ties by the stride of the largest source operand.
- **`flip`.** Load contiguous registers, reverse the lanes with a permute, and
  store. Reading back to front one element at a time is not acceptable.

#### Tier 2 specialization

A runtime flag per leaf inside the hot loop is often not moved out of the
loop when several leaves and packet unrolling are involved.

- Build the compile-time specialization for "exactly one leaf has inner stride
  0" (the bias-add case) in P3, as PyTorch's scalar-operand `vectorized_loop`
  does.
- Other combinations use `loop2d` with the broadcast value splatted once per
  row.

#### Prefetch and alignment

- **Prefetch.** The plan owns prefetch distance. Sequential unit-stride loops
  either issue no prefetch or prefetch 256–512 bytes ahead, chosen by
  benchmark. Tier 1 adds a prefetch of the next row's start when the row
  stride is large. The per-leaf, per-iteration prefetch of one register ahead
  is removed; the hardware prefetcher covers that distance on sequential
  streams.
- **Alignment.** Remove the `src_aligned` template variant. Keep the
  destination peel, which avoids stores that split a cache line on AVX-512.
  This roughly halves the instantiations per expression type.

#### Streaming stores (P6, optional)

When the destination does not overlap any source and is larger than about half
the last-level cache, use non-temporal stores. This avoids reading every
destination line before writing it and typically gives 1.2–1.5× on
write-heavy elementwise work. Neither Eigen nor PyTorch does this; glibc and
MKL `memcpy` do. This is an opt-in measured at P6.

### GPU and Metal

#### CUDA / HIP — offset calculator and indexing width

- **One size array.** After broadcasting, every operand has the same sizes;
  only strides differ. Store the sizes (as fast divisors) once and the strides
  per operand, like PyTorch's `OffsetCalculator`. A `strided_ref` per leaf with
  `MaxRank = 8` is 144 bytes, so ten leaves would mean about 1.4 KB of kernel
  parameters.
- **Lowering depends on the tier.** Tier 0 leaves lower to `T*`, not
  `strided_ref`.
- **32-bit indexing.** Use 32-bit indices when the element count is below 2³¹,
  and split larger launches, like PyTorch's `can_use_32bit_indexing`.
  `IntDivider` only speeds up 32-bit division; 64-bit division is very slow on
  GPUs.
- **Vector loads.** When every leaf is 16-byte aligned and contiguous, each
  thread processes 4 elements with 16-byte loads and stores, like PyTorch's
  `vectorized_elementwise_kernel`. Other cases use the scalar unrolled kernel.
- **Reductions.** A fixed grid (multiprocessor count × k) with grid-stride
  accumulation per thread and warp shuffles inside a block, then one final
  launch. There is no per-pass allocation ([section 10](#10-reductions)).

#### Metal — pipeline cache, synchronization and buffers

- **Cache key.** The pipeline is held in a function-local static per
  expression type `E`, tier and rank. MSL source is generated only on the
  first call, so later calls build no strings and take no global lock. Scalar
  values already travel in a buffer, so they do not change the source.
- **Batched command buffers.** Evaluations are encoded into a shared command
  buffer and committed without waiting. The host waits only when it reads the
  data or when a stream event requires it, as PyTorch MPS's `MPSStream` does
  with commit-and-continue.
- **One `MTLBuffer` per allocation.** Because storage is a `retained_ptr`
  ([section 3](#3-layers-storage-geometry-kernel-descriptor)), the `MTLBuffer`
  is created once at allocation and kept with it. A view is passed as
  `setBuffer:offset:` using `tensor_impl::offset_`. This removes per-call
  wrapping and makes interior slices work, which `newBufferWithBytesNoCopy`
  cannot do because it requires page alignment.

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

- **Lattice test.** For two views with equal strides that each divide the
  next, decompose the offset difference greedily across
  dimensions. The views share an element exactly when the difference
  decomposes and each coefficient is smaller than its dimension size. This
  covers shifted and interleaved slices.
- **Deviation from PyTorch.** PyTorch lets `too_hard` cases run silently. This
  design errors on them.
- **Escape hatches.** For `partial`, the error message names
  `dst = src.clone();`. For `too_hard`, it also names `dst.noalias() = …` for
  callers who know the operands are disjoint.
- **Internal overlap.** A destination that overlaps itself (stride 0 from
  `expand`) is always an error, with or without `noalias()`.

### `noalias()`

```cpp
c.noalias() = a + b;          // skip the overlap classifier
c.noalias() = matmul(a, b);   // GEMM straight into c, no temporary, check skipped in Release
c.noalias() += matmul(a, b);  // beta = 1
c.noalias() -= matmul(a, b);  // alpha = -1, beta = 1
```

- **Contract.** The caller asserts that the destination shares no element
  with any source. Only the destination/source classification changes; shape,
  device and internal-overlap checks still run in every build type.
- **Checking.**
  - Debug: the classifier still runs and errors on a proven overlap
    (`partial`, or `full` for a non-elementwise operation). `too_hard` is
    accepted because the caller has vouched for it.
  - Release: the classifier is skipped. A false assertion is undefined
    behavior, as in Eigen.
- **Proxy.** `noalias()` returns a short-lived `noalias_proxy<tensor_view<T>>`
  holding a reference to the destination; it is not copyable, so it cannot be
  stored. `+=` and `-=` on an elementwise expression evaluate
  `dst = dst ± expr`, which is `full` overlap and safe.
- **Shape.** The proxy never resizes: the result shape must equal the
  destination's, for `tensor<T>` as well as `tensor_view<T>`.
- **What it buys.**
  - Elementwise: the overlap classifier costs O(leaves × rank); `full`
    aliasing already runs in place. `noalias()` saves that check and lets
    `too_hard` layouts run.
  - Products: `c = matmul(a, b)` already writes directly into `c` when the
    classifier reports `none` ([section 11](#assignment-and-noalias)). The
    O(rank) check is negligible next to an O(n³) product, so the main use of
    `noalias()` here is to allow `too_hard` layouts and to skip that check in
    Release.
  - SIMD loops issue explicit loads and stores, so `noalias()` does not change
    their code generation.
- **Reference points.** Eigen's `noalias()` has the same spelling and lets
  products evaluate directly into the destination, but never checks anything.
  PyTorch has no user-facing equivalent: `out=` variants still check, and
  `TensorIteratorConfig::check_mem_overlap(false)` is internal. The Debug check
  is this design's deviation from Eigen.

---

## 7. Slice steps

Negative steps are forbidden, following PyTorch. Every stride is
non-negative, which keeps the planner, the overlap test and the BLAS mapping
free of reversed dimensions.

- **Slice API.** The current `stop = -1` meaning "to end" collides with
  Python's -1 index. The signature becomes
  `slice(dim, std::optional<int64_t> start, std::optional<int64_t> stop,
  int64_t step = 1)`:
  - `step < 1` is an error in every build type; the message names `flip`;
  - negative `start` / `stop` count from the end, as in Python, and are then
    clamped to `[0, size]`;
  - `offset += start * stride`, then `stride *= step`.
- **`flip(dims)`** returns a new `tensor<T>`, like PyTorch's `flip`. It goes
  through the planner and uses a lane-reversing SIMD kernel: contiguous
  registers are loaded, lanes are permuted to reverse order, then stored. This
  is not a scalar read-back-to-front loop. Negative strides stay internal to
  that kernel and never appear in a view.
- **Arithmetic.** `logical_offset`, `at(i, j)`, `linearized_index` and
  `gather_logical` cast strides to `size_t`. That is correct under the
  non-negative invariant, so no signed rewrite is needed. Every view
  constructor checks the invariant with `VECTORIZATION_CHECK`.
- **Address range.** A non-empty view touches
  `[offset, offset + Σ (nᵢ−1)·sᵢ]`. Both ends are checked against the
  allocation; overlap checks and transfers use the same range.
- **Cost.** Code that needs a reversed traversal pays for a `flip` copy,
  where NumPy would return a free view.

---

## 8. View operations

- **`view(shape)`**: metadata only. Uses PyTorch's `computeStride` rule,
  which is enough because strides are never negative. Errors when no view
  exists; never copies.
- **`reshape(shape)`**: `view` when possible, otherwise a packed copy viewed
  with the new shape.
- **`contiguous()`**: a view of the same storage when already packed (safe
  now that views retain storage), otherwise a packed copy.
- **New**: `expand` / `broadcast_to` (stride-0 views, never valid
  destinations), `squeeze`, `unsqueeze`, `flatten`; `flip` as a copy
  ([section 7](#7-slice-steps)).
- **Validation**: every argument check uses `VECTORIZATION_CHECK`, not the
  Debug-only variant. These operations only touch metadata, so the cost is not
  measurable.

---

## 9. Copies and transfers

`clone()`, `contiguous()` and conversion from a view all evaluate an
**identity expression through the planner**. One mechanism gives:

- CPU gathers on the best tier; tier-3 copies use the tiled transpose kernel
  when exactly one 2-D pair of dimensions is swapped, and the lane-reversing
  SIMD kernel for `flip` ([section 5](#tiled-transpose-and-flip-kernels));
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
  ([section 12](#12-upstream-api-changes)).

---

## 10. Reductions

`accumulate`, `hmin` and `hmax` require contiguous input today, and axis
reductions (`sum(dim)`) are not designed. The planner serves both elementwise
operations and reductions, exactly as TensorIterator does.

A reduction plan marks the reduced dimensions and extends the same
`iteration_plan` struct. The evaluator routes to the appropriate reduction
path based on tier.

### CPU (P3)

A reduction over the inner dimension uses the existing per-register
accumulators. A reduction over an outer dimension uses vertical SIMD
accumulation across rows. Threading follows the same grain-size threshold as
elementwise work ([section 5](#iteration-plan-and-threading)).

### GPU (P4)

A fixed grid (multiprocessor count × k) with grid-stride accumulation per
thread and warp shuffles inside a block, then one final launch. There is no
per-pass allocation; the partials buffer is sized once and reused.

**`dot` and `accumulate` into a 0-d tensor.** Returning `T` directly forces a
device-to-host synchronization. `dot_out` writes into a 0-d `tensor<T>`, and
the same for `accumulate`, so the caller controls when the synchronization
happens ([section 11](#11-linearalgebra-adapter)).

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

`matmul` and `bmm` return a `product_expression`: it captures both operands
like any expression leaf (rvalues are moved in and keep their storage alive)
and computes nothing until it is assigned. This follows Eigen's lazy
`Product`; PyTorch computes eagerly and offers `out=` instead.

The overlap classifier decides whether a temporary is needed. The check is
O(rank) next to an O(n³) product, so it is never elided for correctness
reasons; `noalias()` skips it only in Release.

| Statement | Lowering |
|---|---|
| `tensor<T> c(matmul(a, b))` | Allocate `c`, then `matmul_out(c, a, b)` |
| `c = matmul(a, b)`, classifier reports `none` | `matmul_out(c, a, b, 1, 0)` |
| `c = matmul(a, b)`, any overlap | Packed temporary, then copy into `c` |
| `c.noalias() = matmul(a, b)` | `matmul_out(c, a, b, 1, 0)`, check skipped in Release |
| `c.noalias() += matmul(a, b)` | `matmul_out(c, a, b, 1, 1)` |
| `c.noalias() -= matmul(a, b)` | `matmul_out(c, a, b, -1, 1)` |
| `matmul(a, b) + d` assigned to `c` | Copy `d` into `c`, then GEMM with β = 1 |

The temporary in `c = matmul(a, b)` with overlap makes `a = matmul(a, b)`
correct without `noalias()`.

### Product epilogues

Recognized forms, evaluated without a temporary or an extra pass:

| Expression assigned to `c` | Lowering |
|---|---|
| `matmul(a, b) + d` | Copy `d` into `c`, then GEMM with β = 1 |
| `s * matmul(a, b)` | GEMM with α = s |
| `c += s * matmul(a, b)` | GEMM with α = s, β = 1 |

Any other tree containing a product still evaluates the product into a
temporary leaf. Eigen recognizes `scalar * product` the same way.

### Routing by size and shape

| Case | Route | Reference point |
|---|---|---|
| Matrix × vector | GEMV | PyTorch `matmul` |
| Vector · vector | DOT | PyTorch `matmul` |
| Tiny matrices (n ≲ 16–20, tuned) | Native unrolled kernel, or MKL JIT / direct call | Eigen `EIGEN_GEMM_TO_COEFFBASED_THRESHOLD` |
| `bmm` on CPU | `gemm_batch_strided` where available; otherwise split the batch across threads | MKL |
| Everything else | Backend GEMM | — |

Small fixed shapes (Eigen's stack-allocated fixed-size matrices) are out of
scope. Benchmarks report them separately so they are not compared with Eigen
by accident.

### Stride mapping

`linalg::multiplication::apply` is row-major with explicit leading dimensions:

| Operand layout | `transpose` | `ld` |
|---|---|---|
| `stride(1) == 1` | `false` | `stride(0)` |
| `stride(0) == 1` | `true` | `stride(1)` |
| Otherwise | Pack with an identity copy | `cols` |

- **Output.** `stride(1) == 1`: write directly. `stride(0) == 1`: compute
  Cᵀ = Bᵀ·Aᵀ by swapping the operands and flipping the flags. Otherwise compute
  into a temporary and copy.
- **Size-1 dimensions.** BLAS rejects a leading dimension smaller than the row
  length even where the stride is irrelevant, so `ld` is clamped to at least
  the row length.
- **Aliasing.** Any overlap between `c` and `a` or `b`, including `full`, is
  an error for `matmul_out` ([section 6](#6-aliasing-and-overlap)); under
  `noalias()` the same check runs in Debug only. `a = matmul(a, b)` is fine
  because it goes through a temporary.

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

## 12. Upstream API changes

Memory and LinearAlgebra are separate repositories (submodules under
`ThirdParty/`). These changes are designed, tested and landed there first;
Vectorization then bumps the submodule.

### Memory (U1)

- **One allocation per tensor.** For sized allocations, place the control
  block in the same allocation as the data, as `std::make_shared` does. The
  separate control block is kept only for adopted external buffers. This
  removes the second heap allocation for every temporary and is the
  prerequisite for the P1 performance gate.
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

### LinearAlgebra (U2)

- `alpha` / `beta` on `multiplication::apply`; the existing signature stays as
  the `alpha = 1, beta = 0` overload. `c.noalias() += / -= matmul(a, b)` need
  them.
- GEMV and DOT entry points with the same leading-dimension rules as GEMM.
- Rectangular strided-batched GEMM: `rows`, `columns`, `depth`, a batch stride
  per operand (0 for broadcasting), positive leading dimensions, a stream and
  a workspace. The square `batched_matrix_multiplication(dim, count, …)` stays
  as a wrapper.

---

## 13. Implementation phases

Each phase carries both a correctness gate and a performance gate. The
performance gates are starting points confirmed against a measured baseline
before each phase begins.

| Phase | Repository | Scope | Correctness gate | Performance gate |
|---|---|---|---|---|
| P0 | Vectorization | Result shape when leaf shapes are equal; validate every leaf's shape; Release checks on view operations; overlap classifier raising errors; copy constructor keeps source strides and copies the view span at offset 0 (`clone()` packs) | Every probe in [section 1](#1-problems-with-the-current-tensor) has a passing Release test | Release benchmarks at today's speed; shape checks run outside the inner loop |
| P1 | Vectorization | `retained_ptr` storage, explicit offset, views retain storage | Lifetime tests (view outlives owner, deferred expression) clean under AddressSanitizer | Construction and destruction of a tensor: one heap allocation after U1 |
| P2 | Vectorization | `tensor` / `tensor_view` split, rvalue capture into expressions, `noalias()` proxy for elementwise assignment | View assignment writes through; `noalias()` overlap caught in Debug; existing tests pass unchanged | Expression building with `tensor<T>` leaves adds no atomic operations |
| P3 | Vectorization | CPU planner tiers 0–2 with `iteration_plan` and `run_range`; `parallel_for` above grain size; contiguous fast path; `loop2d`; tiled transpose; lane-reversing `flip`; tier-2 specialization; planner-owned prefetch; broadcasting; PyTorch reshape rule; CPU reductions ([section 10](#10-reductions)) | Tier 0 benchmark within noise of today; strided, broadcast and `flip` tests pass | Tier 0 ≥ 90% of STREAM triad bandwidth on one thread and scales with threads; tier 1 ≥ 80% for rows ≥ 256 elements; latency at n ≤ 64 within 10% of today; `t().contiguous()` ≥ 50% of tier-0 copy bandwidth |
| U1 | Memory | [Section 12](#memory-u1) APIs | Landed upstream with tests; submodule bumped | Offset-aware copy at device-copy bandwidth |
| U2 | LinearAlgebra | [Section 12](#linearalgebra-u2) APIs | Landed upstream with tests; submodule bumped | GEMV, DOT and batched GEMM entry points at backend-library speed |
| P4 | Vectorization | CUDA/HIP offset-calculator kernel with 32-bit indexing and vector loads; Metal pipeline cache, batched command buffers, per-allocation `MTLBuffer`; GPU reductions ([section 10](#10-reductions)); offset-aware transfers | Device clone of an interior slice passes without host memory (Metal locally; CUDA/HIP in CI) | Tier 0 and strided tiers measured in GB/s on each GPU backend; Metal per-call host overhead after the first call in µs and free of string building |
| P5 | Vectorization | `matmul_out`, `dot`, factorization objects; `matmul` / `bmm` as lazy products with overlap-based lowering and `noalias()` skipping the check in Release; product epilogues; routing by size | Rectangular, transposed-view, strided and aliased-output GEMM tests pass; `c = matmul(a, b)` on disjoint operands allocates nothing | `c = matmul(a, b)` on disjoint operands allocates nothing; small-matrix GEMM within 2× of Eigen fixed-size products |
| P6 | Vectorization | Streaming stores for large, non-aliased destinations (optional, benchmarked) | Streaming stores behind a flag; no regression on existing tiers | 1.2–1.5× improvement on write-heavy tier-0 work on machines where non-temporal stores win |

U1 and U2 run in parallel with P1–P3; only P4 and P5 depend on them. P0 is
independent of everything else and fixes incorrect results, so it goes first.

---

## 14. Verification

### Correctness tests

In addition to numerical values:

- Result shape of expression construction, including broadcasting.
- Invalid shapes, axes, reshape sizes and slice steps rejected in Release.
- View lifetime: a view returned from a function, and deferred expressions
  holding temporaries.
- Overlap: exact aliasing allowed; shifted, interleaved and `too_hard` cases
  rejected.
- `noalias()`: proven overlap rejected in Debug; `too_hard` layouts run;
  internal overlap rejected in every build type; `+=` / `-=` match the
  equivalent non-`noalias()` expression.
- Zero and negative slice steps rejected in Release; negative `start` /
  `stop` indices; `flip` on every dimension returns a reversed copy and
  leaves the source untouched.
- Strided and broadcast arithmetic on every tier.
- Rectangular and batched GEMM with transposed and strided views.
- `c = matmul(a, b)` on disjoint operands writes directly; on overlapping
  operands uses a temporary.
- `c.noalias() = / += / -= matmul(a, b)` against a reference result, with an
  allocation counter confirming no temporary.
- `matmul(a, b) + d` and `s * matmul(a, b)` epilogues match two-pass
  references.
- Stream transitions: allocation reuse after `record_stream`, and ordering
  through events.
- CPU and GPU reductions on contiguous and strided inputs over each axis.

### Performance measurements

- **Bandwidth.** Achieved GB/s per tier, against a STREAM triad baseline on
  the same machine, on CPU and on each GPU backend.
- **Latency.** Nanoseconds per call at n = 1, 16, 64 and 256 elements,
  including planning and overlap checks.
- **Thread scaling.** Tier 0 and tier 1 throughput from one thread up to all
  cores.
- **Layouts.** Transpose copy, `flip`, column broadcast and outer broadcast.
- **Allocation counts.** Construction and destruction of a tensor (gate: one
  heap allocation after U1); peak memory for complete expression chains.
- **Products.** Square and rectangular GEMM from n = 2 to n = 4,096;
  matrix × vector; batched small matrices; `c = matmul(a, b)` with and without
  `noalias()`; epilogue forms against two-pass references.
- **Metal.** First-call and steady-state host overhead per evaluation; chained
  expressions with and without a host read between them.
- **Comparisons.** Eigen and LibTorch, both with preallocated outputs and with
  allocation included.

---

## 15. Key files

- `include/terminals/tensor.h` — `tensor<T>`, views, `clone()`,
  `contiguous()`, `init_from_expression()`, the `#if 0` matrix block.
- `include/terminals/sizes_and_strides.h` — inline sizes/strides storage.
- `include/expressions/expression_interface.h` — `store_operand()`,
  expression nodes, `buffers_overlap()` / `expression_aliases_ptr()`.
- `include/expressions/expressions_evaluator.h` — CPU SIMD loop (Tier 0),
  `loop2d`, `iteration_plan`, `run_range`.
- `include/expressions/expressions_evaluator_gpu.h` — CUDA/HIP kernels,
  offset calculator, 32-bit indexing, vector loads.
- `include/expressions/expressions_evaluator_metal.h` — Metal emitter,
  pipeline cache.
- `include/expressions/expressions_builder.h` — operators, including
  elementwise `*`.
- `include/expressions/reduce_op.h` — reduction operations.
- `ThirdParty/Memory/include/common/retained_ptr.h`, `data_ptr.h`,
  `data_view.h`, `copy_token.h` — storage and transfer primitives.
- `ThirdParty/Memory/include/gpu/cuda_caching_allocator.h`,
  `gpu_workspace.h` — GPU allocation and workspace.
- `ThirdParty/LinearAlgebra/include/matrix_operation/multiplication.h`,
  `multiplication_batched.h` — GEMM entry points.
- `ThirdParty/LinearAlgebra/include/common/backend.h` — `linalg_backend`.
