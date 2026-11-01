# Vectorization Design Plan

Scope: `include/terminals/tensor.h`, `include/expressions/`, the CPU, CUDA,
HIP and Metal evaluators, and the parts of Memory, Scalars and LinearAlgebra
they build on, updated November 1, 2026. This is the **single implementation
plan** for the tensor and evaluator redesign. The status table and
implementation notes record delivered work; remaining requirements describe
targets, not the current contract.

It covers storage ownership, the `tensor` / `tensor_view` split,
broadcasting and strided evaluation, aliasing and `noalias()`, slice steps,
transfers, reductions, the PyTorch-style matrix LinearAlgebra adapter, backend launch
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

Updated by source inspection on November 1, 2026. "Done" means implemented;
it does not imply that every backend or performance gate has passed.
Recorded verification and remaining gates are listed in
[section 17](#recorded-verification).

| Item | Status | Where |
|---|---|---|
| Expression result takes the broadcast shape; leaf shapes validated in every build | Done | `expression_shape.h`, `tensor::init_from_expression()` |
| CPU planner: sort, merge, tiers, `loop2d` | Done | `iteration_plan.h` |
| CPU `parallel_for` above a 32,768-element grain | Done; serial SIMD fallback fixed in `c4fde41`, not yet benchmarked | `iteration_plan.h`, `expressions_evaluator.h` |
| Strided expression leaves (`strided_ref`) — transposes need no materialization | Done | `strided_ref.h`, `store_operand()` |
| Stride-0 destination rejected | Done | `make_iteration_plan()` |
| General destination injectivity proof (`non_overlapping_` flag) | Done: cached stride-sorted proof; unproven destinations rejected even with `noalias()` | `tensor.h`, `overlap.h` |
| `slice(dim, optional start, optional stop, step ≥ 1)` | Done | `tensor.h` |
| `expand`, `broadcast_to`, `squeeze`, `unsqueeze`, `flatten`; Release checks on view operations | Done | `tensor.h` |
| Copy semantics | Done: owners pack logical values into independent C-order storage; view copies share storage and geometry. Supersedes the source-stride-preserving copy from `c7f436f` | `tensor.h` |
| CUDA/HIP strided sources | Partial: leaves are rebased, destination must be contiguous | `rebase_leaves()` |
| `retained_ptr` storage; views retain storage | Done for owned/adopted storage; explicit raw-pointer and `data_view` construction remains an unsafe borrow | `tensor.h` |
| Expression leaves keep operands alive | Done: `retained_ref` snapshots descriptors and retains lvalue and rvalue storage; `lower_raw()` removes ownership before dispatch | `strided_ref.h`, `expression_interface.h` |
| `tensor` / `tensor_view` split, const views, `rebind()`, elementwise `noalias()` | Done: distinct `tensor_impl<T, IsView>` instantiations; write-through view assignment; proxy offers `=`, `+=`, `-=` | `tensor.h`, `tensor_view.h`, `noalias.h` |
| Destination/source overlap classifier and signed-coefficient lattice test | Done: address ranges, identical mappings, dense spans, stride GCD and nested-stride carry intervals | `overlap.h` |
| Provenance-aware storage shortcut and explicit offset field | Not started: classifier compares addresses conservatively; offsets are derived from `data_view` | `tensor.h`, Memory U1 |
| `flip`, tiled transpose | Partial: CPU logical copies have an 8×8 transpose path; planner-wide transpose, GPU copy kernels and `flip` remain | `tensor.h`, `iteration_plan.h` |
| Planner-owned prefetch; removal of the aligned-source variant | Not started: per-leaf prefetch and both alignment variants remain | `expressions_evaluator.h` |
| Axis reductions; strided reductions | Partial: CPU whole-tensor strided/broadcast reductions exist; axis reductions and general strided GPU reductions remain | `expressions_evaluator.h`, `reduce_op.h` |
| GPU `clone()` of a strided slice without host staging | Not started: stages through host | `tensor::clone()` |
| CUDA/HIP offset calculator, 32-bit indexing, vector loads | Not started: one thread per element, scalar loads | `expressions_evaluator_gpu.h` |
| GPU reduction without per-pass allocation | Not started | `reduce_gpu_partials()` |
| Metal pipeline cache, batched command buffers, per-allocation `MTLBuffer` | Not started: MSL rebuilt per call, commit-and-wait, per-call buffer wrapping | `expressions_evaluator_metal.h`, Scalars `metal_dispatch.mm` |
| Unified launch contract across backends | Not started | [section 13](#13-backend-launch-unification) |
| Tensor-level `matmul`, `dot` and PyTorch-style matrix linear algebra | Not started: matrix block under `#if 0`; dependency algorithms still need tensor adapters | [section 11](#11-linearalgebra-adapter), `tensor.h` |
| Memory U1 and LinearAlgebra U2 APIs | Not started | [section 12](#12-upstream-api-changes) |
| clang-tidy coverage of `include/` and `Testing/` | Repository-wide work remains: changed-code analysis is clean for this delivery, but CI coverage and the historical baseline are not cleared | [section 15](#15-clang-tidy-conformance) |

---

## 2. Decisions

| Question | Decision | Reference point |
|---|---|---|
| Copy semantics | Separate `tensor<T>` (owning, deep copy) and `tensor_view<T>` (shallow copy, assignment writes through) | Eigen `Matrix` vs `Block` / `Ref`; not PyTorch's shared handles |
| Partial overlap between destination and source | Raise an error | PyTorch `assert_no_partial_overlap`; not Eigen's silent temporary |
| Opting out of the overlap check | `dst.noalias() = expr`: the caller asserts no overlap. It skips the O(rank) check in Release and lets `too_hard` layouts run. It does not decide whether a product needs a temporary; the classifier does | Eigen `noalias()`; PyTorch has only `out=` and the internal `check_mem_overlap(false)` |
| Negative slice steps | Forbidden: `step < 1` is an error, strides are never negative, `flip` returns a copy | PyTorch; NumPy supports them |
| `*` on tensors | Elementwise; matrix multiplication only through the `matmul` family | PyTorch, Eigen `Array` |
| Math on a matrix product | `exp`, `sqr`, `sqrt` and `log` accept `matmul` / `bmm` results directly and act elementwise; `sqr` corresponds to PyTorch `square` | PyTorch pointwise semantics; lazy evaluation here |
| Matrix linear algebra | Functional `vectorization::linalg` API follows `torch.linalg` names, matrix/batch dimensions, options and result fields; reusable factorization objects are additional conveniences | PyTorch `torch.linalg` |
| Backend selection | Compile-time; no runtime-polymorphic launch interface | Eigen `TensorDevice`; deviates from PyTorch's runtime dispatcher |
| GPU synchronization | Asynchronous launches; host waits on reads, explicit events, or numerical status in checked linear algebra. `_ex(check_errors=false)` keeps status on device | CUDA streams; PyTorch checked / `_ex` linear algebra; MPS commit-and-continue |
| Memory / LinearAlgebra / Scalars API changes | Designed and landed in their own repositories first, then the submodule is bumped | — |
| Static analysis | All first-party C++ is clang-tidy clean under `.clang-tidy`, warnings as errors, in every configuration CI builds; suppressions name the check and a reason | PyTorch gates on clang-tidy through lintrunner with named `NOLINTNEXTLINE`; Eigen does not gate on clang-tidy |

Value semantics are intentional: `tensor b = a` copies elements, unlike a
PyTorch handle copy. Code that only needs to borrow takes a view.

---

## 3. Layers: storage, geometry, kernel descriptor

The compact state below is the **target layout**. The current public aliases
use `tensor_impl<T, false>` and `tensor_impl<T, true>` to share implementation.
They retain storage and cache `contiguous_` and `non_overlapping_` as separate
booleans, while a `data_view` carries the current window. The explicit offset,
packed flags, provenance shortcut and CPU control-block coallocation remain
targets. Read-only descriptors use `const_data_view` to preserve Memory's
allocation identity without instantiating its unsupported `data_ptr<const T>`.

```cpp
// Shared state of tensor<T> and tensor_view<T>.
template <typename T>
struct tensor_impl
{
    memory::retained_ptr<std::remove_const_t<T>> storage_;  // whole allocation
    sizes_and_strides                            sas_;      // inline up to 5 dims
    int64_t                                      offset_{0};  // elements from base()
    std::size_t                                  numel_{0};
    uint8_t flags_{};  // contiguous | dense_non_overlapping | non_overlapping | has_zero_stride
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
  - **Provenance — target:** record `allocated` (issued by a Memory
    allocator) or `adopted`. Two distinct `allocated` storages never share
    an address. Adopting overlapping external buffers independently creates
    distinct control blocks over the same memory, so storage identity proves
    nothing about adopted storage ([section 6](#6-aliasing-and-overlap)).
  - Every view of owned/adopted storage keeps its allocation alive, at the
    cost of one atomic increment per shared handle. A no-op deleter retains
    only the borrow handle; callers still keep external memory alive.
  - PyTorch's `TensorImpl` holds `intrusive_ptr<StorageImpl>` the same way.
    Eigen `Block` / `Map` hold raw pointers, which only works because they
    are expression types not meant to be stored.
- **Explicit offset — target:** store it as a field, not recomputed by pointer subtraction, so
  copies, overlap checks and GPU transfers work from `(base, offset)`.
- **Kernels only see `strided_ref`.** Reference-counted handles never reach
  a device. Worker threads also use `strided_ref` and raw pointers, so they
  never contend on a shared control block.
- **One allocation per CPU tensor — target** (Memory U1): for sized CPU allocations
  the control block sits in the same allocation as the data, as
  `std::make_shared` does. GPU and adopted storage keep a separate control
  block ([section 14](#14-allocation-and-workspace)).

---

## 4. `tensor` and `tensor_view`

| | `tensor<T>` | `tensor_view<T>` | `tensor_view<const T>` |
|---|---|---|---|
| Invariant | Owns, packed C-order, offset 0 | Any non-negative strides (including zero); retains storage. Usable as a destination only when `non_overlapping` is proven ([section 6](#internal-overlap)) | Same, read-only |
| Copy constructor | Deep copy | Shallow (shares storage) | Shallow |
| `operator=` | Deep copy / evaluate | Writes through | Deleted |
| Converts to | Either view type, implicitly | `tensor<T>`, explicitly (packed copy) | `tensor<T>`, explicitly |

The split, const access and write-through semantics are implemented. A const
owner converts only to a read-only view. `clone()`, `to_cpu()` and expression
construction return independent owning tensors. Raw-pointer or `data_view`
construction copies for `tensor<T>` and deliberately borrows for
`tensor_view<T>`; a view can also retain an adopted `retained_ptr` directly.

- **Return types.**
  - Views: `as_view`, `slice`, `t`, `permute`, `expand`, `broadcast_to`,
    `view`, `squeeze`, `unsqueeze`, `flatten`.
  - Tensors: `clone`, `to_cpu`, construction from an expression; `flip`
    remains planned.
  - Planned lazy products: `matmul`, `bmm` ([section 11](#11-linearalgebra-adapter)).
  - `reshape` and `contiguous` return a view: of the existing storage when
    possible, otherwise of a new packed copy.
- **Parameters.** Non-template functions with a concrete element type take
  inputs as `tensor_view<const float> const&` and outputs as
  `tensor_view<float>`, like Eigen's `Ref<const MatrixXd>`. The implicit
  conversion from `tensor<float>` applies there. It does not apply to a
  function template that deduces `T` from a `tensor_view<const T>`
  parameter: deduction ignores user-defined conversions, so
  `matmul(a, b)` on two `tensor<float>` would not compile. Library entry
  points that deduce `T` are constrained forwarding templates instead
  ([section 11](#11-linearalgebra-adapter)).
- **Rebinding.** Because `operator=` writes through, re-pointing a view uses
  `v.rebind(other)`. Assigning elements of a `std::vector<tensor_view<T>>`
  writes data, the same caveat as Eigen `Ref`.
- **`noalias()`.** `tensor<T>` and `tensor_view<T>` have `noalias()`, which
  returns a proxy offering only `=`, `+=` and `-=` ([section 6](#6-aliasing-and-overlap)).
  It is deleted on `tensor_view<const T>`.
- **Fast path by type — target.** When the destination and every expression leaf
  are `tensor<T>`, contiguity is known at compile time and planning reduces
  to a shape comparison. A `tensor_view` destination is checked at run time
  ([section 5](#contiguous-fast-path)). Current evaluation still checks
  cached contiguity and leaf element counts at run time.

---

## 5. Expressions and the iteration planner

### Capture

- **Implemented.** `store_operand()` stores every tensor/view leaf as a
  `retained_ref`: a snapshot of its `strided_ref` plus a shared storage
  handle. This applies to lvalues and rvalues, including temporary views
  and temporary owners. Rebinding or destroying the source handle does
  not invalidate the saved expression's descriptor or allocation.
- **Cost.** Capturing an lvalue currently increments the reference count;
  nested expression copies can also copy retained handles. The original
  atomic-free lvalue capture gate is not met.
- **Remaining optimization.** Borrowing stable lvalue descriptors and moving
  rvalues into nodes is a possible optimization, subject to explicit
  lifetime semantics and benchmarks. If that route is taken, every builder,
  including planned `matmul` and `bmm`, must preserve value categories with
  forwarding references and `std::forward`; temporary conversions must be
  retained by value.

### Lowering

**Implemented:** `raw_lowered_t<E>` and `lower_raw()` rebuild host trees with
trivially copyable `strided_ref` leaves before evaluator dispatch. CPU worker
trees and CUDA/HIP kernel arguments do not contain retained storage handles;
the Metal emitter also consumes the lowered tree. `flat_lowered()` then
rebases strides for broadcasting where needed.

**Product target:** product nodes will be scheduled and evaluated through the
LinearAlgebra adapter before elementwise lowering
([section 11](#elementwise-math-on-products)). Their results become tensor
leaves. The remaining tree is converted at compile time (`lower_t<E>`)
into a tree whose leaves are `strided_ref`, or `T*` for tier-0 leaves on
GPU. The lowered tree is what CUDA/HIP kernels take by value and what the
Metal emitter walks. A product never performs a dot product inside an
elementwise coefficient evaluator.

### Planner

Modelled on PyTorch's `TensorIterator`; runs once per evaluation at
O(leaves × rank) cost.

1. Collect each leaf's shape, strides, offset, device, storage identity and
   provenance.
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

The storage-provenance ordering and type-based shortcut below remain
optimization targets. Current assignment checks overlap before calling the
evaluator; the evaluator's runtime flat-path check avoids sort/merge planning
for packed, compatible operands. The classifier uses addresses even when
storage handles differ, so independent adoption cannot incorrectly prove
disjointness.

Planning costs more than the arithmetic for small tensors. An O(leaves)
check runs first:

1. The destination and every leaf have the `contiguous` flag from
   `tensor_impl::flags_`. A `tensor<T>` destination has it by invariant.
2. Every leaf has the destination's shape.

The destination is part of condition 1: packed sources of matching shape
assigned into a transpose or a stepped slice would otherwise be written at
packed positions. When both hold, evaluation goes straight to tier 0 with
no sort, merge or lattice test. The overlap check is ordered cheapest first:

1. Different storage, both with `allocated` provenance
   ([section 3](#3-layers-storage-geometry-kernel-descriptor)) → `none`.
2. Same data address (`base + offset`), sizes and strides → `full`.
3. Disjoint address ranges → `none`. This is the first test that can clear
   `adopted` storage.
4. Only then the lattice test.

Two distinct `tensor<T>` objects with `allocated` storage can never overlap,
so step 1 always decides for them. A freshly constructed `tensor<T>`
destination skips the check. The target is Eigen-like cost for trees of
packed tensors, not `TensorIterator`'s microsecond-level setup.

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
| `none` | Different storage with `allocated` provenance, disjoint address ranges, or the lattice test proves no shared element | Run | Run |
| `full` | Same data address, sizes and strides | Run in place (`a = a + 1`) | Error |
| `partial` | The lattice test finds a shared element | Error | Error |
| `too_hard` | The lattice test's preconditions fail, so it cannot decide | Error | Error |

**Implemented for elementwise assignment:** `classify_overlap()` and
`check_assignment_overlap()` in `overlap.h`. Disjoint byte ranges are cleared
first; identical injective mappings are `full`; intersecting dense spans are
`partial`. A stride-GCD test can prove that interleaved slices are disjoint
before the nested-stride lattice test. The provenance shortcut and all
product-specific behavior in this section remain targets.

- **Storage provenance.** Different storage proves `none` only when both
  storages are `allocated`. Adopted storage is compared by address: an
  adopted buffer may overlap another adopted buffer or an allocation under
  a different control block. Address ranges are compared in bytes on the
  same device, from `base + offset` over the span in
  [section 7](#7-slice-steps). When ranges intersect across different
  storage, the lattice test runs on element offsets taken from the byte
  difference. If that difference is not a multiple of `sizeof(T)`, or the
  element types differ, the result is `too_hard`.
- **Lattice test.** Two views share an element when
  `off_b − off_a = Σ dᵢ·sᵢ` for integers `dᵢ = aᵢ − bᵢ`, bounded by
  `−(n_b,ᵢ − 1) ≤ dᵢ ≤ n_a,ᵢ − 1`. The coefficients are signed. A greedy
  decomposition that accepts only non-negative coefficients misses shared
  elements. For example, shape `(2, 2)` with strides `(3, 1)` at offsets 0
  and 2 touches `{0, 1, 3, 4}` and `{2, 3, 5, 6}`. The views share element
  3, since `2 = 1·3 − 1·1`.
  - *Preconditions.* Both views have equal strides after size-1 dimensions
    are dropped. Sorted ascending, each stride divides the next, and both
    views are injective in that order (`sᵢ₊₁ ≥ nᵢ·sᵢ`; see
    [internal overlap](#internal-overlap)).
  - *Implemented procedure.* Work from the innermost dimension. Divide the
    residual by `s₀`; a remainder means `none`. Track the possible residuals
    as an integer interval `[lo, hi]`. For signed coefficient bounds `[L, U]`
    and period `p = sᵢ₊₁ / sᵢ`, propagate the carry interval as
    `[ceil((lo − U) / p), floor((hi − L) / p)]`. An empty interval means
    `none`; at the outermost dimension the interval must intersect its
    coefficient bounds. Overflow uncertainty returns `too_hard`. The carry
    walk costs O(rank) after axes have been sorted.
  - *Coverage.* This covers shifted, interleaved and row-offset slices.
    Unequal or non-nested strides and non-injective layouts return
    `too_hard` unless an earlier range or GCD test already proves
    disjointness.
- **Deviation from PyTorch.** PyTorch lets `too_hard` cases run silently;
  this design errors on them.
- **Escape hatches.** For `partial` the error names `dst = src.clone();`.
  For `too_hard` it also names `dst.noalias() = …`.

### Internal overlap

A destination must be proven injective: no two logical indices may map to
the same element. Rejecting zero strides is not enough. Shape `(2, 2)` with
strides `(1, 1)` touches offsets `{0, 1, 1, 2}` with no zero stride.

- **Proof.** Drop size-1 dimensions and sort the rest by stride. The layout
  is injective when every stride exceeds the extent of the dimensions
  inside it: `sᵢ₊₁ > Σ_{j≤i} (nⱼ − 1)·sⱼ`. A zero stride on a non-trivial
  dimension fails this test. When strides are nested, the test is exact. In
  general it is sufficient but not necessary: sizes `(3, 2)` with strides
  `(2, 3)` are injective but fail. Such layouts are treated as unproven.
- **Flag.** View geometry construction computes the proof in
  O(rank log rank) and caches it as `tensor_impl::non_overlapping_`.
  Assignment and fill validate it before dispatch. Consolidating geometry
  booleans into `flags_` remains a layout optimization. Views derived from
  a packed tensor by `slice`, `t`, `permute`, `squeeze`, `unsqueeze` and `view` keep the
  property, and the check confirms it. `expand`, `broadcast_to` and
  explicit-stride construction are where it can be lost.
- **Enforcement.** Writing to a destination without the flag is an error in
  every build type, with or without `noalias()`, because `noalias()` only
  covers destination/source overlap. An injective layout that fails the
  proof must be rebuilt with nested strides before it is written. Sources
  have no such requirement. A non-injective source matters only when its
  address range meets the destination's, and the lattice test then
  returns `too_hard`.
- **Status.** Implemented, including `(2, 2)` / `(1, 1)` rejection and
  rejection of injective layouts that fail the sufficient proof. CPU scalar
  filling now supports proven strided destinations; GPU destinations still
  require contiguity.

### `noalias()`

```cpp
c.noalias() = a + b;         // implemented; classifier skipped in Release
c.noalias() += a;            // implemented elementwise update
c.noalias() -= b;            // implemented elementwise update

// Planned with P5 products:
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
- **Proxy.** Implemented in `noalias.h` as a non-copyable
  `noalias_proxy<Destination>` holding a reference to either an owning
  tensor or a writable view.
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
- **`contiguous()`**: implemented as a retaining view of the same storage
  when already packed, otherwise a retaining view of a new packed copy.
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

The tensor-facing matrix API follows the functional design of
[`torch.linalg`](https://docs.pytorch.org/docs/stable/linalg.html).
Public functions live in `vectorization::linalg`; they adapt tensors to the
separate dependency's `::linalg` kernels. `linalg::matmul` also exposes the
existing `vectorization::matmul` entry point with identical behavior.
Backend enums, workspace pointers and destructive LAPACK-style buffers
remain implementation details.

```cpp
// True for tensor<T>, tensor_view<T> and tensor_view<const T> (any cv/ref)
// whose element types match; mirrors is_pure_expression in tensor.h.
template <typename... Ops>
inline constexpr bool are_tensor_operands_v = /* ... */;

template <typename C>
inline constexpr bool is_writable_tensor_operand_v = /* ... */;

// Lazy: operands are forwarded into the node and captured by section 5's rule.
template <typename A, typename B,
          std::enable_if_t<are_tensor_operands_v<A, B>, bool> = true>
product_expression<std::decay_t<A>, std::decay_t<B>> matmul(A&& a, B&& b);

template <typename A, typename B,
          std::enable_if_t<are_tensor_operands_v<A, B>, bool> = true>
product_expression<std::decay_t<A>, std::decay_t<B>> bmm(A&& a, B&& b);

// Eager: nothing is captured, so these convert and call a non-deducing
// implementation taking tensor_view<const T> const& / tensor_view<T>.
template <typename C, typename A, typename B,
          std::enable_if_t<are_tensor_operands_v<C, A, B> &&
                               is_writable_tensor_operand_v<C>, bool> = true>
void matmul_out(C&& c, A const& a, B const& b,
                operand_value_t<A> alpha = 1, operand_value_t<A> beta = 0);

template <typename A, typename B,
          std::enable_if_t<are_tensor_operands_v<A, B>, bool> = true>
operand_value_t<A> dot(A const& a, B const& b);
```

- **Why not `tensor_view<const T> const&`.** Deduction does not consider
  the implicit `tensor<T>` → view conversion, so that signature rejects
  owning tensors. Forwarding also preserves temporary value categories if
  product capture later uses the borrowed-lvalue optimization
  ([section 5](#capture)). Current elementwise capture retains every leaf;
  atomic-free lvalue capture is a remaining performance target, not an
  implemented property.
- **Element types.** Mixed element types are a compile error; there is no
  implicit promotion.
- **Expressions as operands.** Elementwise expressions do not satisfy
  `are_tensor_operands_v`; `matmul(a + b, c)` must materialize `a + b`
  first. Eigen evaluates nested product operands into temporaries
  implicitly; requiring it here keeps the allocation visible.
- **Reference points.** Eigen's `MatrixBase::operator*` takes
  `MatrixBase<Derived> const&` and stores nested operands through
  `ref_selector`, by value for expressions and by reference for plain
  objects. PyTorch takes `const Tensor&` but computes eagerly, so lifetime
  never arises.

### PyTorch matrix semantics

`matmul` follows [PyTorch's rank and batch rules](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html):

| Left shape | Right shape | Result shape |
|---|---|---|
| `(k)` | `(k)` | `()` (0-d tensor expression) |
| `(m, k)` | `(k)` | `(m)` |
| `(k)` | `(k, n)` | `(n)` |
| `(m, k)` | `(k, n)` | `(m, n)` |
| `(..., m, k)` | `(..., k, n)` | `(broadcast_batch..., m, n)` |

A left vector is temporarily promoted to `(1, k)` and a right vector to
`(k, 1)`; remove only those introduced dimensions after multiplication.
Batch dimensions align from the right and broadcast by the usual
equal-or-size-1 rule. Matrix dimensions never broadcast. Scalar operands,
unequal contraction dimensions and incompatible batches are errors in
every build type. Empty output dimensions are valid; a zero contraction
dimension produces zeros before any elementwise math is applied.

`bmm` requires two rank-3 operands `(batch, m, k)` and `(batch, k, n)`
with equal batch sizes, returning `(batch, m, n)`.
[Like PyTorch `bmm`](https://docs.pytorch.org/docs/stable/generated/torch.bmm.html),
it does not broadcast. `*` remains elementwise at every rank. These shape
and math rules do not change the library's same-type operand requirement,
value semantics or lazy evaluation.

### Elementwise math on products

`product_expression` participates in the public expression traits, shape
inference and unary/binary builders. Callers can compose products directly:

```cpp
out = ::exp(matmul(a, b));
out = ::sqr(matmul(a, b));
out = ::sqrt(matmul(a, b));
out = ::log(matmul(a, b));
out = ::log(::sqrt(::sqr(matmul(a, b)) + epsilon));
out = ::exp(matmul(a, b) + bias);  // bias broadcasts over the product
out = ::log(bmm(batch_a, batch_b));
```

These functions preserve the product's shape and apply to each element.
`sqr(x)` means `x * x`, corresponding to
[PyTorch `square`](https://docs.pytorch.org/docs/stable/generated/torch.square.html);
`sqrt(x)` takes the
[elementwise square root](https://docs.pytorch.org/docs/stable/generated/torch.sqrt.html).
`exp` is the
[elementwise exponential](https://docs.pytorch.org/docs/stable/generated/torch.exp.html)
and `log` is the
[elementwise natural logarithm](https://docs.pytorch.org/docs/stable/generated/torch.log.html).
For real floating-point inputs, negative `sqrt` / `log` inputs yield NaN,
`log(0)` yields negative infinity, and overflow can yield infinity.
No implicit clamping is added. Dtype support follows the selected backend.

- **Build and capture.** Unary/binary builders accept a product wherever
  they accept an elementwise expression and forward operands by the
  capture rules in section 5. Construction computes metadata only. A
  deferred `auto e = ::exp(matmul(a.t(), b));` owns its temporary nodes
  and view descriptors; the lvalue owners must still outlive evaluation.
  This does not relax the explicit-materialization rule for expressions
  used as inputs to `matmul` / `bmm` above.
- **Evaluate once.** Before any output write, validate the whole tree's
  shapes, devices and destination overlap. Evaluate each product node
  occurrence once through GEMM, GEMV, DOT or batched dispatch, replace it
  with a leaf retaining the result storage, then run the remaining
  elementwise tree in one traversal or kernel. Repeated product syntax
  does not imply common-subexpression elimination; callers can explicitly
  materialize a product to reuse its values. Vector/vector products use
  `dot_out` into a 0-d tensor, so composing unary math on a GPU product
  does not extract a host scalar.
- **Storage and aliasing.** The default for a product nested under unary
  math is a packed intermediate. Keep it alive through the consumer's
  completion; GPU producer and consumer run in order on the same stream,
  with storage recorded there and no host read between stages. Before
  considering destination reuse, classify every original product input
  and remaining leaf. `a = ::exp(matmul(a, b))` must finish the product
  before overwriting `a`. `noalias()` never removes a required evaluation
  stage or bypasses destination injectivity checks.
- **Fusion boundary.** P5 guarantees composition through a product stage
  followed by an elementwise stage. For `::exp(matmul(a, b) + bias)`, the
  addition and exponential share the elementwise pass. A backend fused
  epilogue or safe destination reuse is an optional measured optimization;
  arbitrary unary math is not assumed to be supported by GEMM.

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
- Any other tree containing a product, including the unary math above,
  evaluates the product into a temporary leaf before elementwise lowering.
  Eigen recognizes `scalar * product` the same way.

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

### Matrix linear algebra API

The following is the planned tensor API, not a claim that these adapters
are implemented. Each operation must have an explicit backend/dtype
capability entry before it is enabled. Let `batch...` denote leading
dimensions and `k = min(m, n)`.

| Family | Public operations | Matrix and result contract |
|---|---|---|
| Square systems | `solve`, `solve_ex` | `A: (..., n, n)`; solve `A X = B` by default or `X A = B` with `left=false`; preserve vector versus matrix RHS conventions and broadcast compatible batches |
| Triangular systems | `solve_triangular` | `upper` is required; `left=true`, `unitriangular=false` by default; read only the selected triangle and ignore its diagonal when unit triangular |
| Reuse of factors | `lu_factor`, `lu_factor_ex`, `lu_solve`, `cholesky_solve` | Packed LU plus integer pivots, or a triangular Cholesky factor, reused for multiple RHS; `cholesky_solve(B, L, upper=false)` retains PyTorch's argument order |
| Least squares | `lstsq` | Rectangular `A: (..., m, n)` and matrix RHS `B: (..., m, nrhs)`; result fields `solution`, `residuals`, `rank`, `singular_values` |
| Inverses | `inv`, `inv_ex`, `pinv` | `inv` requires square invertible input; `pinv` maps `(..., m, n)` to `(..., n, m)` with explicit `atol`, `rtol`, `hermitian` options |
| Cholesky | `cholesky`, `cholesky_ex` | Symmetric/Hermitian positive-definite `(..., n, n)` to the same shape; lower factor by default, `upper=false` |
| LU | `lu`, `lu_factor` | `lu` returns `P`, `L`, `U` with `A = P L U`; `lu_factor` returns `LU`, `pivots`; rectangular inputs supported |
| QR | `qr` | Fields `Q`, `R`; `mode=reduced` by default, with `complete` and `r` modes |
| Singular values | `svd`, `svdvals` | Fields `U`, `S`, `Vh`; `full_matrices=true` by default; `S: (..., k)` is real and descending; `svdvals` computes only `S` |
| Symmetric spectrum | `eigh`, `eigvalsh` | `UPLO=L` by default; eigenvalues real and ascending, eigenvectors in columns; read only the selected triangle |
| General spectrum | `eig`, `eigvals` | Square inputs; complex eigenvalues and eigenvectors even for real inputs; no assumed eigenvalue ordering |
| Determinants | `det`, `slogdet` | Square inputs; `det: batch...`; `slogdet` fields `sign`, `logabsdet`, each of shape `batch...` |
| Norms and conditioning | `vector_norm`, `matrix_norm`, `cond` | Preserve the operation's `ord`, `dim`, `keepdim` rules; matrix norm defaults to Frobenius over `(-2, -1)`; condition number defaults to the 2-norm |
| Numerical rank | `matrix_rank` | Rectangular matrices to integer `batch...`; `atol`, `rtol`, `hermitian` follow the same threshold conventions as `pinv` |
| Matrix functions | `matrix_exp`, `matrix_power` | Square matrices to the same shape; exponential is a matrix function, power takes an integer exponent |
| Symmetric matrix root | `matrix_sqrth` | Principal symmetric/Hermitian positive-definite root, using the lower triangle; availability follows the pinned PyTorch reference version |
| Matrix chains | `multi_dot` | PyTorch's rank-2 matrix chain with optional vector endpoints; choose parenthesization to reduce arithmetic; no batch broadcasting |

`cholesky_solve` is a top-level function in PyTorch; placing it in this
library's `linalg` namespace is an explicit organizational difference.
LDL, polar, Householder, tensor inverse/solve and other `torch.linalg`
families remain separate follow-up adapters. Their absence must be visible
in the capability table; adding a backend algorithm alone does not expose
a tensor operation.

#### Shapes, defaults and results

- **Matrix axes.** Unless an operation exposes `dim`, the last two axes
  are matrix axes and preceding axes are batches. A unary operation keeps
  those batches; multiple inputs broadcast only where that specific
  PyTorch operation permits. Do not apply `matmul`'s vector promotion to
  all solvers. A single-matrix scalar result is a 0-d tensor on the input
  device, not a host `T`.
- **Solves.** Follow the exact RHS rank rules of
  [`torch.linalg.solve`](https://docs.pytorch.org/docs/stable/generated/torch.linalg.solve.html),
  including a vector RHS and `left=false`. Solve through factors rather
  than constructing an inverse. `lu_solve` exposes `left` and `adjoint`;
  the latter means conjugate transpose for complex data.
- **Named results.** Return structs supporting field access and structured
  binding in PyTorch's order: `svd_result{U, S, Vh}`,
  `qr_result{Q, R}`, `lu_factor_result{LU, pivots}`,
  `eigh_result{eigenvalues, eigenvectors}` and
  `slogdet_result{sign, logabsdet}`. `_ex` adds the operation's `info`
  tensor. Pivots use PyTorch's
  [1-based row-interchange convention](https://docs.pytorch.org/docs/stable/generated/torch.linalg.lu_factor.html);
  normalize backend index width/base at the adapter boundary. Rank outputs
  are int64; pivots and `info` use the pinned reference's integer dtypes.
- **Decomposition dimensions.**
  [Reduced QR](https://docs.pytorch.org/docs/stable/generated/torch.linalg.qr.html)
  returns `Q: (..., m, k)`, `R: (..., k, n)`; complete QR returns
  `Q: (..., m, m)`, `R: (..., m, n)`. Mode `r` returns an empty `Q`.
  [Full SVD](https://docs.pytorch.org/docs/stable/generated/torch.linalg.svd.html)
  returns `U: (..., m, m)` and `Vh: (..., n, n)`; reduced SVD returns
  `U: (..., m, k)`, `Vh: (..., k, n)`. Reorder singular vectors along
  with singular values when the backend does not sort them.
- **Tolerances and drivers.** Match
  [`pinv`](https://docs.pytorch.org/docs/stable/generated/torch.linalg.pinv.html)
  and `matrix_rank` defaults, including dtype-dependent tolerances and
  broadcastable tolerance tensors. Match
  [`lstsq`](https://docs.pytorch.org/docs/stable/generated/torch.linalg.lstsq.html)
  driver-dependent residual/rank/singular-value fields, including empty
  fields when unavailable. Unsupported drivers/options raise an explicit
  error; never silently substitute a different numerical contract.
- **Dtypes.** Start with supported float/double backends and publish the
  limits, including Metal float-only. Complex input and output require
  tensor, Scalars and LinearAlgebra support together. Until complex output
  exists, general `eig` / `eigvals` are unavailable rather than dropping
  imaginary components. Real singular values, Hermitian eigenvalues,
  norms and `logabsdet` remain real when complex input is supported.

#### Matrix functions and elementwise math

The API keeps both meanings explicit:

| Elementwise expression | Matrix operation |
|---|---|
| `::exp(a)` | `linalg::matrix_exp(a)` computes the matrix exponential |
| `::sqr(a)` | `linalg::matrix_power(a, 2)` computes `a @ a` |
| `::sqrt(a)` | `linalg::matrix_sqrth(a)` computes the positive-definite matrix root |
| `::log(a)` | `linalg::matrix_log(a)` is a separately documented library extension |

[`matrix_power`](https://docs.pytorch.org/docs/stable/generated/torch.linalg.matrix_power.html)
with exponent 0 returns identity; a negative exponent requires an
invertible matrix. [`matrix_exp`](https://docs.pytorch.org/docs/stable/generated/torch.linalg.matrix_exp.html)
operates on square matrices and preserves batches.
[`matrix_sqrth`](https://docs.pytorch.org/docs/main/generated/torch.linalg.matrix_sqrth.html)
has the symmetric/Hermitian positive-definite domain of the referenced
PyTorch API; the dependency's general Schur square-root algorithm does not
define that public contract.

`matrix_log` and a general `matrix_sqrt` expose existing dependency
algorithms as extensions, outside the PyTorch parity set. Their principal
branch/domain restrictions follow `logarithm.h` and `square_root.h`, and
their names never alias pointwise `log` / `sqrt`. Pin and record the
PyTorch reference version for parity tests; version-specific functions
such as `matrix_sqrth` need a version gate, not a fabricated reference.

#### Evaluation, outputs and numerical errors

These functional operations evaluate eagerly and return owning tensors or
named tensor results. They preserve input data even if a backend factors
destructively; copy into factor/output storage as required. Entry points
accept tensors and const/mutable views through constrained templates,
following the deduction rules above. Expression inputs require explicit
materialization. The returned tensors compose with the elementwise API:

```cpp
namespace la = vectorization::linalg;
auto x = la::solve(a, b);
auto factors = la::lu_factor(a);
auto y = la::lu_solve(factors.LU, factors.pivots, another_rhs);
auto decomposition = la::svd(a);  // U, S, Vh; full matrices by default
out = ::log(la::matrix_exp(a));   // matrix exponential, then pointwise log
```

- **Output variants.** Provide `_out` counterparts where the matching
  PyTorch function supports `out`, with output arguments first, matching
  this library's `matmul_out` convention. Require exact output shapes,
  device/dtypes and injective destinations. Classify all input/output
  pairs and pairs of outputs before writing; reject overlap by default.
  Support strided output through a packed temporary and copy when needed.
  Any permitted in-place case must be documented per operation.
- **Checked calls.** Validate shapes/options on the host in every build.
  Singular solves/inverses, failed positive-definiteness and backend
  non-convergence raise numerical errors with operation and batch index
  when backend status is available. Preserve PyTorch's per-operation
  behavior: a singular determinant is zero, and singular `slogdet`
  returns sign zero and `logabsdet=-inf` rather than raising.
- **Asynchronous status.** `solve_ex`, `inv_ex`, `cholesky_ex` and
  `lu_factor_ex` return device `info` with `check_errors=false` by default.
  They avoid a host status read; `check_errors=true` and checked wrappers
  may synchronize to report numerical failure. Invalid metadata always
  errors immediately. Upstream must expose status instead of replacing
  failure with NaNs if an `_ex` adapter is to claim this contract.
- **Layouts and lifetime.** Reuse section 11's stride packing, stream
  recording and workspace rules for every operation. Batched fallbacks
  must preserve batch order and status per matrix. No implicit CPU
  fallback or device transfer for an unsupported operation.

#### Reusable factorization objects

`lu_factorization<T>`, `cholesky_factorization<T>` and
`qr_factorization<T>` are convenience wrappers over the functional tensor
results, with owned factors, explicit metadata and supported `.solve(b)`
methods. They do not replace `lu_factor`, `cholesky`, `qr` or `svd`.
Construction factors once; subsequent solves reuse storage/workspace and
never refactor. Measure factorization separately from repeated solves.

#### P5 delivery order

Keep the phase numbers stable and deliver P5 in reviewable parts:

1. Products, product/elementwise composition and tensor layout adaptation.
2. `solve`, triangular/factor-reuse solves, LU, Cholesky, inverse,
   determinants and their listed `_ex` status paths.
3. QR, SVD, Hermitian spectrum, least squares, pseudoinverse, rank,
   norms and conditioning; general spectrum once complex output is ready.
4. Matrix exponential/power, version-gated symmetric root, matrix chains
   and the separately specified logarithm/general-root extensions.

P5 is complete only when the planned set has tested capability entries
and all supported configurations meet the contracts above. Deferred
families and unsupported backend/dtype combinations are recorded
explicitly; their absence must not be represented as full PyTorch parity.

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
- **Provenance query** on `retained_ptr` (`allocated` or `adopted`), so the
  overlap classifier knows when distinct storage proves disjointness
  ([section 6](#6-aliasing-and-overlap)).
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
- Tensor-adapter coverage for the [matrix API](#matrix-linear-algebra-api):
  explicit matrix sizes, leading dimensions, multiple RHS, batches,
  transpose/adjoint/triangle options and per-backend capabilities. Audit
  existing algorithms first; add missing contracts upstream rather than
  assuming the pointer APIs already match PyTorch.
- Full and reduced SVD/QR modes, sorted spectra with associated vector
  permutations, normalized LU pivots, rank/tolerance controls and
  driver-dependent least-squares outputs.
- Device numerical-status outputs for the listed `_ex` calls, with no
  forced host synchronization when `check_errors=false`; separate host
  launch validation from numerical status. Add complex result support
  before enabling general eigendecomposition.

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
| Sync | Asynchronous launch; the host waits on a read, an event or a checked linear-algebra status read. `_ex(check_errors=false)` leaves status on device ([section 11](#evaluation-outputs-and-numerical-errors)) |
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
| `tensor` / `tensor_view` moves ([section 4](#4-tensor-and-tensor_view)) | `performance-noexcept-move-constructor`, `bugprone-exception-escape` | Move construction transfers handles without throwing. View move assignment writes through and can reject shape or overlap, so it must be allowed to throw; its named suppression explains this. Owning moves and `retained_ptr` moves should not throw |
| `tensor::operator=` (deep copy), `tensor_view::operator=` (write-through) | `bugprone-unhandled-self-assignment` | Explicit self-assignment check; for a view, self-assignment is `full` overlap and returns immediately |
| Expression constructors and forwarding-reference builders (`product_expression`, `store_operand`) | `bugprone-forwarding-reference-overload` | Constrain templates with expression/operand traits. The implemented owning expression constructor takes `E const&`; planned forwarding overloads must preserve copy/move overload resolution |
| Rvalue capture into expression nodes ([section 5](#capture)) | `bugprone-use-after-move`, `performance-move-const-arg` | Move each operand once into the node; never `std::move` a `const` operand. Node-building functions take forwarding references and `std::forward` each operand exactly once |
| `overlap_status`, loop tier, `reduce_op` | `performance-enum-size` | `enum class … : std::uint8_t` |
| `noexcept` loops (`loop_row`, `loop2d`), destructors (`stream_guard`, `copy_token`, `noalias_proxy`) | `bugprone-exception-escape` | No `VECTORIZATION_CHECK` inside; validate in the planner before the loop; destructors never throw |
| 32-bit GPU indexing and offset arithmetic ([section 5](#cuda--hip--offset-calculator-and-indexing-width)) | `bugprone-implicit-widening-of-multiplication-result`, `bugprone-too-small-loop-variable`, `bugprone-misplaced-widening-cast` | Widen before multiplying (`static_cast<int64_t>(i) * stride`); the loop counter type matches its bound after the index-width decision. `bugprone-narrowing-conversions` is disabled, so the span check, not clang-tidy, guards narrowing to 32 bits |
| Overlap classifier address comparisons ([section 6](#6-aliasing-and-overlap)) | `bugprone-casting-through-void` | `reinterpret_cast<std::uintptr_t>(ptr)`, not a cast through `void*` |
| Metal pipeline cache ([section 5](#metal--pipeline-cache-synchronization-and-buffers)) | `bugprone-dynamic-static-initializers` | Keep the cache a function-local static, never a namespace-scope variable in a header |
| `tensor_view<const T>` storage | `portability-std-allocator-const` | Allocate as `std::remove_const_t<T>`, as `tensor_impl` does ([section 3](#3-layers-storage-geometry-kernel-descriptor)) |
| Input views ([section 4](#4-tensor-and-tensor_view)) | `performance-unnecessary-value-param` | Non-template inputs take `tensor_view<const T> const&`; outputs are taken by value only because they are written through. Deducing entry points use constrained forwarding templates ([section 11](#11-linearalgebra-adapter)) |
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
| P0 | Vectorization | Result shape from broadcasting; every leaf's shape validated; Release checks on view operations; overlap classifier raising errors; copy semantics aligned with P2; clang-tidy baseline cleared and coverage gaps closed ([section 15](#15-clang-tidy-conformance)) | Partly done: shape, view checks and classifier implemented; repository-wide clang-tidy and performance gates remain | Local Debug/Release tests pass; changed-code analysis clean. CI must still analyze the full test target and clear the repository baseline | Release benchmarks at today's speed; shape checks outside the inner loop |
| P1 | Vectorization | `retained_ptr` storage with provenance, explicit offset, views retain storage, safe expression capture, `non_overlapping` flag | Partly done: retaining views/leaves and cached injectivity implemented; provenance, explicit offset and allocation gate remain | Local AddressSanitizer lifetime/adopted-deleter tests and internal-overlap rejection pass; CUDA/HIP still need verification | CPU tensor construction and destruction: one heap allocation after U1; current promotion uses a separate control block |
| P2 | Vectorization | `tensor` / `tensor_view` split; const views; `rebind()`; `noalias()` proxy for elementwise assignment | Semantics implemented and locally verified; performance gate open | Updated tests cover packed owner copies, shallow view copies, write-through copy/move assignment, const access, signed lattice intersections and adopted storage. Debug/Release CPU and Metal GPU pass; CUDA/HIP and LibTorch remain unverified | Expression building with `tensor<T>` leaves adds no atomic operations: not met by current retain-all capture; benchmark and resolve before closing the phase |
| P3 | Vectorization | CPU planner tiers 0–2 with `run_range`; contiguous fast path; `loop2d`; threading policy (bytes and op cost, intra-row splits, BLAS coordination); tiled transpose; `flip`; tier-2 specialization; planner-owned prefetch; aligned-source variant removed; CPU reductions | Partly done: planner, `loop2d`, row-split threading | Strided, broadcast, `flip` and reduction tests pass; serial SIMD fallback taken with parallel support disabled | Tier 0 ≥ 90% of STREAM triad bandwidth on one thread and scales with threads; tier 1 ≥ 80% for rows ≥ 256 elements; latency at n ≤ 64 within 10% of today; `t().contiguous()` ≥ 50% of tier-0 copy bandwidth |
| U1 | Memory | [Section 12](#memory-u1) | Not started | Landed upstream with tests; submodule bumped | Offset-aware copy at device-copy bandwidth |
| U2 | LinearAlgebra | [Section 12](#linearalgebra-u2) | Not started | Landed upstream with tests; submodule bumped | GEMV, DOT and batched GEMM at backend-library speed |
| U3 | Scalars | [Section 12](#scalars-u3) | Not started | Landed upstream with tests; submodule bumped | Metal interior-slice dispatch with no per-call buffer wrapping |
| P4 | Vectorization | CUDA/HIP offset calculator with span-checked 32-bit indexing and vector loads; strided GPU destinations; Metal pipeline cache, batched command buffers, per-allocation `MTLBuffer`; GPU reductions; offset-aware transfers; unified launch contract ([section 13](#13-backend-launch-unification)) | Not started | Device clone of an interior slice without host memory (Metal locally; CUDA/HIP in CI); large-stride small tensors take the wide-index path; same launch signature on every backend; macOS `neon` + `metal` clang-tidy job passes | Tier 0 and strided tiers in GB/s on each GPU backend; Metal per-call host overhead after the first call in µs and free of string building |
| P5 | Vectorization | Lazy products, elementwise composition, epilogues and size routing; functional PyTorch-style matrix API, named results, `_ex` status and reusable factors, delivered in [four parts](#p5-delivery-order) | Not started | Product shapes, layouts, aliasing and deferred lifetimes pass; matrix operations match pinned LibTorch shapes/defaults/result fields; reconstruction/residual and numerical-status tests pass for each enabled capability | Warmed-up `c = matmul(a, b)` with reusable output, compatible layouts and sufficient workspace performs no tensor-layer allocation; small-matrix GEMM within 2× of Eigen fixed-size products; composed math evaluates each product occurrence once, then uses one elementwise pass; factor-reuse solves never refactor; packing, workspace and status-read costs reported separately |
| P6 | Vectorization | Optional, measured: streaming stores, true fused epilogues (cuBLASLt), tiny-matrix kernel tuning | Not started | Behind flags; no regression on existing tiers | Streaming stores 1.2–1.5× on write-heavy tier-0 work where they win; no regression on read-after-write chains |

Dependencies: P0 fixes incorrect results and goes first. U1, U2 and U3 run
in parallel with P1–P3. P1's allocation gate needs U1; P4 needs U1 and U3;
P5 needs U2. Only P6 is a tuning phase; every other phase meets its own
performance gate.

Remaining work after the tensor/view and `noalias()` delivery:

1. Close P0's repository-wide clang-tidy baseline and CI coverage gaps;
   changed-line analysis alone does not close that gate.
2. Complete P1's explicit-offset representation and provenance-aware
   shortcut with Memory U1, then verify allocation counts.
3. Measure expression construction and small-tensor latency. Resolve the
   retain-all capture cost against P2's atomic-free target without weakening
   the documented deferred-expression lifetime contract.
4. Run CUDA/HIP and optional LibTorch coverage for the split, including
   stream bookkeeping and view/noalias assignment.
5. Continue P3's CPU kernels/reductions and P4's GPU indexing, copies and
   Metal dispatch work; P5 products and matrix APIs remain separate work.

---

## 17. Verification and benchmark protocol

### Recorded verification

The tensor/view and elementwise `noalias()` delivery was verified locally on
November 1, 2026. These results cover that delivery, not completion of every
phase or subsequent unrelated edits.

| Check | Recorded result |
|---|---|
| CMake Debug, NEON CPU | Full CTest suite passed |
| CMake Release, scalar CPU, C++17 | Full CTest suite passed; exercises the Release `noalias()` path |
| CMake Debug with AddressSanitizer | Full CTest suite passed, including retained view/expression lifetimes and adopted-storage release |
| CMake Debug, NEON + Metal, Apple Clang | Full CTest suite passed with GPU access outside the sandbox, including `TensorGpu.ViewNoaliasFloat` and assignment profiler markers |
| Bazel `//Testing/Cxx:VectorizationCxxTests` | Passed |
| clang-tidy on `TestTensorView.cpp` with first-party headers and changed-line filtering | No diagnostics in the selected changed code; pre-existing diagnostics suppressed, so repository-wide conformance is still open |
| Formatting and whitespace | Checked touched implementation/test files with clang-format; `git diff --check` passed |
| CUDA/HIP and optional LibTorch | Not run |
| Performance and allocation gates | Not measured; no bandwidth, latency, allocation-count or atomic-free capture claim |

`TestTensorView.cpp` covers retaining view copies, write-through assignment,
const views, explicit packed owner construction, temporary expressions,
adopted deleter lifetime, reshape materialization, partial/unknown overlap,
interleaved disjoint slices and destination injectivity. Its lattice test
compares nested two-axis layouts with explicit address sets, including signed
inner coefficients. `TestTensorGpu.cpp` adds retaining GPU views, direct
view assignment, const-view expressions and `noalias()` updates. Existing
copy/contiguity tests and LibTorch comparison helper signatures were migrated
to the split; LibTorch runtime parity remains unverified.

### Correctness tests

In addition to numerical values:

- Result shape of expression construction, including broadcasting.
- Invalid shapes, axes, reshape sizes and slice steps rejected in Release.
- View lifetime: a view returned from a function; deferred expressions
  holding temporaries.
- Overlap: exact elementwise aliasing allowed; shifted intersections and
  `too_hard` cases rejected; provably disjoint interleaved slices allowed.
  Shape `(2, 2)`, strides `(3, 1)`, offsets 0 and 2 is `partial`.
  Two adopted storages over overlapping memory are not `none`.
- Internal overlap: a destination with strides `(1, 1)` on shape `(2, 2)` is
  rejected in every build type, with and without `noalias()`.
- Contiguous fast path: packed sources assigned into a transposed or
  stepped destination take the planner and land at the right positions.
- `noalias()`: proven partial overlap rejected in Debug; `too_hard` layouts run;
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
- PyTorch shape parity: vector/vector gives rank 0, matrix/vector,
  vector/matrix, matrix/matrix, unequal-rank broadcast batches and singleton
  dimensions. Reject scalar operands, bad contraction dimensions,
  incompatible batches, and `bmm` broadcasting or non-rank-3 inputs in
  Release. Cover empty outputs and zero contraction dimensions.
- `matmul`, `bmm`, `matmul_out` and `dot` accept `tensor<T>`,
  `tensor_view<T>` and `tensor_view<const T>` in every combination; a
  deferred product of temporary views evaluates correctly after its
  construction statement.
- `::exp`, `::sqr`, `::sqrt` and `::log` directly on `matmul` and `bmm`,
  nested unary chains and broadcast bias against LibTorch two-stage
  references. Compare `sqr` to `torch::square`; compare shapes as well as
  values, using dtype/backend tolerances and explicit NaN/infinity checks.
  Include negative and zero product entries and exponential overflow.
- Deferred `auto e = ::exp(matmul(a.t(), b));` survives its construction
  statement under AddressSanitizer. Instrument dispatch to verify one
  product evaluation per node occurrence and one remaining elementwise
  traversal/kernel; on GPU, verify stream ordering without a host read.
- Composed assignments with the output aliasing either matrix input or a
  broadcast leaf, and with transposed/stepped output views, follow the
  classifier and produce the same values as an explicit intermediate.
- `c = matmul(a, b)` on disjoint operands writes directly; on overlapping
  operands uses a temporary.
- `c.noalias() = / += / -= matmul(a, b)` against a reference, with an
  allocation counter confirming no temporary.
- Epilogues against two-pass references, including `d` aliasing `c`, `a`
  or `b`.
- Matrix API parity against a pinned LibTorch version: all listed
  operations' shapes, defaults, named fields, integer/real/complex result
  dtypes and permitted batch broadcasting. Include vector versus multiple
  RHS solves, left/right and adjoint solves, triangular options, QR modes,
  full/reduced SVD, triangle selection and driver-dependent empty fields.
- Numerical invariants: `A X ≈ B`, `A ≈ P L U`, `A ≈ Q R`,
  `A ≈ U diag(S) Vh`, orthogonality/unitarity and eigenpair residuals.
  Compare invariant subspaces for repeated spectra rather than requiring
  identical eigenvector/singular-vector signs, phases or bases.
- Singular/indefinite and rank-deficient inputs: checked failures versus
  per-batch `_ex.info`, determinant/slogdet singular behavior, tolerance
  cutoffs for rank/pseudoinverse, and least-squares residuals. Test empty
  matrices/batches per operation rather than assuming GEMM's rules apply.
- Matrix-function identities: exponential of zero is identity; integer
  power zero is identity and power two matches `matmul`; symmetric root
  squares to the original SPD matrix. Test non-diagonal matrices to
  distinguish these from pointwise operations. For extension matrix
  logarithm/root, use independent references and documented domain tests.
- Functional calls preserve inputs; factor objects survive input
  destruction and repeated solves reuse factors. `_out` rejects overlap
  between outputs or with inputs and handles supported strided layouts.
- GPU checked versus `_ex` synchronization and per-batch `info`; unsupported
  backend/dtype/driver combinations fail explicitly without host fallback.
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
  the steady-state call; factorization separate from solve. Measure
  `exp`, `sqr`, `sqrt` and `log` after products against explicit two-stage
  references and eager LibTorch, including intermediate allocation/traffic
  and GPU enqueue overhead. Report optional fused epilogues separately.
- **Matrix algorithms.** Solves with one and many RHS; factorization
  versus factor reuse; rectangular QR/SVD/least squares; batched
  inverses, determinants and matrix functions. Compare the same modes,
  triangle, tolerances and drivers against LibTorch. Report checked
  status synchronization separately from `_ex` enqueue/device time, and
  adapter copies/packing separately from numerical kernels.
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

- `include/terminals/tensor.h` — shared `tensor_impl<T, IsView>`, owning/view
  semantics, `clone()`, `contiguous()`, cached injectivity, checked assignment,
  `init_from_expression()`, the `#if 0` matrix block.
- `include/terminals/tensor_view.h` — public view include.
- `include/terminals/noalias.h` — non-copyable assignment/update proxy.
- `include/terminals/sizes_and_strides.h` — inline sizes/strides storage.
- `include/expressions/strided_ref.h` — kernel-side leaf descriptor,
  host-side `retained_ref` and read-only Memory view facade.
- `include/expressions/expression_interface.h` — `store_operand()`,
  expression nodes, `raw_lowered_t` / `lower_raw()`; legacy
  `buffers_overlap()` / `expression_aliases_ptr()` helpers.
- `include/expressions/overlap.h` — active overlap classifier, signed carry
  lattice test and assignment checks.
- `include/expressions/expression_shape.h` — broadcast shape inference,
  `rebase_leaves()`.
- `include/expressions/iteration_plan.h` — CPU planner, tiers, `loop2d`,
  threaded row split.
- `include/expressions/expressions_evaluator.h` — `run()` / `fill()`
  dispatch, contiguous SIMD loop, prefetch and alignment variants.
- `include/expressions/expressions_evaluator_gpu.h` — CUDA/HIP kernels and
  reductions.
- `include/expressions/expressions_evaluator_metal.h` — MSL emitter and
  direct leaf assignment dispatch.
- `include/expressions/expressions_builder.h` — operators, including
  elementwise `*`.
- `include/expressions/reduce_op.h` — reduction operations.
- `Testing/Cxx/TestTensorView.cpp`, `TestTensorGpu.cpp` — split, ownership,
  overlap and `noalias()` regression coverage on CPU and supported GPUs.
- `ThirdParty/Scalars/include/backend/gpu/metal/metal_dispatch.mm` — Metal
  buffers and dispatch.
- `ThirdParty/Memory/include/common/retained_ptr.h`, `data_ptr.h`,
  `data_view.h` — storage primitives.
- `ThirdParty/Memory/include/gpu/cuda_caching_allocator.h`,
  `gpu_workspace.h` — GPU allocation and workspace.
- `ThirdParty/LinearAlgebra/include/matrix_operation/multiplication.h`,
  `multiplication_batched.h` — GEMM entry points.
- `ThirdParty/LinearAlgebra/include/matrix_operation/` — solve,
  factorization, inverse, norm, spectral and matrix-function algorithms;
  tensor adapters must normalize their contracts before public exposure.
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
