# Tensor Speed Plan

Scope: performance requirements for the tensor redesign in
[vectorization_tensor_design.md](vectorization_tensor_design.md), as of
October 2026. This is a **proposed** companion plan, not the current contract.
It changes decisions in the design's planner, GPU, Metal, storage and
LinearAlgebra sections. It also moves performance work out of the final
tuning phase (P6) and into the phase that builds each component.

Section references of the form "design §N" point to the design document.

---

## Contents

1. [Why a separate plan](#1-why-a-separate-plan)
2. [Current hot-path costs](#2-current-hot-path-costs)
3. [Summary of changes](#3-summary-of-changes)
4. [CPU elementwise evaluation](#4-cpu-elementwise-evaluation)
5. [CUDA and HIP](#5-cuda-and-hip)
6. [Metal](#6-metal)
7. [Storage and allocation](#7-storage-and-allocation)
8. [Products](#8-products)
9. [Reductions](#9-reductions)
10. [Phase gates](#10-phase-gates)
11. [Benchmarks](#11-benchmarks)

---

## 1. Why a separate plan

The design document settles correctness: lifetimes, overlap, shapes and
slice steps. It leaves performance to P6, but several P3 and P4 choices set
the ceiling and are expensive to change afterwards:

- the shape of the planner's output, which decides whether work can be split
  across threads;
- the inner-loop interface shared by tiers 0–2;
- the GPU indexing width and kernel-parameter layout;
- the Metal pipeline cache key and synchronization model.

This plan fixes those choices up front and gives each phase a measurable
performance gate.

---

## 2. Current hot-path costs

Confirmed by source inspection:

| Behavior | Where | Effect |
|---|---|---|
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

## 3. Summary of changes

| # | Change | Design section affected | Phase |
|---|---|---|---|
| C1 | Planner output is a splittable range; CPU `parallel_for` | §5 | P3 |
| C2 | O(leaves) contiguous fast path before planning | §4, §5 | P3 |
| C3 | One `loop2d` inner kernel for tiers 0–2 | §5 | P3 |
| C4 | Tiled transpose and lane-reversing `flip` kernels | §5, §7, §9 | P3 |
| C5 | Compile-time single-broadcast specialization for tier 2 | §5 | P3 |
| C6 | Planner-owned prefetch; aligned-source variant removed | §5 | P3 |
| C7 | Streaming stores for large, non-aliased destinations | §5 | P6 |
| C8 | Shared offset calculator, 32-bit indexing, vector loads on GPU | §3, §5 | P4 |
| C9 | Metal: type-keyed pipeline cache, batched command buffers, per-allocation `MTLBuffer` | §5 | P4 |
| C10 | Control block allocated together with the data | §11 (Memory) | U1 |
| C11 | `c = matmul(a, b)` writes directly when the classifier reports no overlap | §6, §10 | P5 |
| C12 | Product epilogues: `+ d`, scalar multiples | §10 | P5 |
| C13 | Size-based GEMM routing: tiny, vector, batched | §10, §11 (LinearAlgebra) | U2, P5 |
| C14 | Reductions go through the planner | New | P3, P4 |
| C15 | Per-phase performance gates replace P6 | §12, §13 | All |

---

## 4. CPU elementwise evaluation

### C1. Splittable plan and threading

The planner's result describes the merged iteration space as a range that
can be split into chunks:

```cpp
// Produced once per evaluation; consumed by one or more threads.
struct iteration_plan
{
    int     rank;                       // after merging, usually 1–3
    int64_t sizes[MaxRank];             // shared by every operand
    int64_t strides[MaxOperands][MaxRank];
    int     tier;
};

// Runs the plan over [begin, end) of the linearized iteration space.
template <typename Lowered>
void run_range(Lowered const& expr, iteration_plan const& plan, int64_t begin, int64_t end);
```

- **Threading.** Above a grain size (start at 32,768 elements, as PyTorch
  does, and tune per backend), the range is split with a `parallel_for`.
- **Chunk edges.** Chunk edges are rounded to multiples of
  `simd::size × VECTORIZATION_PACKET_SIZE`, so each chunk keeps the aligned
  body loop and only the first and last chunks peel.
- **Reference points.** PyTorch: `TensorIterator::for_each` with
  `at::parallel_for`. Eigen: `TensorExecutor` with `ThreadPoolDevice`.
- **Thread-local data.** Worker threads use `strided_ref` and raw pointers,
  never `tensor_view`. This avoids contended reference-count updates on a
  shared control block (see [C10](#c10-one-allocation-per-tensor)).

### C2. Contiguous fast path

Planning costs O(leaves × rank) plus a sort, which is more than the
arithmetic for small tensors. Before planning, an O(leaves) check is run:

1. Every leaf has the `contiguous` flag from `tensor_impl::flags_`.
2. Every leaf has the destination's shape.

When both hold, evaluation goes straight to tier 0 with no sort, merge or
lattice test. The overlap check is ordered cheapest first:

1. Different storage → `none`.
2. Same base pointer, sizes and strides → `full`.
3. Only then the lattice test.

Two distinct `tensor<T>` objects each own their storage and can never
partially overlap, so for them step 1 or 2 always decides. A freshly
constructed `tensor<T>` destination skips the check.

The target is Eigen-like cost (no runtime planning) for trees of packed
tensors, not TensorIterator's microsecond-level setup.

### C3. One `loop2d` inner kernel

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

### C4. Transpose and `flip` kernels

`clone`, `contiguous`, conversion from a view, and the GEMM packing step
(design §10) all go through the planner. A purely scalar tier 3 would make
`a.t().contiguous()`, the most common tier-3 case, the slowest copy in the
library.

- **Tiled transpose.** When exactly one 2-D pair of dimensions is swapped
  between destination and source, copy in 8×8 or 16×16 tiles, transposed in
  registers. Other tier-3 layouts stay scalar.
- **Inner dimension choice.** Planner step 4 sorts by output stride, then
  breaks ties by the stride of the largest source operand.
- **`flip`.** Load contiguous registers, reverse the lanes with a permute, and
  store. Reading back to front one element at a time is not acceptable.

### C5. Tier 2 specialization

A runtime flag per leaf inside the hot loop is often not moved out of the
loop when several leaves and packet unrolling are involved.

- Build the compile-time specialization for "exactly one leaf has inner stride
  0" (the bias-add case) in P3, as PyTorch's scalar-operand `vectorized_loop`
  does.
- Other combinations use `loop2d` with the broadcast value splatted once per
  row.

### C6. Prefetch and alignment variants

- **Prefetch.** The plan owns prefetch distance. Sequential unit-stride loops
  either issue no prefetch or prefetch 256–512 bytes ahead, chosen by
  benchmark. Tier 1 adds a prefetch of the next row's start when the row
  stride is large.
- **Alignment.** Remove the `src_aligned` template variant. Keep the
  destination peel, which avoids stores that split a cache line on AVX-512.
  This roughly halves the instantiations per expression type.

### C7. Streaming stores

When the destination does not overlap any source and is larger than about
half the last-level cache, use non-temporal stores. This avoids reading every
destination line before writing it, which typically gives 1.2–1.5× on
write-heavy elementwise work. Neither Eigen nor PyTorch does this; glibc and
MKL `memcpy` do. This is optional and is benchmarked in P6.

---

## 5. CUDA and HIP

### C8. Offset calculator and indexing width

- **One size array.** After broadcasting, every operand has the same sizes;
  only strides differ. Store the sizes (as fast divisors) once and the strides
  per operand, like PyTorch's `OffsetCalculator`. A `strided_ref` per leaf with
  `MaxRank = 8` is 144 bytes, so ten leaves would mean about 1.4 KB of kernel
  parameters.
- **Lowering depends on the tier.** Tier 0 leaves lower to `T*`, not
  `strided_ref`.
- **32-bit indexing.** Use 32-bit indices when the element count is below
  2³¹, and split larger launches, like PyTorch's `can_use_32bit_indexing`.
  `IntDivider` only speeds up 32-bit division; 64-bit division is very slow on
  GPUs.
- **Vector loads.** When every leaf is 16-byte aligned and contiguous, each
  thread processes 4 elements with 16-byte loads and stores, like PyTorch's
  `vectorized_elementwise_kernel`. Other cases use the scalar unrolled kernel.

---

## 6. Metal

### C9. Pipeline cache, synchronization and buffers

- **Cache key.** The pipeline is held in a function-local static per
  expression type `E`, tier and rank. MSL source is generated only on the
  first call, so later calls build no strings and take no global lock. Scalar
  values already travel in a buffer, so they do not change the source.
- **Batched command buffers.** Evaluations are encoded into a shared command
  buffer and committed without waiting. The host waits only when it reads the
  data or when a stream event requires it, as PyTorch MPS's `MPSStream` does
  with commit-and-continue.
- **One `MTLBuffer` per allocation.** Because storage is a `retained_ptr`
  (design §3), the `MTLBuffer` is created once at allocation and kept with it.
  A view is passed as `setBuffer:offset:` using `tensor_impl::offset_`. This
  removes per-call wrapping and makes interior slices work, which
  `newBufferWithBytesNoCopy` cannot do because it requires page alignment.

---

## 7. Storage and allocation

### C10. One allocation per tensor

Promoting `data_ptr` to `retained_ptr` creates a separate control block, so
every temporary costs two heap allocations. In Memory (U1):

- for sized allocations, place the control block in the same allocation as
  the data, as `std::make_shared` does;
- keep the separate control block for adopted external buffers.

Small fixed shapes (Eigen's stack-allocated fixed-size matrices) are out of
scope. Benchmarks report them separately so they are not compared with Eigen
by accident.

---

## 8. Products

### C11. Direct writes without `noalias()`

Design §10 lowers `c = matmul(a, b)` through a temporary. Eigen needs one
because it cannot see storage identity; this design can.

| Statement | Lowering |
|---|---|
| `c = matmul(a, b)`, classifier reports `none` | `matmul_out(c, a, b, 1, 0)` |
| `c = matmul(a, b)`, any overlap | Packed temporary, then copy into `c` |
| `c.noalias() = matmul(a, b)` | `matmul_out(c, a, b, 1, 0)`, check skipped in Release |

The check is O(rank) next to an O(n³) product. `noalias()` now only skips
the check; it no longer decides whether a temporary is allocated. Design §6
("What it buys") and the §10 lowering table are updated to match.

### C12. Product epilogues

Recognized forms, evaluated without a temporary or an extra pass:

| Expression assigned to `c` | Lowering |
|---|---|
| `matmul(a, b) + d` | Copy `d` into `c`, then GEMM with β = 1 |
| `s * matmul(a, b)` | GEMM with α = s |
| `c += s * matmul(a, b)` | GEMM with α = s, β = 1 |

Any other tree containing a product still evaluates the product into a
temporary leaf. Eigen recognizes `scalar * product` the same way.

### C13. Routing by size and shape

| Case | Route | Reference point |
|---|---|---|
| Matrix × vector | GEMV | PyTorch `matmul` |
| Vector · vector | DOT | PyTorch `matmul` |
| Tiny matrices (n ≲ 16–20, tuned) | NATIVE unrolled kernel, or MKL JIT / direct call | Eigen `EIGEN_GEMM_TO_COEFFBASED_THRESHOLD` |
| `bmm` on CPU | `gemm_batch_strided` where available; otherwise split the batch across threads | MKL |
| Everything else | Backend GEMM (design §10) | — |

- **GPU `dot`.** Returning `T` forces a device-to-host synchronization. Add
  `dot_out` into a 0-d tensor, and the same for `accumulate`.
- **Upstream.** LinearAlgebra (U2) exposes GEMV, DOT and strided-batched entry
  points with the same leading-dimension rules as GEMM.

---

## 9. Reductions

### C14. Reductions through the planner

`accumulate`, `hmin` and `hmax` require contiguous input today, and axis
reductions (`sum(dim)`) are not designed. TensorIterator serves both
elementwise operations and reductions; this planner does too.

- **CPU (P3).** A reduction plan marks reduced dimensions. A reduction over the
  inner dimension uses the existing per-register accumulators; a reduction
  over an outer dimension uses vertical SIMD accumulation across rows.
- **GPU (P4).** A fixed grid (multiprocessor count × k) with grid-stride
  accumulation per thread and warp shuffles inside a block, then one final
  launch. There is no per-pass allocation.

---

## 10. Phase gates

Phase P6 ("performance tuning") is removed. Each phase instead has a
performance gate alongside its correctness criteria from design §12.

| Phase | Performance gate |
|---|---|
| P0 | Release benchmarks at today's speed; the new shape checks run outside the inner loop |
| P1 | Construction and destruction of a tensor: one heap allocation after U1 |
| P2 | Expression building with `tensor<T>` leaves adds no atomic operations |
| P3 | Tier 0 ≥ 90% of STREAM triad bandwidth on one thread and scales with threads; tier 1 ≥ 80% for rows ≥ 256 elements; latency at n ≤ 64 within 10% of today; `t().contiguous()` ≥ 50% of tier-0 copy bandwidth |
| U1 | Offset-aware copy at device-copy bandwidth |
| U2 | GEMV, DOT and batched GEMM entry points at backend-library speed |
| P4 | Tier 0 and strided tiers measured in GB/s on each GPU backend; Metal per-call host overhead after the first call measured in µs and free of string building |
| P5 | `c = matmul(a, b)` on disjoint operands allocates nothing; small-matrix GEMM within 2× of Eigen fixed-size products |

Gate thresholds are starting points; each is confirmed against a measured
baseline before the phase starts.

---

## 11. Benchmarks

Additions to the measurements in design §13:

- **Bandwidth.** Achieved GB/s per tier, against a STREAM triad baseline on
  the same machine, on CPU and on each GPU backend.
- **Latency.** Nanoseconds per call at n = 1, 16, 64 and 256 elements,
  including planning and overlap checks.
- **Thread scaling.** Tier 0 and tier 1 throughput from one thread up to all
  cores.
- **Layouts.** Transpose copy, `flip`, column broadcast and outer broadcast.
- **Products.** Square and rectangular GEMM from n = 2 to n = 4,096;
  matrix × vector; batched small matrices; `c = matmul(a, b)` with and without
  `noalias()`.
- **Metal.** First-call and steady-state host overhead per evaluation; chained
  expressions with and without a host read between them.
- **Comparisons.** Eigen and LibTorch, both with preallocated outputs and with
  allocation included.
