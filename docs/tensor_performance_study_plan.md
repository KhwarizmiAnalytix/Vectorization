# Tensor Performance Study Plan

Date: October 27, 2026.

Scope: review the proposed tensor architecture against PyTorch and Eigen,
then establish an implementation and measurement plan for allocation,
elementwise operations, reductions, and linear algebra.

This document records a source review and proposed studies. It does not
report measured performance or replace the contracts in
[Tensor Design](vectorization_tensor_design.md). The
[Tensor Speed Plan](speed_tensor_plan.md) is retained as historical context;
its proposals have been merged into the design document.

## 1. Architectural direction

Combine Eigen-style expression fusion with PyTorch-style storage, strides,
broadcasting, and backend dispatch. The main performance opportunities are
avoiding allocations, intermediate memory passes, unnecessary packing, and
CPU/GPU synchronization. SIMD and threading support those goals.

Use a cheap contiguous path for ordinary packed tensors and a general
iteration planner for views and broadcasting. Keep numerical backend calls
behind a thin tensor adapter that resolves shape, layout, aliasing, and
workspace requirements.

## 2. Review findings

### 2.1 Refresh the implementation status

The design's list of current problems is partly outdated. The reviewed code
already contains broadcast shape inference, CPU iteration planning, and
optional threading. Storage-retaining views, safe capture of temporary
tensor operands, and tensor-level matrix multiplication remain work items.

Before estimating implementation effort, classify each feature as
implemented, incomplete, or proposed. Existing code paths still need their
correctness and performance gates verified.

Relevant files:

- [tensor.h](../include/terminals/tensor.h)
- [expression_interface.h](../include/expressions/expression_interface.h)
- [expressions_evaluator.h](../include/expressions/expressions_evaluator.h)
- [iteration_plan.h](../include/expressions/iteration_plan.h)

### 2.2 Preserve the serial SIMD fallback

The initial review identified a path where large contiguous expressions
skipped the explicit SIMD loop when parallel support was disabled and fell
through to scalar evaluation. The working tree at the time this study was
written has moved the preprocessor guard around the parallel branch, so the
serial SIMD body remains reachable. This change has not been benchmarked
as part of this study.

Verify that enabling or disabling parallel support preserves vectorized
serial execution. Measure sizes immediately below and above the threading
threshold as well as large arrays.

### 2.3 Separate CPU and GPU allocation targets

Coallocating the reference-count control block and payload is appropriate
for ordinary CPU storage. For device-only CUDA/HIP storage, a host-managed
reference count cannot simply reside beside the device payload.

Study these separate targets:

- CPU: one aligned allocation for common owning storage, including its
  control block; inline shape metadata for common ranks.
- GPU: pooled host metadata plus cached device storage, with no new device
  allocation or synchronization during steady-state reuse.
- External buffers: explicit ownership/deleter rules and a separate control
  block where needed.

PyTorch's caching allocator is a reference for reuse and stream-aware
reclamation. See [CUDA memory management][pytorch-memory].

### 2.4 Validate GPU index spans

The proposed condition `numel < 2^31` is insufficient for 32-bit indexing.
A small tensor with large strides can access offsets beyond that range.

Check both element count and the maximum reachable offset of every operand,
including the destination. Use the units of the selected offset calculator
and overflow-checked arithmetic. Rebase and split work, or use wider
indexing, when the bounds do not fit.

PyTorch checks both element count and operand byte offsets. See
[TensorIterator implementation][pytorch-iterator].

### 2.5 Distinguish temporary elimination from epilogue fusion

The proposed lowering of `c = matmul(a, b) + d` to copying `d` into `c`
followed by GEMM with `beta = 1` eliminates the product temporary, but the
copy remains an extra memory pass.

Classify aliasing before any write: copying into `c` must not destroy an
input to GEMM, and overlap with `d` also needs defined handling. Reusing
`c` itself as the addend can avoid the copy when valid.

Study true fused epilogues separately, using supported backend operations
such as cuBLASLt bias and activation epilogues. Their applicability depends
on dtype, layout, and batching constraints. See
[cuBLASLt epilogues][cublas-epilogues].

### 2.6 Qualify allocation-free linear algebra

Disjoint operands alone do not eliminate input packing, output conversion,
or workspace requirements. Use the following performance contract:

> A warmed-up multiplication with reusable output, compatible layouts, and
> sufficient workspace performs no tensor-layer allocation.

Measure packing and workspace growth separately from the steady-state
backend call. Backend-internal allocation is a separate measurement from
tensor-layer allocation.

## 3. Comparison with PyTorch and Eigen

| Area | PyTorch | Eigen | Direction for this library |
|---|---|---|---|
| Ownership | Shared tensor/storage handles | Owning matrices and borrowing views | Owning tensors and retaining views; document deep-copy costs |
| Elementwise execution | Eager operators; compilation can optimize larger regions | Lazy expression templates | One CPU traversal or GPU kernel per supported expression |
| Layout handling | General strides and broadcasting | Matrix/array expressions; separate Tensor module | Contiguous fast path plus general stride planner |
| Small matrices | General runtime dispatch | Fixed dimensions permit specialization and inline storage | Runtime small-matrix kernels first; optional static shapes later |
| Large linear algebra | Specialized numerical backends | Native kernels or external BLAS/LAPACK | Reuse LinearAlgebra and tuned backend libraries |

Eigen introduces temporaries when aliasing or subexpression reuse makes them
useful. Apply the same reasoning around products instead of requiring all
expressions to fuse. See [Eigen lazy evaluation][eigen-evaluation].

Compare eager PyTorch and compiled PyTorch separately where applicable;
they are different execution baselines. See
[PyTorch compilation][pytorch-compile].

Keep the proposed deep-copy semantics if value semantics are intentional.
However, `tensor b = a` then copies elements, unlike a PyTorch handle copy.
Make this cost explicit and use views or references for borrowing.

## 4. Allocation study

Establish these contracts and measurements:

1. Provide an uninitialized allocation path for numerical storage. Kernels
   that overwrite every element should not first zero the payload.
2. Reuse storage when assigning into a correctly sized destination.
3. Keep common shapes, strides, views, and iteration plans free of metadata
   heap allocations.
4. Reuse CPU scratch arenas and per-stream GPU workspaces for packing and
   reduction partials. Release or reset scratch only when its users finish.
5. Keep queued GPU operands and temporaries valid through completion.
   Distinguish storage ownership, allocator reuse protection, and event-based
   ordering between streams.

Measure construction/destruction latency, allocation counts, cached versus
new device allocations, peak live bytes, retained/reserved bytes, and
steady-state reuse. Separate common low-rank tensors from ranks that spill
metadata to the heap.

## 5. Elementwise and reduction study

For an expression such as:

```cpp
out = a * b + c;
```

The target is one traversal with three input reads and one output write,
without an intermediate tensor. Avoiding the intermediate reduces memory
traffic, especially for arrays larger than cache.

Study and verify:

- Shape, device, and overlap validation once before kernel execution.
- Allocation-free planner and expression construction for common cases.
- Threading thresholds based on bytes processed and operation cost;
  32,768 elements is an initial tuning point, not a universal threshold.
- Splitting within long rows as well as between rows, so a layout with few
  long rows can still use multiple workers.
- Coordination between tensor workers and BLAS threads to avoid
  oversubscription.
- Contiguous, row-broadcast, column-broadcast, transpose, and general strided
  cases, including non-contiguous destinations.
- Prefetch and non-temporal stores as measured options. Non-temporal stores
  can hurt when the next operation immediately reads the result.

For reductions, define accumulator precision, empty-input behavior, and a
determinism policy before performance tuning. Parallel summation changes
floating-point addition order. Reuse partial buffers and keep GPU scalar
results on the device until the caller explicitly requests a host value.

## 6. Linear algebra study

| Workload | Preferred route |
|---|---|
| Vector inner product | DOT |
| Matrix-vector product | GEMV |
| Tiny matrix product | Specialized small kernel |
| General matrix product | GEMM |
| Many small products | Batched GEMM |
| Repeated solves with the same matrix | Factor once and reuse factors |

Use strides, leading dimensions, and transpose flags to pass compatible
views directly. Pack only unsupported layouts. Eigen's external backend
integration is a useful comparison; see
[Using BLAS/LAPACK from Eigen][eigen-blas].

The current LinearAlgebra API already accepts caller-owned workspace. Build
on that interface and add the planned `alpha`/`beta`, GEMV/DOT, and rectangular
strided-batched entry points. Reuse output and workspace across calls.

For factorizations, retain factors and pivots in reusable objects with solve
operations. Measure factorization separately from subsequent solves.

Treat Eigen fixed-size products as a separate benchmark category: dynamic
tensor allocation and metadata costs are not comparable to inline storage
with compile-time dimensions. Match backend and thread settings for large
matrix comparisons.

## 7. Implementation sequence and evidence

| Step | Work | Evidence required |
|---|---|---|
| 1 | Refresh implementation status and verify serial SIMD fallback | Accurate status table; Release behavior and timings around the threading threshold |
| 2 | Finish retained views and safe temporary-expression capture | Lifetime cases clean under AddressSanitizer; no per-element ownership work |
| 3 | Establish output reuse and reusable scratch | Allocation counters demonstrate steady-state reuse |
| 4 | Complete CPU contiguous, broadcast, strided, transpose, and reduction paths | Correctness across layouts; latency, bandwidth, and thread-scaling measurements |
| 5 | Add tensor-level GEMM, batching, and reusable factorizations | Backend comparisons including packing, workspace, and overlap cases |
| 6 | Complete asynchronous GPU execution and lifetime handling | Device timings, host overhead, stream-ordering and reclamation checks |
| 7 | Tune epilogues, tiny matrices, and streaming stores | Measured improvements without regression in representative workloads |

Memory and LinearAlgebra API changes follow the upstream workflow specified
in the main design. Coordinate dependencies explicitly; storage allocation
gates cannot be completed before their required Memory APIs exist.

## 8. Benchmark protocol

- Separate allocation-inclusive execution from preallocated output reuse.
- Separate first-call setup, compilation, and workspace growth from warmed-up
  steady-state execution.
- Match thread counts, dtype, numerical precision policy, layouts, numerical
  backend, and CPU instruction set across comparisons.
- Report latency for small tensors, effective bandwidth for elementwise
  operations, and throughput for matrix products.
- Record the byte-count convention used for bandwidth; distinguish useful
  algorithmic traffic from measured hardware traffic.
- Measure GPU kernels with device timing and complete-operation latency with
  synchronization at the measurement boundary. Report host enqueue overhead
  separately.
- Cover empty and scalar tensors, small arrays, cache-resident arrays, and
  arrays larger than cache; cover square, rectangular, transposed, and batched
  products.
- Compare Eigen dynamic and fixed-size cases separately, and eager PyTorch
  and compiled PyTorch separately where applicable.
- Record hardware, compiler/build flags, dependency versions, thread settings,
  and timing variability with results.

Use the results to attribute remaining cost to allocation, planning, memory
traffic, packing, synchronization, or the numerical kernel. Performance
thresholds remain hypotheses until measured against a matched baseline.

[pytorch-memory]: https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management
[pytorch-iterator]: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/TensorIterator.cpp
[pytorch-compile]: https://docs.pytorch.org/docs/stable/generated/torch.compile.html
[eigen-evaluation]: https://libeigen.gitlab.io/eigen/docs-nightly/TopicLazyEvaluation.html
[eigen-blas]: https://libeigen.gitlab.io/eigen/docs-nightly/TopicUsingBlasLapack.html
[cublas-epilogues]: https://docs.nvidia.com/cuda/cublas/index.html#cublasltepilogue-t
