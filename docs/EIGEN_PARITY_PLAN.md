
# Eigen parity plan

Remaining optimizations where Vectorization falls behind Eigen (the copy
bundled under `ThirdParty/LinearAlgebra/ThirdParty/eigen`).

## Baseline

Measured on 2026-11-17 with clang-cl 22, AVX2, SVML, and
`VECTORIZATION_PACKET_SIZE=1`, in a Release build on one thread. Values are
medians of 3 runs. The strided cases use a 512×509 view of a 512×520 row-major
tensor. The scratch benchmark that produced these numbers is not in the
repository; P4 adds equivalent cases to `benchmark_tensorbaselines`.

| Case | Vectorization | Eigen | Gap |
|---|---:|---:|---:|
| `hmax` / `hmin`, f32, n = 32,771 | 2.80 µs | 0.72 µs | 3.9× |
| `hmax` / `hmin`, f64, n = 32,771 | 5.73 µs | 1.44 µs | 4.0× |
| `accumulate(a)`, f32 (`sum(a)`: 0.55 µs) | 1.41 µs | 0.55 µs | 2.6× |
| `sum` of strided view, f32 | 543 µs | 5.3 µs | ~100× |
| `sum` of strided view, f64 | 546 µs | 11.6 µs | ~47× |
| `hmax` of strided view, f32 | 542 µs | 7.5 µs | ~72× |
| `sum(va * vb)` over strided views, f32 | 1086 µs | 13.4 µs | ~80× |
| `va * vb + va` into a 2-D tensor, f32 | 47.2 µs | 46.4 µs | level |
| `va * vb + va` into a 2-D tensor, f64 | 105 µs | 80 µs | 1.3× |

These already match Eigen and need no work:

- fusion of a whole expression into a single pass
- the scalar head up to the first aligned element, with aligned stores and co-aligned loads
- FMA contraction
- 4 accumulators in `sum`, `prod`, `mean`, `nansum` and `sum(a * b)`
- SIMD on rows whose inner dimension is contiguous, for element-wise assignment
- the blocked GEMM with `alpha * A * B + beta * C` epilogue fusion

## Status (2026-11-20)

Re-measured and profiled with a sampling profiler; see the README results table.

- P1 done: `hmin` / `hmax` / `accumulate` are level with Eigen (were 2.6–4×). The
  old loops had one accumulator, so each add or max waited on the previous one.
- P2 done: strided reductions went from 553 µs to about 6 µs (f32 sum; Eigen
  5.3 µs). The old path divided the flat index by every dimension for each
  element. Strided max is about 20% faster than Eigen; strided f32 sum and dot
  are 10–25% behind. The remainder is per-row loop overhead: a hand-written
  loop with a runtime row length measured level with Eigen.
- P3: no work needed; strided `va * vb + va` is level with Eigen in both types.
- P4 done: `Max`, `Min` and `Strided{Sum,Max,Dot,Fma}` rows are in
  `benchmark_tensorbaselines`.
- Small arrays: fixed cost reduced from about 55 ns to 22 ns. About 17 ns of the
  rest is the four atomic reference-count updates of the operand capture design.
- Transcendentals (P5): 92–97% of the time is inside SVML, and exp / log vary by
  up to 1.8× with data placement relative to a 4 KiB page. Closing this gap
  means inline polynomial kernels in Scalars, which owns the math backends.

## P1: Min/max and `accumulate` reductions

`hmin`, `hmax` and `accumulate` use the older loops in
[expressions_evaluator.h](../include/expressions/expressions_evaluator.h), which unroll
by `VECTORIZATION_PACKET_SIZE` (default 1). Each add or max then waits for the
previous one.

- [x] Handle `reduce_op::min` and `reduce_op::max` in `reduce_expression_impl`
  ([expression_reductions.h](../include/expressions/expression_reductions.h)), using
  `S::min` / `S::max` with at least 4 accumulators.
  - Keep the empty-input results the same as today: `max()` for min and `-max()` for max.
  - Keep the NaN behavior of the current `std::fmin` / `std::fmax` scalar path.
- [x] Change the public `hmin`, `hmax` and `accumulate` to call `reduce_expression<Op>`.
- [x] Delete the old `detail::accumulate`, `detail::hmin` and `detail::hmax` bodies.
- [x] **Target:** `hmin` and `hmax` within 10% of Eigen's `minCoeff()` and `maxCoeff()`,
  and `accumulate` level with `sum`.

## P2: Reductions over strided views

`reduce_strided` in
[expressions_evaluator.h](../include/expressions/expressions_evaluator.h) adds up
strided and broadcast operands one logical element at a time, at about
2 ns per element. Eigen runs SIMD along each contiguous inner row.

- [x] Rewrite `reduce_strided` to walk the iteration planner's rows
  ([iteration_plan.h](../include/expressions/iteration_plan.h)).
  - Rows with stride 1 run in SIMD with 4 accumulators carried from one row to the next.
  - Only stride other than 1 or broadcast cases stay scalar.
- [x] Cover strided tensors, strided expressions (`sum(va * vb)`), and min/max once P1 is done.
- [ ] Confirm that the GPU path (`try_reduce_gpu`) is unaffected. It still runs first and is
  unchanged, but no CUDA / HIP / Metal build was run.
- [ ] **Target:** within 10–20% of Eigen's `block().sum()`, `maxCoeff()` and
  `(a * b).sum()` on the 512×509 view.

## P3: Strided element-wise rows in f64

`loop_row` in [iteration_plan.h](../include/expressions/iteration_plan.h) has no
unrolling and no aligned head, unlike `run_flat`.

- [ ] Add `VECTORIZATION_PACKET_SIZE` unrolling and an aligned head on the destination to `loop_row`.
- [x] **Target:** f64 `va * vb + va` within 10% of Eigen (already level; no change needed).

## P4: Benchmark coverage

- [x] Add rows to `benchmark_tensorbaselines`
  ([BenchmarkTensorBaselines.cpp](../Testing/Cxx/BenchmarkTensorBaselines.cpp)) for:
  - Max and Min
  - strided sum
  - strided dot
  - strided FMA

  Check each result against a reference, as the existing rows do.
- [x] Update the README results table and the list of remaining gaps.

## P5: Later, lower priority

- [ ] Finish partial-SIMD tails with masked loads and stores (AVX `maskload`, AVX-512 masks, SVE
  predicates), as Eigen's `UsePacketSegment` does. This affects small arrays only.
- [ ] Look into inline polynomial math for float log, sin, cos and tanh, and double tanh,
  which are 1.8–2.8× slower than Eigen with SVML. This needs an accuracy decision first.
- [ ] Decide whether to add axis reductions (`sum(dim)`), the equivalent of Eigen's
  `colwise()` / `rowwise()`. This is an API addition, not a speed fix.

## Verification for each item

- [x] Add unit tests under `Testing/Cxx/` (`TestTensorReductions`: Extrema, Planned,
  PackedAssignment), covering:
  - empty, one-element and NaN inputs
  - offset views that are not aligned
  - lengths that are not a multiple of the SIMD width
- [ ] Run CTest with clang-cl in three fresh build directories: SVML, SLEEF and `psize4`.
- [x] Run the Bazel build, or record why it could not run: on 2026-11-20 Bazel could not
  fetch `https://bcr.bazel.build/` (TLS certificate validation failed, PKIX path building).
- [x] Run clang-tidy on the changed headers (no new findings), then follow the session checklist.
