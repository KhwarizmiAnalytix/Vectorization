# Tensor Contiguity

Scope: `Library/Vectorization/terminals/tensor.h`, `contiguous_` and
`recompute_cpu_simd_alignment_state()`, as of August 2026. What "contiguous"
means for `tensor<T>`, how it's computed and kept up to date, which
operations preserve or break it, and where it's enforced.

Backend dispatch (CPU/CUDA/HIP/Metal) is in
[vectorization_backends.md](vectorization_backends.md). Storage
(`data_ptr`/`data_view`) is in [memory_design.md](memory_design.md).

---

## Contents

1. [What "contiguous" means here](#1-what-contiguous-means-here)
2. [Where it's tracked](#2-where-its-tracked)
3. [The algorithm](#3-the-algorithm)
4. [Per-operation summary](#4-per-operation-summary)
5. [Where it's enforced](#5-where-its-enforced)
6. [Forcing contiguity](#6-forcing-contiguity)
7. [Key files](#7-key-files)

---

## 1. What "contiguous" means here

`tensor<T>::is_contiguous()` answers one specific question: does
`data()[i]` for `i` in `[0, size())`, taken in row-major order, address
exactly the same element as the logical multi-index it unravels to? That is
the **flat-indexing invariant** — not "does PyTorch's `is_contiguous()`
agree" as a goal in itself, though the two turn out to coincide (see §3).
It's the exact property `expressions_evaluator::run()`/`fill()` and the CPU
SIMD loops require before they're allowed to read/write through a single
`value_t*` with a plain `+= 1` stride, instead of walking
`sizes_and_strides_` one dimension at a time.

A tensor that fails this test isn't broken — `at(i, j)`/`operator[]` and
`logical_offset()` (§5) always compute the correct strided offset regardless
of `contiguous_`. It's slower (no flat-pointer fast path, no SIMD alignment
peeling) and, since this session's hardening of `expressions_evaluator`, it
is now a hard, catchable `logging::exception` on assignment into or out of
such a tensor as a *destination*, rather than silently reinterpreting
strided memory as packed.

## 2. Where it's tracked

`contiguous_` (`tensor.h:1565`) is a cached `bool`, not computed on demand.
It — along with `numel_`, `align_start_`, `align_end_` — is recomputed by
`recompute_cpu_simd_alignment_state()` every time `sizes_and_strides_`
changes:

- every constructor (default, shape, initializer-list, expression, the
  private view constructor used by `t()`/`permute()`/`view()`/`reshape()`/
  `slice()`),
- copy construction and copy assignment (recomputed from the copied shape —
  it does **not** just copy the source's `contiguous_` bit, see §4),
- `stamp_contiguous_shape()` (used by `clone()`).

Move construction/assignment is the one exception: `contiguous_` (like
`numel_`, `align_start_`, `align_end_`) is copied directly from the moved-
from tensor rather than recomputed, and the moved-from tensor is reset to
its default empty state (`contiguous_ = true`, `numel_ = 0`) — cheaper than
a redundant recompute, and correct since nothing about the shape changed.

## 3. The algorithm

```cpp
// recompute_cpu_simd_alignment_state(), tensor.h:1509
if (numel_ == 0) {
    contiguous_ = true;                    // no element can be "out of place"
} else {
    contiguous_ = true;
    int64_t expected_stride = 1;
    for (int i = rank() - 1; i >= 0; --i) {
        int64_t size_i = size(i);
        if (size_i == 1) continue;         // a size-1 dim's stride is never read
        if (stride(i) != expected_stride) { contiguous_ = false; break; }
        expected_stride *= size_i;
    }
}
```

This mirrors PyTorch's `TensorImpl::compute_contiguous()` exactly: track a
running *expected* stride from the innermost dimension outward, and skip
size-1 dimensions entirely rather than checking their stride against
anything. A size-1 dimension has exactly one valid index (0), so whatever
value its stride holds is multiplied by 0 in every offset computation —
it cannot affect whether the tensor is flat-contiguous, and must not be
allowed to gate the result.

**Before an August 2026 fix**, the check instead compared every dimension's
stride directly against its *immediate neighbor's* `stride * size`,
including size-1 dimensions:

```cpp
// old, incorrect version
contiguous_ = stride(n - 1) == 1;
for (i = n - 2; i >= 0; --i)
    if (stride(i) != stride(i + 1) * size(i + 1)) { contiguous_ = false; break; }
```

This is strictly *more* conservative than the true criterion — provably
sound (never reports `true` for a genuinely non-contiguous tensor, since a
stricter per-dimension check can only reject more, never accept a case the
relaxed check would reject) but incomplete: it produces false negatives
whenever a size-1 dimension's stride doesn't happen to equal its
neighbor's `stride * size`, which is exactly what `permute()`/`t()` produce
when they relocate a size-1 axis next to a *different* neighbor than the one
its stride was originally canonical for (they copy strides verbatim, they
don't recompute them). Worked example: a naturally-constructed `(4, 1, 5)`
tensor has canonical strides `(5, 5, 1)`. `permute({1, 0, 2})` reorders this
to shape `(1, 4, 5)`, strides `(5, 5, 1)` (same values, just relabeled) —
genuinely still flat-contiguous (`offset(k) == k` for every `k`), but the
old check saw `stride(0)=5 != stride(1)*size(1) = 5*4 = 20` and rejected it.
See `TestTensor.cpp`'s "permute() moving a size-1 axis..." test for the
regression case, and `t()`'s `(N,1)` → `(1,N)` case in §4 for a second,
simpler instance of the same bug shape.

Since the old check could only be *more* conservative than correct, this
was never a memory-safety issue — but the hard `VECTORIZATION_CHECK`s added
to `expressions_evaluator::run()`/`fill()` in the same session made the
symptom sharper: what used to just silently skip an optimization now throws.

## 4. Per-operation summary

| Operation | Result contiguity | Why |
|---|---|---|
| Any fresh-shape constructor | always `true` | `make_contiguous_sas()` builds canonical strides from scratch |
| Copy ctor / copy assignment | mirrors source shape, recomputed | same strides copied in, contiguity independently re-derived |
| Move ctor / move assignment | mirrors source's cached flag (not recomputed); moved-from resets to `true`, empty | shape didn't change, no need to redo the O(rank) walk |
| `clone()` | always `true` | `stamp_contiguous_shape()` rebuilds canonical strides over the packed copy, regardless of the source's contiguity |
| `contiguous()` | always `true` | borrows (`is_contiguous()` already `true`) or falls back to `clone()` |
| `view()` / `reshape()` | always `true` | both require an already-contiguous source and call `make_contiguous_sas()` for the new shape |
| `t()` (rank-2 transpose) | `false` in general; `true` if the tensor is shaped `(N,1)`/`(1,N)` | swaps size/stride pairs verbatim; a vector-shaped tensor's non-unit axis keeps stride 1 either way |
| `permute()` | `false` if it reorders any two size>1 axes; `true` for the identity permutation or one that only relocates size-1 axes | copies size/stride pairs verbatim in the new order |
| `slice(dim, start, stop, step)` | depends — see below | shrinks one dimension's size and multiplies its stride by `step`; leaves all other dimensions untouched |

`slice()` case-by-case (all against a `(4, 5)` source, strides `(5, 1)`):

| Call | Result shape | Contiguous? | Why |
|---|---|---|---|
| `slice(0, 0, 4, 1)` | `(4, 5)` | `true` | no-op, identical to the source |
| `slice(0, 1, 3, 1)` | `(2, 5)` | `true` | a contiguous block of whole rows |
| `slice(0, 1, 2, 1)` | `(1, 5)` | `true` | outer dim becomes size 1 → skipped; the remaining real dim is one packed row |
| `slice(1, 0, 3, 1)` | `(4, 3)` | `false` | each row now only uses 3 of its 5 slots — a real gap between rows |
| `slice(1, 1, 2, 1)` | `(4, 1)` | `false` | inner dim becomes size 1 → skipped, but the *outer* dim's stride (5) no longer equals the expected packed stride (1) — genuine stride-5 gaps between elements |
| `slice(1, 0, 5, 2)` | `(4, 3)` | `false` | `step=2` multiplies the inner stride to 2 ≠ 1 |
| `slice(dim, k, k, 1)` (empty) | size 0 | `true` | `numel_ == 0` short-circuit |

The `(1,5)` vs `(4,1)` pair is the clearest illustration of why size-1
dimensions must be *skipped*, not just *treated leniently*: in both cases a
dimension collapses to size 1, but only one of the two results is actually
packed in memory.

## 5. Where it's enforced

| Site | Check | Debug-only or hard? |
|---|---|---|
| `tensor::begin()`/`end()` (const and non-const) | `is_contiguous()` | `VECTORIZATION_CHECK_DEBUG` — debug only |
| `view()` / `reshape()` (source) | `is_contiguous()` | `VECTORIZATION_CHECK_DEBUG` — debug only |
| `copy_from_host()` (destination) | `is_contiguous()` | `VECTORIZATION_CHECK_DEBUG` — debug only |
| `store_operand()` (expression *sources*, `expression_interface.h`) | `x.is_contiguous()` | hard `VECTORIZATION_CHECK` — always |
| `expressions_evaluator::run()`/`fill()` (expression *destinations*) | `rhs.is_contiguous()` | hard `VECTORIZATION_CHECK` — always, added this session (see [vectorization_backends.md](vectorization_backends.md) for the surrounding device-dispatch guard it sits next to) |
| `logical_offset()` | branches on `contiguous_` (flat passthrough vs. per-dimension divmod) | not a check — always correct either way, just a fast path |

The asymmetry is deliberate: `begin()`/`view()`/`reshape()` are debug-only
because they're primarily internal-invariant assertions on well-formed
call sites, while `store_operand()` and `expressions_evaluator::run()`/
`fill()` sit directly on `data[i]` flat-pointer paths where a silent
contiguity violation means either dereferencing memory that isn't there or
writing through a stride pattern the destination doesn't actually have —
so those stay hard checks in release builds too.

## 6. Forcing contiguity

`t.contiguous()` is the one call that's always safe regardless of `t`'s
current state: identity (borrow, same pointer) when already contiguous,
`clone()` (new packed buffer) otherwise. Prefer it over manually branching
on `is_contiguous()` before an expression assignment or a `begin()`-based
loop.

## 7. Key files

- `Library/Vectorization/terminals/tensor.h` — `contiguous_`,
  `recompute_cpu_simd_alignment_state()`, `stamp_contiguous_shape()`,
  `logical_offset()`, `t()`/`permute()`/`view()`/`reshape()`/`slice()`/
  `clone()`/`contiguous()`.
- `Library/Vectorization/expressions/expression_interface.h` —
  `store_operand()`'s source-side hard check.
- `Library/Vectorization/expressions/expressions_evaluator.h` —
  `run()`/`fill()`'s destination-side hard check and `check_cpu_reachable()`.
- `Library/Vectorization/Testing/Cxx/TestTensor.cpp` — contiguity test
  coverage (construction, copy/move, `t()`/`permute()`/`view()`/`reshape()`/
  `slice()`/`clone()`/`contiguous()`, the destination hard-check regression).
