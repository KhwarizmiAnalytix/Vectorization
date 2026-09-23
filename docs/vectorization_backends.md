# Vectorization Backend Interfaces

Scope: `Library/Vectorization` as of August 2026, after Metal expression fusion
and the removal of unfused per-node lowering. This is the **current** backend
contract: how CPU, CUDA, HIP, and Metal evaluate expressions, where they
already agree, and where the launch APIs still diverge.

CPU SIMD ISA selection (SSE / AVX / NEON / …) is in
[readme/vectorization.md](readme/vectorization.md). Tensor storage is in
[memory_design.md](memory_design.md).

---

## Contents

1. [Two layers](#1-two-layers)
2. [Fusion](#2-fusion)
3. [Launch contracts](#3-launch-contracts)
4. [`simd<T>` vs Metal](#4-simdt-vs-metal)
5. [Unification](#5-unification)
6. [Recommended order](#6-recommended-order)
7. [Key files](#7-key-files)

---

## 1. Two layers

The public API is already one surface: `tensor<T>`, `a + b`, `operator=`.
Behind it there are two compile-time layers, plus a named-kernel Metal API
that expression assignment does not use.

```
tensor / expressions_builder                 ← one public surface
        │
        ▼
expressions_evaluator::run / fill            ← dispatcher
        │
        ├── CPU:      simd<T> + expression_loader in a host SIMD loop
        ├── CUDA/HIP: gpu_eval_kernel<E,T>  (nvcc/hipcc compiles the C++ tree)
        └── Metal:    host emits MSL → dispatch_fused  (separate shader toolchain)

simd<T>                                      ← packet math (CPU ISAs, or GPU size=1)
metal_backend::dispatch("add", …)            ← named kernels; tests only
```

`simd<T>` is the CPU packet ISA (SSE / AVX / AVX2 / AVX-512 / NEON / SVE)
and, on the CUDA/HIP device pass, a scalar (`size == 1`) specialization in
`backend/gpu/{float,double}/simd.h`. Metal expression evaluation does not
go through `simd<T>`.

Configure-time: one CPU SIMD tree and at most one GPU backend (`none` /
`cuda` / `hip` / `metal`) per binary. `tensor::device()` is a runtime check
inside `run()` / `fill()`.

---

## 2. Fusion

Every `tensor` assignment of a pure expression is **one** fused evaluation.
There is no unfused Metal lowering.

| Backend | Mechanism | Per-node temps |
|---|---|---|
| CPU | `expression_loader` in one SIMD loop | No |
| CUDA / HIP | Same templates in one `__global__` kernel; one thread per element | No |
| Metal | Host walks the tree, JIT-compiles one MSL kernel (`fused_float`), cached by source | No |

Unsupported ops, or a Metal tree that exceeds the 31-buffer argument cap,
throw. Named kernels in `backend/gpu/metal/kernels.metal` remain for fill,
multi-block `reduce_{sum,min,max}`, and `metal_backend::dispatch()` tests —
not for expression assignment.

Metal fused coverage is the full expression-template set **except** `cdf` /
`inv_cdf` (MSL has no `erf` / `erfinv`). CUDA/HIP implement those via device
`erfc` / `erfinv`. Metal is float-only; `allocator<double>` already throws for
`device_enum::METAL`.

---

## 3. Launch contracts

`expressions_evaluator::run` / `fill` call three different signatures:

```cpp
// CUDA / HIP
run_gpu(E const& expr, T* data, size_t n, gpu_stream_t stream);
fill_gpu(T* data, T value, size_t n, gpu_stream_t stream);

// Metal
run_metal(E const& expr, T& rhs);     // no stream; needs the tensor, not a raw pointer
fill_metal(T& rhs, float value);      // float hard-coded

// CPU
// inlined SIMD/scalar loop over rhs.begin(); stream ignored
```

How the dispatcher chooses a path:

- CUDA/HIP: only when the TU is compiled with `__CUDACC__` / `__HIPCC__`
  (`TestTensorGpu.cpp` is tagged LANGUAGE CUDA/HIP). Ordinary host C++ never
  instantiates `gpu_eval_kernel`.
- Metal: `VECTORIZATION_HAS_METAL` on ordinary clang++; `if constexpr` float
  plus runtime `rhs.device() == METAL`.
- Else: CPU loop.

`gpu_stream_t` is `cudaStream_t` / `hipStream_t` in device-compiler TUs and
`void*` otherwise. Metal and CPU ignore `stream`. `tensor::assign_async` /
`fill_async` therefore overlap work only on CUDA/HIP.

| Topic | CPU | CUDA / HIP | Metal |
|---|---|---|---|
| Fusion | SIMD loop | One kernel | One JIT kernel |
| Stream | unused | real stream | unused (`void*`) |
| Sync | in-process | launch async; host copies wait | `waitUntilCompleted` every dispatch |
| Types | float, double | float, double | float only |
| Ops | full set | full set | full set minus `cdf` / `inv_cdf` |
| `accumulate` / `hmin` / `hmax` | fused SIMD host loop | fused `gpu_reduce_expr_kernel` + multi-block partials | leaf buffer or fused materialize + multi-block `reduce_{sum,min,max}_float` |
| Errors | `VECTORIZATION_CHECK` | mostly unchecked CUDA/HIP runtime | `throw std::runtime_error` from the evaluator / `.mm` glue |

---

## 4. `simd<T>` vs Metal

CPU backends implement one `simd<T>` surface (`load` / `store` / `add` /
`sin` / …). The GPU scalar `simd<float>` / `simd<double>` is that same
surface with `size = 1` so `expression_loader` and the functors compile as
`__device__` code.

A fake Metal `simd<float>` would not unify anything: MSL is a separate
toolchain and never sees the C++ tree. The seam to unify is **evaluator
launch**, not packet math.

---

## 5. Unification

Do not add runtime-polymorphic `IComputeBackend::launch`. Expression fusion
is a compile-time tree; a vtable on the CPU hot path is the wrong model.
Memory already selects the GPU backend at configure time; Vectorization
should stay the same.

### Target launch contract

Every backend implements the same functions; `run()` only chooses CPU vs GPU:

```cpp
template <typename E, typename T>
void eval_expr(E const& expr, T* out, size_t n, gpu_stream_t stream);

template <typename T>
void eval_fill(T* out, T value, size_t n, gpu_stream_t stream);

template <typename E>
auto eval_reduce(E const& expr, /* sum | min | max */, gpu_stream_t stream);
```

- **CPU** — current SIMD loop; `stream` unused.
- **CUDA/HIP** — current `gpu_eval_kernel<E,T>`; keep nvcc/hipcc compiling the
  C++ tree.
- **Metal** — keep JIT MSL internally; call it as `eval_expr(expr, out, n,
  stream)` so the dispatcher does not special-case `T&`.

`metal_backend::dispatch(name, …)` stays test-only. Fill/reduce named kernels
can remain as Metal’s implementation of `eval_fill` / `eval_reduce`.

### Semantics to make identical

| Topic | Unify to |
|---|---|
| Streams | A real `gpu_stream_t` on every GPU backend, **or** document Metal as a single implicit queue and make `assign_async` a no-op there. Silently ignoring `stream` is the bug. |
| Sync | CUDA-style async launch + wait on host read, **or** Metal-style sync dispatch — pick one for all GPU evals. |
| Types | Metal float-only stays an allocator constraint. No second Metal double evaluator. |
| Ops | Same functor set. Metal `cdf`/`inv_cdf`: MSL approximation **or** `VECTORIZATION_CHECK`, not a unique `std::runtime_error`. |
| Reductions | `accumulate` / `hmin` / `hmax` must branch on `device()` the way `run` does. |
| Errors | `VECTORIZATION_CHECK` for evaluator-level failures; driver/JIT failures only at the CUDA/HIP/ObjC boundary. |

### What not to unify

- CPU `simd<T>` ISAs (SSE vs AVX vs NEON) — already one interface.
- GPU `simd<T>` size=1 — CUDA/HIP device math only.
- Runtime backend objects for expression eval.

The public `tensor` API does not need to change. Unification lives behind
`expressions_evaluator::run` / `fill` / `accumulate`.

---

## 6. Recommended order

1. Normalize signatures: `run_metal` takes `(expr, T* out, n, stream)` like
   `run_gpu`. The dispatcher owns the tensor.
2. Device-dispatch reductions: `accumulate` / `hmin` / `hmax` branch on leaf
   `device()` the way `run` does (CUDA/HIP fused reduce kernels; Metal
   multi-block `reduce_{sum,min,max}_float`).
3. One stream/sync story: Metal command buffers named by `gpu_stream_t`, or a
   documented no-op `assign_async` on Metal.
4. One error wrapper at the evaluator layer.
5. Keep `kernels.metal` as fill + reduce + tests; fusion remains the only
   expression assignment path.

---

## 7. Key files

Paths are under `Library/Vectorization/`.

| Path | Role |
|---|---|
| `expressions/expressions_evaluator.h` | `run` / `fill` dispatcher; CPU SIMD loop; `accumulate` / `hmin` / `hmax` (CPU or GPU) |
| `expressions/expressions_evaluator_gpu.h` | `gpu_eval_kernel` / `run_gpu` / `fill_gpu` / `reduce_gpu` |
| `expressions/expressions_evaluator_metal.h` | MSL emit + `run_metal` / `fill_metal` / `reduce_metal` |
| `expressions/reduce_op.h` | Shared `reduce_op` + identity/combine for CPU-dispatch and GPU backends |
| `expressions/expression_interface_loader.h` | Recursive fused evaluate (CPU + CUDA/HIP device) |
| `backend/simd.h` | Selects CPU ISA `simd<T>` or GPU scalar `simd<T>` |
| `backend/gpu/{float,double}/simd.h` | CUDA/HIP scalar packet |
| `backend/gpu/metal/metal_dispatch.h` | C++ launch surface (`dispatch_fused`, `dispatch_fill`, `reduce_sum` / `reduce_min` / `reduce_max`) |
| `backend/gpu/metal/kernels.metal` | Fill, named test kernels, multi-block reduce sum/min/max |
| `terminals/tensor.h` | `operator=` → `run`; `assign_async` / `fill_async` (stream) |

Tests: `Testing/Cxx/TestTensorGpu.cpp` (including `FusedCatalogFloat`),
`TestMetalDispatch.mm`, `TestTensorGpuProfiler.cpp`.
