# Vectorization

[![CI](https://github.com/KhwarizmiAnalytix/Vectorization/actions/workflows/ci.yml/badge.svg)](https://github.com/KhwarizmiAnalytix/Vectorization/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/KhwarizmiAnalytix/Vectorization/branch/main/graph/badge.svg)](https://codecov.io/gh/KhwarizmiAnalytix/Vectorization)
[![License: GPL v3 / Commercial](https://img.shields.io/badge/license-GPL--3.0--or--later%20%2F%20commercial-blue.svg)](LICENSE)

Standalone C++20 Vectorization library extracted from XSigma. Public target: `Vectorization::Vectorization`.

```sh
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVECTORIZATION_ENABLE_BENCHMARK=OFF
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Dependencies are pinned Git submodules in `ThirdParty`: Memory, Logging, Scalars, LinearAlgebra,
googletest, and benchmark. Logging owns fmt, Memory owns mimalloc, and Scalars
owns SVML, SLEEF and Accelerate; initialize nested submodules with `--recursive`.
Profiler is optional and resolved through `find_package(Profiler)`.
Existing dependency targets are reused when embedded. Tests remain owned by this repository;
set `VECTORIZATION_ENABLE_TESTING=OFF` when consuming only the library.

```cmake
add_subdirectory(ThirdParty/Vectorization)
target_link_libraries(my_app PRIVATE Vectorization::Vectorization)
```

Public headers are in `include/`, implementation files are in `src/`, and tests are in
`Testing/Cxx/`. Build configuration and project guidance live at the repository root.

GPU toolchains and optional numerical backends require their corresponding SDKs.

## Build scripts

`Scripts/setup.py` wraps the CMake configure/build/test/coverage pipeline behind
dotted tokens (mirrors `KhwarizmiAnalytix/Logging`'s own `Scripts/setup.py`).
Run it from inside `Scripts/`:

```sh
cd Scripts
python setup.py config.build.test              # default CPU backend, GPU backend none
python setup.py config.build.test.avx2          # select a CPU SIMD backend
python setup.py config.build.test.avx2.cuda     # CPU + GPU backend combined
python setup.py config.build.test.metal         # macOS Metal GPU backend
python setup.py --help                          # full option list
```

`Scripts/setup_bazel.py` drives the equivalent Bazel build (`MODULE.bazel`,
`BUILD.bazel`, `bazel/`). CPU SIMD backends and the Metal GPU
backend are wired alongside CUDA/HIP. CUDA requires an installed toolkit
(`CUDA_PATH`); HIP requires a ROCm SDK (`ROCM_PATH`) and ROCm Clang selected
with Bazel's `--repo_env=CC` and `--repo_env=CXX`. See `.bazelrc` for commands
and architecture overrides. Tensor expression tests use the device compiler:

```sh
python Scripts/setup_bazel.py build.test.avx2
python Scripts/setup_bazel.py build.test.metal
python Scripts/setup_bazel.py --help
```

## CI

`.github/workflows/ci.yml` builds every CPU SIMD backend (`no`/`sse`/`avx`/`avx2`/
`avx512` on Linux+Windows x86_64, `no`/`neon` on macOS AArch64), the Metal GPU
backend (build + run on macOS's own GPU), and CUDA/HIP jobs. CUDA uses Scalars'
`nvidia/cuda:12.6.2-devel-ubuntu22.04` container to build tests and benchmarks.
CUDA execution requires the NVIDIA driver library (`libcuda.so.1`); CUDA and HIP
are build-only on GitHub-hosted runners. Sanitizers, coverage, Valgrind, static analysis
(clang-tidy/IWYU/cppcheck), the Bazel build, and toolchain axes (LTO, linker,
C++ standard, compiler cache) each run in their own job rather than a
combinatorial matrix — see the comments in `ci.yml` for the exact coverage
strategy and its limits (SVE has no hosted runner; SVML/MKL need proprietary
SDKs and are validated locally instead).

## Static analysis

Enable clang-tidy with `python3 Scripts/setup.py config.build.clangtidy`.
The library, C++ tests, Metal Objective-C++ tests, and enabled benchmarks run
with warnings treated as errors. `ThirdParty/` diagnostics are excluded.
CI pins clang-tidy to LLVM 19 on Linux AVX2 and macOS NEON + Metal, with
`--testing=ON`. Local tools may report additional checks in newer LLVM
versions; select the intended binary with `CLANG_TIDY_PATH` in CMake.
CUDA/HIP device translation units are built in their GPU jobs; they are not
analyzed by the clang-tidy jobs. Optional LibTorch/MKL paths require their SDKs.

## Matrix product expressions

```cpp
#include "terminals/tensor.h"

vectorization::tensor<float> a{{1, 2}, {3, 4}};
vectorization::tensor<float> b{{1, 0}, {0, 1}};
vectorization::tensor<float> bias{1, 2};
vectorization::tensor<float> out(2, 2);
out = ::exp(vectorization::linalg::matmul(a, b) + bias);
out.noalias() += vectorization::matmul(a, b);
vectorization::matmul_out(out, a, b, 2.0f, 1.0f); // out = 2 * (a @ b) + out
```

`matmul` and `bmm` build lazy expressions; each product occurrence is evaluated
once before the surrounding elementwise pass. `exp`, `sqr`, `sqrt`, `log`,
arithmetic, broadcasting and whole-expression reductions accept products.
Saved expressions retain their input storage, including temporary owners and views.
`*` remains elementwise.

The initial adapter supports CPU float/double. `matmul` promotes vectors and
broadcasts leading batch dimensions; `bmm` requires two rank-3 inputs with equal
batch counts. `dot` requires two vectors and returns a host scalar. Owners,
mutable views and const views are accepted with matching element types.

Assignment preserves the destination's shape and storage. Product/input overlap
uses a temporary; `matmul_out` rejects overlap. `noalias()` asserts disjoint
inputs (checked in Debug) and always validates destination injectivity. Compatible
row-major and transposed layouts pass their leading dimensions to LinearAlgebra;
other layouts are packed or copied through an intermediate.

The selected CPU LinearAlgebra backend performs GEMM. Vectors and batches use
promoted or repeated GEMM until the planned upstream GEMV/DOT and batched APIs
arrive. Non-default alpha/beta and compound product expressions currently use
intermediates; epilogue optimization and allocation-count gates remain open.
GPU products fail explicitly. Remaining capabilities are tracked in the
[LinearAlgebra integration plan](docs/vectorization_design_plan.md#11-linearalgebra-adapter).

## Square matrix linear algebra

CPU float/double tensors and views support square LU and Cholesky, solves,
inverses and determinants through `vectorization::linalg` (also available in
`vectorization`):

```cpp
auto x = vectorization::linalg::solve(a, b);
vectorization::linalg::lu_factorization<float> factors(a);
auto another_x = factors.solve(b); // reuse LU; matrix RHS
auto inverse = vectorization::linalg::inv(a);
auto [sign, logabsdet] = vectorization::linalg::slogdet(a);
auto status = vectorization::linalg::solve_ex(a, b); // status.result, status.info
```

`solve` accepts vector or matrix RHS inputs, supports `left=false` for matrix
RHS inputs, and broadcasts matrix batches. `lu_solve` supports `left` and
`adjoint`; `solve_triangular` supports `upper`, `left` and `unitriangular`.
`cholesky`, `cholesky_solve` and reusable Cholesky factors support either
triangle. Factor-reuse and triangular solves require matrix RHS inputs.
Inputs may be strided; outputs are packed independent tensors. Empty matrices,
empty RHS columns and zero-size batches are supported.

`lu_factor_ex`, `cholesky_ex`, `solve_ex` and `inv_ex` return per-matrix status
and default to `check_errors=false`. Their checked counterparts throw on
numerical failure; `lu`, `det` and `slogdet` accept singular inputs. `solve_ex`
status has the coefficient matrix's batch shape before RHS broadcasting.
Pivots are 1-based. Pivots and status currently use the input's float/double
dtype; integer metadata tensors and rectangular LU remain planned. GPU
adapters are unavailable, and repeated solves still copy factors/RHS buffers.

## Whole-expression reductions

```cpp
#include "terminals/tensor.h"

vectorization::tensor<float> x(1025);
x = 1.0f;
auto average = vectorization::mean(x * 2.0f + 1.0f); // 3.0f
std::uint64_t nonzeros = vectorization::count_nonzero(x); // 1025
bool positive = vectorization::all(x > 0.0f); // true
```

Alongside `accumulate`, `hmin`, and `hmax`, the expression API provides:

| Operation | Return type | Empty input |
|---|---|---|
| `sum` | input scalar type | 0 |
| `prod` | input scalar type | 1 |
| `mean` | input scalar type | NaN |
| `all` | bool | true |
| `any` | bool | false |
| `count_nonzero` | `std::uint64_t` | 0 |
| `nansum` | input scalar type | 0 |
| `nanmean` | input scalar type | NaN |

These reduce every logical element to a host scalar; axis reductions are not
provided. Float and double inputs retain their scalar type for numeric results.
`all`, `any`, and `count_nonzero` treat NaNs and infinities as nonzero; both signs
of zero are false. `nansum` ignores input NaNs, while `nanmean` divides by the
number of non-NaN inputs and returns NaN if that count is zero. Infinities still
participate in arithmetic. The existing empty extrema identities (`hmin` gives
`max()`, `hmax` gives `-max()`) remain unchanged.

The design follows PyTorch's separation of input reduction and partial-result
combination ([shared operations](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/SharedReduceOps.h),
[GPU reduction](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/Reduce.cuh)).
CPU loops fuse expressions into independent SIMD accumulators, with a scalar
fallback for strided and broadcast inputs. Counts merge into integers before
float lanes can lose precision. CUDA/HIP fuse expression evaluation into a
bounded grid of blocks with register accumulation and device partial passes;
only the final scalar is copied to the host. Metal's new product, logical,
count, and NaN-sum kernels fuse expressions directly into SIMD-group partials.
Metal remains float-only and requires flat operands. `nanmean` uses two
reductions (sum and valid count). Floating-point results may differ slightly
with backend and reduction order.

`BenchmarkTensorReductions.cpp` and `BenchmarkTensorGpu.cpp` cover the new
operations from 257 to more than 16 million elements. Optional LibTorch tests
compare their results directly with PyTorch.

## Performance against baselines

`benchmark_tensorbaselines` (`Testing/Cxx/BenchmarkTensorBaselines.cpp`) runs
each operation on the same inputs through four implementations:

| Row suffix | Implementation |
|---|---|
| `Vectorization` | `tensor<T>` expression templates |
| `Loop` | Plain scalar loop, auto-vectorized by the compiler where it can |
| `Eigen` | `Eigen::Array` expression (the copy bundled with LinearAlgebra) |
| `LibTorch` | `torch::*_out` kernel on one thread (only with `VECTORIZATION_ENABLE_LIBTORCH=ON`) |

Rows are named `<Op>/<f32|f64>/<n>/<implementation>` and run at 1,027, 32,771
and 1,048,579 elements (L1-, L2- and DRAM-sized). Strided rows,
`<StridedOp>/<f32|f64>/512x509/<implementation>`, reduce or combine a 512 × 509
window of a 512 × 520 row-major array: a `slice()` view, an Eigen `block()`, or a
double loop over the parent buffer. Outputs are preallocated. Before timing,
each result is checked against a double-precision reference (min and max
exactly), and a mismatch reports an error instead of a time.

```sh
benchmark_tensorbaselines --benchmark_filter='Exp/f32/'   # one case, all implementations
benchmark_tensorbaselines --benchmark_filter='^Strided'   # the strided views
benchmark_tensorbaselines --benchmark_min_time=0.5s       # full run
```

Wall time in µs at n = 32,771 (strided rows: the whole 512 × 509 window).
Measured with clang-cl 22, AVX2 and SVML (the default x86 math provider when
SLEEF is off) in a Release build, one thread pinned to one performance core of
an 8P + 16E-core x86-64 desktop CPU (48 KiB L1D and 2 MiB L2 per core), median
of 5 repetitions. Treat differences under about 10% as noise. LibTorch rows
were not part of this run.

| Op | f32 Vec. | f32 Loop | f32 Eigen | f64 Vec. | f64 Loop | f64 Eigen |
|---|---:|---:|---:|---:|---:|---:|
| Add `a + b` | 2.02 | 1.97 | 1.96 | 4.01 | 3.91 | 3.92 |
| Mul `a * b` | 2.02 | 1.97 | 1.96 | 4.02 | 3.91 | 3.92 |
| Saxpy `2.5a + b` | 2.02 | 1.96 | 1.98 | 4.03 | 3.92 | 3.96 |
| Fma `a * b + c` | 2.46 | 2.31 | 2.40 | 4.89 | 4.63 | 4.86 |
| Composite `sqrt(a²+b²) + c` | 4.49 | 4.34 | 4.34 | 17.5 | 17.4 | 17.4 |
| Exp | 9.6–16.4 | 50.2 | 8.57 | 20.4–32.9 | 76.7 | 19.7 |
| Log | 9.6–16.5 | 68.7 | 5.56 | 28.4–37.4 | 81.1 | 24.2 |
| Sqrt | 4.33 | 17.3 | 4.33 | 17.4 | 34.7 | 17.4 |
| Sin | 17.7 | 195 | 7.65 | 20.3 | 227 | 40.9 |
| Cos | 17.9 | 203 | 7.63 | 21.8 | 225 | 40.9 |
| Tanh | 15.1 | 202 | 5.36 | 53.0 | 252 | 18.9 |
| Sum | 0.562 | 11.6 | 0.556 | 1.11 | 11.5 | 1.11 |
| Dot `sum(a * b)` | 1.16 | 11.6 | 1.11 | 2.29 | 11.6 | 2.23 |
| Max `hmax(a)` | 0.728 | 23.1 | 0.722 | 1.45 | 23.1 | 1.45 |
| Min `hmin(a)` | 0.727 | 23.1 | 0.722 | 1.45 | 23.1 | 1.45 |
| Strided sum | 5.85 | 92.0 | 5.34 | 12.3–15.4 | 92.3 | 10.7–13.0 |
| Strided max | 5.99 | 184 | 7.58 | 13.2–15.5 | 184 | 13.1–13.4 |
| Strided dot `sum(va * vb)` | 12.8–15.9 | 184 | 11.5–12.8 | 46.8–59.8 | 185 | 48.7–61.9 |
| Strided fma `va * vb + va` | 41.8–52.8 | 40.0–50.1 | 41.1–50.5 | 84.5–101 | 81.4–103 | 84.6–102 |

Ranges are the spread over several full runs on the same build where it
exceeded the noise level; see the notes on transcendentals and strided views
below.

What the numbers show:

- **Element-wise arithmetic, fused expressions and reductions:** Vectorization
  is within about 5% of the faster of the auto-vectorized loop and Eigen at
  n = 32,771 (7% for `a * b + c` in float) and within 5% at n = 1,048,579.
  These cases are bound by load/store or cache bandwidth, which all three
  implementations saturate. Reductions are 10–30× faster than the scalar loop,
  which cannot reorder floating-point additions.
- **Min and max:** `hmin`, `hmax` and `accumulate` run on the same four
  independent SIMD accumulators as `sum` and are level with Eigen's
  `minCoeff()` / `maxCoeff()`. On x86 they skip NaN elements, as `std::fmin` /
  `std::fmax` do.
- **Strided views:** reductions over strided views and broadcast expressions
  walk the iteration planner's rows: each contiguous row runs the packed SIMD
  body, rows with a broadcast operand use splatted loads, and accumulators are
  carried from row to row, so no element index is divided. Strided max is about
  20% faster than Eigen in float; strided sum and dot are within about 10–25%
  of Eigen in float. In double the 2.1 MiB parent buffer is just over the 2 MiB
  L2, and all three implementations vary by up to 30% between runs.
- **Multiply-add:** `a * b + c`, `c + a * b` and `c += a * b` build
  `fma(a, b, c)` when the target has a hardware FMA, as the compiler does for
  the loop (see `VECTORIZATION_FUSE_MUL_ADD` in
  `include/common/vectorization_macros.h`). The fused result is rounded once,
  so it can differ from separate multiply-then-add in the last bit. Define
  `VECTORIZATION_FUSE_MUL_ADD=0` to keep separate rounding.
- **Small arrays:** a two-operand assignment has about 22 ns of fixed cost
  (Eigen: about 2 ns). At 1,027 elements, Add takes 0.042 µs against 0.021 µs
  for the loop. When every operand is packed with the destination's shape and
  is the destination itself or disjoint from it, assignment skips shape
  inference, re-lowering and the general overlap check. About three quarters
  of what remains is the atomic reference counting of the operands: every
  tensor leaf retains its storage so a saved expression stays valid after its
  operands are destroyed (design plan, section 5, capture).
- **Transcendentals:** Vectorization is level with Eigen for sqrt in both
  types and about 2× faster for sin and cos in double. Eigen is 2.3–3× faster
  for log, sin, cos and tanh in float and for tanh in double. Vectorization
  spends 92–97% of that time inside SVML (one library call per SIMD register);
  Eigen inlines polynomial approximations. Exp and log also vary by up to 1.8×
  with where the input and output arrays fall relative to each other within a
  4 KiB page (the ranges in the table), an effect of the per-register SVML
  call that inlined code does not show. On this benchmark's input ranges
  Eigen's float approximations measured as accurate as SVML or better (max
  error vs. a double reference: tanh 3.8 vs 3.2 ULP, sin 1.4 vs 2.3, log 1.3
  vs 3.9, exp 0.6 vs 2.4). The reference check accepts relative errors up to
  2e-5 (float), so it does not separate faster approximations from more
  accurate ones.
- **SLEEF:** with `VECTORIZATION_ENABLE_SLEEF=ON` (`sleef` token), float log
  (24.5 µs) and tanh (47.8 µs) are 2.3× and 3.2× slower than with SVML, float
  exp, sin and cos are within about 15%, and every double function is 1.5–2.2×
  slower. On x86, SVML is the faster provider.

Results depend on the CPU backend, the math provider (SLEEF, SVML or the
built-in fallback), the compiler and `VECTORIZATION_PACKET_SIZE`. Rerun the
benchmark after changing any of them.

## CMake options

### Primary selectors

| CMake variable | Default | Summary |
|----------------|---------|---------|
| `VECTORIZATION_CPU_BACKEND` | host-dependent | `no`, `sse`, `avx`, `avx2`, `avx512`, `neon`, `sve`; AVX2 on recognised x86, NEON on AArch64, otherwise `no` |
| `VECTORIZATION_GPU_BACKEND` | `none` | `none`, `cuda`, `hip`, `metal` — which GPU runtime the evaluator compiles against; independent of `VECTORIZATION_CPU_BACKEND`, both can be active at once |
| `VECTORIZATION_CXX_STANDARD` | `20` | `11`, `14`, `17`, `20`, `23` |

### GPU backend (CUDA / HIP / Metal)

`VECTORIZATION_GPU_BACKEND` selects the GPU evaluator compiled into the library (one per binary). CUDA and HIP share scalar (`size==1`) `simd<T>` plus `expressions_evaluator_gpu.h`. Metal JIT-compiles one MSL kernel per expression (`expressions_evaluator_metal.h`) and does not use `simd<T>`. All three fuse a pure expression into a single kernel or loop. Launch-signature gaps: [backend contracts](docs/vectorization_backends.md).

`tensor<T>` dispatches on `device_enum::CUDA` / `HIP` / `METAL` (`ThirdParty/Memory/include/common/device.h`).

| CMake variable | Default | Notes |
|----------------|---------|-------|
| `VECTORIZATION_CUDA_ARCHITECTURES` | `75;80;86;89;90` | CUDA compute-capability list, set when `VECTORIZATION_GPU_BACKEND=cuda` |
| `VECTORIZATION_GPU_ARCHITECTURES` | `gfx900;gfx906;gfx908;gfx90a;gfx1030;gfx1100` | HIP/ROCm `gfxXXX` target list, set when `VECTORIZATION_GPU_BACKEND=hip` |

`VECTORIZATION_GPU_BACKEND` also configures Memory's GPU allocator when building the dependency. An existing `Memory::Memory` target must support the selected backend. `setup.py` accepts `cuda` / `hip` / `metal` as chained tokens. Example:

```
python3 Scripts/setup.py config.build.test.ninja.clang.avx2.cuda
python3 Scripts/setup.py config.build.test.ninja.clang.avx2.metal
```

GPU tests live in `Testing/Cxx/`; CUDA/HIP tests need the matching device to run.
See `Testing/Cxx/CMakeLists.txt` for compiler-specific GPU configuration.

### Math providers and tensor storage

Use the public `VECTORIZATION_ENABLE_*` flags in CMake and `setup.py`.
CMake forwards each feature only to the dependency that implements it:

| Public flags | Dependency options |
|--------------|--------------------|
| `VECTORIZATION_ENABLE_SVML`, `_SLEEF`, `_ACCELERATE` | `SCALARS_ENABLE_SVML`, `_SLEEF`, `_ACCELERATE` |
| `VECTORIZATION_ENABLE_MIMALLOC`, `_MIMALLOC_STATS`, `_NUMA`, `_MEMKIND`, `_TBB` | Corresponding `MEMORY_ENABLE_*` options |
| `VECTORIZATION_ENABLE_MAGICENUM`, `_PORTABLE_FLOAT_FORMAT` | Corresponding `LOGGING_ENABLE_*` options |
| Shared cache, sanitizer, coverage and analysis flags | Corresponding options on dependencies built in this tree |
| `VECTORIZATION_ENABLE_MKL` | MKL reference tests and benchmarks only; leaves Scalars' MKL backend disabled |

The `svml`, `sleef`, `mkl` and `accelerate` setup tokens emit public
`VECTORIZATION_ENABLE_*` flags. Scalars owns SIMD math discovery and linkage;
SVML requires its runtime under `ThirdParty/Scalars/ThirdParty/svml`.
On x86, when the compiler has no built-in SVML intrinsics and SLEEF is off,
Scalars enables SVML automatically if that runtime is present, so
transcendentals stay vectorized; the summary then shows `ON (auto)`.
`VECTORIZATION_ENABLE_SVML=ON` makes the runtime mandatory.
MKL comparisons use the SDK's `MKLConfig.cmake` and link only the comparison
targets. Dependency tests, examples and benchmarks remain disabled, and existing
dependency targets are reused without reconfiguring them.

`tensor<T>` owns packed C-order storage. Copying it, or explicitly constructing
it from a view, copies logical values into independent storage. `tensor_view<T>`
shallow-copies its handle; assignment writes through without changing its shape.
Use `rebind()` to point a view at another tensor or view. Const tensors produce
`tensor_view<const T>` inputs, which cannot be assigned to.

Slices, transposes, permutations, reshapes and `contiguous()` return views.
Views retain their allocation through `memory::retained_ptr<T>`, including when
`reshape()` or `contiguous()` needs to materialize a packed copy. Saved
expressions also retain their tensor operands. `clone()` and `to_cpu()` return
independent owning tensors. Kernel descriptors preserve Memory's allocation
identity and base pointer for stream tracking.

```cpp
vectorization::tensor<float> a{{1, 2}, {3, 4}};
auto transposed = a.t();                    // retaining, writable view
vectorization::tensor<float> copy(transposed); // independent packed copy
transposed = copy;                          // writes into a
copy.noalias() = a + 1.0f;                  // caller asserts no source overlap
copy.noalias() += a;
```

Ordinary expression/view assignment permits identical elementwise mappings and
rejects partial overlap or layouts whose overlap cannot be proven. Use
`src.clone()` for an independent source. `noalias()` offers `=`, `+=` and `-=`;
Release skips source overlap classification, while Debug still rejects proven
partial overlap. Shape, device and destination injectivity checks always run.
Expanded or otherwise internally overlapping destinations cannot be written.

External buffers can be deliberately borrowed with `tensor_view<T>(ptr, dims)`
or `tensor_view<T>(storage.view(), dims)`; the caller must keep that external
memory alive. Pass a `memory::retained_ptr<T>` to retain adopted storage, or
transfer a `memory::data_ptr<T>` into an owning tensor:

```cpp
memory::data_ptr<float> storage(12, memory::device_enum::CPU);
vectorization::tensor_view<float> borrowed(storage.view(), {3, 4});
vectorization::tensor<float> owned(std::move(storage), {3, 4});
```

### Feature and tooling

| CMake variable | Default | Summary |
|----------------|---------|---------|
| `VECTORIZATION_LTO_MODE` / `VECTORIZATION_ENABLE_COVERAGE` | build-type dependent / OFF | LTO mode (`off`, `thin`, `full`, `ipo`, `auto`) / coverage |
| `VECTORIZATION_ENABLE_TESTING` | ON | Test subtree |
| `VECTORIZATION_ENABLE_BENCHMARK` | ON | `BenchmarkSimd*.cpp` → individual `benchmark_simd*` targets |
| `VECTORIZATION_ENABLE_LIBTORCH` | OFF | Probe `find_package(Torch)` (via `CMAKE_PREFIX_PATH`) and build the LibTorch comparison tests/benchmarks, including the LibTorch rows of `benchmark_tensorbaselines` (`torch` flag in `setup.py`) |
| `USE_NATIVE_ARCH` | OFF | `-march=native` (Clang/GCC) |
| `VECTORIZATION_ENABLE_ICECC` / `VECTORIZATION_ENABLE_CACHE` / `VECTORIZATION_ENABLE_CLANGTIDY` / `VECTORIZATION_ENABLE_FIX` / `VECTORIZATION_ENABLE_IWYU` / `VECTORIZATION_ENABLE_SANITIZER` / `VECTORIZATION_ENABLE_SPELL` / `VECTORIZATION_ENABLE_VALGRIND` | see `CMakeLists.txt` | Tooling |

### `CACHE STRING` (tooling)

| CMake variable | Default | Notes |
|----------------|---------|-------|
| `VECTORIZATION_SANITIZER_TYPE` | address | if sanitizer enabled |
| `VECTORIZATION_LINKER_CHOICE` | default | `default`, `lld`, `mold`, `gold`, `lld-link` |
| `VECTORIZATION_CACHE_BACKEND` | none | `none`, `ccache`, `sccache`, `buildcache` |

Logging is **required** (`Logging::Logging`). Memory is **required**
(`Memory::Memory`). Scalars is **required** (`Scalars::Scalars`). There is no `VECTORIZATION_HAS_LOGGING` or
`VECTORIZATION_HAS_MEMORY` gate.

### Reproducing CI with the setup helper

CI and local builds use the documented dotted actions and named options from
`python Scripts/setup.py --help`. Use `--parallel N` to set build jobs and
`--enable-parallel on` to enable the Parallel backend:

```sh
python Scripts/setup.py config.build.test.release.ninja.clang --build-dir build
python Scripts/setup.py install.release --build-dir build --install-prefix install
python Scripts/setup.py config.build.test --build-dir build-debug --compiler clang-cl --build-type Debug --testing on --benchmark off
```

Only declared options are accepted. Tool commands, raw build-system arguments,
and a `--` argument separator are rejected. Paths, backend selections, GPU
architectures, test filters, and installation settings have named options in
the usage text. `--source-dir` selects a dependency or installed-package consumer
project. `verifycache --build-dir build --target TARGET` checks the launcher and
clean-rebuild cache hits. `verifytoolchain` checks the compiler selected by Ninja.
Run Windows Ninja builds in a Visual Studio developer environment.

Bazel CI uses the same action style and the options listed by
`python Scripts/setup_bazel.py --help`, for example:

```sh
python Scripts/setup_bazel.py build.test --target //Testing/... --compiler default --test-output errors
```
