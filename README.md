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

Dependencies are pinned Git submodules in `ThirdParty`: Memory, Logging, Scalars,
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
backend are fully wired; CUDA/HIP select a `@local_config_{cuda,hip}` label
that this workspace does not define yet (see the script's `--help`):

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
MKL comparisons use the SDK's `MKLConfig.cmake` and link only the comparison
targets. Dependency tests, examples and benchmarks remain disabled, and existing
dependency targets are reused without reconfiguring them.

Tensors allocate through `memory::data_ptr<T>` and expose the underlying
`memory::data_view<T>` with `memory_view()`. Slices, reshapes and expression
operands preserve Memory's allocation identity and base pointer for stream
tracking. Views borrow storage; their owner must outlive them. Copying a tensor
creates independent Memory storage and preserves its shape and strides.
Use `clone()` to request a packed copy.

```cpp
memory::data_ptr<float> storage(12, memory::device_enum::CPU);
vectorization::tensor<float> borrowed(storage.view(), {3, 4});
vectorization::tensor<float> owned(std::move(storage), {3, 4});
```

### Feature and tooling

| CMake variable | Default | Summary |
|----------------|---------|---------|
| `VECTORIZATION_LTO_MODE` / `VECTORIZATION_ENABLE_COVERAGE` | build-type dependent / OFF | LTO mode (`off`, `thin`, `full`, `ipo`, `auto`) / coverage |
| `VECTORIZATION_ENABLE_TESTING` | ON | Test subtree |
| `VECTORIZATION_ENABLE_BENCHMARK` | ON | `BenchmarkSimd*.cpp` → individual `benchmark_simd*` targets |
| `VECTORIZATION_ENABLE_LIBTORCH` | OFF | Probe `find_package(Torch)` (via `CMAKE_PREFIX_PATH`) and build the LibTorch comparison tests/benchmarks (`torch` flag in `setup.py`) |
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
