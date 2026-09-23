# Vectorization (`Library/Vectorization`)

**SIMD** library: `simd<T>` register wrappers, expression templates, CPU backends
(**SSE**, **AVX**, **AVX2**, **AVX-512**, **NEON**, **SVE**), and one GPU evaluator
(**CUDA**, **HIP**, or **Metal**). Evaluator contracts:
[Docs/vectorization_backends.md](../../Docs/vectorization_backends.md).

## Layout

- `CMakeLists.txt` — `VECTORIZATION_CPU_BACKEND`, SVML, testing/benchmark/tooling.
- `BUILD.bazel` — `//Library/Vectorization:Vectorization`, `VectorizationCxxTests`, `benchmark_simdunary`, `benchmark_simdbinaryfn`, `benchmark_simdbinaryop`, `benchmark_simdternaryhorizontal`.
- `Cmake/` — SIMD capability checks (`utils.cmake`, etc.).
- `Testing/` — generated test headers and C++ tests.

---

## CMake options

### Primary selectors

| CMake variable | Default | Summary |
|----------------|---------|---------|
| `VECTORIZATION_CPU_BACKEND` | host-dependent | `no`, `sse`, `avx`, `avx2`, `avx512`, `neon`, `sve`; AVX2 on recognised x86, NEON on AArch64, otherwise `no` |
| `VECTORIZATION_GPU_BACKEND` | `none` | `none`, `cuda`, `hip`, `metal` — which GPU runtime the evaluator compiles against; independent of `VECTORIZATION_CPU_BACKEND`, both can be active at once |
| `VECTORIZATION_CXX_STANDARD` | `20` | `11`, `14`, `17`, `20`, `23` |

### GPU backend (CUDA / HIP / Metal)

`VECTORIZATION_GPU_BACKEND` selects the GPU evaluator compiled into the library (one per binary). CUDA and HIP share scalar (`size==1`) `simd<T>` plus `expressions_evaluator_gpu.h`. Metal JIT-compiles one MSL kernel per expression (`expressions_evaluator_metal.h`) and does not use `simd<T>`. All three fuse a pure expression into a single kernel or loop. Launch-signature gaps: [Docs/vectorization_backends.md](../../Docs/vectorization_backends.md).

`tensor<T>` dispatches on `device_enum::CUDA` / `HIP` / `METAL` (`Library/Memory/common/device.h`).

| CMake variable | Default | Notes |
|----------------|---------|-------|
| `VECTORIZATION_CUDA_ARCHITECTURES` | `75;80;86;89;90` | CUDA compute-capability list, set when `VECTORIZATION_GPU_BACKEND=cuda` |
| `VECTORIZATION_GPU_ARCHITECTURES` | `gfx900;gfx906;gfx908;gfx90a;gfx1030;gfx1100` | HIP/ROCm `gfxXXX` target list, set when `VECTORIZATION_GPU_BACKEND=hip` |

`setup.py` accepts `cuda` / `hip` / `metal` as chained tokens and forwards `-DVECTORIZATION_GPU_BACKEND=...` and `-DMEMORY_GPU_BACKEND=...` together (the two libraries have independent but must-agree GPU backend selectors). Example:

```
python3 setup.py config.build.test.ninja.clang.avx2.cuda --project.vectorization
python3 setup.py config.build.test.ninja.clang.avx2.metal --project.vectorization
```

GPU unit tests (`Testing/Cxx/TestGpuSimd.cu`, `TestTensorGpu.cpp`, `TestMetalDispatch.mm`) and the `benchmark_tensorgpu` target skip themselves at runtime (`GTEST_SKIP`) when no GPU device is present. CUDA-with-Clang builds currently exclude the CUDA sources at configure time (nvcc-only texture-intrinsics incompatibility — see `Testing/Cxx/CMakeLists.txt`).

### Feature and tooling

| CMake variable | Default | Summary |
|----------------|---------|---------|
| `VECTORIZATION_LTO_MODE` / `VECTORIZATION_ENABLE_COVERAGE` | build-type dependent / OFF | LTO mode (`off`, `thin`, `full`, `ipo`, `auto`) / coverage |
| `VECTORIZATION_ENABLE_TESTING` | ON | Test subtree |
| `VECTORIZATION_ENABLE_EXAMPLES` | OFF | Examples |
| `VECTORIZATION_ENABLE_GTEST` | ON | GoogleTest |
| `VECTORIZATION_ENABLE_BENCHMARK` | ON | `BenchmarkSimd*.cpp` → `benchmark_simdunary`, `benchmark_simdbinaryfn`, `benchmark_simdbinaryop`, `benchmark_simdternaryhorizontal` |
| `VECTORIZATION_ENABLE_LIBTORCH` | OFF | Probe `find_package(Torch)` (via `CMAKE_PREFIX_PATH`) and build the LibTorch comparison tests/benchmarks (`torch` flag in `setup.py`) |
| `USE_NATIVE_ARCH` | OFF | `-march=native` (Clang/GCC) |
| `VECTORIZATION_ENABLE_SVML` | *(probe)* | Set in `Cmake/utils.cmake`: ON if the SVML intrinsic probe **fails** for the active SIMD flags (link ThirdParty `svml`); OFF if the compiler already provides those intrinsics |
| `VECTORIZATION_ENABLE_ICECC` / `VECTORIZATION_ENABLE_CACHE` / `VECTORIZATION_ENABLE_CLANGTIDY` / `VECTORIZATION_ENABLE_FIX` / `VECTORIZATION_ENABLE_IWYU` / `VECTORIZATION_ENABLE_SANITIZER` / `VECTORIZATION_ENABLE_SPELL` / `VECTORIZATION_ENABLE_VALGRIND` | see `CMakeLists.txt` | Tooling |

### `CACHE STRING` (tooling)

| CMake variable | Default | Notes |
|----------------|---------|-------|
| `VECTORIZATION_SANITIZER_TYPE` | address | if sanitizer enabled |
| `VECTORIZATION_LINKER_CHOICE` | default | `default`, `lld`, `mold`, `gold`, `lld-link` |
| `VECTORIZATION_CACHE_BACKEND` | none | `none`, `ccache`, `sccache`, `buildcache` |

Logging is **required** (`Logging::Logging`). Memory is **required**
(`Memory::Memory`). There is no `VECTORIZATION_HAS_LOGGING` or
`VECTORIZATION_HAS_MEMORY` gate.

---

## Bazel flags

Starlark: [`bazel/vectorization.bzl`](../../bazel/vectorization.bzl). SIMD × OS groups: [`bazel/vectorization_settings.bzl`](../../bazel/vectorization_settings.bzl).

### SIMD tier and ISA flags

| Define | Effect |
|--------|--------|
| `vectorization_type` | `no` / `sse` / `avx` / `avx2` / `avx512` / `neon` / `sve` — selects `//bazel:vectorization_type_*`, backend globs, and `VECTORIZATION_HAS_*` / `VECTORIZATION_VECTORIZED` |
| *(unset)* | Defaults to AVX2 behavior (`//conditions:default` in selects + `.bazelrc` sets `vectorization_type=avx2`) |

Compiler ISA flags come from `vectorization_simd_copts()` (MSVC `/arch:*` or GCC/Clang `-m*`).

### Other preprocessor defines

| Define / rule | Effect |
|---------------|--------|
| `vectorization_enable_svml` | **`true`** → force `//bazel:enable_svml` (always `VECTORIZATION_HAS_SVML=1`, `@svml//:SVML`, `svml.h`). **`false`** → `//bazel:disable_svml` (no SVML). **Unset** → autodetect via `//bazel:svml_configure.bzl` (`vectorization_svml_autodetect` repo): host compile probe per SIMD tier (same snippet as `utils.cmake`); Windows assumes third-party SVML is needed |
| `enable_gtest` | `false` → `//bazel:disable_gtest` → `VECTORIZATION_HAS_GTEST=0` on the library |
| `vectorization_enable_testing` | `false` → `//bazel:vectorization_disable_testing` — **skips** `VectorizationCxxTests` (`target_compatible_with`) |
| `vectorization_enable_benchmark` | `false` → `//bazel:vectorization_disable_benchmark` — **skips** `benchmark_simdunary`, `benchmark_simdbinaryfn`, `benchmark_simdbinaryop`, `benchmark_simdternaryhorizontal` |

Fixed in Starlark today: Logging and Memory are always linked.

### Library-only copts

`vectorization_library_copts()` adds MSVC `/WX` and Unix `-include cstdlib` on `vectorization_lib` only (matches CMake `PRIVATE` on the Vectorization target).

### CMake-only

`VECTORIZATION_CXX_STANDARD` is not a Bazel `--define` (`c++20` is fixed in `vectorization.bzl`). LTO, coverage, sanitizers, cache/linker/tooling — **CMake only**. SVML autodetect is mirrored under Bazel via `WORKSPACE` + `svml_configure` (re-run when `CXX`/`CC` changes).
