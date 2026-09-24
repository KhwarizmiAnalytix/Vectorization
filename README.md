# Vectorization

Standalone C++20 Vectorization library extracted from XSigma. Public target: `Vectorization::Vectorization`.

```sh
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DVECTORIZATION_ENABLE_BENCHMARK=OFF
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Dependencies are pinned Git submodules in `ThirdParty`: Memory, Logging, Profiler, fmt, mimalloc, sleef, googletest, benchmark.
Logging and Profiler use product-only CMake overlays, matching XSigma's integration.
Existing dependency targets are reused when embedded. Tests remain owned by this repository;
set `VECTORIZATION_ENABLE_TESTING=OFF` when consuming only the library.

```cmake
add_subdirectory(ThirdParty/Vectorization)
target_link_libraries(my_app PRIVATE Vectorization::Vectorization)
```

Source headers and implementation are in `include/`; tests are in `Testing/Cxx/`.

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

`Scripts/setup_bazel.py` drives the equivalent Bazel build (`WORKSPACE.bazel`,
`MODULE.bazel`, `BUILD.bazel`, `bazel/`). CPU SIMD backends and the Metal GPU
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
backend (build + run on macOS's own GPU), and build-only CUDA/HIP jobs (no GPU
present on GitHub-hosted runners, so `nvcc`/`hipcc` install and configure+build
only, skipping `ctest`). Sanitizers, coverage, Valgrind, static analysis
(clang-tidy/IWYU/cppcheck), the Bazel build, and toolchain axes (LTO, linker,
C++ standard, compiler cache) each run in their own job rather than a
combinatorial matrix — see the comments in `ci.yml` for the exact coverage
strategy and its limits (SVE has no hosted runner; SVML/MKL need proprietary
SDKs and are validated locally instead).
