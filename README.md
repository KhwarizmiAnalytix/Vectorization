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

The supported standalone build is CMake. XSigma maintains its Bazel overlays separately.
Source headers and implementation are in `Vectorization/`; tests are in `Vectorization/Testing/`.

GPU toolchains and optional numerical backends require their corresponding SDKs.
