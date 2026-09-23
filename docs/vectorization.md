# CPU SIMD (Vectorization)

XSigma’s Vectorization library compiles **one** CPU SIMD ISA per binary
(`VECTORIZATION_CPU_BACKEND`). The packet type `simd<T>` and the expression
evaluator then use that ISA. Picking the wrong ISA for the machine you run
on yields an illegal-instruction crash, not a runtime fallback.

GPU expression evaluation (CUDA, HIP, Metal) is a separate compile-time
backend on the same `tensor` API. Contracts, fusion, and launch-interface
gaps: [vectorization_backends.md](../vectorization_backends.md).

`Scripts/setup.py` is the recommended whole-project entry point. Direct CMake
is supported; the cache names below are what the helper forwards.

## Table of Contents

- [Supported ISAs](#supported-isas)
- [Building](#building)
- [Packet size](#packet-size)
- [Choosing an ISA](#choosing-an-isa)
- [Platform notes](#platform-notes)
- [Troubleshooting](#troubleshooting)
- [Related](#related)

## Supported ISAs

| Token / `VECTORIZATION_CPU_BACKEND` | Typical flags (GCC/Clang) | MSVC | Hardware |
|------|---------------------------|------|----------|
| `no` | none | none | any |
| `sse` | `-msse -msse2` | `/arch:SSE2` | x86-64 (Pentium 4+) |
| `avx` | `-mavx` | `/arch:AVX` | Sandy Bridge+ (2011+) |
| `avx2` | `-mavx -mavx2` | `/arch:AVX2` | Haswell+ (2013+); default on recognised x86 hosts |
| `avx512` | `-mavx -mavx2 -mavx512f` | `/arch:AVX512` | Skylake-X+ (2017+) |
| `neon` | `-march=armv8-a` | — | Apple Silicon, Linux AArch64 |
| `sve` | `-march=armv8-a+sve -msve-vector-bits=128` | — | SVE-capable AArch64 |

The `sve` preset uses a **fixed** 128-bit vector length so lane counts match
the NEON kernels. Wider `-msve-vector-bits` values are not supported yet.

CMake: `-DVECTORIZATION_CPU_BACKEND=<token>`. Bazel:
`--define=vectorization_type=<token>` (also `neon` / `sve`). See
[Library/Vectorization/README.md](../../Library/Vectorization/README.md).

## Building

From `Scripts/`:

```bash
# AVX2 (default on recognised x86 hosts)
python3 setup.py config.build.test.ninja.clang.avx2 --project.vectorization

# Older x86
python3 setup.py config.build.test.ninja.clang.sse --project.vectorization

# Apple Silicon / AArch64
python3 setup.py config.build.test.ninja.clang.neon --project.vectorization

# AVX-512
python3 setup.py config.build.test.ninja.clang.avx512 --project.vectorization

# Scalar (no SIMD backend)
python3 setup.py config.build.test.ninja.clang.no --project.vectorization
```

GPU backends are independent of the CPU ISA (one GPU backend per binary):

```bash
python3 setup.py config.build.test.ninja.clang.avx2.cuda --project.vectorization
python3 setup.py config.build.test.ninja.clang.avx2.metal --project.vectorization
```

`setup.py` forwards CPU and GPU selectors to both Vectorization and Memory
so the two libraries agree.

## Packet size

`--packet-size=N` or the token `psizeN` sets `VECTORIZATION_PACKET_SIZE`
(default 4). Changing it requires a rebuild, not reuse of an existing
`build_*` directory.

```bash
python3 setup.py config.build.test.ninja.clang.avx2.psize8 --project.vectorization
```

## Choosing an ISA

| Level | Use when |
|-------|----------|
| `no` | Debugging codegen, or maximum portability |
| `sse` | Oldest x86 machines you still support |
| `avx2` | Default for Intel/AMD hosts from ~2013 |
| `avx512` | Machines that actually have AVX-512 (not all “AVX2-class” laptops) |
| `neon` | Apple Silicon and other AArch64 |
| `sve` | Hosts with SVE; still 128-bit lanes |

AVX-512 binaries will SIGILL on CPUs that only have AVX2. NEON/SVE tokens
on x86, and SSE/AVX tokens on Apple Silicon, are the wrong tree for that
host.

## Platform notes

- **Windows (MSVC):** SSE2 is implied on x64. `/arch:AVX512` needs VS 2017 15.7+.
- **Linux:** `native` in `setup.py` adds `-march=native` for the build machine.
- **macOS:** Apple Silicon uses `neon` (not `avx2`). Intel Macs from 2013+ can use `avx2`.

## Troubleshooting

**Compile errors for the selected ISA.** Drop one tier (`avx512` → `avx2` →
`avx` → `sse`) or use `no`. Confirm the compiler actually supports the ISA.

**Illegal instruction at runtime.** The binary was built for an ISA the CPU
does not have. Rebuild with a lower token (often `avx2` → `sse` on old x86,
or `avx2` → `neon` on Apple Silicon).

**No speedup.** ISA selection only affects Vectorization’s `simd<T>` path.
Confirm the build actually passed the SIMD flags, that the work is in
expression/`simd` kernels, and that data is aligned when the kernels require
it.

## Related

- [Vectorization backends](../vectorization_backends.md) — CPU / CUDA / HIP / Metal evaluator contracts
- [Library/Vectorization/README.md](../../Library/Vectorization/README.md) — CMake and Bazel options
- [Setup Guide](setup.md) — `setup.py` token reference
- [Build Configuration](build/build-configuration.md)
- [Cross-Platform Building](cross-platform-building.md)
