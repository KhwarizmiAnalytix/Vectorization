# Vectorization

SIMD abstraction layer over multiple CPU and GPU backends. See root
`/CLAUDE.md` for general coding/testing/build rules — this file only covers
what's specific to this library.

## Backends

- CPU: `backend/cpu/{sse,avx,avx512,neon,sve,mkl}` — selected via `setup.py`
  tokens `sse`, `avx`, `avx2`, `avx512`, `neon`, `sve` (`cpu backend: no,
  sse, avx, avx2, avx512, neon, or sve`).
- GPU: `backend/gpu/{metal,float,double}` — selected via
  `--gpu_backend=none|hip|cuda|metal`.
- SLEEF (`sleef` token) provides NEON/SVE transcendentals and is
  auto-enabled by CMake when Apple Accelerate's vForce isn't available —
  don't hand-add it unless you're overriding that default.

## Packet size

`VECTORIZATION_PACKET_SIZE` (SIMD lane count, default 4) is set via
`--packet-size=N` or the shorthand token `psizeN` (e.g. `psize8`). Changing
it affects codegen across the whole backend, not just Vectorization — when
testing a packet-size change, rebuild rather than reusing an existing build
directory.

## GPU feature-guard macros — use `MEMORY_HAS_*`, not `PROJECT_HAS_*`

Test/source code must guard GPU-specific code with `MEMORY_HAS_CUDA` /
`MEMORY_HAS_HIP` / `MEMORY_HAS_METAL` (and the equivalent per-module macros
CMake defines, e.g. `VECTORIZATION_HAS_CUDA`, `VECTORIZATION_HAS_METAL`).
`PROJECT_HAS_CUDA` / `PROJECT_HAS_HIP` look plausible but **are never defined
anywhere in this repo** — code guarded by them silently compiles out and the
GPU path never runs, which is exactly the bug fixed in commit `f15cf987`
(14 test files had this wrong). If you add a new GPU-conditional block,
verify the guard macro is actually defined by grepping the relevant
`Cmake/*.cmake` file before trusting it compiles the intended branch.

## HIP is Unix-only

`hip.cmake` and this library's HIP CMake block fail fast with a clear error
on `WIN32` by policy — ROCm-on-Windows is unvalidated here. Don't try to
make a HIP path "work" on Windows CI; if HIP support is needed there, that's
a decision for the user, not a silent workaround.
