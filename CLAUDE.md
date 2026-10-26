# Vectorization

SIMD abstraction layer over CPU and GPU backends. Public headers live in `include/`, implementation files in `src/`, and tests in `Testing/Cxx/`. Build configuration lives in the root `CMakeLists.txt` and `Cmake/`. Use namespace `vectorization` and `VECTORIZATION_*` macros. Dependencies are under `ThirdParty/`.

## Shared agent guidance

Adapted from the public [XSigma rules and skills](https://github.com/KhwarizmiAnalytix/XSigma/tree/89848c54492abef57fd0d0dc53b9da96b7cd1d5d)
at revision `89848c54492abef57fd0d0dc53b9da96b7cd1d5d`. Local API, dependency, language, and build
conventions below specialize that guidance for this standalone repository.

Read the applicable rules before editing. They apply to Claude as well as
Augment; C++ rules apply only when working on C++:

- [C++ coding](.augment/rules/coding.md) and [builders](.augment/rules/builder.md)
- [Python](.augment/rules/python.md)
- [Testing](.augment/rules/testing.md) and [builds](.augment/rules/build%20rule.md)
- [Dependencies](.augment/rules/ThirdParty.md)
- [Portability](.augment/rules/must-have.md) and [documentation](.augment/rules/markdown.md)

Use these task-specific skills as needed:

- [project-build](.claude/skills/project-build/SKILL.md): configure, build, and test
- [new-test](.claude/skills/new-test/SKILL.md): add tests using local conventions
- [clang-tidy](.claude/skills/clang-tidy/SKILL.md): analyze first-party C++ when applicable
- [session-checklist](.claude/skills/session-checklist/SKILL.md): verify completed work

## Build and test

Use the setup helper from `Scripts/`; inspect its help before adding
feature flags:

```sh
cd Scripts
python3 setup.py --help
python3 setup.py config.build.test
```

For compiler or generator requirements, follow `README.md` and CI.
The repository also documents direct CMake commands for integration and CI.

For Bazel, also run from `Scripts/`:

```sh
python3 setup_bazel.py config.build.test
```

## Test conventions

Match adjacent test cases and testing framework conventions in the repository.
Tests use `Test*.cpp` or `Test*.cxx` under `Testing/Cxx/`; CMake uses a recursive glob
while Bazel uses a package-local glob. Check exclusions and register new subdirectories
in both systems when adding tests.

## Verification and scope

For non-trivial source or build changes, run affected tests, review the diff,
and run configured lint/static-analysis checks relevant to touched files.
Check both build systems where provided. Follow the session checklist and
report checks run, failures, and unavailable tools explicitly. Guidance-only
changes need frontmatter/link/whitespace validation, not compilation.

Keep unrelated user edits and dependency sources intact. Share review
findings in the response or pull request; do not create unsolicited status
documents. Follow this repository's existing license and contribution policy.

## CMake Configuration Message Alignment

All `message("  LABEL : value")` configuration summary messages in CMakeLists.txt
and `Cmake/lto.cmake` must align colons at exactly **24 characters from the opening
quote** (inclusive).

Format: `message("  LABEL{PADDING}: VALUE")`
- Opening `"`: position 1
- Two spaces + label + padding: positions 2-23 (22 chars total)
- Colon `:`: position 24

Example:
```cmake
message("  Icecc               : ${VECTORIZATION}_ENABLE_ICECC}")
```

This ensures all colons in configuration output form a vertical line for readability.

## Backends

- CPU: `ThirdParty/Scalars/include/backend/cpu/{sse,avx,avx512,neon,sve,mkl}` — selected via `setup.py`
  tokens `sse`, `avx`, `avx2`, `avx512`, `neon`, `sve` (`cpu backend: no,
  sse, avx, avx2, avx512, neon, or sve`).
- GPU: `ThirdParty/Scalars/include/backend/gpu/{metal,float,double}` — selected via
  `--gpu_backend=none|hip|cuda|metal`.
- SLEEF (`sleef` token) is an optional math backend owned by Scalars; enable it
  through `VECTORIZATION_ENABLE_SLEEF` so the dependency receives the same setting.

## Packet size

`VECTORIZATION_PACKET_SIZE` (SIMD registers per unrolled evaluator loop, default 1) is set via
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
`CMakeLists.txt` or `Cmake/` module file before trusting it compiles the intended branch.

## HIP is Unix-only

`hip.cmake` and this library's HIP CMake block fail fast with a clear error
on `WIN32` by policy — ROCm-on-Windows is unvalidated here. Don't try to
make a HIP path "work" on Windows CI; if HIP support is needed there, that's
a decision for the user, not a silent workaround.
