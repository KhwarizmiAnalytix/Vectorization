#!/usr/bin/env python3
"""Vectorization Bazel Build Configuration Script.

Design follows KhwarizmiAnalytix/Logging's Scripts/setup_bazel.py (dotted-token
CLI, a Configuration class, the llvm-profdata/llvm-cov coverage pipeline that
works around Bazel's own broken C++ coverage merging) scoped to what this
repo's bazel/BUILD.bazel and .bazelrc actually wire up: one flat //:Vectorization
target plus the --define= keys bazel/BUILD.bazel's own config_settings
consume (vectorization_cpu_backend, vectorization_gpu_backend,
build_shared_libs, vectorization_enable_profiler).

Unlike Logging, this repo owns two independent backend selectors —
VECTORIZATION_CPU_BACKEND and VECTORIZATION_GPU_BACKEND in CMake, mirrored
here as --define=vectorization_cpu_backend=... and
--define=vectorization_gpu_backend=.... Selecting a GPU backend also passes
--define=memory_gpu_backend=... so the vendored @memory external repository
(ThirdParty/Memory, wired via bazel/memory_repository.bzl) picks the same
backend — this mirrors include/CMakeLists.txt's own
`set(MEMORY_GPU_BACKEND "${VECTORIZATION_GPU_BACKEND}" ...)` propagation.

CUDA/HIP are wired the same way Memory's own bazel/BUILD.bazel wires them
(select() on @local_config_cuda / @local_config_hip) but neither this repo's
WORKSPACE.bazel nor Memory's defines those external repositories yet, so
`--gpu_backend=cuda` / `--gpu_backend=hip` will fail to resolve until a CUDA/
HIP Bazel toolchain is added — same limitation Memory itself currently has.
Metal is fully wired (Apple frameworks only, no external toolchain repo
needed) and builds today on macOS.

Usage:
    python Scripts/setup_bazel.py config.build.test
    python Scripts/setup_bazel.py build.test.release
    python Scripts/setup_bazel.py build.test.avx2
    python Scripts/setup_bazel.py build.test.metal
    python Scripts/setup_bazel.py coverage
"""

import os
import platform
import re
import shutil
import subprocess
import sys
import time
from typing import Optional

try:
    import colorama

    colorama.init()
    COLOR_CYAN = colorama.Fore.CYAN
    COLOR_GREEN = colorama.Fore.GREEN
    COLOR_RED = colorama.Fore.RED
    COLOR_WHITE = colorama.Fore.WHITE
    COLOR_YELLOW = colorama.Fore.YELLOW
    COLOR_RESET = colorama.Style.RESET_ALL
except ImportError:  # Bazel CI jobs may not pip-install colorama
    COLOR_CYAN = ""
    COLOR_GREEN = ""
    COLOR_RED = ""
    COLOR_WHITE = ""
    COLOR_YELLOW = ""
    COLOR_RESET = ""


def print_status(message: str, status: str = "INFO") -> None:
    colors = {"INFO": COLOR_CYAN, "SUCCESS": COLOR_GREEN, "WARNING": COLOR_YELLOW, "ERROR": COLOR_RED}
    print(f"{colors.get(status, COLOR_WHITE)}[{status}]{COLOR_RESET} {message}")


def _find_bazel_executable() -> Optional[str]:
    """Return the absolute path of bazelisk or bazel, or None."""
    for cmd in ["bazelisk", "bazel"]:
        path = shutil.which(cmd)
        if path:
            return path
    return None


def get_bazel_command() -> str:
    cmd = _find_bazel_executable()
    if cmd:
        return cmd
    raise RuntimeError("Neither bazel nor bazelisk found in PATH")


def bazel_prefix() -> list[str]:
    """Argv prefix that can spawn Bazel on every platform.

    Windows ``.cmd``/``.bat`` launchers need ``cmd /c``; CreateProcess
    cannot execute them directly (WinError 2).
    """
    exe = get_bazel_command()
    if os.name == "nt" and exe.lower().endswith((".cmd", ".bat")):
        return ["cmd.exe", "/c", exe]
    return [exe]


_CPU_BACKENDS = {"no", "sse", "avx", "avx2", "avx512", "neon", "sve"}
_GPU_BACKENDS = {"none", "cuda", "hip", "metal"}

# Maps setup.py-style sanitizer names to the token this script tracks. Not
# wired to a Bazel --config here (this repo's .bazelrc defines no asan/tsan/…
# config groups) — tracked only so `sanitizer.address` etc. degrade to a clear
# "CMake-only" warning instead of a silent no-op.
_SANITIZER_NAMES = {"address", "undefined", "thread", "memory", "leak"}


def _merge_dotted_segments(parts: list[str]) -> list[str]:
    """Merge split segments like sanitizer.address into single tokens."""
    out: list[str] = []
    pl = [p.lower() for p in parts]
    i = 0
    while i < len(pl):
        if pl[i] == "sanitizer" and i + 1 < len(pl) and pl[i + 1] in _SANITIZER_NAMES:
            out.append(f"sanitizer.{pl[i + 1]}")
            i += 2
        elif pl[i] == "lto" and i + 1 < len(pl) and pl[i + 1] in ("off", "thin", "full", "ipo", "auto"):
            out.append(f"lto.{pl[i + 1]}")
            i += 2
        elif pl[i] == "linker" and i + 1 < len(pl):
            out.append(f"linker.{pl[i + 1]}")
            i += 2
        else:
            out.append(pl[i])
            i += 1
    return out


class VectorizationBazelConfiguration:
    """Manages Bazel build configuration and execution for the Vectorization repo."""

    def __init__(self, args: list[str]) -> None:
        self.args = args
        self.build_type = "debug"
        self.cxx_standard: Optional[str] = None
        self.targets: list[str] = ["//..."]
        self.run_tests = False
        self.run_build = False
        self.run_clean = False
        self.run_config = False
        self.run_coverage = False
        self.timing_data: dict[str, float] = {}
        self.slow_phase_warn_seconds: float = 60.0
        self.use_batch: bool = False
        self.verbose_tests = False
        self.subprocess_timeout: int = 600
        self.system = platform.system()

        # Compiler: None = Bazel's autoconfigured default toolchain (unlike CMake,
        # switching this only takes effect via --repo_env=CC=/CXX=, which forces
        # Bazel to re-run its C/C++ autoconfiguration).
        self.compiler: Optional[str] = None
        self.compiler_c: Optional[str] = None
        self.compiler_cxx: Optional[str] = None

        # Mirrors CMake VECTORIZATION_CPU_BACKEND / VECTORIZATION_GPU_BACKEND; None
        # = bazel/BUILD.bazel's own default (cpu: no select() branch taken, i.e. no
        # SIMD tree; gpu: none).
        self.cpu_backend: Optional[str] = None
        self.gpu_backend: Optional[str] = None

        # bazel/BUILD.bazel's :shared_libs config_setting; Bazel's own default (no
        # --define) is a *static* link, the opposite of CMake's standalone default.
        self.shared_libs: bool = False

        # CMake-only flags — not wired to any Bazel --config/--define in this repo's
        # bazel/BUILD.bazel or .bazelrc; tracked only so the summary/warnings are accurate.
        self.spell: bool = False
        self.clangtidy: bool = False
        self.fix: bool = False
        self.iwyu: bool = False
        self.valgrind: bool = False
        self.icecc: bool = False
        self.examples: bool = False
        self.cppcheck: bool = False
        self.cache: bool = False
        self.svml: bool = False
        self.mkl: bool = False
        self.sleef: bool = False
        self.accelerate: bool = False
        self.libtorch: bool = False
        self.sanitizer: Optional[str] = None
        self.lto_mode: Optional[str] = None
        self.linker: Optional[str] = None

        self._parse_arguments()
        self._apply_defaults()

    def _cmake_only(self, token: str, attr: Optional[str] = None) -> None:
        if attr:
            setattr(self, attr, True)
        print_status(
            f"Token '{token}': CMake-only (Scripts/setup.py) — not wired to a Bazel "
            "--config/--define in this repo's bazel/BUILD.bazel or .bazelrc.",
            "WARNING",
        )

    def _parse_arguments(self) -> None:
        for arg in self.args:
            arg_lower = arg.lower()

            if arg_lower in ("clang", "gcc") or "clang" in arg_lower or "gcc" in arg_lower or "g++" in arg_lower:
                if arg_lower not in ("clangtidy", "clang-tidy", "clang_tidy", "cppcheck"):
                    self._set_compiler(arg_lower)
                    continue

            if arg_lower in ["debug", "release", "relwithdebinfo"]:
                self.build_type = arg_lower
            elif arg_lower in ["cxx17", "cxx20", "cxx23"]:
                self.cxx_standard = arg_lower
            elif arg_lower in _CPU_BACKENDS:
                self.cpu_backend = arg_lower
            elif arg_lower in _GPU_BACKENDS:
                self.gpu_backend = arg_lower
            elif arg_lower.startswith("cpu_backend."):
                backend = arg_lower.split(".", 1)[1]
                if backend in _CPU_BACKENDS:
                    self.cpu_backend = backend
                else:
                    print_status(f"Invalid CPU backend '{backend}'. Use no, sse, avx, avx2, avx512, neon, or sve.", "ERROR")
                    sys.exit(1)
            elif arg_lower.startswith("gpu_backend."):
                backend = arg_lower.split(".", 1)[1]
                if backend in _GPU_BACKENDS:
                    self.gpu_backend = backend
                else:
                    print_status(f"Invalid GPU backend '{backend}'. Use none, cuda, hip, or metal.", "ERROR")
                    sys.exit(1)
            elif arg_lower in ("shared", "static"):
                # CMake's BUILD_SHARED_LIBS defaults ON for this repo; Bazel's own
                # default (no define) is static, so both spellings are accepted and
                # only "static" (or omitting the token) keeps Bazel's real default.
                self.shared_libs = arg_lower == "shared"
            elif arg_lower == "svml":
                self._cmake_only(arg_lower, "svml")
            elif arg_lower == "mkl":
                self._cmake_only(arg_lower, "mkl")
            elif arg_lower == "sleef":
                self._cmake_only(arg_lower, "sleef")
            elif arg_lower == "accelerate":
                self._cmake_only(arg_lower, "accelerate")
            elif arg_lower == "torch":
                self._cmake_only(arg_lower, "libtorch")
            elif arg_lower.startswith("sanitizer."):
                self.sanitizer = arg_lower.split(".", 1)[1]
                self._cmake_only(arg_lower)
            elif arg_lower.startswith("lto"):
                self.lto_mode = arg_lower.split(".", 1)[1] if "." in arg_lower else "auto"
                self._cmake_only(arg_lower)
            elif arg_lower.startswith("linker."):
                self.linker = arg_lower.split(".", 1)[1]
                self._cmake_only(arg_lower)
            elif arg_lower == "spell":
                self._cmake_only(arg_lower, "spell")
            elif arg_lower in ("clangtidy", "clang-tidy", "clang_tidy"):
                self._cmake_only(arg_lower, "clangtidy")
            elif arg_lower == "fix":
                self._cmake_only(arg_lower, "fix")
            elif arg_lower == "iwyu":
                self._cmake_only(arg_lower, "iwyu")
            elif arg_lower == "valgrind":
                self._cmake_only(arg_lower, "valgrind")
            elif arg_lower == "icecc":
                self._cmake_only(arg_lower, "icecc")
            elif arg_lower == "examples":
                self._cmake_only(arg_lower, "examples")
            elif arg_lower == "cppcheck":
                self._cmake_only(arg_lower, "cppcheck")
            elif arg_lower in ("cache", "cache_type"):
                self._cmake_only(arg_lower, "cache")
            elif arg_lower == "vv":
                self.verbose_tests = True
            elif arg_lower == "batch":
                self.use_batch = True
            elif arg_lower == "build":
                self.run_build = True
            elif arg_lower == "test":
                self.run_tests = True
            elif arg_lower == "coverage":
                self.run_coverage = True
            elif arg_lower == "clean":
                self.run_clean = True
            elif arg_lower == "config":
                self.run_config = True

    def _set_compiler(self, arg: str) -> None:
        if "clang" in arg and arg not in ("clang-cl",):
            self.compiler = "clang"
            self.compiler_c = arg
            self.compiler_cxx = arg.replace("clang", "clang++")
        elif "gcc" in arg or "g++" in arg:
            self.compiler = "gcc"
            if "g++" in arg:
                self.compiler_cxx = arg
                self.compiler_c = arg.replace("g++", "gcc")
            else:
                self.compiler_c = arg
                self.compiler_cxx = arg.replace("gcc", "g++")

    def _apply_defaults(self) -> None:
        if not self.compiler:
            self.compiler = "clang"
            self.compiler_c = "clang"
            self.compiler_cxx = "clang++"
        print_status(f"Using default compiler: {self.compiler.upper()}", "INFO")

    def _config_flags(self) -> list[str]:
        """Flags that select *which build configuration* Bazel resolves.

        Shared between build_bazel_command() and _bazel_info(): `bazel info
        bazel-bin`/`output_path` resolve to a different directory per
        compilation_mode (fastbuild/dbg/opt) and per --repo_env (a different
        CC/CXX re-triggers cc_autoconf) — querying `bazel info` without these
        same flags silently returns a sibling configuration's path.
        """
        flags: list[str] = []
        if self.build_type == "release":
            flags.extend(["-c", "opt"])
        elif self.build_type == "relwithdebinfo":
            flags.extend(["-c", "opt", "--copt=-g"])
        else:
            flags.extend(["-c", "dbg"])

        if self.compiler_c:
            flags.append(f"--repo_env=CC={self.compiler_c}")
        if self.compiler_cxx:
            flags.append(f"--repo_env=CXX={self.compiler_cxx}")

        if self.cpu_backend:
            flags.append(f"--define=vectorization_cpu_backend={self.cpu_backend}")

        if self.gpu_backend:
            flags.append(f"--define=vectorization_gpu_backend={self.gpu_backend}")
            # Mirrors include/CMakeLists.txt propagating VECTORIZATION_GPU_BACKEND
            # into MEMORY_GPU_BACKEND when Memory is embedded (see
            # "if(NOT DEFINED MEMORY_GPU_BACKEND) set(MEMORY_GPU_BACKEND ...)").
            flags.append(f"--define=memory_gpu_backend={self.gpu_backend}")

        if self.shared_libs:
            flags.append("--define=build_shared_libs=true")

        if self.cxx_standard:
            std_version = self.cxx_standard.replace("cxx", "")
            flags.append(f"--cxxopt=-std=c++{std_version}")
            flags.append(f"--host_cxxopt=-std=c++{std_version}")

        return flags

    def build_bazel_command(self, action: str) -> list[str]:
        """Build the Bazel command with all configurations."""
        cmd = bazel_prefix()
        if self.use_batch:
            cmd.append("--batch")
        cmd.append(action)
        cmd.extend(self._config_flags())
        cmd.extend(self.targets)
        return cmd

    def _kill_stale_bazel_processes(self) -> None:
        """Force-kill any stale Bazel server processes holding the output base lock."""
        print_status("Killing any stale Bazel server processes...", "INFO")
        for proc_name in ["bazel-real.exe", "bazel.exe"]:
            try:
                result = subprocess.run(
                    ["taskkill", "/F", "/IM", proc_name], capture_output=True, timeout=10, check=False
                )
                if result.returncode == 0:
                    print_status(f"Killed stale process: {proc_name}", "WARNING")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

    def _shutdown_bazel_for_batch(self) -> None:
        if not self.use_batch:
            return
        print_status("Running `bazel shutdown` before `--batch` (avoids startup-option mismatch).", "INFO")
        try:
            subprocess.run(bazel_prefix() + ["shutdown"], capture_output=True, text=True, timeout=120, check=False)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    def _on_off(self, condition: bool) -> str:
        return f"{COLOR_GREEN}ON{COLOR_RESET}" if condition else f"{COLOR_RED}OFF{COLOR_RESET}"

    def _na(self) -> str:
        return f"{COLOR_YELLOW}N/A (CMake-only){COLOR_RESET}"

    def _pf(self, label: str, value: str, width: int = 20) -> None:
        print(f"  {label:{width}}: {value}")

    def print_configuration_summary(self) -> None:
        """Print a summary of the resolved build configuration to stdout."""
        print("\n" + "=" * 80)
        print("VECTORIZATION BAZEL BUILD CONFIGURATION SUMMARY")
        print("=" * 80)

        print(f"\n{COLOR_CYAN}Compiler & Build Tool:{COLOR_RESET}")
        self._pf("Platform", self.system)
        self._pf("Compiler", (self.compiler or "clang").upper())
        self._pf("Build type", self.build_type.upper())
        self._pf("Cxx standard", self.cxx_standard.replace("cxx", "C++") if self.cxx_standard else "C++20 (default)")

        print(f"\n{COLOR_CYAN}Vectorization module (Bazel flags):{COLOR_RESET}")
        self._pf("CPU backend", (self.cpu_backend or "no (default)").upper())
        self._pf("GPU backend", (self.gpu_backend or "none").upper())
        if self.gpu_backend in ("cuda", "hip"):
            print_status(
                f"GPU backend '{self.gpu_backend}' selects @local_config_{self.gpu_backend}, which this "
                "workspace does not define yet — the build will fail until a CUDA/HIP Bazel toolchain "
                "is wired (same limitation as Memory's own bazel/BUILD.bazel).",
                "WARNING",
            )
        self._pf("Shared libs", self._on_off(self.shared_libs))
        self._pf("Coverage", self._on_off(self.run_coverage))
        self._pf("Testing", self._on_off(self.run_tests))
        self._pf("Gtest", self._on_off(True))
        self._pf("Benchmark", self._on_off(True))
        self._pf("Svml", self._na() if self.svml else self._on_off(False))
        self._pf("Mkl", self._na() if self.mkl else self._on_off(False))
        self._pf("Sleef", self._na() if self.sleef else self._on_off(False))
        self._pf("Accelerate", self._na() if self.accelerate else self._on_off(False))
        self._pf("LibTorch", self._na() if self.libtorch else self._on_off(False))
        self._pf("Examples", self._na() if self.examples else self._on_off(False))
        self._pf("Clang-tidy", self._na() if self.clangtidy else self._on_off(False))
        self._pf("Fix", self._na() if self.fix else self._on_off(False))
        self._pf("Iwyu", self._na() if self.iwyu else self._on_off(False))
        self._pf("Cppcheck", self._na() if self.cppcheck else self._on_off(False))
        self._pf("Sanitizer", self._na() if self.sanitizer else self._on_off(False))
        self._pf("Spell", self._na() if self.spell else self._on_off(False))
        self._pf("Valgrind", self._na() if self.valgrind else self._on_off(False))
        self._pf("Icecc", self._na() if self.icecc else self._on_off(False))
        self._pf("Cache", self._na() if self.cache else self._on_off(False))
        self._pf("Linker", self._na() if self.linker else "default")
        self._pf("Lto", self._na() if self.lto_mode else self._on_off(False))

        if self.run_build or self.run_tests or self.run_coverage:
            action = "test" if self.run_tests else ("coverage" if self.run_coverage else "build")
            cmd = self.build_bazel_command(action)
            print(f"\n{COLOR_CYAN}Bazel Command:{COLOR_RESET}")
            print(f"  {' '.join(cmd)}")

        print("\n" + "=" * 80 + "\n")

    def config(self) -> None:
        """Handle config action (summary already printed by execute())."""

    def clean(self) -> None:
        if not self.run_clean:
            return
        print_status("Cleaning Bazel build artifacts...", "INFO")
        try:
            start_time = time.time()
            subprocess.run(bazel_prefix() + ["clean", "--expunge"], check=True, timeout=300)
            elapsed = time.time() - start_time
            self.timing_data["clean"] = elapsed
            print_status(f"Clean completed successfully ({elapsed:.2f}s)", "SUCCESS")
        except subprocess.CalledProcessError as e:
            print_status(f"Clean failed with exit code {e.returncode}", "ERROR")
            sys.exit(1)
        except subprocess.TimeoutExpired:
            print_status("Clean operation timed out (exceeded 5 minutes)", "ERROR")
            sys.exit(1)

    def build(self) -> None:
        if not self.run_build:
            return
        print_status("Starting Bazel build...", "INFO")
        self._shutdown_bazel_for_batch()
        cmd = self.build_bazel_command("build")
        print_status(f"Running: {' '.join(cmd)}", "INFO")
        try:
            start_time = time.time()
            subprocess.run(cmd, check=True, timeout=self.subprocess_timeout)
            elapsed = time.time() - start_time
            self.timing_data["build"] = elapsed
            print_status(f"Build completed successfully ({elapsed:.2f}s)", "SUCCESS")
            if elapsed > self.slow_phase_warn_seconds:
                print_status(f"Build took longer than {self.slow_phase_warn_seconds:.0f}s.", "WARNING")
        except subprocess.CalledProcessError as e:
            print_status(f"Build failed with exit code {e.returncode}", "ERROR")
            sys.exit(1)
        except subprocess.TimeoutExpired:
            print_status(f"Build timed out (exceeded {self.subprocess_timeout}s)", "ERROR")
            sys.exit(1)

    def test(self) -> None:
        if not self.run_tests:
            return
        print_status("Running Bazel tests...", "INFO")
        self._shutdown_bazel_for_batch()
        cmd = self.build_bazel_command("test")
        cmd.append("--test_output=all" if self.verbose_tests else "--test_output=errors")
        if self.verbose_tests:
            cmd.append("--verbose_failures")
        print_status(f"Running: {' '.join(cmd)}", "INFO")
        try:
            start_time = time.time()
            result = subprocess.run(cmd, check=False, timeout=self.subprocess_timeout)
            elapsed = time.time() - start_time
            self.timing_data["test"] = elapsed
            if result.returncode == 0:
                print_status(f"Tests completed successfully ({elapsed:.2f}s)", "SUCCESS")
            elif result.returncode == 4:
                print_status(f"No test targets found ({elapsed:.2f}s)", "WARNING")
            else:
                print_status(f"Tests failed with exit code {result.returncode}", "ERROR")
                sys.exit(1)
        except subprocess.TimeoutExpired:
            print_status(f"Tests timed out (exceeded {self.subprocess_timeout}s)", "ERROR")
            sys.exit(1)

    def coverage(self) -> None:
        if not self.run_coverage:
            return
        print_status("Running Bazel coverage...", "INFO")
        self._shutdown_bazel_for_batch()
        cmd = self.build_bazel_command("coverage")
        cmd.append("--test_output=all" if self.verbose_tests else "--test_output=errors")
        print_status(f"Running: {' '.join(cmd)}", "INFO")
        try:
            start_time = time.time()
            result = subprocess.run(cmd, check=False, timeout=self.subprocess_timeout)
            elapsed = time.time() - start_time
            self.timing_data["coverage"] = elapsed
            if result.returncode == 0:
                print_status(f"Coverage completed successfully ({elapsed:.2f}s)", "SUCCESS")
                dat_path = self._build_llvm_coverage_report(start_time)
                if dat_path is None:
                    dat_path = self._resolve_coverage_dat_path()
                print_status(f"Coverage report (LCOV): {dat_path}", "INFO")
                self._generate_coverage_html(dat_path)
            elif result.returncode == 4:
                print_status(f"No coverage targets found ({elapsed:.2f}s)", "WARNING")
            else:
                print_status(f"Coverage failed with exit code {result.returncode}", "ERROR")
                sys.exit(1)
        except subprocess.TimeoutExpired:
            print_status(f"Coverage timed out (exceeded {self.subprocess_timeout}s)", "ERROR")
            sys.exit(1)

    def _bazel_info(self, key: str, fallback: str, match_config: bool = False) -> str:
        """Resolve a `bazel info <key>` value (absolute path, cwd-independent)."""
        cmd = bazel_prefix() + ["info"]
        if match_config:
            cmd.extend(self._config_flags())
        cmd.append(key)
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
            return result.stdout.strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return fallback

    def _resolve_coverage_dat_path(self) -> str:
        """Resolve Bazel's own (known-empty; see coverage() docstring below) LCOV report."""
        output_path = self._bazel_info("output_path", "bazel-out", match_config=True)
        return os.path.join(output_path, "_coverage", "_coverage_report.dat")

    def _find_recent_profraw_files(self, since: float) -> list[str]:
        """Find .profraw files the just-run coverage tests produced."""
        output_base = self._bazel_info("output_base", "")
        if not output_base:
            return []
        stash_root = os.path.join(output_base, "sandbox", "sandbox_stash", "TestRunner")
        if not os.path.isdir(stash_root):
            return []
        profraw_files = []
        for root, _dirs, files in os.walk(stash_root):
            if "_coverage" not in root:
                continue
            for name in files:
                if not name.endswith(".profraw"):
                    continue
                path = os.path.join(root, name)
                try:
                    if os.path.getmtime(path) >= since:
                        profraw_files.append(path)
                except OSError:
                    continue
        return profraw_files

    def _find_coverage_test_binaries(self) -> list[str]:
        """Resolve bazel-bin paths for every cc_test target covered by this run."""
        targets_expr = " + ".join(self.targets)
        try:
            query_result = subprocess.run(
                bazel_prefix()
                + [
                    "query",
                    f"kind(cc_test, {targets_expr})",
                    "--output=label",
                    "--enable_workspace",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return []

        bazel_bin = self._bazel_info("bazel-bin", "", match_config=True)
        if not bazel_bin:
            return []

        binaries = []
        for label in query_result.stdout.splitlines():
            label = label.strip()
            if not label.startswith("//"):
                continue
            pkg, _, name = label[2:].partition(":")
            binaries.append(os.path.join(bazel_bin, pkg, name))
        return binaries

    def _build_llvm_coverage_report(self, since: float) -> Optional[str]:
        """Build a real LCOV trace directly via llvm-profdata/llvm-cov (Clang only)."""
        if self.compiler != "clang":
            print_status(
                f"Bazel coverage HTML generation is only implemented for Clang "
                f"(compiler is '{self.compiler}'); falling back to Bazel's own "
                "(known-empty) report.",
                "WARNING",
            )
            return None

        profraw_files = self._find_recent_profraw_files(since)
        if not profraw_files:
            print_status(
                "No .profraw files found from this coverage run; falling back to Bazel's own (known-empty) report",
                "WARNING",
            )
            return None

        binaries = [b for b in self._find_coverage_test_binaries() if os.path.isfile(b)]
        if not binaries:
            print_status(
                "Could not resolve test binaries for llvm-cov -object; falling back to Bazel's own (known-empty) report",
                "WARNING",
            )
            return None

        llvm_profdata = shutil.which("llvm-profdata")
        llvm_cov = shutil.which("llvm-cov")
        if not llvm_profdata or not llvm_cov:
            print_status(
                "llvm-profdata/llvm-cov not found on PATH; falling back to Bazel's own (known-empty) report",
                "WARNING",
            )
            return None

        output_path = self._bazel_info("output_path", "bazel-out", match_config=True)
        coverage_dir = os.path.join(output_path, "_coverage")
        os.makedirs(coverage_dir, exist_ok=True)
        merged_profdata = os.path.join(coverage_dir, "_merged.profdata")
        dat_path = os.path.join(coverage_dir, "_llvm_coverage_report.dat")

        try:
            merge_result = subprocess.run(
                [llvm_profdata, "merge", "-o", merged_profdata, *profraw_files],
                capture_output=True,
                text=True,
                timeout=self.subprocess_timeout,
            )
            if merge_result.returncode != 0:
                print_status(f"llvm-profdata merge failed: {merge_result.stderr.strip()}", "WARNING")
                return None

            export_cmd = [
                llvm_cov,
                "export",
                "-instr-profile",
                merged_profdata,
                "-format=lcov",
                "-ignore-filename-regex=(/external/|/ThirdParty/|^/opt/|^/usr/|^/Applications/)",
            ]
            for binary in binaries:
                export_cmd += ["-object", binary]
            export_result = subprocess.run(
                export_cmd, capture_output=True, text=True, timeout=self.subprocess_timeout
            )
            if export_result.returncode != 0:
                print_status(f"llvm-cov export failed: {export_result.stderr.strip()}", "WARNING")
                return None

            workspace_root = self._bazel_info("workspace", "")
            lcov_text = export_result.stdout
            if workspace_root:
                lcov_text = re.sub(
                    r"^SF:.*?/execroot/_main/", f"SF:{workspace_root}/", lcov_text, flags=re.MULTILINE
                )
            with open(dat_path, "w", encoding="utf-8") as f:
                f.write(lcov_text)
        except OSError as e:
            print_status(f"Failed to build coverage report: {e}", "WARNING")
            return None

        return dat_path

    def _generate_coverage_html(self, dat_path: str) -> None:
        """Convert the Bazel LCOV coverage report into an HTML report via genhtml."""
        if not os.path.isfile(dat_path):
            print_status(f"Coverage LCOV file not found at {dat_path}; skipping HTML report", "WARNING")
            return
        genhtml = shutil.which("genhtml")
        if genhtml is None:
            print_status(
                "genhtml not found on PATH; skipping HTML report (install lcov, e.g. 'brew install lcov')",
                "WARNING",
            )
            return
        html_dir = os.path.join(os.path.dirname(dat_path), "html")
        cmd = [
            genhtml,
            dat_path,
            "--output-directory",
            html_dir,
            "--ignore-errors",
            "inconsistent,corrupt,unsupported",
        ]
        try:
            subprocess.run(cmd, check=True, timeout=self.subprocess_timeout, capture_output=True, text=True)
            index_html = os.path.abspath(os.path.join(html_dir, "index.html"))
            print_status(f"Coverage HTML report: {index_html}", "SUCCESS")
        except subprocess.CalledProcessError as e:
            print_status(f"Failed to generate HTML coverage report: {(e.stderr or '').strip()[-500:]}", "WARNING")
        except subprocess.TimeoutExpired as e:
            print_status(f"Failed to generate HTML coverage report: {e}", "WARNING")

    def print_timing_summary(self) -> None:
        if not self.timing_data:
            return
        print("\n" + "=" * 80)
        print("BUILD TIMING SUMMARY")
        print("=" * 80)
        total_time = sum(self.timing_data.values())
        for phase, elapsed in self.timing_data.items():
            percentage = (elapsed / total_time * 100) if total_time > 0 else 0
            print(f"  {phase.capitalize():20} {elapsed:8.2f}s ({percentage:5.1f}%)")
        print(f"  {'-' * 40}")
        print(f"  {'Total':20} {total_time:8.2f}s (100.0%)")
        print("=" * 80 + "\n")

    def execute(self) -> None:
        """Execute the build pipeline."""
        self.print_configuration_summary()
        self.config()
        if self.run_config and (self.run_build or self.run_tests or self.run_coverage):
            print_status("Config requested: forcing clean build (bazel clean --expunge).", "INFO")
            self.run_clean = True
        self.clean()
        if self.run_build or self.run_tests or self.run_coverage:
            self._kill_stale_bazel_processes()
        if self.run_coverage:
            if self.run_build or self.run_tests:
                print_status(
                    "Coverage requested: skipping separate build/test steps (bazel coverage "
                    "builds and tests everything itself under different instrumented flags).",
                    "INFO",
                )
            self.coverage()
        else:
            self.build()
            self.test()
        self.print_timing_summary()


def parse_args(args: list[str]) -> list[str]:
    """Parse argv like Scripts/setup.py: long flags, dotted shortcuts."""
    processed: list[str] = []
    for arg in args:
        if arg.startswith("--cpu_backend=") or arg.startswith("--cpu-backend="):
            bt = arg.split("=", 1)[1].lower()
            if bt in _CPU_BACKENDS:
                processed.append(bt)
            else:
                print_status(f"Invalid CPU backend: {bt}. Valid: no, sse, avx, avx2, avx512, neon, sve", "ERROR")
                sys.exit(1)
            continue
        if arg.startswith("--gpu_backend=") or arg.startswith("--gpu-backend="):
            bt = arg.split("=", 1)[1].lower()
            if bt in _GPU_BACKENDS:
                processed.append(bt)
            else:
                print_status(f"Invalid GPU backend: {bt}. Valid: none, cuda, hip, metal", "ERROR")
                sys.exit(1)
            continue
        if arg.startswith("--sanitizer."):
            processed.append(f"sanitizer.{arg.split('.', 1)[1].lower()}")
            continue
        if arg.startswith("--lto."):
            processed.append(f"lto.{arg.split('.', 1)[1].lower()}")
            continue
        if arg.startswith("--linker."):
            processed.append(f"linker.{arg.split('.', 1)[1].lower()}")
            continue
        if re.search(r"[/\\]", arg) and re.search(r"[Cc]lang|[Gg][Cc][Cc]|[Gg]\+\+", arg):
            processed.append(arg)
            continue
        if "." in arg and not arg.startswith("--"):
            processed.extend(_merge_dotted_segments(arg.split(".")))
            continue
        processed.append(arg.lower())
    return processed


def print_help() -> None:
    print_status("Vectorization Bazel Build Configuration Helper", "INFO")
    print("\n" + "=" * 80)
    print("BAZEL BUILD SYSTEM")
    print("=" * 80)
    print("\nUsage examples:")
    print("  1. Show configuration (no build):")
    print("     python setup_bazel.py config")
    print("  2. Default debug build (CPU backend 'no'):")
    print("     python setup_bazel.py build.test")
    print("  3. Release build with a CPU SIMD backend:")
    print("     python setup_bazel.py build.test.avx2.release")
    print("  4. Metal GPU backend (macOS only; fully wired):")
    print("     python setup_bazel.py build.test.metal")
    print("  5. CUDA/HIP GPU backend (NOT YET wired end-to-end; see module docstring):")
    print("     python setup_bazel.py build.test.cuda")
    print("     python setup_bazel.py build.test --gpu_backend=hip")
    print("  6. Coverage (llvm-profdata/llvm-cov + genhtml report; Clang only):")
    print("     python setup_bazel.py coverage")
    print("  7. Clean build:")
    print("     python setup_bazel.py clean.build.test.release")
    print("\nBuild types:")
    print("  debug          - Debug build, -c dbg (default)")
    print("  release        - Release build, -c opt")
    print("  relwithdebinfo - -c opt --copt=-g")
    print("\nCompiler (via --repo_env=CC=/CXX=; default: Clang):")
    print("  clang | gcc | clang-15 | gcc-13 | ...")
    print("\nC++ Standard:")
    print("  cxx17 | cxx20 (default, from .bazelrc) | cxx23")
    print("\nCPU SIMD backend (mirrors CMake VECTORIZATION_CPU_BACKEND; default 'no'):")
    print("  no | sse | avx | avx2 | avx512 | neon | sve")
    print("  --cpu_backend=avx2  (same, long form)")
    print("\nGPU backend (mirrors CMake VECTORIZATION_GPU_BACKEND; default 'none'):")
    print("  none | cuda | hip | metal")
    print("  --gpu_backend=metal  (same, long form)")
    print("\nLinking:")
    print("  shared         - --define=build_shared_libs=true")
    print("  static         - Bazel's own default (no define needed)")
    print("\nMisc:")
    print("  vv             - Verbose Bazel test output (--test_output=all)")
    print("  batch          - Pass --batch to Bazel (runs `bazel shutdown` first)")
    print("\nActions:")
    print("  config         - Show configuration summary (no build)")
    print("  build          - Build the project")
    print("  test           - Run tests")
    print("  coverage       - Run tests with coverage instrumentation (lcov report)")
    print("  clean          - Clean build artifacts")
    print("\nCMake-only (Scripts/setup.py), not wired to Bazel in this repo:")
    print("  svml | mkl | sleef | accelerate | torch | sanitizer.* | valgrind |")
    print("  cppcheck | clangtidy | fix | iwyu | spell | icecc | examples | cache |")
    print("  cache_type | linker.* | lto.*")
    print("\nEquivalent to CMake setup.py:")
    print("  CMake:  python setup.py config.build.test.avx2.release")
    print("  Bazel:  python setup_bazel.py config.build.test.avx2.release")
    print()


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "--help":
        print_help()
        return

    if len(sys.argv) < 2:
        print_status("No build configuration specified. Use --help for usage information.", "ERROR")
        sys.exit(1)

    try:
        arg_list = parse_args(sys.argv[1:])
        print_status(f"Starting Bazel build for {platform.system()}", "INFO")
        config = VectorizationBazelConfiguration(arg_list)

        if not (config.run_build or config.run_tests or config.run_clean or config.run_config or config.run_coverage):
            config.run_build = True

        config.execute()
        print_status("Build process completed successfully!", "SUCCESS")

    except KeyboardInterrupt:
        print_status("\nBuild process interrupted by user", "WARNING")
        sys.exit(1)
    except Exception as e:
        print_status(f"An unexpected error occurred: {e}", "ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main()
