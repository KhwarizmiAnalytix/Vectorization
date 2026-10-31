"""Exercise public option forwarding against the pinned dependency projects."""

import json
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(shutil.which("cmake"), "CMake is required")
class DependencyTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ninja"), "Ninja is required")
    def test_optional_dependencies_do_not_add_unused_modules_or_tests(self):
        source = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([
                "cmake", "-S", str(source), "-B", directory, "-G", "Ninja",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DVECTORIZATION_ENABLE_TESTING=ON",
                "-DVECTORIZATION_ENABLE_BENCHMARK=OFF",
                "-DVECTORIZATION_ENABLE_CACHE=OFF",
                "-DVECTORIZATION_ENABLE_SLEEF=ON",
                "-DVECTORIZATION_ENABLE_SVML=OFF",
                # Exercise stale cache values as well as fresh configurations.
                "-DFMT_MODULE=ON", "-DSLEEF_ENABLE_SSL=ON",
            ], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([
                "ctest", "--test-dir", directory, "--show-only=json-v1",
            ], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            tests = {test["name"] for test in json.loads(result.stdout)["tests"]}
            self.assertEqual(tests, {"VectorizationCxxTests"})
            result = subprocess.run([
                "cmake", "--build", directory, "--target", "help",
            ], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn("fmt-module", result.stdout)

    def test_backend_and_standard_follow_public_options_on_reconfigure(self):
        source = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            def configure(*flags):
                result = subprocess.run([
                    "cmake", "-S", str(source), "-B", directory,
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DVECTORIZATION_ENABLE_TESTING=OFF",
                    "-DVECTORIZATION_ENABLE_BENCHMARK=OFF",
                    "-DVECTORIZATION_ENABLE_CACHE=OFF",
                    *flags,
                ], capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                cache = {}
                for line in (Path(directory) / "CMakeCache.txt").read_text().splitlines():
                    if "=" in line and not line.startswith(("//", "#")):
                        key, value = line.split("=", 1)
                        cache[key.split(":", 1)[0]] = value
                return cache

            cache = configure(
                "-DVECTORIZATION_CPU_BACKEND=no", "-DLINALG_CPU_BACKEND=avx512",
                "-DSCALARS_CPU_BACKEND=avx512", "-DLINALG_ENABLE_VECTORIZATION=ON",
                "-DLINALG_ENABLE_CUBLAS=ON", "-DLINALG_GPU_BACKEND=cuda",
                "-DVECTORIZATION_CXX_STANDARD=23",
            )
            for prefix in ("VECTORIZATION", "SCALARS", "LINALG"):
                self.assertEqual(cache[f"{prefix}_CPU_BACKEND"], "no")
                self.assertEqual(cache[f"{prefix}_GPU_BACKEND"], "none")
            for prefix in ("LOGGING", "MEMORY", "SCALARS", "LINALG"):
                self.assertEqual(cache[f"{prefix}_CXX_STANDARD"], "23")
                self.assertEqual(cache[f"{prefix}_ENABLE_TESTING"], "OFF")

            # Switch the public selection in the same build tree, including a
            # backend unsupported on this host. Every consumer must follow the
            # actual Scalars fallback rather than retaining its previous flags.
            requested = "avx2" if platform.machine().lower() in ("arm64", "aarch64") else "neon"
            cache = configure(f"-DVECTORIZATION_CPU_BACKEND={requested}",
                              "-DVECTORIZATION_CXX_STANDARD=17")
            for prefix in ("VECTORIZATION", "SCALARS", "LINALG"):
                self.assertEqual(cache[f"{prefix}_CPU_BACKEND"], "no")
            for prefix in ("SCALARS", "LINALG"):
                self.assertEqual(cache[f"{prefix}_CXX_STANDARD"], "20")


if __name__ == "__main__":
    unittest.main()
