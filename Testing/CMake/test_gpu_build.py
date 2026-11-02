"""Check GPU configuration contracts without requiring a GPU SDK."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2]


@unittest.skipUnless(shutil.which("cmake"), "CMake is required")
class GpuBuildTests(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "HIP is supported on Linux only")
    def test_hip_compiler_architectures_and_host_flags(self):
        for selection in ("sdk", "explicit", "environment"):
            with self.subTest(
                selection=selection
            ), tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                sdk = work / "rocm"
                (sdk / "include/hip").mkdir(parents=True)
                (sdk / "include/hip/hip_runtime.h").touch()
                compiler = sdk / "llvm/bin/clang++"
                compiler.parent.mkdir(parents=True)
                compiler.write_text("#!/bin/sh\nexit 0\n")
                compiler.chmod(0o755)
                explicit = (
                    'set(CMAKE_HIP_COMPILER "/explicit/clang++")'
                    if selection == "explicit"
                    else ""
                )
                expected = {
                    "sdk": compiler.as_posix(),
                    "explicit": "/explicit/clang++",
                    "environment": "",
                }[selection]
                body = r"""
cmake_minimum_required(VERSION 3.24)
project(HipProbe LANGUAGES CXX)
set(VECTORIZATION_CXX_STANDARD 20)
set(VECTORIZATION_GPU_ARCHITECTURES gfx1100)
list(PREPEND CMAKE_MODULE_PATH "@SOURCE@/Cmake/tools")
@EXPLICIT@
function(find_package name)
  if(NOT CMAKE_HIP_ARCHITECTURES STREQUAL "gfx1100")
    message(FATAL_ERROR "Architecture must be set before package discovery")
  endif()
  if(name STREQUAL "hip")
    add_library(hip::host INTERFACE IMPORTED GLOBAL)
    add_library(hip::device INTERFACE IMPORTED GLOBAL)
    foreach(target hip::host hip::device)
      set_property(TARGET ${target} PROPERTY INTERFACE_COMPILE_OPTIONS
        "-x;hip;SHELL:-x hip;$<$<COMPILE_LANGUAGE:CXX>:--offload-arch=gfx906>;-DKEEP")
      set_property(TARGET ${target} PROPERTY INTERFACE_LINK_OPTIONS "--hip-link;-Wl,keep")
    endforeach()
    set(hip_FOUND TRUE PARENT_SCOPE)
    set(hip_VERSION "6.2.1" PARENT_SCOPE)
  endif()
endfunction()
function(enable_language language)
  if(NOT CMAKE_HIP_ARCHITECTURES STREQUAL "gfx1100")
    message(FATAL_ERROR "Architecture must be set before compiler detection")
  endif()
  if(NOT "${CMAKE_HIP_COMPILER}" STREQUAL "@EXPECTED@")
    message(FATAL_ERROR "Incorrect HIP compiler: ${CMAKE_HIP_COMPILER}")
  endif()
endfunction()
include("@SOURCE@/Cmake/tools/hip.cmake")
file(WRITE "${CMAKE_BINARY_DIR}/probe.cpp" "int probe() { return 0; }")
add_library(probe STATIC "${CMAKE_BINARY_DIR}/probe.cpp")
vectorization_apply_hip_flags(probe)
foreach(target hip::host hip::device)
  get_target_property(options ${target} INTERFACE_COMPILE_OPTIONS)
  if(NOT options STREQUAL "-DKEEP")
    message(FATAL_ERROR "Host-incompatible flags survived: ${options}")
  endif()
  get_target_property(options ${target} INTERFACE_LINK_OPTIONS)
  if(NOT options STREQUAL "-Wl,keep")
    message(FATAL_ERROR "Host-incompatible linker flags survived: ${options}")
  endif()
endforeach()
get_target_property(architectures probe HIP_ARCHITECTURES)
if(NOT architectures STREQUAL "gfx1100")
  message(FATAL_ERROR "Incorrect target architectures: ${architectures}")
endif()
"""
                for key, value in {
                    "SOURCE": SOURCE.as_posix(),
                    "EXPLICIT": explicit,
                    "EXPECTED": expected,
                }.items():
                    body = body.replace(f"@{key}@", value)
                (work / "CMakeLists.txt").write_text(body)
                env = dict(os.environ, ROCM_PATH=str(sdk), HIP_PATH=str(sdk))
                env.pop("HIPCXX", None)
                if selection == "environment":
                    env["HIPCXX"] = "/environment/clang++"
                result = subprocess.run(
                    ["cmake", "-S", str(work), "-B", str(work / "build")],
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_tensor_device_translation_units_match_cmake(self):
        # Mixing plain C++ and device definitions of the same expression
        # templates causes an ODR violation. Check the maintained Bazel list
        # against the same source-content predicate used by CMake.
        import ast
        import re

        build = (SOURCE / "Testing/Cxx/BUILD.bazel").read_text()
        match = re.search(r"_GPU_TEST_SRCS = (\[[^\]]*\])", build, re.DOTALL)
        self.assertIsNotNone(match)
        expected = {
            path.name
            for path in (SOURCE / "Testing/Cxx").glob("Test*.cpp")
            if not path.name.startswith("TestTensorLibTorch")
            and "tensor<" in path.read_text()
        }
        self.assertEqual(set(ast.literal_eval(match.group(1))), expected)

    def test_cuda_architectures_precede_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            body = r"""
cmake_minimum_required(VERSION 3.24)
project(CudaProbe LANGUAGES NONE)
set(LINALG_CXX_STANDARD 20)
set(VECTORIZATION_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(WIN32 TRUE)
set(CMAKE_CXX_COMPILER_ID Clang)
set(CMAKE_CXX_COMPILER_FRONTEND_VARIANT GNU)
function(find_package name)
  set(CUDAToolkit_LIBRARY_DIR "/sdk/lib" PARENT_SCOPE)
endfunction()
macro(enable_language language)
  if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "86")
    message(FATAL_ERROR "Architecture must be set before CUDA detection")
  endif()
  # Compiler modules overwrite runtime defaults during enable_language.
  set(CMAKE_CUDA_RUNTIME_LIBRARY_LINK_OPTIONS_SHARED cudadevrt cudart)
endmacro()
include("@SOURCE@/Cmake/tools/cuda.cmake")

"""
            body = body.replace("@SOURCE@", SOURCE.as_posix())
            (work / "CMakeLists.txt").write_text(body)
            result = subprocess.run(
                ["cmake", "-S", str(work), "-B", str(work / "build")],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
