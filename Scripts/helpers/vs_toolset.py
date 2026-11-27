"""
Visual Studio toolset selection shared by setup.py's dotted and named modes.

Visual Studio generators compile CUDA through NVIDIA's MSBuild customization
(nvcc with cl.exe as its host compiler) whatever the C++ toolset is, so the
toolset (-T) has to suit both the C++ compiler and the CUDA toolkit:

* cuda=<root> pins the toolkit that find_package(CUDAToolkit) also uses, so a
  build does not depend on which Visual Studio instances the CUDA installer
  integrated into (CI images, side-by-side toolkits or Visual Studio versions).
* Visual Studio 2026's default STL (v145, MSVC 14.5x) rejects CUDA older than
  13.4 (yvals_core.h STL1002), and CUDA 13.2's cudafe++ crashes (0xC0000005)
  while reporting it. Older toolkits are paired with the v143 toolset
  (MSVC 14.4x) when that Visual Studio instance has it installed; otherwise
  Cmake/tools/cuda.cmake stops with instructions.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

# Generator -> (vswhere version range, oldest CUDA its default STL accepts).
_VISUAL_STUDIO: dict[str, tuple[str, Optional[tuple[int, int]]]] = {
    "Visual Studio 16 2019": ("[16.0,17.0)", None),
    "Visual Studio 17 2022": ("[17.0,18.0)", None),
    "Visual Studio 18 2026": ("[18.0,19.0)", (13, 4)),
}
_CUDA_DEFINITION = "-DVECTORIZATION_GPU_BACKEND=cuda"


def cuda_toolkit() -> Optional[tuple[Path, tuple[int, int]]]:
    """Return (root, (major, minor)) of the toolkit CMake's FindCUDAToolkit uses."""
    # FindCUDAToolkit's order: CUDAToolkit_ROOT, CUDACXX, nvcc on PATH, CUDA_PATH.
    nvcc = os.environ.get("CUDACXX") or shutil.which("nvcc")
    root = (
        os.environ.get("CUDAToolkit_ROOT")
        or (str(Path(nvcc).resolve().parent.parent) if nvcc else None)
        or os.environ.get("CUDA_PATH")
    )
    if not root:
        return None
    try:
        with open(Path(root) / "version.json", encoding="utf-8") as stream:
            version = json.load(stream)["cuda"]["version"]
        major, minor = (int(part) for part in version.split(".")[:2])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return Path(root), (major, minor)


def v143_tools_version(version_range: str) -> Optional[str]:
    """Return the v143 MSVC version (e.g. "14.44") installed in that Visual Studio."""
    program_files = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = (
        Path(program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    )
    if not vswhere.is_file():
        return None
    try:
        installation = subprocess.run(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-version",
                version_range,
                "-property",
                "installationPath",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    if not installation:
        return None
    marker = (
        Path(installation)
        / "VC"
        / "Auxiliary"
        / "Build"
        / "Microsoft.VCToolsVersion.v143.default.txt"
    )
    try:
        version = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return ".".join(version.split(".")[:2]) or None


def select(generator: str, clang_cl: bool, definitions: list[str]) -> Optional[str]:
    """Return the -T value for a Visual Studio generator, or None for the default."""
    if not generator.startswith("Visual Studio"):
        return None
    toolset = "ClangCL" if clang_cl else None
    if os.environ.get("CMAKE_GENERATOR_TOOLSET"):
        return toolset
    toolkit = cuda_toolkit() if _CUDA_DEFINITION in definitions else None
    if toolkit is None:
        return toolset
    root, cuda = toolkit
    fields = []
    version_range, minimum = _VISUAL_STUDIO.get(generator, ("", None))
    if minimum and cuda < minimum:
        tools = v143_tools_version(version_range)
        cuda_text = f"CUDA {cuda[0]}.{cuda[1]}"
        if tools:
            print(
                f"[INFO] {cuda_text} predates the {generator} STL (needs CUDA "
                f"{minimum[0]}.{minimum[1]}+); using MSVC {tools} (v143)"
            )
            if clang_cl:
                fields.append(f"version={tools}")
            else:
                toolset = "v143"
        else:
            print(
                f"[WARNING] {cuda_text} predates the {generator} STL and its MSVC "
                "v143 build tools are not installed; install them or CUDA "
                f"{minimum[0]}.{minimum[1]}+"
            )
    if (root / "extras" / "visual_studio_integration" / "MSBuildExtensions").is_dir():
        fields.append(f"cuda={root}")
    if fields:
        toolset = ",".join([toolset or "", *fields]).lstrip(",")
    return toolset
