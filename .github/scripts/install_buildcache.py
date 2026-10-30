"""Build a pinned buildcache on the runner instead of relying on package aliases."""

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    args = parser.parse_args()
    prefix = args.prefix.resolve()
    source, build = prefix / "source", prefix / "build"
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            "v0.33.1",
            "--recurse-submodules",
            "https://gitlab.com/bits-n-bites/buildcache.git",
            str(source),
        ],
        check=True,
    )
    configure = [
        "cmake",
        "-S",
        str(source / "src"),
        "-B",
        str(build),
        "-DCMAKE_BUILD_TYPE=Release",
        # The pinned MinGW thread headers still request CMake 3.0 policies.
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
        # CI uses local storage; avoid runner-specific OpenSSL DLLs.
        "-DDISABLE_SSL=ON",
        f"-DCMAKE_INSTALL_PREFIX={prefix}",
    ]
    # Let CMake discover Visual Studio on Windows. Ninja otherwise selects the
    # runner's MinGW compiler, whose bundled thread headers fail on newer GCC.
    if os.name != "nt":
        configure.extend(["-G", "Ninja"])
    subprocess.run(configure, check=True)
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build),
            "--config",
            "Release",
            "--target",
            "buildcache",
            "--parallel",
            "4",
        ],
        check=True,
    )
    executable = "buildcache.exe" if os.name == "nt" else "buildcache"
    binaries = [path for path in build.rglob(executable) if path.is_file()]
    if not binaries:
        raise RuntimeError(
            "buildcache built without producing the requested executable"
        )
    destination = prefix / "bin"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(binaries[0], destination / executable)
    for pattern in ("*.dll", "*.dylib", "*.so*"):
        for library in build.rglob(pattern):
            if library.is_file():
                shutil.copy2(library, destination / library.name)
    subprocess.run([str(destination / executable), "--version"], check=True)
    with Path(os.environ["GITHUB_PATH"]).open("a", encoding="utf-8") as output:
        output.write(str(destination) + "\n")


if __name__ == "__main__":
    main()
