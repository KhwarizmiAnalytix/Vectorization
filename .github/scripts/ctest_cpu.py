"""Run CPU tests when the runner supports the compiled ISA; keep build coverage otherwise."""

import argparse
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--config", default="Release")
    args = parser.parse_args()
    build = Path(args.build_dir).resolve()
    if args.backend == "avx512":
        source = Path(__file__).resolve().parents[1] / "cpu-probe"
        probe = build / "cpu-probe"
        subprocess.run(["cmake", "-S", str(source), "-B", str(probe)], check=True)
        if (probe / "supported.txt").read_text() != "0":
            print("::notice::AVX-512 build passed; execution requires AVX-512 CPU and OS support.")
            return 0
    return subprocess.call([
        "ctest", "--test-dir", str(build), "-C", args.config,
        "--output-on-failure", "--no-tests=error", "--timeout", "300",
    ])


if __name__ == "__main__":
    raise SystemExit(main())
