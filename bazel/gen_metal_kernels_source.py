#!/usr/bin/env python3
"""Substitutes @METAL_KERNEL_SOURCE@ in metal_kernels_source.h.in with the raw
contents of kernels.metal — the Bazel equivalent of CMake's
`configure_file(... metal_kernels_source.h.in metal_kernels_source.h @ONLY)`.

Usage: gen_metal_kernels_source.py <kernels.metal> <template.h.in> <out.h>
"""

import sys


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        return 1

    kernels_path, template_path, out_path = sys.argv[1:4]

    with open(kernels_path, encoding="utf-8") as f:
        kernel_source = f.read()
    with open(template_path, encoding="utf-8") as f:
        template = f.read()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(template.replace("@METAL_KERNEL_SOURCE@", kernel_source))

    return 0


if __name__ == "__main__":
    sys.exit(main())
