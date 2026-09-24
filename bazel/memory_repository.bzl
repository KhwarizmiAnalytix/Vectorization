"""Local checkout of ThirdParty/Memory with this repo's overlay BUILD.

The vendored repo ships its own BUILD.bazel for standalone use (allocator,
data_ptr, data_view, device_enum). This rule copies first-party Memory
sources, drops nested BUILD/WORKSPACE files and Memory's own vendored
ThirdParty/ (this workspace supplies fmt/mimalloc/loguru/magic_enum/
googletest/benchmark itself, at the same submodule paths Memory uses), and
writes //ThirdParty:memory.BUILD at the root so @vectorization//bazel flags
apply.

Vectorization depends on @memory//:Memory only.
"""

def _local_memory_repository_impl(repository_ctx):
    src = repository_ctx.path(str(repository_ctx.workspace_root) + "/" + repository_ctx.attr.path)
    if not src.exists:
        fail("Memory checkout missing at %s (git submodule update --init ThirdParty/Memory)" % src)

    python = repository_ctx.which("python3")
    if python == None:
        python = repository_ctx.which("python")
    if python == None:
        fail("python3 is required to stage @memory")

    rsync = repository_ctx.which("rsync")
    if rsync != None:
        result = repository_ctx.execute([
            rsync,
            "-a",
            "--copy-links",
            "--exclude", ".git",
            "--exclude", "ThirdParty",
            str(src) + "/",
            "./",
        ])
        if result.return_code != 0:
            fail("Failed to rsync Memory sources: %s%s" % (result.stdout, result.stderr))
    else:
        copy_script = """
import os
import shutil
import sys

src, dst = sys.argv[1], sys.argv[2]
for name in os.listdir(src):
    if name in (".git", "ThirdParty"):
        continue
    s = os.path.join(src, name)
    d = os.path.join(dst, name)
    if os.path.isdir(s):
        shutil.copytree(s, d, symlinks=False)
    elif not os.path.islink(s):
        shutil.copy2(s, d)
"""
        result = repository_ctx.execute([python, "-c", copy_script, str(src), "."])
        if result.return_code != 0:
            fail("Failed to copy Memory sources: %s%s" % (result.stdout, result.stderr))

    for rel in [
        "BUILD.bazel",
        "WORKSPACE.bazel",
        "Testing/BUILD.bazel",
        "Testing/Cxx/BUILD.bazel",
        ".bazelignore",
    ]:
        nested = repository_ctx.path(rel)
        if nested.exists:
            repository_ctx.delete(nested)

    repository_ctx.file(
        "WORKSPACE",
        'workspace(name = "%s")\n' % repository_ctx.name,
    )
    repository_ctx.file(
        "BUILD.bazel",
        repository_ctx.read(repository_ctx.attr.build_file),
    )

local_memory_repository = repository_rule(
    implementation = _local_memory_repository_impl,
    attrs = {
        "path": attr.string(mandatory = True),
        "build_file": attr.label(mandatory = True, allow_single_file = True),
    },
    local = True,
)
