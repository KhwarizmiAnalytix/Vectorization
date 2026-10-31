"""Local checkout of ThirdParty/LinearAlgebra with this repo's overlay BUILD.

The vendored repo ships its own BUILD.bazel for standalone use (matrix
decompositions, solvers, BLAS/LAPACK backends). This rule copies first-party
LinearAlgebra sources, drops nested BUILD/WORKSPACE files and LinearAlgebra's
own vendored ThirdParty/ (WORKSPACE.bazel resolves transitive dependencies from
their owning checkouts), and writes //bazel:linearalgebra.BUILD at the
root so @xsigma_vectorization//bazel flags apply.

Vectorization depends on @linear_algebra//:LinearAlgebra only.
"""

def _local_linear_algebra_repository_impl(repository_ctx):
    src = repository_ctx.path(str(repository_ctx.workspace_root) + "/" + repository_ctx.attr.path)
    if not src.exists:
        fail("LinearAlgebra checkout missing at %s (git submodule update --init ThirdParty/LinearAlgebra)" % src)

    python = repository_ctx.which("python3")
    if python == None:
        python = repository_ctx.which("python")
    if python == None:
        fail("python3 is required to stage @linear_algebra")

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
            fail("Failed to rsync LinearAlgebra sources: %s%s" % (result.stdout, result.stderr))
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
            fail("Failed to copy LinearAlgebra sources: %s%s" % (result.stdout, result.stderr))

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

local_linear_algebra_repository = repository_rule(
    implementation = _local_linear_algebra_repository_impl,
    attrs = {
        "path": attr.string(mandatory = True),
        "build_file": attr.label(mandatory = True, allow_single_file = True),
    },
    local = True,
)
