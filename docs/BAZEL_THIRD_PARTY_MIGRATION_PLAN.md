# Bazel Third-Party Plan

Applies to every KhwarizmiAnalytix repo, not just this one.

## Goal

Each repo owns the Bazel overlay for the third-party dependencies it
vendors. Consumers depend only on the owning repo's module (`@logging`,
`@memory`, ...) — never directly on a third-party repo they don't own. This
mirrors PyTorch: `third_party/<dep>.BUILD` overlays belong to the module
that vendors the submodule; downstream consumers never re-declare them.

## Problem

A consuming repo's `MODULE.bazel` often declares third-party repositories
(e.g. `fmt`, `magic_enum`, `mimalloc`, `loguru`) it doesn't actually own, just
to satisfy another KhwarizmiAnalytix repo's transitive closure — even though
its own code and `BUILD.bazel` never reference them directly.

## Plan (per repo)

1. List every third-party repository declared in the repo's `MODULE.bazel`.
2. For each one, grep the repo's own `include/`/`src/`/`BUILD.bazel` for a
   direct reference. If there is none, the declaration is there only to
   satisfy a sibling dependency and does not belong here.
3. Remove declarations that fail step 2. Keep only:
   - Third-party repos the repo's own code references directly.
   - Sibling KhwarizmiAnalytix module repos it depends on directly
     (`logging`, `memory`, `profiler`, `scalars`, `linear_algebra`, ...).
4. Confirm each sibling module exposes its own third-party overlays
   internally, so removing the redundant declaration changes nothing about
   what actually gets built — those symbols were always consumed
   transitively, never directly.
5. Build and test the repo standalone and as a downstream dependency.

## Rule going forward

Before adding a third-party repository declaration to `MODULE.bazel`, check
whether the repo's own code references it directly. If not, it belongs to
whichever repo vendors it — don't declare it here.

## Completion criteria

- [ ] `MODULE.bazel` declares no repository that the repo's own code doesn't
      reference directly.
- [ ] Build and tests pass, standalone and downstream, for every repo
      touched.
- [ ] `.bazelignore` / staging rules still exclude nested-submodule BUILD
      files for every sibling repo consumed this way.

## Reference

- [PyTorch `WORKSPACE`](https://github.com/pytorch/pytorch/blob/main/WORKSPACE) — per-module `third_party/<dep>.BUILD` ownership model
