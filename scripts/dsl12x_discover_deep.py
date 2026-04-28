#!/usr/bin/env python3
"""dsl12x cutlass.cute deep API surface discovery probe (round 2).

The first discover script (scripts/dsl12x_discover.py) gave us the top-level
surface: cute.MmaAtom, cute.make_mma_atom, cute.gemm, cute.copy, cute.nvgpu,
cute.arch, cute.atom, etc. all exist; cute.make_shared_memory does NOT exist.

This second pass drills specifically into:

    1. cutlass.utils  - looking for SmemAllocator / shared-memory-struct patterns
    2. cute.nvgpu.cpasync - cp.async copy op variants
    3. cute.nvgpu.warp - SM80/89 warp-level MMA op variants
    4. cute.nvgpu.common - shared utilities, address-space helpers
    5. cute.nvgpu.helpers - convenience builders
    6. cute.atom - all CopyOp / MmaOp variants (the actual op selection)
    7. cute.arch (filtered) - SMEM/alloc/mbarrier/tmem/copy/mma names
    8. cute.tensor (filtered) - allocation / fragment / pointer builders
    9. cute.AddressSpace - enum members (.smem, .gmem, .rmem expected)
    10. cute.make_ptr / cute.make_tensor - signature + docstring
    11. cute.make_fragment / cute.make_rmem_tensor - signature + docstring
    12. on-disk walk: find example/tutorial .py files in the cutlass-dsl install
        so I can reference real working idioms, not guess.

Run:

    python3 scripts/dsl12x_discover_deep.py 2>&1 | tee /tmp/dsl12x_discover_deep.txt

Pipe the file back. Section headers are grep-friendly.
"""

from __future__ import annotations

import os
import sys
import inspect

_WORKSPACE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
)
if _WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, _WORKSPACE_ROOT)


def _hr(title: str) -> None:
    print(f"\n[SECTION {title}]")
    print("-" * (len(title) + 12))


def _safe_get(obj, name):
    try:
        return getattr(obj, name)
    except Exception as exc:
        return f"<getattr error: {type(exc).__name__}: {exc}>"


def _print_attrs(obj, prefix_filter=None, name_substring_filter=None, max_items=400):
    attrs = []
    for name in sorted(dir(obj)):
        if name.startswith("_"):
            continue
        if prefix_filter is not None and not name.startswith(prefix_filter):
            continue
        if name_substring_filter is not None:
            lname = name.lower()
            if not any(s in lname for s in name_substring_filter):
                continue
        attrs.append(name)
    if not attrs:
        print("  (none)")
        return
    for name in attrs[:max_items]:
        val = _safe_get(obj, name)
        if isinstance(val, str):
            print(f"  {name}  {val}")
        else:
            kind = type(val).__name__
            extra = ""
            if inspect.isclass(val):
                extra = " (class)"
            elif inspect.ismodule(val):
                extra = " (module)"
            elif callable(val):
                try:
                    sig = inspect.signature(val)
                    extra = f" {sig}"
                except (TypeError, ValueError):
                    extra = "(<no inspectable signature>)"
            print(f"  {name}  ({kind}){extra}")
    if len(attrs) > max_items:
        print(f"  ... (+{len(attrs) - max_items} more)")


def _print_signature_and_doc(obj, name):
    val = _safe_get(obj, name)
    if isinstance(val, str):
        print(f"  {name} -> {val}")
        return
    if val is None:
        print(f"  {name} -> NOT FOUND")
        return
    try:
        sig = inspect.signature(val)
        print(f"  {name}{sig}")
    except (TypeError, ValueError):
        print(f"  {name}(<no inspectable signature>)")
    doc = inspect.getdoc(val)
    if doc:
        for line in doc.splitlines()[:12]:
            print(f"      | {line}")
        if len(doc.splitlines()) > 12:
            print("      | ...")


def main() -> int:
    try:
        import cutlass
        import cutlass.cute as cute
    except ImportError as exc:
        print(f"FATAL: cutlass(.cute) not importable: {exc}")
        return 3

    _hr("cutlass.utils.* (looking for SmemAllocator)")
    try:
        import cutlass.utils as cutlass_utils
        print(f"  module path: {cutlass_utils.__file__}")
        _print_attrs(cutlass_utils)
        for hit in ("SmemAllocator", "smem_allocator", "make_smem_struct",
                    "SharedStorage", "shared_storage", "make_shared_storage"):
            if hasattr(cutlass_utils, hit):
                print(f"\n  >>> found likely SMEM allocator: cutlass.utils.{hit}")
                _print_signature_and_doc(cutlass_utils, hit)
    except ImportError as exc:
        print(f"  cutlass.utils NOT IMPORTABLE: {exc}")

    _hr("cute.nvgpu.cpasync.*")
    try:
        cpasync = cute.nvgpu.cpasync
        _print_attrs(cpasync)
    except AttributeError as exc:
        print(f"  cute.nvgpu.cpasync NOT FOUND: {exc}")

    _hr("cute.nvgpu.warp.*")
    try:
        warp = cute.nvgpu.warp
        _print_attrs(warp)
    except AttributeError as exc:
        print(f"  cute.nvgpu.warp NOT FOUND: {exc}")

    _hr("cute.nvgpu.warpgroup.*")
    try:
        wg = cute.nvgpu.warpgroup
        _print_attrs(wg)
    except AttributeError as exc:
        print(f"  cute.nvgpu.warpgroup NOT FOUND: {exc}")

    _hr("cute.nvgpu.tcgen05.*")
    try:
        tcgen05 = cute.nvgpu.tcgen05
        _print_attrs(tcgen05)
    except AttributeError as exc:
        print(f"  cute.nvgpu.tcgen05 NOT FOUND: {exc}")

    _hr("cute.nvgpu.common.*")
    try:
        common = cute.nvgpu.common
        _print_attrs(common)
    except AttributeError as exc:
        print(f"  cute.nvgpu.common NOT FOUND: {exc}")

    _hr("cute.nvgpu.helpers.*")
    try:
        helpers = cute.nvgpu.helpers
        _print_attrs(helpers)
    except AttributeError as exc:
        print(f"  cute.nvgpu.helpers NOT FOUND: {exc}")

    _hr("cute.atom.* (full)")
    try:
        atom = cute.atom
        _print_attrs(atom)
    except AttributeError as exc:
        print(f"  cute.atom NOT FOUND: {exc}")

    _hr("cute.arch.* filtered: smem|alloc|mbarrier|tmem|sync|copy|mma|cluster|barrier")
    try:
        arch = cute.arch
        _print_attrs(
            arch,
            name_substring_filter=[
                "smem", "alloc", "mbarrier", "tmem", "sync",
                "copy", "mma", "cluster", "barrier", "warp",
                "thread", "addr", "ptr",
            ],
        )
    except AttributeError as exc:
        print(f"  cute.arch NOT FOUND: {exc}")

    _hr("cute.tensor.* filtered: tensor|fragment|alloc|smem|ptr|make")
    try:
        tensor_mod = cute.tensor
        _print_attrs(
            tensor_mod,
            name_substring_filter=["tensor", "fragment", "alloc", "smem", "ptr", "make"],
        )
    except AttributeError as exc:
        print(f"  cute.tensor NOT FOUND: {exc}")

    _hr("cute.AddressSpace enum members")
    try:
        AS = cute.AddressSpace
        members = []
        for name in dir(AS):
            if name.startswith("_"):
                continue
            try:
                val = getattr(AS, name)
                members.append((name, val))
            except Exception:
                continue
        for name, val in members[:60]:
            print(f"  AddressSpace.{name} = {val!r}")
    except AttributeError as exc:
        print(f"  cute.AddressSpace NOT FOUND: {exc}")

    _hr("cute.make_ptr / cute.make_tensor / cute.make_fragment signatures + docs")
    for name in (
        "make_ptr",
        "make_tensor",
        "make_fragment",
        "make_fragment_like",
        "make_rmem_tensor",
        "make_rmem_tensor_like",
        "make_layout",
        "make_swizzle",
        "make_copy_atom",
        "make_mma_atom",
        "make_tiled_copy",
        "make_tiled_mma",
        "gemm",
        "copy",
    ):
        _print_signature_and_doc(cute, name)
        print()

    _hr("cute.kernel / cute.jit / cute.compile signatures + docs")
    for name in ("kernel", "jit", "compile"):
        _print_signature_and_doc(cute, name)
        print()

    _hr("on-disk: walk cutlass-dsl install for examples / tutorials / docs")
    cutlass_pkg_dir = os.path.dirname(cutlass.__file__)
    cutlass_root_dir = os.path.dirname(os.path.dirname(cutlass_pkg_dir))
    print(f"  cutlass package dir:    {cutlass_pkg_dir}")
    print(f"  cutlass-dsl wheel root: {cutlass_root_dir}")
    interesting_files = []
    interesting_dirs = []
    for base in (cutlass_pkg_dir, cutlass_root_dir):
        if not os.path.exists(base):
            continue
        for root, dirs, files in os.walk(base):
            depth = root[len(base):].count(os.sep)
            if depth > 4:
                dirs[:] = []
                continue
            rl = root.lower()
            if any(tag in rl for tag in (
                "example", "tutorial", "test", "doc", "share/cutlass",
                "demo", "sample",
            )):
                interesting_dirs.append(root)
            for f in files:
                fl = f.lower()
                if not (fl.endswith(".py") or fl.endswith(".md") or fl.endswith(".rst")):
                    continue
                if any(tag in fl for tag in ("example", "tutorial", "demo", "sample", "readme", "smem", "mma", "gemm", "flash")):
                    interesting_files.append(os.path.join(root, f))
    print("\n  interesting directories:")
    if not interesting_dirs:
        print("    (none found)")
    else:
        for d in sorted(set(interesting_dirs))[:50]:
            print(f"    {d}")
    print("\n  interesting files (.py / .md / .rst):")
    if not interesting_files:
        print("    (none found)")
    else:
        for f in sorted(set(interesting_files))[:80]:
            try:
                size = os.path.getsize(f)
            except OSError:
                size = -1
            print(f"    {f}  ({size} bytes)")

    _hr("on-disk: try common nvidia-cutlass-dsl example locations")
    candidate_paths = [
        "/usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/examples",
        "/usr/local/lib/python3.12/dist-packages/nvidia_cutlass_dsl/share",
        "/usr/local/share/cutlass",
        "/usr/local/share/cutlass-dsl",
        "/opt/cutlass",
        "/opt/cutlass-dsl",
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            try:
                contents = sorted(os.listdir(p))[:20]
            except OSError:
                contents = ["<unreadable>"]
            print(f"  EXISTS: {p}")
            for c in contents:
                print(f"    .{os.sep}{c}")
        else:
            print(f"  no:     {p}")

    _hr("done")
    print("Send /tmp/dsl12x_discover_deep.txt back. The next phase will:")
    print("  1. Pick the right SMEM allocation idiom (cutlass.utils.SmemAllocator")
    print("     or make_ptr(AddressSpace.smem, dtype) + make_tensor(...)).")
    print("  2. Pick the right MMA op for SM120 BF16->F32 (likely a cute.nvgpu.warp")
    print("     SM80/SM89 m16n8k16 op since SM120 supports the SM80/89 ISA).")
    print("  3. Pick the right cp.async op variant from cute.nvgpu.cpasync.")
    print("  4. Rewrite dsl12x/hello_mma/kernel.py against the verified idiom.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
