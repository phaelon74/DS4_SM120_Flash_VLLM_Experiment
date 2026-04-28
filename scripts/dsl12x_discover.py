#!/usr/bin/env python3
"""dsl12x cutlass.cute API surface discovery probe.

This is a READ-ONLY introspection script. It does NOT compile a kernel and
does NOT touch GPU memory beyond a basic CUDA availability check. Its job is
to tell us EXACTLY which cutlass.cute APIs are available in the installed
nvidia-cutlass-dsl-libs-cu13 version, so the next phase can call the right
names instead of guessing.

The output is structured (sectioned, grep-friendly). Pipe it to a file and
send the file back:

    python3 scripts/dsl12x_discover.py > dsl12x_discover.txt 2>&1

Sections it prints:

    [SECTION env]           - Python, torch, cuda, cutlass, cute versions
    [SECTION top]           - cutlass top-level attrs
    [SECTION cute_top]      - cutlass.cute top-level attrs
    [SECTION cute_mma]      - cute.MmaAtom / cute.SM* enums (this is the key one)
    [SECTION cute_arch]     - architecture-specific submodules
    [SECTION cute_compile]  - cute.compile / cute.kernel signatures
    [SECTION cute_smem]     - shared memory primitives
    [SECTION cute_copy]     - cp.async / TMA copy primitives
    [SECTION cute_layouts]  - layout / swizzle helpers
    [SECTION cute_math]     - exp2 / fast-math / online-softmax-friendly ops
    [SECTION cute_arith]    - arithmetic types (FP8, BF16, TF32 wrappers)

If any section is missing or empty, that tells me a particular feature is
not in this version and I need to fall back to inline PTX or a different
abstraction.

Exit codes:
    0 = probe completed (even if some sections are empty).
    3 = environment error (no torch / no cutlass / etc.).
"""

from __future__ import annotations

import sys
import inspect


def _hr(title: str) -> None:
    print(f"\n[SECTION {title}]")
    print("-" * (len(title) + 12))


def _safe_attrs(obj, prefix_filter: str | None = None) -> list[str]:
    out = []
    for name in sorted(dir(obj)):
        if name.startswith("_"):
            continue
        if prefix_filter is not None and not name.startswith(prefix_filter):
            continue
        out.append(name)
    return out


def _print_attrs(obj, prefix_filter: str | None = None, max_items: int = 200) -> None:
    attrs = _safe_attrs(obj, prefix_filter=prefix_filter)
    if not attrs:
        print("  (none)")
        return
    for name in attrs[:max_items]:
        try:
            val = getattr(obj, name)
            kind = type(val).__name__
        except Exception as exc:
            kind = f"<getattr error: {type(exc).__name__}>"
        print(f"  {name}  ({kind})")
    if len(attrs) > max_items:
        print(f"  ... ({len(attrs) - max_items} more elided)")


def _print_signature(obj, name: str) -> None:
    try:
        fn = getattr(obj, name)
        sig = inspect.signature(fn)
        print(f"  {name}{sig}")
    except (TypeError, ValueError):
        print(f"  {name}(<no inspectable signature>)")
    except AttributeError:
        print(f"  {name} -> NOT FOUND")


def main() -> int:
    _hr("env")
    print(f"  python: {sys.version.split()[0]}")
    try:
        import torch
        print(f"  torch: {torch.__version__}")
        print(f"  torch.cuda.is_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            name = torch.cuda.get_device_name(0)
            print(f"  device 0: {name}  cc={cap[0]}.{cap[1]}")
            try:
                print(f"  cuda runtime: {torch.version.cuda}")
            except Exception:
                pass
    except ImportError:
        print("  torch: NOT INSTALLED")
        return 3

    try:
        import cutlass
        print(f"  cutlass: {getattr(cutlass, '__version__', '<no __version__>')}")
        print(f"  cutlass module: {cutlass.__file__}")
    except ImportError as exc:
        print(f"  cutlass: NOT IMPORTABLE ({exc})")
        return 3

    try:
        import cutlass.cute as cute
        print(f"  cutlass.cute: importable, module path: {cute.__file__}")
    except ImportError as exc:
        print(f"  cutlass.cute: NOT IMPORTABLE ({exc})")
        return 3

    _hr("top")
    print("All public attrs on `cutlass`:")
    _print_attrs(cutlass)

    _hr("cute_top")
    print("All public attrs on `cutlass.cute`:")
    _print_attrs(cute)

    _hr("cute_mma")
    print("Looking for MMA atoms (the key thing for the prefill kernel inner):")
    print("  -> any of: MmaAtom, SM80_*, SM89_*, SM100_*, SM120_*, mma, mma_atom")
    candidates = []
    for name in dir(cute):
        if name.startswith("_"):
            continue
        if (
            "mma" in name.lower()
            or "Mma" in name
            or name.startswith("SM")
            or name.startswith("Sm")
        ):
            candidates.append(name)
    if not candidates:
        print("  (no obvious MMA-related top-level names; try cute.arch instead)")
    else:
        for name in sorted(candidates):
            try:
                val = getattr(cute, name)
                if inspect.isclass(val):
                    members = []
                    for m in dir(val):
                        if m.startswith("_"):
                            continue
                        members.append(m)
                    members_str = ", ".join(members[:20])
                    if len(members) > 20:
                        members_str += f", ... (+{len(members) - 20} more)"
                    print(f"  cute.{name} (class)  members: {members_str}")
                else:
                    print(f"  cute.{name}  ({type(val).__name__})")
            except Exception as exc:
                print(f"  cute.{name}  ({type(exc).__name__}: {exc})")

    _hr("cute_arch")
    print("Architecture submodules (cute.arch.*, cute.sm120, etc.):")
    arch_submods = []
    for name in dir(cute):
        if name.startswith("_"):
            continue
        try:
            val = getattr(cute, name)
            if inspect.ismodule(val):
                arch_submods.append(name)
        except Exception:
            continue
    for name in arch_submods:
        try:
            val = getattr(cute, name)
            inner = [m for m in dir(val) if not m.startswith("_")]
            print(f"  cute.{name}  -> {len(inner)} public attrs")
            for m in inner[:30]:
                print(f"    .{m}")
            if len(inner) > 30:
                print(f"    ... (+{len(inner) - 30} more)")
        except Exception as exc:
            print(f"  cute.{name}  ({type(exc).__name__}: {exc})")
    if not arch_submods:
        print("  (no public submodules; check cute.arch via direct import)")
        for arch_path in ("cutlass.cute.arch", "cutlass.cute.nvgpu", "cutlass.cute.atom"):
            try:
                mod = __import__(arch_path, fromlist=["*"])
                inner = [m for m in dir(mod) if not m.startswith("_")]
                print(f"  {arch_path}  -> {len(inner)} public attrs")
                for m in inner[:30]:
                    print(f"    .{m}")
                if len(inner) > 30:
                    print(f"    ... (+{len(inner) - 30} more)")
            except ImportError as exc:
                print(f"  {arch_path}  NOT IMPORTABLE ({exc})")

    _hr("cute_compile")
    print("Looking for kernel / host JIT compile entry points:")
    for name in ("kernel", "compile", "jit", "jit_compile", "make_kernel"):
        if hasattr(cute, name):
            _print_signature(cute, name)
        else:
            print(f"  {name} -> NOT FOUND on cute")

    _hr("cute_smem")
    print("Shared memory builders / addressing:")
    for name in (
        "make_shared_memory",
        "shared_memory",
        "smem",
        "make_smem_buffer",
        "make_tensor",
        "make_layout",
        "Swizzle",
        "swizzle",
        "tile_to_shape",
    ):
        if hasattr(cute, name):
            _print_signature(cute, name)

    _hr("cute_copy")
    print("Copy primitives (cp.async / TMA):")
    for name in (
        "copy",
        "cp_async",
        "async_copy",
        "make_copy_atom",
        "CopyAtom",
        "Copy",
        "TmaDescriptor",
        "make_tma_copy",
    ):
        if hasattr(cute, name):
            _print_signature(cute, name)

    _hr("cute_layouts")
    print("Layout / swizzle helpers:")
    for name in (
        "Layout",
        "make_layout",
        "make_shape",
        "make_stride",
        "Shape",
        "Stride",
        "Swizzle",
        "ComposedLayout",
        "tile_to_shape",
        "logical_divide",
        "blocked_product",
        "raked_product",
    ):
        if hasattr(cute, name):
            print(f"  cute.{name} present ({type(getattr(cute, name)).__name__})")

    _hr("cute_math")
    print("Math / online-softmax friendly ops:")
    candidates = []
    for name in dir(cute):
        if name.startswith("_"):
            continue
        lname = name.lower()
        if any(tag in lname for tag in ("exp", "log", "max", "sum", "softmax", "rsqrt", "fast")):
            candidates.append(name)
    if not candidates:
        print("  (no obvious math/softmax names at top level; check cute.arch)")
    for name in sorted(candidates)[:60]:
        print(f"  cute.{name}  ({type(getattr(cute, name)).__name__})")

    _hr("cute_arith")
    print("Arithmetic / dtype wrappers (FP8, BF16, TF32):")
    candidates = []
    for name in dir(cute):
        if name.startswith("_"):
            continue
        lname = name.lower()
        if any(tag in lname for tag in ("fp8", "fp4", "bf16", "tf32", "f32", "e4m3", "e5m2", "dtype", "dtype_t")):
            candidates.append(name)
    if not candidates:
        print("  (no obvious dtype wrappers at top level)")
    for name in sorted(candidates)[:60]:
        try:
            val = getattr(cute, name)
            print(f"  cute.{name}  ({type(val).__name__})")
        except Exception as exc:
            print(f"  cute.{name}  ({type(exc).__name__}: {exc})")

    _hr("cute_examples_path")
    print("Where to look on disk for cutlass.cute examples / tests / docs:")
    try:
        import os
        cute_dir = os.path.dirname(cute.__file__)
        print(f"  cute.__file__ dir: {cute_dir}")
        for sub in ("examples", "test", "tests"):
            cand = os.path.join(cute_dir, sub)
            if os.path.exists(cand):
                print(f"  found: {cand}")
            else:
                print(f"  no:    {cand}")
        site_pkg = os.path.dirname(os.path.dirname(cute_dir))
        for sub in ("examples", "test", "tests", "share/cutlass"):
            cand = os.path.join(site_pkg, sub)
            if os.path.exists(cand):
                print(f"  also:  {cand}")
    except Exception as exc:
        print(f"  (could not introspect: {type(exc).__name__}: {exc})")

    _hr("done")
    print("Discovery complete. Send the full output back so the next phase can")
    print("write the MMA inner against the verified API surface.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
