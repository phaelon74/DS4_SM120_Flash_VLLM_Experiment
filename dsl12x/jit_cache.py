"""dsl12x.jit_cache: LRU cache for JIT-compiled @cute.kernel host launchers.

CuTe DSL kernels JIT-compile on first call per (kernel_function, type
signature, constexpr arg values) tuple. The compile cost is 1-3 seconds for
attention-shaped kernels. For chunked-prefill that calls the kernel ~64
times with the same shape per 16k prompt, the overhead would be 1-3s on the
first chunk and 0 on subsequent chunks -- if we cache the compiled launcher.

Without a cache, cute.compile is called on every host invocation, which
defeats the point of JIT (and adds ~3s per request).

The cache key is the tuple of:
    1. The kernel function itself (its Python id; different @cute.kernel
       decorated functions get separate entries).
    2. A "metadata key" derived from input tensor types, shapes (the static
       parts), and any constexpr arguments. The dynamic shape parts (e.g.
       sequence length) are passed at launch time as cute.Int parameters
       and do NOT participate in the cache key.

The cache uses functools.lru_cache semantics with a configurable max size.
b12x's _EAGER_HOST_LAUNCHER_CACHE_SIZE is 32; we mirror that default.

Usage from a host wrapper:

    from dsl12x.jit_cache import run_cached_host_launcher

    def my_host_wrapper(q, k, v, ...):
        run_cached_host_launcher(
            my_at_cute_kernel,
            metadata_key=(q.dtype, k.dtype, q.shape[-1], v.shape[-1]),
            launch_args=(q, k, v, ...),
        )

Note that this file does NOT import cutlass.cute at module level -- the
imports happen inside the cached call. This lets dsl12x be importable
even on systems without cutlass-dsl installed (the kernels themselves
will then fail at first call, not at import).
"""

from __future__ import annotations

import functools
import logging
import threading
from typing import Any, Callable, Hashable, Tuple

logger = logging.getLogger(__name__)

# Mirrors b12x._EAGER_HOST_LAUNCHER_CACHE_SIZE = 32.
DEFAULT_CACHE_SIZE = 32

# The cache is keyed by (kernel_function_id, metadata_key). We store the
# compiled launcher (cute.compile output) which is callable as a regular
# function with the runtime-shape arguments.
_launcher_cache: dict[Tuple[int, Hashable], Any] = {}
_cache_lock = threading.Lock()
_cache_max_size = DEFAULT_CACHE_SIZE
_cache_access_order: list[Tuple[int, Hashable]] = []


def set_cache_size(new_size: int) -> None:
    """Set the LRU cache max size. Default 32. Call before any kernel runs."""
    global _cache_max_size
    if new_size < 1:
        raise ValueError(f"cache size must be >= 1, got {new_size}")
    _cache_max_size = new_size


def cache_clear() -> None:
    """Clear all cached launchers. Used by tests to start with a cold cache."""
    with _cache_lock:
        _launcher_cache.clear()
        _cache_access_order.clear()


def cache_info() -> dict:
    """Return a dict describing the current cache state (for diagnostics)."""
    with _cache_lock:
        return {
            "size": len(_launcher_cache),
            "max_size": _cache_max_size,
            "keys": list(_launcher_cache.keys()),
        }


def _evict_if_needed() -> None:
    """Evict oldest entries until size <= max. Caller holds _cache_lock."""
    while len(_launcher_cache) > _cache_max_size and _cache_access_order:
        oldest = _cache_access_order.pop(0)
        _launcher_cache.pop(oldest, None)


def _touch(key: Tuple[int, Hashable]) -> None:
    """Mark a key as most recently used. Caller holds _cache_lock."""
    try:
        _cache_access_order.remove(key)
    except ValueError:
        pass
    _cache_access_order.append(key)


def get_or_compile(
    kernel_fn: Callable,
    metadata_key: Hashable,
    compile_fn: Callable[[], Any],
) -> Any:
    """Look up or compile a kernel launcher.

    Args:
        kernel_fn: The @cute.kernel decorated function. Used to derive the
            cache key (its Python id is unique per kernel definition).
        metadata_key: Hashable summary of all static (shape/dtype/constexpr)
            arguments. Two calls with the same metadata_key reuse the same
            compiled launcher.
        compile_fn: Zero-arg callable that returns a compiled launcher when
            called. Invoked exactly once per cache miss.

    Returns:
        The compiled launcher (callable).

    The function is structured this way (rather than as a decorator) so the
    metadata_key construction stays explicit at the call site -- the right
    cache key for a sparse MLA prefill call differs from a decode call,
    and bugs in the key construction can lead to either silent
    over-compilation (key too coarse) or silent wrong output (key too
    coarse and we reuse a launcher built for a different shape).
    """
    cache_key = (id(kernel_fn), metadata_key)

    with _cache_lock:
        cached = _launcher_cache.get(cache_key)
        if cached is not None:
            _touch(cache_key)
            return cached

    # Compile outside the lock -- it's slow (1-3 s) and we don't want to
    # serialize all kernel launches on the cache lock.
    launcher = compile_fn()

    with _cache_lock:
        # Re-check in case another thread compiled it concurrently.
        existing = _launcher_cache.get(cache_key)
        if existing is not None:
            _touch(cache_key)
            return existing
        _launcher_cache[cache_key] = launcher
        _touch(cache_key)
        _evict_if_needed()

    logger.info(
        "dsl12x.jit_cache: compiled launcher for %s metadata=%r (cache size=%d/%d)",
        getattr(kernel_fn, "__name__", repr(kernel_fn)),
        metadata_key,
        len(_launcher_cache),
        _cache_max_size,
    )

    return launcher


@functools.lru_cache(maxsize=128)
def tensor_meta_key(
    shape: Tuple[int, ...],
    stride: Tuple[int, ...] | None,
    dtype_str: str,
) -> Tuple:
    """Build a tensor metadata key suitable for use in metadata_key tuples.

    Args:
        shape: Static shape components (use -1 or None for dynamic dims so
            the key is shape-agnostic on those dims).
        stride: Static strides, or None to ignore stride in the key.
        dtype_str: ``str(tensor.dtype)`` (e.g., "torch.float8_e4m3fn").

    Returns:
        A hashable tuple suitable to combine with other metadata into a
        cache key.
    """
    return (shape, stride, dtype_str)
