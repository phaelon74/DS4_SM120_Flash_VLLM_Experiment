"""dsl12x.runtime: Stream + hardware-info helpers shared across kernels.

This module wraps a few cuda-bindings calls (current stream, SM count, device
properties) so that the rest of dsl12x does not have to repeat the same
boilerplate. The cache decorators avoid re-querying the driver on every kernel
launch -- a hot path that adds up when chunked-prefill calls the prefill
kernel 64 times for a 16k prompt.

Mirrors b12x's small public surface (current_cuda_stream, get_num_sm,
get_hardware_info) but does not import from b12x; the implementations are
re-derived from cuda.bindings.driver and cutlass.utils. The functions are
intentionally not @cute.kernel decorated -- they run on the host before any
JIT-compiled kernel launches.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import cuda.bindings.driver as cuda_driver
    import cutlass.utils


@torch._dynamo.disable
def current_cuda_stream():
    """Return the current Torch CUDA stream as a CUDA driver stream handle.

    The cutlass.cute compiled launchers expect a ``cuda.CUstream`` (the
    driver-level stream handle), not a ``torch.cuda.Stream``. PyTorch exposes
    the underlying driver handle as ``.cuda_stream`` on its Stream object.

    @torch._dynamo.disable prevents Dynamo from tracing through this function
    when called from a torch.compile graph (the cuda-bindings types are
    Dynamo-opaque and would force a graph break with a confusing message).
    """
    import cuda.bindings.driver as cuda_driver

    return cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)


@functools.lru_cache(maxsize=8)
def get_num_sm(device_index: int) -> int:
    """Return the streaming multiprocessor count for the given CUDA device.

    Cached per device index so repeat calls do not re-query the driver.
    RTX PRO 6000 reports 188 SMs; the kernel grid uses this to size launches
    that should exactly fill the GPU (e.g., persistent kernels).
    """
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@functools.lru_cache(maxsize=8)
def get_smem_per_sm(device_index: int) -> int:
    """Return the maximum SMEM per SM in bytes (with carveout) for the device.

    Used by traits to compute occupancy: per-CTA SMEM budget = SMEM/SM /
    target_ctas_per_sm. SM120 reports 100 KB SMEM/SM with the default
    carveout; b12x targets ~9-22 KB per CTA for ~4-11 CTAs/SM.
    """
    props = torch.cuda.get_device_properties(device_index)
    return props.shared_memory_per_block_optin


_hardware_info_cache: "cutlass.utils.HardwareInfo | None" = None


def get_hardware_info():
    """Get cached cutlass HardwareInfo singleton.

    HardwareInfo queries CUDA device capabilities, which is expensive (~ms).
    Cached singleton avoids repeated queries on every kernel launch.
    """
    global _hardware_info_cache
    if _hardware_info_cache is None:
        import cutlass.utils

        _hardware_info_cache = cutlass.utils.HardwareInfo()
    return _hardware_info_cache


@functools.lru_cache(maxsize=64)
def get_max_active_clusters(cluster_size: int) -> int:
    """Get max active clusters for a given cluster size.

    Cluster size is the product of cluster_shape_mn dimensions. SM120 supports
    cluster sizes up to 16 in the standard mode. dsl12x does not currently
    use clusters > 1 (we run single-CTA tiles), so this is mostly a forward
    compatibility hook for future MoE-style kernels.
    """
    return get_hardware_info().get_max_active_clusters(cluster_size)


def is_sm120() -> bool:
    """True if the current device is SM120 (Blackwell workstation, RTX PRO 6000).

    dsl12x kernels are SM120-specific. The tile geometry, MMA instructions,
    and SMEM budget are all tuned for SM120's 100 KB SMEM/SM and 188 SMs.
    Calling on a non-SM120 device should fail loudly rather than silently
    producing wrong results.
    """
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] == 12


def assert_sm120(operation: str) -> None:
    """Raise RuntimeError if the current device is not SM120.

    Helper for kernel host wrappers; gives a single clear error rather than
    a downstream cutlass MLIR failure that is hard to diagnose.
    """
    if not is_sm120():
        cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (-1, -1)
        raise RuntimeError(
            f"dsl12x.{operation} requires SM120 (Blackwell workstation), "
            f"got compute capability {cap[0]}.{cap[1]}. dsl12x kernels are "
            f"tuned for SM120's tile geometry and MMA instructions and will "
            f"not run on other architectures."
        )
