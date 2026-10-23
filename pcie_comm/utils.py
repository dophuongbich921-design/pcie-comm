"""Utilities extracted from FlashInfer; Copyright FlashInfer team.
Apache-2.0, see LICENSE."""
import os
from functools import lru_cache
from typing import Callable, Iterable, Optional, Sequence, Union
import torch
from .jit.core import logger

def next_positive_power_of_2(x: int) -> int:
    if x < 1:
        return 1

    # Following code is equivalent to 1 << (x - 1).bit_length()
    # But this impl does not contain bit_length() so can be used by torch compile.
    # It can correctly handle 64bit number which should be enough for now.
    n = x - 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    return n + 1

def register_custom_op(name: str, fn: Optional[Callable]=None, /, *, mutates_args: Union[str, Iterable[str]], device_types: Optional[Union[str, Sequence[str]]]=None, schema: Optional[str]=None) -> Callable:
    return lambda x: x

@lru_cache(maxsize=1)
def is_confidential_compute() -> bool:
    """Whether the GPU is running in NVIDIA Confidential Computing (CC) mode.

    Detected once via NVML and cached.
    Overridable with ``PCIE_COMM_CONFIDENTIAL_COMPUTE=1/0``.
    """
    forced = os.environ.get("PCIE_COMM_CONFIDENTIAL_COMPUTE")
    if forced is not None:
        if forced not in ("0", "1"):
            raise ValueError(
                f"PCIE_COMM_CONFIDENTIAL_COMPUTE must be '0' or '1', got {forced!r}"
            )
        return forced == "1"
    if not torch.cuda.is_available():
        return False
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            state = pynvml.nvmlSystemGetConfComputeState()
            # ccFeature != 0 means CC is enabled (ON or devtools).
            return int(getattr(state, "ccFeature", 0)) != 0
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:
        logger.debug("[Flashinfer]: Confidential-compute detection failed: %r", e)
        return False

@lru_cache(maxsize=1)
def get_globaltimer_kernel():
    """Same timestamp kernel; compile via the shared DeepJIT artifact cache."""
    try:
        from .jit.comm import gen_timing_module
        return gen_timing_module().build_and_load().get_globaltimer_timestamp
    except Exception as exc:
        logger.warning("Global timer unavailable; falling back to CUDA events: %s", exc)
        return None
