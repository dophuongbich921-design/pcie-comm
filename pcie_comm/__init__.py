"""Standalone FlashInfer PCIe collectives with DeepJIT artifact caching."""
from .version import __version__
from .autotuner import AutoTuner, autotune
from .comm import (
    PcieIpcAllReduceWorkspace,
    PcieIpcAllGatherWorkspace,
    PcieIpcReduceScatterWorkspace,
)

__all__ = ["AutoTuner", "autotune", "PcieIpcAllReduceWorkspace",
           "PcieIpcAllGatherWorkspace", "PcieIpcReduceScatterWorkspace"]
