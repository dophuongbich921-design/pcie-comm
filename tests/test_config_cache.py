import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from pcie_comm.comm.pcie_ipc_collectives._ag_rs_tuning import (
    PcieIpcCollectiveTuningState, config_to_tactic,
)
from pcie_comm.comm.pcie_ipc_collectives import all_gather, reduce_scatter


def test_autotuner_selects_minimum_and_reloads_after_restart(tmp_path):
    path = str(tmp_path / "autotune.json")
    results = []
    for mode in ("cold", "warm"):
        result = subprocess.run(
            [sys.executable, "-m", "tests.config_worker", path, mode],
            text=True, capture_output=True, timeout=60, check=True,
        )
        results.append(json.loads(result.stdout.strip().splitlines()[-1]))
    assert results == [{"tactic": 1, "measurements": 3}, {"tactic": 1, "measurements": 0}]


@pytest.mark.parametrize("module", [all_gather, reduce_scatter])
def test_collective_config_persists_and_placement_invalidates(tmp_path, module, monkeypatch):
    # CPU workspace stand-in: validate the original persistence/lookup code.
    spec = module._TUNING_SPEC
    seed = spec.config_type(blocks=1, threads=128, variant=spec.grid_variants[0])
    winner = spec.config_type(blocks=2, threads=128, variant=spec.grid_variants[0])
    ws = SimpleNamespace(
        rank=0, world_size=4, dtype=torch.bfloat16, element_size=2,
        ordered_4plus4=False, placement_fingerprint="placement-a",
        max_blocks=4, max_numel=1024,
        launch_config=lambda _: seed, _joint_check=lambda *a, **kw: None,
    )
    path = str(tmp_path / "collective.json")
    inp = torch.empty(1, 1024, dtype=ws.dtype)
    state = PcieIpcCollectiveTuningState(ws, spec, tune_batches=(1,), tune_cache=path)
    key = state._cache_key(inp)
    state._dirty_entries[key] = list(config_to_tactic(winner))
    state._save(path)
    restored = PcieIpcCollectiveTuningState(ws, spec, tune_batches=(1,), tune_cache=path)
    assert restored.tuned_launch_config(inp) == winner

    def forbidden(*_):
        raise AssertionError("resolved configuration must not reread disk")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    restored._entries.clear()
    assert restored.tuned_launch_config(inp) == winner
    monkeypatch.undo()
    ws.placement_fingerprint = "placement-b"
    changed = PcieIpcCollectiveTuningState(ws, spec, tune_batches=(1,), tune_cache=path)
    assert changed.tuned_launch_config(inp) == seed
