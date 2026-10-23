"""Controlled-latency test of upstream AutoTuner selection/persistence.

The timer is replaced in this CPU test; it makes no GPU performance claim.
"""
import json
import sys
import torch
from pcie_comm.autotuner import AutoTuner, TunableRunner, TuningConfig


class ProbeRunner(TunableRunner):
    def get_valid_tactics(self, inputs, profile):
        return [0, 1, 2]

    def get_cache_key_extras(self, inputs):
        return (str(inputs[0].dtype), "probe-v1")

    def forward(self, inputs, tactic=-1):
        return inputs[0]


path, mode = sys.argv[1:]
tuner = AutoTuner()
torch.cuda.is_current_stream_capturing = lambda: False
measured = []


def controlled_timer(runner, tensors, tactic, config, **kwargs):
    if mode == "warm":
        raise AssertionError("persisted winner must not be profiled")
    measured.append(tactic)
    return [3.0, 1.0, 2.0][tactic]


tuner._profile_single_kernel = controlled_timer
runner = ProbeRunner()
config = TuningConfig()
inputs = [torch.ones(1, 1024)]
if mode == "cold":
    tuner.is_tuning_mode = True
else:
    assert tuner.load_configs(path)
_, tactic = tuner.choose_one("pcie_comm::probe", [runner], config, inputs)
assert tactic == 1
if mode == "cold":
    assert measured == [0, 1, 2]
    tuner.save_configs(path)
else:
    assert not measured
before = list(measured)
assert tuner.choose_one("pcie_comm::probe", [runner], config, inputs)[1] == 1
assert measured == before
print(json.dumps({"tactic": tactic, "measurements": len(measured)}))
