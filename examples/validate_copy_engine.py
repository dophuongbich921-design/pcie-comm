"""Explicit Copy Engine coverage, independent of which variant wins tuning.

Run with torchrun on 4 or 8 ranks. Eight ranks require ordered 4+4 PCIe islands.
Uses public explicit-config APIs; it does not change the tuning protocol.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from pcie_comm import PcieIpcAllReduceWorkspace, PcieIpcAllGatherWorkspace
from pcie_comm.comm.pcie_ipc_policy import IpcLaunchConfig, IpcVariant
from pcie_comm.comm.pcie_ipc_collectives.all_gather_policy import (
    PcieIpcAllGatherLaunchConfig, PcieIpcAllGatherVariant,
)
from pcie_comm.jit.core import cache_stats


def check(execute, inp, expected, config):
    out = torch.empty_like(expected)
    for _ in range(3):
        execute(inp, out=out, config=config)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        execute(inp, out=out, config=config)
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    del graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    if os.getuid() == 0:
        raise SystemExit("Validation must run as a non-root user.")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world not in (4, 8):
        raise ValueError("Copy Engine validation requires 4 or 8 ranks")
    args.results.mkdir(parents=True, exist_ok=True)
    records = []
    sizes = (1024, 262144)
    try:
        for dtype in (torch.bfloat16, torch.float16):
            with PcieIpcAllReduceWorkspace(
                dist.group.WORLD, max_numel=max(sizes), dtype=dtype,
                max_blocks=4, tune_cache=str(args.results / "unused-ar.json"),
            ) as ws:
                variants = [IpcVariant.COPY_ENGINE_RING]
                for size in sizes:
                    inp = torch.arange(size, device="cuda").remainder(8).add(rank).to(dtype).reshape(1, size)
                    expected = inp.clone()
                    dist.all_reduce(expected)
                    for variant in variants:
                        for pieces in (1, 2, 4):
                            config = IpcLaunchConfig(pieces, 256, variant)
                            check(ws.all_reduce, inp, expected, config)
                            ws.rebind_stream()
                            records.append({"op": "all_reduce", "dtype": str(dtype),
                                            "numel": size, "variant": variant.name,
                                            "pieces": pieces, "correct": True,
                                            "graph_replays": 10})
                print(f"rank={rank} Copy Engine AR {dtype}: passed", flush=True)
        if world == 8:
            for dtype in (torch.bfloat16, torch.float16, torch.float32):
                with PcieIpcAllGatherWorkspace(
                    dist.group.WORLD, max_numel=max(sizes), dtype=dtype,
                    max_blocks=4, tune_cache=str(args.results / "unused-ag.json"),
                ) as ws:
                    assert ws.ordered_4plus4
                    config = PcieIpcAllGatherLaunchConfig(1, 32, PcieIpcAllGatherVariant.COPY_ENGINE)
                    for size in sizes:
                        inp = torch.arange(size, device="cuda").remainder(8).add(rank).to(dtype).reshape(1, size)
                        expected = torch.empty(world, size, dtype=dtype, device="cuda")
                        dist.all_gather_into_tensor(expected, inp)
                        check(ws.all_gather, inp, expected, config)
                        ws.rebind_stream()
                        records.append({"op": "all_gather", "dtype": str(dtype),
                                        "numel": size, "variant": "COPY_ENGINE",
                                        "correct": True, "graph_replays": 10})
                print(f"rank={rank} Copy Engine AG {dtype}: passed", flush=True)
        report = {"rank": rank, "uid": os.getuid(), "world_size": world,
                  "gpu": torch.cuda.get_device_name(), "stats": cache_stats(),
                  "records": records}
        (args.results / f"ce-rank{rank}.json").write_text(json.dumps(report, indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
