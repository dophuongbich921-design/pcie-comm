"""Run with torchrun. Cold tunes/saves; a separate warm run forbids compilation."""
import argparse
import json
import os
from pathlib import Path
from datetime import timedelta

import torch
import torch.distributed as dist

from pcie_comm import (
    PcieIpcAllReduceWorkspace, PcieIpcAllGatherWorkspace, PcieIpcReduceScatterWorkspace,
)
from pcie_comm.autotuner import AutoTuner
from pcie_comm.comm.pcie_ipc_tuning import PCIE_IPC_CUSTOM_OP, pcie_ipc_tuning_config
from pcie_comm.jit.core import cache_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("cold", "warm"), required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    if os.getuid() == 0:
        raise SystemExit("Validation must run as a non-root user.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required; this is not a CPU simulation.")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=15))
    world = dist.get_world_size()
    if world not in (2, 4, 8):
        raise ValueError("FlashInfer PCIe kernels support 2, 4 or 8 ranks")
    args.results.mkdir(parents=True, exist_ok=True)
    if args.phase == "warm":
        import tvm_ffi.cpp

        def forbidden(*args, **kwargs):
            raise AssertionError("warm restart unexpectedly invoked compiler")

        tvm_ffi.cpp.build = forbidden
    records = []
    classes = [
        ("all_reduce", PcieIpcAllReduceWorkspace),
        ("all_gather", PcieIpcAllGatherWorkspace),
        ("reduce_scatter", PcieIpcReduceScatterWorkspace),
    ]
    try:
        for op, workspace_type in classes:
            dtypes = (torch.bfloat16, torch.float16) if op == "all_reduce" else (
                torch.bfloat16, torch.float16, torch.float32)
            for dtype in dtypes:
                print(f"[{args.phase}] rank={rank} {op} {dtype}: start", flush=True)
                hidden = 1024
                cache = str(args.results / f"{op}-{str(dtype).split('.')[-1]}.json")
                with workspace_type(
                    dist.group.WORLD, max_numel=hidden, dtype=dtype,
                    max_blocks=4, tune_batches=(1,), tune_cache=cache,
                ) as ws:
                    if args.phase == "cold":
                        if op == "all_reduce":
                            ws.tune([hidden], dtype=dtype)
                        else:
                            ws.tune([hidden])
                    rows = world if op == "reduce_scatter" else 1
                    inp = (torch.arange(rows * hidden, device="cuda").reshape(rows, hidden)
                           .remainder(8).add(rank).to(dtype))
                    if op == "all_reduce":
                        hit = AutoTuner.get().search_cache(
                            PCIE_IPC_CUSTOM_OP, [ws._runner], (tuple(inp.shape),),
                            pcie_ipc_tuning_config((1,)), inputs=[inp],
                        )[0]
                        assert hit, "missing persisted AllReduce config"
                        expected = inp.clone()
                        dist.all_reduce(expected)
                        ws.prepare([(1, hidden)], dtype=dtype)
                    elif op == "all_gather":
                        assert ws._tuning_state._cache_key(inp) in ws._tuning_state._entries
                        expected = torch.empty(world, hidden, dtype=dtype, device="cuda")
                        dist.all_gather_into_tensor(expected, inp)
                    else:
                        assert ws._tuning_state._cache_key(inp) in ws._tuning_state._entries
                        expected = torch.empty(1, hidden, dtype=dtype, device="cuda")
                        dist.reduce_scatter_tensor(expected, inp)
                    execute = getattr(ws, op)
                    out = torch.empty_like(expected)
                    config = ws.tuned_launch_config(inp)
                    for _ in range(3):
                        execute(inp, out=out)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(out, expected, rtol=0, atol=0)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        execute(inp, out=out)
                    for _ in range(10):
                        graph.replay()
                    torch.cuda.synchronize()
                    ws.rebind_stream()
                    torch.testing.assert_close(out, expected, rtol=0, atol=0)
                    del graph
                    records.append({
                        "op": op, "dtype": str(dtype),
                        "config": [int(config.variant), config.blocks, config.threads],
                        "correct": True, "graph_replays": 10,
                    })
                    print(f"[{args.phase}] rank={rank} {op} {dtype}: passed {config}", flush=True)
        stats = cache_stats()
        if args.phase == "warm":
            assert stats["builds"] == 0
            assert stats["disk_hits"] >= 2
            cold = json.loads((args.results / f"cold-rank{rank}.json").read_text())
            assert records == cold["records"], "restart selected different configs"
        report = {
            "phase": args.phase, "rank": rank, "uid": os.getuid(), "world_size": world,
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "stats": stats, "records": records,
        }
        (args.results / f"{args.phase}-rank{rank}.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
