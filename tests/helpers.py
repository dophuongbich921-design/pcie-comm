"""FlashInfer test helpers; Copyright FlashInfer team; Apache-2.0."""
import multiprocessing as mp
import socket
import time
from typing import Any
_JOIN_TIMEOUT_S = 600

def get_open_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

def multi_process_parallel(
    world_size: int,
    target: Any,
    args: tuple = (),
    timeout_s: float = _JOIN_TIMEOUT_S,
) -> None:
    """Run ``target`` on ``world_size`` ranks, or fail with a bounded wait.

    ``timeout_s`` is lowered by the negative-control tests, which *expect* the
    deadlock and would otherwise pay the full timeout to observe it.
    """
    mp.set_start_method("spawn", force=True)
    port = get_open_port()
    procs = []
    for rank in range(world_size):
        p = mp.Process(
            target=target, args=(world_size, rank, port) + args, name=f"Worker-{rank}"
        )
        p.start()
        procs.append(p)
    # A bounded join is essential here: the failure mode this suite is built to
    # catch is a spin-wait deadlock, and an unbounded join would hang CI rather
    # than report it.
    #
    # Every exit runs the reaper, not just the timeout one. When one rank dies
    # the others are left spinning inside a collective their peer will never
    # reach, so returning without them would leak processes that hold a GPU
    # context for the rest of the session -- and the next test in the file
    # would then fail for reasons that have nothing to do with it.
    deadline = time.monotonic() + timeout_s
    try:
        for rank, p in enumerate(procs):
            p.join(timeout=max(1.0, deadline - time.monotonic()))
            if p.is_alive():
                raise AssertionError(
                    f"rank {rank} did not finish within {timeout_s}s; "
                    "the collective most likely deadlocked"
                )
            assert p.exitcode == 0, f"rank {rank} failed with exit code {p.exitcode}"
    finally:
        for p in procs:
            if p.is_alive():
                p.kill()
        for p in procs:
            p.join(timeout=30.0)
