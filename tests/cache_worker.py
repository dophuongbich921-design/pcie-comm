"""Fresh-process probe used by the artifact-cache tests."""
import ctypes
import json
from pathlib import Path
import subprocess
import sys

from pcie_comm.jit.core import cache_stats, get_or_build_artifact


root, signature, mode = sys.argv[1:]


def build(directory):
    if mode == "forbid-build":
        raise AssertionError("cache hit invoked the compiler")
    source = directory / "answer.cpp"
    source.write_text('extern "C" int answer() { return 42; }\n')
    subprocess.run(
        ["c++", "-shared", "-fPIC", str(source), "-o", str(directory / "module.so")],
        check=True,
    )


path = get_or_build_artifact("cpu_probe", {"version": signature}, build, roots=[root])
assert ctypes.CDLL(str(path)).answer() == 42
print(json.dumps({"path": str(path), "stats": cache_stats()}))
