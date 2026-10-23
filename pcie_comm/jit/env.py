import os
from pathlib import Path

PCIE_COMM_WORKSPACE_DIR = Path(
    os.environ.get("PCIE_COMM_CACHE_DIR", Path.home() / ".cache" / "pcie_comm")
).expanduser()
