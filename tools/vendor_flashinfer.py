"""Reproduce the pinned PCIe source extraction (no network or compiler required)."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

PIN = "595bc5746cd57efc0006ff888b4bf8144f8235b9"
ROOT = Path(__file__).resolve().parents[1]


def adapt(text):
    # Drop only the optional API logging / trace decorators. No collective or
    # autotuning control flow is changed. Use our own import/op/env namespace.
    text = re.sub(r"^from \.+api_logging import flashinfer_api\n", "", text, flags=re.M)
    text = re.sub(r"^from \.+trace\.templates\.comm import \w+\n", "", text, flags=re.M)
    text = re.sub(r"^\s*@flashinfer_api(?:\(trace=\w+\))?\n", "\n", text, flags=re.M)
    return text.replace("flashinfer.", "pcie_comm.").replace("flashinfer::", "pcie_comm::").replace("FLASHINFER_", "PCIE_COMM_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path)
    args = parser.parse_args()
    src = args.checkout.resolve()
    rev = subprocess.check_output(["git", "-C", str(src), "rev-parse", "HEAD"], text=True).strip()
    if rev != PIN:
        raise SystemExit(f"Expected {PIN}, found {rev}")
    manifest = {}

    def record(rel, dest, content=None, kind="verbatim"):
        original = (src / rel).read_bytes()
        dest.parent.mkdir(parents=True, exist_ok=True)
        output = content.encode() if content is not None else original
        dest.write_bytes(output)
        manifest[str(dest.relative_to(ROOT))] = {
            "source": str(rel), "source_sha256": hashlib.sha256(original).hexdigest(),
            "vendored_sha256": hashlib.sha256(output).hexdigest(), "adaptation": kind,
        }

    files = list((src / "flashinfer/autotuner").glob("*.py"))
    files += list((src / "flashinfer/comm").glob("pcie_ipc*.py"))
    files += list((src / "flashinfer/comm/pcie_ipc_collectives").glob("*.py"))
    files += [src / "flashinfer" / n for n in ["autotune_cache.py", "tactics_blocklist.py"]]
    files += [src / "flashinfer/comm/cuda_ipc.py"]
    for path in files:
        rel = path.relative_to(src)
        record(rel, ROOT / "pcie_comm" / path.relative_to(src / "flashinfer"), adapt(path.read_text()), "namespace; remove optional API tracing decorators")
    headers = list((src / "include/flashinfer/comm").glob("pcie_ipc*.cuh"))
    headers += [src / "include/flashinfer/layout.cuh", src / "csrc/tvm_ffi_utils.h"]
    headers += list((src / "csrc").glob("pcie_ipc*.cu"))
    for path in headers:
        rel = path.relative_to(src)
        record(rel, ROOT / "pcie_comm" / rel)
    # Only the utility functions used by the extracted tuner. Keep their bodies.
    utility = (src / "flashinfer/utils.py").read_text()
    tree = ast.parse(utility)
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {"next_positive_power_of_2", "is_confidential_compute"}:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            selected.append("\n".join(utility.splitlines()[start - 1:node.end_lineno]))
        if isinstance(node, ast.If) and "TorchVersion" in ast.unparse(node.test):
            for fn in node.orelse:
                if isinstance(fn, ast.FunctionDef) and fn.name == "register_custom_op":
                    selected.append(ast.unparse(fn))
    prefix = '"""Utilities extracted from FlashInfer; Copyright FlashInfer team.\nApache-2.0, see LICENSE."""\nimport os\nfrom functools import lru_cache\nfrom typing import Callable, Iterable, Optional, Sequence, Union\nimport torch\nfrom .jit.core import logger\n\n'
    suffix = '\n\n@lru_cache(maxsize=1)\ndef get_globaltimer_kernel():\n    """Same timestamp kernel; compile via the shared DeepJIT artifact cache."""\n    try:\n        from .jit.comm import gen_timing_module\n        return gen_timing_module().build_and_load().get_globaltimer_timestamp\n    except Exception as exc:\n        logger.warning("Global timer unavailable; falling back to CUDA events: %s", exc)\n        return None\n'
    record(Path("flashinfer/utils.py"), ROOT / "pcie_comm/utils.py", adapt(prefix + "\n\n".join(selected) + suffix), "selected helpers; globaltimer uses shared JIT adapter")
    for name in ["LICENSE", "NOTICE"]:
        record(Path(name), ROOT / name)
    cute_path = Path("flashinfer/jit/cute_dsl_core.py")
    cute_src = (src / cute_path).read_text()
    cute_fn = next(n for n in ast.parse(cute_src).body if isinstance(n, ast.FunctionDef) and n.name == "_get_cute_dsl_version")
    cute_body = "\n".join(cute_src.splitlines()[cute_fn.decorator_list[0].lineno - 1:cute_fn.end_lineno])
    record(cute_path, ROOT / "pcie_comm/jit/cute_dsl_core.py", '"""Extracted from FlashInfer; Copyright FlashInfer team; Apache-2.0."""\nimport functools\n\n' + cute_body + "\n", "extract compiler-stack fingerprint helper")
    helper_path = Path("tests/comm/test_pcie_ipc_all_reduce.py")
    helper_src = (src / helper_path).read_text()
    helper_parts = []
    for fn in ast.parse(helper_src).body:
        if isinstance(fn, ast.FunctionDef) and fn.name in {"get_open_port", "multi_process_parallel"}:
            helper_parts.append("\n".join(helper_src.splitlines()[fn.lineno-1:fn.end_lineno]))
    record(helper_path, ROOT / "tests/helpers.py", '"""FlashInfer test helpers; Copyright FlashInfer team; Apache-2.0."""\nimport multiprocessing as mp\nimport socket\nimport time\nfrom typing import Any\n_JOIN_TIMEOUT_S = 600\n\n' + "\n\n".join(helper_parts) + "\n", "extract CPU multiprocess helpers")
    for name in ["test_pcie_ipc_policy.py", "test_pcie_ipc_tuning.py", "test_pcie_ipc_admission.py"]:
        rel = Path("tests/comm") / name
        content = adapt((src / rel).read_text())
        if name == "test_pcie_ipc_policy.py":
            content = content.split("def _load_benchmark_module():")[0].rstrip() + "\n"
        content = content.replace("from tests.comm.test_pcie_ipc_all_reduce import", "from tests.helpers import")
        record(rel, ROOT / "tests/upstream" / name, content, "namespace; select library tests; extracted helper import")
    (ROOT / "UPSTREAM.json").write_text(json.dumps({
        "flashinfer": {"repository": "https://github.com/flashinfer-ai/flashinfer", "commit": PIN, "files": manifest},
        "deepep": {"repository": "https://github.com/deepseek-ai/DeepEP", "inspected_commit": "93eb6eb238127e96c6d7a4a625a6dad158348509"},
        "deepjit": {"repository": "https://github.com/deepseek-ai/DeepJIT", "commit": "cba802df081985141eeee36803685775c183a047", "integration": "Unmodified DiskCache and hash headers; shared-library builder/loader adapter"},
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
