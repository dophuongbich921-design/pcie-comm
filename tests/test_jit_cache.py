import ctypes
import json
from pathlib import Path
import subprocess
import sys

import pytest

from pcie_comm import _deepjit_cache
from pcie_comm.jit import core


def worker(root, signature, mode="build"):
    result = subprocess.run(
        [sys.executable, "-m", "tests.cache_worker", str(root), signature, mode],
        text=True, capture_output=True, check=True, timeout=60,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_real_binary_survives_process_restart_without_build(tmp_path):
    cold = worker(tmp_path, "v1")
    warm = worker(tmp_path, "v1", "forbid-build")
    assert cold["path"] == warm["path"]
    assert cold["stats"]["builds"] == 1
    assert warm["stats"]["builds"] == 0
    assert warm["stats"]["disk_hits"] == 1
    changed = worker(tmp_path, "v2")
    assert changed["path"] != cold["path"]
    assert changed["stats"]["builds"] == 1


@pytest.mark.parametrize("failure", ["exception", "missing", "empty"])
def test_failed_build_is_not_published_and_can_retry(tmp_path, failure):
    cache = _deepjit_cache.ArtifactCache([str(tmp_path)])

    def fail(directory):
        if failure == "exception":
            (Path(directory) / "module.so").write_bytes(b"partial")
            raise RuntimeError("compiler failed")
        if failure == "empty":
            (Path(directory) / "module.so").touch()

    with pytest.raises(RuntimeError):
        cache.get_or_build("failure_probe", "s", fail)
    assert not list(tmp_path.glob("cache/*/.committed"))
    assert not list((tmp_path / "tmp").iterdir())

    def succeed(directory):
        (Path(directory) / "module.so").write_bytes(b"complete")

    artifact, _, hit = cache.get_or_build("failure_probe", "s", succeed)
    assert not hit
    assert Path(artifact).read_bytes() == b"complete"


def test_secondary_cache_root_is_read_without_copy_or_build(tmp_path):
    published = tmp_path / "published"
    local = tmp_path / "local"
    cold = worker(published, "v1")

    def forbidden(_):
        raise AssertionError("must reuse secondary root")

    result = core.get_or_build_artifact(
        "cpu_probe", {"version": "v1"}, forbidden, roots=[str(local), str(published)]
    )
    assert str(result) == cold["path"]
    assert not local.exists()


def test_source_headers_flags_and_architecture_change_signature(tmp_path):
    source = tmp_path / "kernel.cu"
    header = tmp_path / "kernel.cuh"
    source.write_text("kernel-v1")
    header.write_text("header-v1")

    def signature(flags=("-O3",), arch="sm_90"):
        return _deepjit_cache.digest(core.canonical_signature({
            "files": core.file_fingerprints([source, header], tmp_path),
            "flags": flags, "target": arch,
        }))

    initial = signature()
    assert initial != signature(flags=("-O0",))
    assert initial != signature(arch="sm_120")
    header.write_text("header-v2")
    assert signature() != initial
    header.write_text("header-v1")
    source.write_text("kernel-v2")
    assert signature() != initial


def test_tvm_ffi_module_is_built_and_loaded_once(tmp_path, monkeypatch):
    """Exercise the same builder/loader used for CUDA, with a real CPU module."""
    package = tmp_path / "package"
    (package / "csrc").mkdir(parents=True)
    (package / "include").mkdir()
    (package / "csrc/probe.cpp").write_text(
        '#include <tvm/ffi/function.h>\n'
        'int answer() { return 42; }\n'
        'TVM_FFI_DLL_EXPORT_TYPED_FUNC(answer, answer);\n'
    )
    monkeypatch.setattr(core, "PACKAGE", package)
    monkeypatch.setenv("PCIE_COMM_JIT_CACHE_DIR", str(tmp_path / "cache"))
    spec = core.JitSpec("ffi_probe", ("probe.cpp",), cuda=False)
    before = core.cache_stats()
    module = spec.build_and_load()
    assert module.answer() == 42
    assert spec.build_and_load() is module
    delta = {key: core.cache_stats()[key] - before[key] for key in before}
    assert delta == {"builds": 1, "disk_hits": 0, "loads": 1}

    # Clear process memory only: persistence must bypass the compiler.
    core._build_and_load.cache_clear()
    core._load_module.cache_clear()
    import tvm_ffi.cpp

    def forbidden(**_):
        raise AssertionError("warm disk path tried to compile")

    monkeypatch.setattr(tvm_ffi.cpp, "build", forbidden)
    assert spec.build_and_load().answer() == 42
    assert core.cache_stats()["disk_hits"] - before["disk_hits"] == 1


def test_concurrent_cold_publish_produces_complete_binary(tmp_path):
    args = [sys.executable, "-m", "tests.cache_worker", str(tmp_path), "race", "build"]
    processes = [subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                 for _ in range(2)]
    try:
        results = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=60)
            assert process.returncode == 0, stderr
            results.append(json.loads(stdout.strip().splitlines()[-1]))
        assert results[0]["path"] == results[1]["path"]
        assert ctypes.CDLL(results[0]["path"]).answer() == 42
        assert len(list(tmp_path.glob("cache/*/.committed"))) == 1
        assert not list((tmp_path / "tmp").iterdir())
        # DeepJIT deliberately permits duplicate compilation on a cold race.
        assert sum(result["stats"]["builds"] for result in results) in (1, 2)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait()
