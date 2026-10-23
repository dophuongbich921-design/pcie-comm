"""Adapter from FlashInfer module specs to DeepJIT's unmodified DiskCache.

DeepJIT owns lookup, temporary directories, commit markers, publication and
cleanup. TVM FFI builds/loads .so files because PCIe kernels have host-side
workspace and Copy Engine code; they are not standalone cubin entrypoints.
"""
from dataclasses import dataclass
import functools
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sysconfig

logger = logging.getLogger("pcie_comm")
PACKAGE = Path(__file__).resolve().parents[1]
_stats = {"builds": 0, "disk_hits": 0, "loads": 0}


def cache_stats():
    """Process-local counters, useful for proving restart reuse."""
    return dict(_stats)


def cache_roots():
    from .env import PCIE_COMM_WORKSPACE_DIR
    raw = os.environ.get("PCIE_COMM_JIT_CACHE_DIR", os.environ.get("DJ_JIT_CACHE_DIR"))
    if raw is None:
        return (str(PCIE_COMM_WORKSPACE_DIR / "jit"),)
    paths = raw.split(os.pathsep)
    if not all(paths):
        raise ValueError("JIT cache search path contains an empty component")
    return tuple(str(Path(p).expanduser().resolve()) for p in paths)


def file_fingerprints(paths, root):
    """Stable across install locations; invalidate on source/header changes."""
    root = Path(root)
    return {str(Path(p).relative_to(root)): hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in sorted(paths)}


def canonical_signature(description):
    return json.dumps(description, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compiler_identity(command):
    args = shlex.split(command)
    if not args:
        raise ValueError("Empty compiler command")
    result = subprocess.run(args + ["--version"], check=True, capture_output=True, text=True)
    return {"command": args, "version": result.stdout + result.stderr}


@functools.cache
def _disk_cache(roots):
    from .._deepjit_cache import ArtifactCache
    return ArtifactCache(list(roots))


@functools.cache
def _load_module(path):
    import tvm_ffi
    module = tvm_ffi.load_module(path)
    _stats["loads"] += 1
    return module


def get_or_build_artifact(tag, description, builder, *, roots=None):
    """Builder receives a private Path and must produce module.so there."""
    roots = tuple(roots) if roots is not None else cache_roots()
    signature = canonical_signature(description)

    def build(directory):
        target = Path(directory)
        _stats["builds"] += 1
        logger.info("JIT build: %s", tag)
        builder(target)
        (target / "signature.json").write_text(signature + "\n")

    path, digest, hit = _disk_cache(roots).get_or_build(tag, signature, build)
    if hit:
        _stats["disk_hits"] += 1
        logger.info("JIT disk hit: %s.%s", tag, digest)
    return Path(path)


@dataclass(frozen=True)
class JitSpec:
    name: str
    sources: tuple[str, ...]
    cuda: bool = True

    def build_and_load(self):
        return _build_and_load(self, cache_roots())


@functools.cache
def _build_and_load(spec, roots):
    import torch
    import tvm_ffi
    import tvm_ffi.cpp
    from tvm_ffi.cpp import extension

    sources = [PACKAGE / "csrc" / name for name in spec.sources]
    headers = list((PACKAGE / "include").rglob("*.cuh"))
    headers += list((PACKAGE / "csrc").glob("*.h"))
    cflags = ["-std=c++17", "-O3", "-DNDEBUG",
              f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}"]
    cflags += shlex.split(os.environ.get("PCIE_COMM_EXTRA_CFLAGS", ""))
    cuda_flags = ["-std=c++17", "-O3", "-DNDEBUG", "--use_fast_math",
                  "--expt-relaxed-constexpr", "-DFLASHINFER_ENABLE_F16",
                  "-DFLASHINFER_ENABLE_BF16",
                  f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}"]
    cuda_flags += shlex.split(os.environ.get("PCIE_COMM_EXTRA_CUDAFLAGS", ""))
    ldflags = shlex.split(os.environ.get("PCIE_COMM_EXTRA_LDFLAGS", ""))
    toolchain = {"cxx": compiler_identity(os.environ.get("CXX", "c++"))}
    target = "cpu"
    if spec.cuda:
        cuda_home = Path(extension._find_cuda_home())
        toolchain["nvcc"] = compiler_identity(str(cuda_home / "bin/nvcc"))
        toolchain["nvcc_host"] = compiler_identity(os.environ.get("NVCC_CCBIN", "g++"))
        # Exactly the target flags used by TVM FFI's builder (including suffixes).
        target = extension._get_cuda_target()
        ldflags += [f"-L{cuda_home / 'lib64/stubs'}", "-lcuda"]
        # Match FlashInfer's behavior with CUDA >= 12.8.
        import re
        match = re.search(r"release (\d+)\.(\d+)", toolchain["nvcc"]["version"])
        if match and tuple(map(int, match.groups())) >= (12, 8):
            cuda_flags.append("-static-global-template-stub=false")

    ffi_include = Path(tvm_ffi.libinfo.find_include_path())
    dlpack_include = Path(tvm_ffi.libinfo.find_dlpack_include_path())
    description = {
        "schema": 1, "module": spec.name, "target": target,
        "source": file_fingerprints(sources + headers, PACKAGE),
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "toolchain": toolchain, "cflags": cflags, "cuda_flags": cuda_flags,
        "ldflags": ldflags, "platform": platform.platform(),
        "python_abi": sysconfig.get_config_var("SOABI"), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "tvm_ffi": tvm_ffi.__version__,
        "ffi_headers": file_fingerprints(ffi_include.rglob("*.h"), ffi_include),
        "dlpack_headers": file_fingerprints(dlpack_include.rglob("*.h"), dlpack_include),
        "build_environment": {k: os.environ.get(k) for k in (
            "CC", "CXX", "CPATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH",
            "CUDA_HOME", "CUDA_PATH", "NVCC_CCBIN", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS",
            "TVM_FFI_CUDA_ARCH_LIST")},
    }

    def build(directory):
        # Supplying a private build_directory bypasses TVM FFI's global cache
        # and interprocess lock. Publication belongs solely to DeepJIT.
        tvm_ffi.cpp.build(
            name="module", cpp_files=[] if spec.cuda else list(map(str, sources)),
            cuda_files=list(map(str, sources)) if spec.cuda else [],
            extra_cflags=cflags, extra_cuda_cflags=cuda_flags,
            extra_ldflags=ldflags,
            extra_include_paths=[str(PACKAGE / "include"), str(PACKAGE / "csrc")],
            build_directory=str(directory),
            backend="cuda" if spec.cuda else None,
        )

    path = get_or_build_artifact(spec.name, description, build, roots=roots)
    return _load_module(str(path))
