import os
import sys
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

if sys.platform == "darwin":
    os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", "14.0")

setup(
    ext_modules=[Pybind11Extension(
        "pcie_comm._deepjit_cache",
        ["csrc/deepjit_cache.cpp"],
        include_dirs=["csrc/compat", "third_party/DeepJIT/include"],
        cxx_std=20,
        extra_compile_args=["-mmacosx-version-min=14.0"] if sys.platform == "darwin" else [],
        extra_link_args=["-mmacosx-version-min=14.0"] if sys.platform == "darwin" else [],
    )],
    cmdclass={"build_ext": build_ext},
)
