from .core import JitSpec


def gen_pcie_ipc_comm_module():
    return JitSpec("pcie_ipc_all_reduce", ("pcie_ipc_all_reduce.cu",))


def gen_pcie_ipc_ag_rs_module():
    return JitSpec("pcie_ipc_ag_rs", ("pcie_ipc_all_gather.cu", "pcie_ipc_reduce_scatter.cu"))


def gen_timing_module():
    return JitSpec("pcie_timing", ("timing.cu",))
