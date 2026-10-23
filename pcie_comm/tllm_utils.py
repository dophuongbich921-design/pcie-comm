import functools
from .jit.comm import gen_timing_module


@functools.cache
def get_trtllm_utils_module():
    return gen_timing_module().build_and_load()


def delay_kernel(stream_delay_micro_secs: int):
    get_trtllm_utils_module().delay_kernel(stream_delay_micro_secs)
