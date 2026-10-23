/*
 * Delay kernel: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * Timestamp kernel: Copyright FlashInfer team.
 * Licensed under the Apache License, Version 2.0; see LICENSE.
 * Extracted from FlashInfer delayStream.cu and utils.py. Only host binding and
 * error handling are adapted to remove unrelated TensorRT-LLM dependencies.
 */
#include <cuda_runtime.h>
#include "tvm_ffi_utils.h"

__global__ void delayStreamKernel(long long delay_micro_secs) {
  for (int i = 0; i < delay_micro_secs; ++i) {
    __nanosleep(1000);
  }
}

void delay_kernel(int64_t delay_micro_secs) {
  delayStreamKernel<<<1, 1, 0, get_current_stream()>>>(delay_micro_secs);
  const auto error = cudaGetLastError();
  TVM_FFI_ICHECK(error == cudaSuccess) << cudaGetErrorString(error);
}

__global__ void _get_globaltimer_timestamp(uint64_t* timestamp) {
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(*timestamp));
}

void get_globaltimer_timestamp(TensorView timestamp) {
  TVM_FFI_ICHECK(timestamp.device().device_type == kDLCUDA);
  TVM_FFI_ICHECK(timestamp.dtype() == dl_int64);
  TVM_FFI_ICHECK(timestamp.numel() >= 1);
  _get_globaltimer_timestamp<<<1, 1, 0, get_current_stream()>>>(
      reinterpret_cast<uint64_t*>(timestamp.data_ptr()));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(delay_kernel, delay_kernel);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_globaltimer_timestamp, get_globaltimer_timestamp);
