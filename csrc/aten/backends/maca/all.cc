// Copyright (c) 2026, BAAI. All rights reserved.
//
// MACA all backend: delegate to mcPytorch's CUDA/MACA all via device boxing.

#include "../../all.h"
#include "../../device_boxing.h"

#include <ATen/ops/all_cuda_dispatch.h>

namespace at::native::flagos {

namespace {

at::Tensor AllKernelMaca(const at::Tensor& self) {
  DeviceBoxingGuard guard(self);
  at::Tensor result = at::cuda::all(self);
  if (result.defined() && result.is_cuda()) {
    UnboxToFlagos(result);
  }
  return result;
}

} // namespace

FLAGOS_REGISTER_DISPATCH(AllFn, all_stub, FlagosDevice::kMaca, AllKernelMaca)

} // namespace at::native::flagos
