// Copyright (c) 2026, BAAI. All rights reserved.
//
// MACA le backend: delegate to mcPytorch's CUDA/MACA le via device boxing.

#include "../../le.h"
#include "../../device_boxing.h"

#include <ATen/ops/le_cuda_dispatch.h>

namespace at::native::flagos {

namespace {

at::Tensor LeKernelMaca(const at::Tensor& self, const at::Tensor& other) {
  DeviceBoxingGuard guard(self, other);
  at::Tensor result = at::cuda::le(self, other);
  UnboxToFlagos(result);
  return result;
}

} // namespace

FLAGOS_REGISTER_DISPATCH(LeTensorFn, le_tensor_stub, FlagosDevice::kMaca, LeKernelMaca)

} // namespace at::native::flagos
