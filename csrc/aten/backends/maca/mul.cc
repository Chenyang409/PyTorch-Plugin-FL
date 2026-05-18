// Copyright (c) 2026, BAAI. All rights reserved.
//
// MACA mul backend: delegate to mcPytorch's CUDA/MACA mul via device boxing.

#include "../../mul.h"
#include "../../device_boxing.h"

#include <ATen/ops/mul_cuda_dispatch.h>

namespace at::native::flagos {

namespace {

at::Tensor MulKernelMaca(const at::Tensor& self, const at::Tensor& other) {
  DeviceBoxingGuard guard(self, other);
  at::Tensor result = at::cuda::mul(self, other);
  UnboxToFlagos(result);
  return result;
}

} // namespace

FLAGOS_REGISTER_DISPATCH(MulTensorFn, mul_tensor_stub, FlagosDevice::kMaca, MulKernelMaca)

} // namespace at::native::flagos
