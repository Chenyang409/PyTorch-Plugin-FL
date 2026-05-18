// Copyright (c) 2026, BAAI. All rights reserved.
//
// MACA add backend: delegate to mcPytorch's CUDA/MACA add (structured_ufunc_add_CUDA)
// via device boxing. flagos and CUDA share the same device memory on MACA.

#include "../../add.h"
#include "../../device_boxing.h"

#include <ATen/ops/add_cuda_dispatch.h>

namespace at::native::flagos {

namespace {

at::Tensor AddKernelMaca(
    const at::Tensor& self, const at::Tensor& other, const at::Scalar& alpha) {
  DeviceBoxingGuard guard(self, other);
  at::Tensor result = at::cuda::add(self, other, alpha);
  UnboxToFlagos(result);
  return result;
}

} // namespace

FLAGOS_REGISTER_DISPATCH(AddTensorFn, add_tensor_stub, FlagosDevice::kMaca, AddKernelMaca)

} // namespace at::native::flagos
