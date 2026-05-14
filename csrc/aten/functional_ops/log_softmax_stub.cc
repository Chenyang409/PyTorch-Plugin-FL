// Copyright (c) 2026, BAAI. All rights reserved.

#include "log_softmax_stub.h"

#include <ATen/ops/_log_softmax_meta.h>
#include <ATen/ops/_log_softmax_native.h>

#include "../device_boxing.h"

namespace at::native::flagos {

FLAGOS_DEFINE_DISPATCH(LogSoftmaxFn, log_softmax_stub, "_log_softmax")

namespace {

at::Tensor LogSoftmaxKernelCuda(const at::Tensor& self, int64_t dim, bool half_to_float) {
  auto output_dtype = half_to_float ? at::ScalarType::Float : self.scalar_type();
  auto output = at::empty(self.sizes(), self.options().dtype(output_dtype));

  BoxToCuda(self);
  BoxToCuda(output);

  struct CudaImpl final : public at::native::structured_log_softmax_cuda_out {
    CudaImpl(at::Tensor& out) : out_(out) {}
    void set_output_raw_strided(int64_t, at::IntArrayRef, at::IntArrayRef,
                                at::TensorOptions, at::DimnameList) override {
    }
    const at::Tensor& maybe_get_output(int64_t) override { return out_; }
    at::Tensor& out_;
  };
  CudaImpl op(output);
  op.meta(self, dim, half_to_float);
  op.impl(self, dim, half_to_float, output);

  UnboxToFlagos(self);
  UnboxToFlagos(output);
  return output;
}

} // namespace

// FlagGems does not currently export a C++ log_softmax kernel
// (flag_gems::log_softmax is unavailable in liboperators.so).
// Only the CUDA backend is registered; routing to flaggems would
// fail with "backend not registered".
FLAGOS_REGISTER_DISPATCH(LogSoftmaxFn, log_softmax_stub, FlagosDevice::kCuda, LogSoftmaxKernelCuda)

} // namespace at::native::flagos
