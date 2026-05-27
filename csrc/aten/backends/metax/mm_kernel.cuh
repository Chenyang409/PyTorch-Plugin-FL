// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <cstdint>

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <c10/util/Exception.h>

#include "metax_elementwise.cuh"

namespace at::native::flagos {

namespace {

template <typename scalar_t, typename acc_t>
__global__ void MmKernel(
    int64_t m,
    int64_t n,
    int64_t k,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ self,
    const scalar_t* __restrict__ mat2) {
  const int64_t row = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) {
    return;
  }
  acc_t sum = acc_t(0);
  for (int64_t kk = 0; kk < k; ++kk) {
    sum += static_cast<acc_t>(self[row * k + kk]) *
           static_cast<acc_t>(mat2[kk * n + col]);
  }
  out[row * n + col] = static_cast<scalar_t>(sum);
}

template <typename scalar_t, typename acc_t>
void LaunchMm(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  const at::Tensor a = self.contiguous();
  const at::Tensor b = mat2.contiguous();
  const int64_t m = a.size(0);
  const int64_t k = a.size(1);
  const int64_t n = b.size(1);
  TORCH_CHECK(b.size(0) == k, "MetaX mm: shape mismatch");

  constexpr int kTile = 16;
  const dim3 threads(kTile, kTile, 1);
  const dim3 blocks(
      static_cast<unsigned int>((n + kTile - 1) / kTile),
      static_cast<unsigned int>((m + kTile - 1) / kTile),
      1);

  MmKernel<scalar_t, acc_t><<<blocks, threads, 0, metax::CurrentStream()>>>(
      m,
      n,
      k,
      out.data_ptr<scalar_t>(),
      a.data_ptr<scalar_t>(),
      b.data_ptr<scalar_t>());
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(
      err == cudaSuccess,
      "MetaX mm kernel launch failed: ",
      cudaGetErrorString(err));
}

}  // namespace

inline void MmKernelMetax(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  TORCH_CHECK(self.dim() == 2 && mat2.dim() == 2, "MetaX mm: inputs must be 2-D");
  at::native::resize_output(out, {self.size(0), mat2.size(1)});

  if (out.numel() == 0) {
    return;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "mm_metax",
      [&]() {
        using acc_t = at::opmath_type<scalar_t>;
        LaunchMm<scalar_t, acc_t>(self, mat2, out);
      });
}

}  // namespace at::native::flagos
