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
__global__ void BmmKernel(
    int64_t batch,
    int64_t m,
    int64_t n,
    int64_t k,
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ self,
    const scalar_t* __restrict__ mat2) {
  const int64_t b = static_cast<int64_t>(blockIdx.z);
  const int64_t row = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (b >= batch || row >= m || col >= n) {
    return;
  }
  const int64_t self_batch_stride = m * k;
  const int64_t mat2_batch_stride = k * n;
  const int64_t out_batch_stride = m * n;
  const scalar_t* a = self + b * self_batch_stride;
  const scalar_t* b_mat = mat2 + b * mat2_batch_stride;
  scalar_t* c = out + b * out_batch_stride;

  acc_t sum = acc_t(0);
  for (int64_t kk = 0; kk < k; ++kk) {
    sum += static_cast<acc_t>(a[row * k + kk]) *
           static_cast<acc_t>(b_mat[kk * n + col]);
  }
  c[row * n + col] = static_cast<scalar_t>(sum);
}

template <typename scalar_t, typename acc_t>
void LaunchBmm(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  const at::Tensor a = self.contiguous();
  const at::Tensor b = mat2.contiguous();
  const int64_t batch = a.size(0);
  const int64_t m = a.size(1);
  const int64_t k = a.size(2);
  const int64_t n = b.size(2);
  TORCH_CHECK(b.size(0) == batch && b.size(1) == k, "MetaX bmm: shape mismatch");

  constexpr int kTile = 16;
  const dim3 threads(kTile, kTile, 1);
  const dim3 blocks(
      static_cast<unsigned int>((n + kTile - 1) / kTile),
      static_cast<unsigned int>((m + kTile - 1) / kTile),
      static_cast<unsigned int>(batch));

  BmmKernel<scalar_t, acc_t><<<blocks, threads, 0, metax::CurrentStream()>>>(
      batch,
      m,
      n,
      k,
      out.data_ptr<scalar_t>(),
      a.data_ptr<scalar_t>(),
      b.data_ptr<scalar_t>());
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(
      err == cudaSuccess,
      "MetaX bmm kernel launch failed: ",
      cudaGetErrorString(err));
}

}  // namespace

inline void BmmKernelMetax(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  TORCH_CHECK(
      self.dim() == 3 && mat2.dim() == 3, "MetaX bmm: inputs must be 3-D");
  at::native::resize_output(out, {self.size(0), self.size(1), mat2.size(2)});

  if (out.numel() == 0) {
    return;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "bmm_metax",
      [&]() {
        using acc_t = at::opmath_type<scalar_t>;
        LaunchBmm<scalar_t, acc_t>(self, mat2, out);
      });
}

}  // namespace at::native::flagos
