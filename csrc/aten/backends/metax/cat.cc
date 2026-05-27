// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../cat.h"

#include <include/flagos.h>

#include <vector>

namespace at::native::flagos {

namespace {

at::Tensor CatKernelMetax(const at::ITensorListRef& tensors, int64_t dim) {
  TORCH_CHECK(dim >= 0, "MetaX cat: dim must be non-negative");

  std::vector<at::Tensor> materialized;
  materialized.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    materialized.push_back(tensor.contiguous());
  }
  TORCH_CHECK(!materialized.empty(), "MetaX cat: expected a non-empty TensorList");

  const auto& first = materialized[0];
  TORCH_CHECK(
      dim < first.dim(),
      "MetaX cat: dimension ",
      dim,
      " out of range");

  int64_t out_dim_size = 0;
  for (const auto& tensor : materialized) {
    TORCH_CHECK(
        tensor.scalar_type() == first.scalar_type(),
        "MetaX cat: all inputs must have the same dtype");
    TORCH_CHECK(
        tensor.device() == first.device(),
        "MetaX cat: all inputs must be on the same device");
    for (int64_t d = 0; d < first.dim(); ++d) {
      if (d == dim) {
        continue;
      }
      TORCH_CHECK(
          tensor.size(d) == first.size(d),
          "MetaX cat: sizes must match except in the concatenation dimension");
    }
    out_dim_size += tensor.size(dim);
  }

  auto out_sizes = first.sizes().vec();
  out_sizes[dim] = out_dim_size;
  at::Tensor out = at::empty(out_sizes, first.options());

  if (out.numel() == 0) {
    return out;
  }

  const int64_t outer =
      dim == 0 ? 1 : first.numel() / first.size(dim);
  const int64_t out_inner =
      dim == first.dim() - 1
          ? 1
          : out.numel() / (outer * out_dim_size);
  const size_t elem_size = static_cast<size_t>(first.element_size());

  int64_t offset_along_dim = 0;
  for (const auto& tensor : materialized) {
    const int64_t chunk = tensor.size(dim);
    const int64_t chunk_elems = outer * chunk * out_inner;
    if (chunk_elems == 0) {
      offset_along_dim += chunk;
      continue;
    }

    const size_t nbytes = static_cast<size_t>(chunk_elems) * elem_size;
    char* dst = static_cast<char*>(out.data_ptr()) +
        static_cast<size_t>(offset_along_dim * out_inner) * elem_size;
    Memcpy(dst, tensor.data_ptr(), nbytes, MemcpyDeviceToDevice);
    offset_along_dim += chunk;
  }

  return out;
}

}  // namespace

FLAGOS_REGISTER_DISPATCH(CatFn, cat_stub, FlagosDevice::kMetax, CatKernelMetax)

}  // namespace at::native::flagos
