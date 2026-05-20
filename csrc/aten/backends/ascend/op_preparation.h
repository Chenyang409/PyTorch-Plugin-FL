// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/ATen.h>
#include <c10/core/DeviceType.h>

namespace at::native::flagos::ascend {

class OpPreparation {
 public:
  static at::Tensor apply_tensor_without_format(
      at::IntArrayRef sizes,
      const at::TensorOptions& options) {
    return at::empty(sizes, options.device(c10::DeviceType::PrivateUse1));
  }

  static at::Tensor apply_tensor_with_format(
      at::IntArrayRef sizes,
      const at::TensorOptions& options,
      int64_t format = 2 /* ACL_FORMAT_ND */) {
    return at::empty(sizes, options.device(c10::DeviceType::PrivateUse1));
  }

  static int8_t get_cube_math_type(bool allow_hf32 = false) {
    return allow_hf32 ? 1 : 0;
  }

  static void check_tensor(
      std::initializer_list<at::Tensor> inputs,
      at::Tensor& output,
      at::ScalarType dtype,
      at::IntArrayRef sizes) {
    if (output.sizes() != sizes) {
      output.resize_(sizes);
    }
  }
};

} // namespace at::native::flagos::ascend
