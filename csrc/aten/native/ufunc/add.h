// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from pytorch/aten/src/ATen/native/ufunc/add.h
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <c10/macros/Macros.h>

namespace at::native::ufunc {

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T add(T self, T other, T alpha) {
  return self + alpha * other;
}

} // namespace at::native::ufunc
