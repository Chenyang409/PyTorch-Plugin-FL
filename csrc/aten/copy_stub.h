// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include <optional>

#include "dispatch_stub.h"

namespace at::native::flagos {

using LocalScalarDenseFn = at::Scalar (*)(const at::Tensor&);
FLAGOS_DECLARE_DISPATCH(LocalScalarDenseFn, local_scalar_dense_stub)

using ToCopyFn = at::Tensor (*)(
    const at::Tensor&,
    std::optional<c10::ScalarType>,
    std::optional<c10::Layout>,
    std::optional<c10::Device>,
    std::optional<bool>,
    bool,
    std::optional<c10::MemoryFormat>);
FLAGOS_DECLARE_DISPATCH(ToCopyFn, to_copy_stub)

}  // namespace at::native::flagos
