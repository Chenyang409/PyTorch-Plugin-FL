// Copyright (c) 2026, BAAI. All rights reserved.

#pragma once

#include <ATen/core/Tensor.h>
#include "../dispatch_stub.h"

namespace at::native::flagos {

using LogSoftmaxFn = at::Tensor (*)(const at::Tensor&, int64_t, bool);
FLAGOS_DECLARE_DISPATCH(LogSoftmaxFn, log_softmax_stub)

} // namespace at::native::flagos
