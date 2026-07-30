#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#ifdef ENABLE_NVIDIA_API
#include "../nvidia/ops_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "../metax/ops_metax.cuh"
#endif

#include <cstring>

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_ARGUMENT(index->ndim() == 1, "Embedding index must be 1D.");
    CHECK_ARGUMENT(weight->ndim() == 2 && out->ndim() == 2, "Embedding weight and output must be 2D.");
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding index must be int64.");
    CHECK_ARGUMENT(out->dtype() == weight->dtype(), "Embedding output dtype must match weight dtype.");
    CHECK_ARGUMENT(out->shape()[0] == index->shape()[0] && out->shape()[1] == weight->shape()[1], "Embedding output shape mismatch.");
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), "Embedding: all tensors must be contiguous.");
    switch (weight->dtype()) {
    case LLAISYS_DTYPE_F32:
    case LLAISYS_DTYPE_F16:
    case LLAISYS_DTYPE_BF16:
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(weight->dtype());
    }

    const size_t rows = weight->shape()[0];
    const size_t hidden = weight->shape()[1];
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        const auto *idx = reinterpret_cast<const int64_t *>(index->data());
        const size_t row_bytes = hidden * weight->elementSize();
        for (size_t i = 0; i < index->shape()[0]; ++i) {
            CHECK_ARGUMENT(idx[i] >= 0 && static_cast<size_t>(idx[i]) < rows, "Embedding index out of range.");
            std::memcpy(out->data() + i * row_bytes, weight->data() + static_cast<size_t>(idx[i]) * row_bytes, row_bytes);
        }
        return;
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::embedding(out->data(), index->data(), weight->data(), weight->dtype(), index->shape()[0], rows, hidden, llaisys::core::context().runtime().stream());
#endif
#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX:
        return metax::embedding(out->data(), index->data(), weight->data(), weight->dtype(), index->shape()[0], rows, hidden, llaisys::core::context().runtime().stream());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
