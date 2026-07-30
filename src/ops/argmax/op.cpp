#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#ifdef ENABLE_NVIDIA_API
#include "../nvidia/ops_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "../metax/ops_metax.cuh"
#endif

#include <limits>

namespace {
template <typename T>
float read_as_float(const std::byte *data, size_t i) {
    return llaisys::utils::cast<float>(reinterpret_cast<const T *>(data)[i]);
}

template <typename T>
void argmax_cpu(llaisys::tensor_t max_idx, llaisys::tensor_t max_val, llaisys::tensor_t vals) {
    const auto *vals_data = vals->data();
    size_t best_idx = 0;
    float best_val = read_as_float<T>(vals_data, 0);
    for (size_t i = 1; i < vals->numel(); ++i) {
        const float val = read_as_float<T>(vals_data, i);
        if (val > best_val) {
            best_val = val;
            best_idx = i;
        }
    }
    reinterpret_cast<int64_t *>(max_idx->data())[0] = static_cast<int64_t>(best_idx);
    reinterpret_cast<T *>(max_val->data())[0] = llaisys::utils::cast<T>(best_val);
}
} // namespace

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_ARGUMENT(vals->ndim() == 1, "Argmax currently supports 1D input.");
    CHECK_ARGUMENT(vals->numel() > 0, "Argmax input must not be empty.");
    CHECK_ARGUMENT(max_idx->numel() == 1 && max_val->numel() == 1, "Argmax outputs must contain one element.");
    CHECK_ARGUMENT(max_idx->dtype() == LLAISYS_DTYPE_I64, "Argmax index output must be int64.");
    CHECK_ARGUMENT(max_val->dtype() == vals->dtype(), "Argmax value output dtype must match input dtype.");
    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(), "Argmax: all tensors must be contiguous.");
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (vals->dtype()) {
        case LLAISYS_DTYPE_F32:
            return argmax_cpu<float>(max_idx, max_val, vals);
        case LLAISYS_DTYPE_F16:
            return argmax_cpu<fp16_t>(max_idx, max_val, vals);
        case LLAISYS_DTYPE_BF16:
            return argmax_cpu<bf16_t>(max_idx, max_val, vals);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(vals->dtype());
        }
    }

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());
    switch (vals->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel(), llaisys::core::context().runtime().stream());
#endif
#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX:
        return metax::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel(), llaisys::core::context().runtime().stream());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
