#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#ifdef ENABLE_NVIDIA_API
#include "../nvidia/ops_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "../metax/ops_metax.cuh"
#endif

#include <cmath>

namespace {
template <typename T>
void swiglu_cpu(llaisys::tensor_t out, llaisys::tensor_t gate, llaisys::tensor_t up) {
    auto *y = reinterpret_cast<T *>(out->data());
    const auto *g = reinterpret_cast<const T *>(gate->data());
    const auto *u = reinterpret_cast<const T *>(up->data());
    for (size_t i = 0; i < out->numel(); ++i) {
        const float gate_val = llaisys::utils::cast<float>(g[i]);
        const float up_val = llaisys::utils::cast<float>(u[i]);
        y[i] = llaisys::utils::cast<T>(up_val * gate_val / (1.0f + std::exp(-gate_val)));
    }
}
} // namespace

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape());
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(), "SwiGLU: all tensors must be contiguous.");
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            return swiglu_cpu<float>(out, gate, up);
        case LLAISYS_DTYPE_F16:
            return swiglu_cpu<fp16_t>(out, gate, up);
        case LLAISYS_DTYPE_BF16:
            return swiglu_cpu<bf16_t>(out, gate, up);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::swiglu(out->data(), gate->data(), up->data(), out->dtype(), out->numel(), llaisys::core::context().runtime().stream());
#endif
#ifdef ENABLE_METAX_API
    case LLAISYS_DEVICE_METAX:
        return metax::swiglu(out->data(), gate->data(), up->data(), out->dtype(), out->numel(), llaisys::core::context().runtime().stream());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
