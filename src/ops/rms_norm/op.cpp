#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#ifdef ENABLE_NVIDIA_API
#include "../nvidia/ops_nvidia.cuh"
#endif

#include <cmath>

namespace {
template <typename T>
void rms_norm_cpu(llaisys::tensor_t out, llaisys::tensor_t in, llaisys::tensor_t weight, float eps) {
    const auto *x = reinterpret_cast<const T *>(in->data());
    const auto *w = reinterpret_cast<const T *>(weight->data());
    auto *y = reinterpret_cast<T *>(out->data());
    const size_t rows = in->shape()[0];
    const size_t hidden = in->shape()[1];

    for (size_t row = 0; row < rows; ++row) {
        float sum_sq = 0.0f;
        for (size_t col = 0; col < hidden; ++col) {
            const float val = llaisys::utils::cast<float>(x[row * hidden + col]);
            sum_sq += val * val;
        }
        const float scale = 1.0f / std::sqrt(sum_sq / static_cast<float>(hidden) + eps);
        for (size_t col = 0; col < hidden; ++col) {
            const float val = llaisys::utils::cast<float>(x[row * hidden + col]) * scale * llaisys::utils::cast<float>(w[col]);
            y[row * hidden + col] = llaisys::utils::cast<T>(val);
        }
    }
}
} // namespace

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_ARGUMENT(out->ndim() == 2 && in->ndim() == 2 && weight->ndim() == 1, "RMSNorm expects 2D input/output and 1D weight.");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_ARGUMENT(weight->shape()[0] == in->shape()[1], "RMSNorm weight shape mismatch.");
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "RMSNorm: all tensors must be contiguous.");
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            return rms_norm_cpu<float>(out, in, weight, eps);
        case LLAISYS_DTYPE_F16:
            return rms_norm_cpu<fp16_t>(out, in, weight, eps);
        case LLAISYS_DTYPE_BF16:
            return rms_norm_cpu<bf16_t>(out, in, weight, eps);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), in->shape()[0], in->shape()[1], eps, llaisys::core::context().runtime().stream());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
