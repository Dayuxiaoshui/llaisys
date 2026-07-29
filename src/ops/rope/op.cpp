#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#ifdef ENABLE_NVIDIA_API
#include "../nvidia/ops_nvidia.cuh"
#endif

#include <cmath>

namespace {
template <typename T>
void rope_cpu(llaisys::tensor_t out, llaisys::tensor_t in, llaisys::tensor_t pos_ids, float theta) {
    const auto *x = reinterpret_cast<const T *>(in->data());
    auto *y = reinterpret_cast<T *>(out->data());
    const auto *pos = reinterpret_cast<const int64_t *>(pos_ids->data());
    const size_t seq_len = in->shape()[0];
    const size_t n_heads = in->shape()[1];
    const size_t head_dim = in->shape()[2];
    const size_t half = head_dim / 2;

    for (size_t s = 0; s < seq_len; ++s) {
        for (size_t h = 0; h < n_heads; ++h) {
            const size_t base = (s * n_heads + h) * head_dim;
            for (size_t i = 0; i < half; ++i) {
                const float freq = static_cast<float>(pos[s]) / std::pow(theta, 2.0f * static_cast<float>(i) / static_cast<float>(head_dim));
                const float sin_v = std::sin(freq);
                const float cos_v = std::cos(freq);
                const float a = llaisys::utils::cast<float>(x[base + i]);
                const float b = llaisys::utils::cast<float>(x[base + half + i]);
                y[base + i] = llaisys::utils::cast<T>(a * cos_v - b * sin_v);
                y[base + half + i] = llaisys::utils::cast<T>(b * cos_v + a * sin_v);
            }
        }
    }
}
} // namespace

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_ARGUMENT(out->ndim() == 3 && in->ndim() == 3, "RoPE expects 3D input/output.");
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_ARGUMENT(pos_ids->ndim() == 1 && pos_ids->shape()[0] == in->shape()[0], "RoPE pos_ids shape mismatch.");
    CHECK_ARGUMENT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE pos_ids must be int64.");
    CHECK_ARGUMENT(out->dtype() == in->dtype(), "RoPE output dtype must match input dtype.");
    CHECK_ARGUMENT(in->shape()[2] % 2 == 0, "RoPE head dimension must be even.");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), "RoPE: all tensors must be contiguous.");
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            return rope_cpu<float>(out, in, pos_ids, theta);
        case LLAISYS_DTYPE_F16:
            return rope_cpu<fp16_t>(out, in, pos_ids, theta);
        case LLAISYS_DTYPE_BF16:
            return rope_cpu<bf16_t>(out, in, pos_ids, theta);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), in->shape()[0], in->shape()[1], in->shape()[2], theta, llaisys::core::context().runtime().stream());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
