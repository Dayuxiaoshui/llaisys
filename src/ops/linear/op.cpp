#include "op.hpp"

#include "../../utils.hpp"

namespace {
template <typename T>
void linear_cpu(llaisys::tensor_t out, llaisys::tensor_t in, llaisys::tensor_t weight, llaisys::tensor_t bias) {
    const auto *x = reinterpret_cast<const T *>(in->data());
    const auto *w = reinterpret_cast<const T *>(weight->data());
    const auto *b = reinterpret_cast<const T *>(bias->data());
    auto *y = reinterpret_cast<T *>(out->data());
    const size_t m = in->shape()[0];
    const size_t k = in->shape()[1];
    const size_t n = weight->shape()[0];

    for (size_t row = 0; row < m; ++row) {
        for (size_t col = 0; col < n; ++col) {
            float acc = llaisys::utils::cast<float>(b[col]);
            for (size_t p = 0; p < k; ++p) {
                acc += llaisys::utils::cast<float>(x[row * k + p]) * llaisys::utils::cast<float>(w[col * k + p]);
            }
            y[row * n + col] = llaisys::utils::cast<T>(acc);
        }
    }
}
} // namespace

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight, bias);
    CHECK_ARGUMENT(out->ndim() == 2 && in->ndim() == 2 && weight->ndim() == 2 && bias->ndim() == 1, "Linear expects 2D input/output/weight and 1D bias.");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype(), bias->dtype());
    CHECK_ARGUMENT(in->shape()[1] == weight->shape()[1], "Linear input feature size mismatch.");
    CHECK_ARGUMENT(out->shape()[0] == in->shape()[0] && out->shape()[1] == weight->shape()[0], "Linear output shape mismatch.");
    CHECK_ARGUMENT(bias->shape()[0] == weight->shape()[0], "Linear bias shape mismatch.");
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous() && bias->isContiguous(), "Linear: all tensors must be contiguous.");
    CHECK_ARGUMENT(out->deviceType() == LLAISYS_DEVICE_CPU, "Linear currently only supports CPU tensors.");

    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return linear_cpu<float>(out, in, weight, bias);
    case LLAISYS_DTYPE_F16:
        return linear_cpu<fp16_t>(out, in, weight, bias);
    case LLAISYS_DTYPE_BF16:
        return linear_cpu<bf16_t>(out, in, weight, bias);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}
} // namespace llaisys::ops
