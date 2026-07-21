#include "op.hpp"

#include "../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace {
template <typename T>
void self_attention_cpu(llaisys::tensor_t attn_val, llaisys::tensor_t q, llaisys::tensor_t k, llaisys::tensor_t v, float scale) {
    const auto *q_data = reinterpret_cast<const T *>(q->data());
    const auto *k_data = reinterpret_cast<const T *>(k->data());
    const auto *v_data = reinterpret_cast<const T *>(v->data());
    auto *out = reinterpret_cast<T *>(attn_val->data());

    const size_t qlen = q->shape()[0];
    const size_t kvlen = k->shape()[0];
    const size_t nh = q->shape()[1];
    const size_t nkvh = k->shape()[1];
    const size_t hd = q->shape()[2];
    const size_t group = nh / nkvh;
    std::vector<float> scores(kvlen);

    for (size_t qi = 0; qi < qlen; ++qi) {
        const size_t last_allowed = qi + kvlen - qlen;
        for (size_t h = 0; h < nh; ++h) {
            const size_t kvh = h / group;
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t kj = 0; kj < kvlen; ++kj) {
                if (kj > last_allowed) {
                    scores[kj] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                float dot = 0.0f;
                for (size_t d = 0; d < hd; ++d) {
                    dot += llaisys::utils::cast<float>(q_data[(qi * nh + h) * hd + d])
                         * llaisys::utils::cast<float>(k_data[(kj * nkvh + kvh) * hd + d]);
                }
                scores[kj] = dot * scale;
                max_score = std::max(max_score, scores[kj]);
            }

            float denom = 0.0f;
            for (size_t kj = 0; kj <= last_allowed; ++kj) {
                scores[kj] = std::exp(scores[kj] - max_score);
                denom += scores[kj];
            }

            for (size_t d = 0; d < hd; ++d) {
                float acc = 0.0f;
                for (size_t kj = 0; kj <= last_allowed; ++kj) {
                    const float prob = scores[kj] / denom;
                    acc += prob * llaisys::utils::cast<float>(v_data[(kj * nkvh + kvh) * hd + d]);
                }
                out[(qi * nh + h) * hd + d] = llaisys::utils::cast<T>(acc);
            }
        }
    }
}
} // namespace

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_ARGUMENT(attn_val->ndim() == 3 && q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3, "SelfAttention expects 3D tensors.");
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    CHECK_SAME_SHAPE(k->shape(), v->shape());
    CHECK_SAME_SHAPE(attn_val->shape(), q->shape());
    CHECK_ARGUMENT(q->shape()[2] == k->shape()[2], "SelfAttention head dimension mismatch.");
    CHECK_ARGUMENT(q->shape()[1] % k->shape()[1] == 0, "SelfAttention query heads must be divisible by kv heads.");
    CHECK_ARGUMENT(k->shape()[0] >= q->shape()[0], "SelfAttention kv length must be >= query length for causal mask.");
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(), "SelfAttention: all tensors must be contiguous.");
    CHECK_ARGUMENT(attn_val->deviceType() == LLAISYS_DEVICE_CPU, "SelfAttention currently only supports CPU tensors.");

    switch (attn_val->dtype()) {
    case LLAISYS_DTYPE_F32:
        return self_attention_cpu<float>(attn_val, q, k, v, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_cpu<fp16_t>(attn_val, q, k, v, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_cpu<bf16_t>(attn_val, q, k, v, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(attn_val->dtype());
    }
}
} // namespace llaisys::ops
