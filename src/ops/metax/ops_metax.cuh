#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::metax {
void add(std::byte *out, const std::byte *a, const std::byte *b, llaisysDataType_t dtype, size_t numel, llaisysStream_t stream);
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t dtype, size_t numel, llaisysStream_t stream);
void embedding(std::byte *out,
               const std::byte *index,
               const std::byte *weight,
               llaisysDataType_t dtype,
               size_t num_indices,
               size_t num_embeddings,
               size_t embedding_dim,
               llaisysStream_t stream);
void linear(std::byte *out,
            const std::byte *in,
            const std::byte *weight,
            const std::byte *bias,
            llaisysDataType_t dtype,
            size_t m,
            size_t n,
            size_t k,
            llaisysStream_t stream);
void rms_norm(std::byte *out,
              const std::byte *in,
              const std::byte *weight,
              llaisysDataType_t dtype,
              size_t rows,
              size_t hidden,
              float eps,
              llaisysStream_t stream);
void rope(std::byte *out,
          const std::byte *in,
          const std::byte *pos_ids,
          llaisysDataType_t dtype,
          size_t seq_len,
          size_t num_heads,
          size_t head_dim,
          float theta,
          llaisysStream_t stream);
void self_attention(std::byte *out,
                    const std::byte *q,
                    const std::byte *k,
                    const std::byte *v,
                    llaisysDataType_t dtype,
                    size_t q_len,
                    size_t kv_len,
                    size_t num_heads,
                    size_t num_kv_heads,
                    size_t head_dim,
                    float scale,
                    llaisysStream_t stream);
void swiglu(std::byte *out,
            const std::byte *gate,
            const std::byte *up,
            llaisysDataType_t dtype,
            size_t numel,
            llaisysStream_t stream);
} // namespace llaisys::ops::metax
