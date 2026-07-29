#include "ops_nvidia.cuh"

#include "../../utils.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace llaisys::ops::nvidia {
namespace {
constexpr int BLOCK_SIZE = 256;

void checkCuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void checkCublas(cublasStatus_t status, const char *operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with cuBLAS status " + std::to_string(status));
    }
}

void checkLaunch(const char *kernel) {
    checkCuda(cudaGetLastError(), kernel);
}

unsigned int gridSize(size_t numel) {
    const size_t blocks = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CHECK_ARGUMENT(blocks <= std::numeric_limits<unsigned int>::max(), "CUDA launch grid is too large.");
    return static_cast<unsigned int>(blocks);
}

template <typename T>
__device__ float toFloat(T value);

template <>
__device__ float toFloat<float>(float value) {
    return value;
}

template <>
__device__ float toFloat<__half>(__half value) {
    return __half2float(value);
}

template <>
__device__ float toFloat<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ T fromFloat(float value);

template <>
__device__ float fromFloat<float>(float value) {
    return value;
}

template <>
__device__ __half fromFloat<__half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __nv_bfloat16 fromFloat<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

template <typename T>
__global__ void addKernel(T *out, const T *a, const T *b, size_t numel) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = fromFloat<T>(toFloat(a[i]) + toFloat(b[i]));
    }
}

template <typename T>
__global__ void argmaxKernel(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    __shared__ float values[BLOCK_SIZE];
    __shared__ size_t indices[BLOCK_SIZE];

    float best_value = -CUDART_INF_F;
    size_t best_index = 0;
    for (size_t i = threadIdx.x; i < numel; i += blockDim.x) {
        const float value = toFloat(vals[i]);
        if (value > best_value || (value == best_value && i < best_index)) {
            best_value = value;
            best_index = i;
        }
    }
    values[threadIdx.x] = best_value;
    indices[threadIdx.x] = best_index;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float other_value = values[threadIdx.x + stride];
            const size_t other_index = indices[threadIdx.x + stride];
            if (other_value > values[threadIdx.x]
                || (other_value == values[threadIdx.x] && other_index < indices[threadIdx.x])) {
                values[threadIdx.x] = other_value;
                indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        max_idx[0] = static_cast<int64_t>(indices[0]);
        max_val[0] = fromFloat<T>(values[0]);
    }
}

template <typename T>
__global__ void embeddingKernel(T *out,
                                const int64_t *index,
                                const T *weight,
                                size_t num_indices,
                                size_t num_embeddings,
                                size_t embedding_dim) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t numel = num_indices * embedding_dim;
    if (i < numel) {
        const size_t row = i / embedding_dim;
        const int64_t source_row = index[row];
        if (source_row >= 0 && static_cast<size_t>(source_row) < num_embeddings) {
            out[i] = weight[static_cast<size_t>(source_row) * embedding_dim + i % embedding_dim];
        }
    }
}

template <typename T>
__global__ void addBiasKernel(T *out, const T *bias, size_t numel, size_t width) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < numel) {
        out[i] = fromFloat<T>(toFloat(out[i]) + toFloat(bias[i % width]));
    }
}

template <typename T>
__global__ void rmsNormKernel(T *out, const T *in, const T *weight, size_t hidden, float eps) {
    __shared__ float partial[BLOCK_SIZE];
    const size_t row = blockIdx.x;
    float sum = 0.0f;
    for (size_t col = threadIdx.x; col < hidden; col += blockDim.x) {
        const float value = toFloat(in[row * hidden + col]);
        sum += value * value;
    }
    partial[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv_rms = rsqrtf(partial[0] / static_cast<float>(hidden) + eps);
    for (size_t col = threadIdx.x; col < hidden; col += blockDim.x) {
        const size_t i = row * hidden + col;
        out[i] = fromFloat<T>(toFloat(in[i]) * inv_rms * toFloat(weight[col]));
    }
}

template <typename T>
__global__ void ropeKernel(T *out,
                           const T *in,
                           const int64_t *pos_ids,
                           size_t num_heads,
                           size_t head_dim,
                           float theta,
                           size_t num_pairs) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_pairs) {
        return;
    }
    const size_t half = head_dim / 2;
    const size_t pair = i % half;
    const size_t vector = i / half;
    const size_t seq = vector / num_heads;
    const size_t base = vector * head_dim;
    const float angle = static_cast<float>(pos_ids[seq])
                      / powf(theta, 2.0f * static_cast<float>(pair) / static_cast<float>(head_dim));
    float sin_value;
    float cos_value;
    sincosf(angle, &sin_value, &cos_value);
    const float a = toFloat(in[base + pair]);
    const float b = toFloat(in[base + half + pair]);
    out[base + pair] = fromFloat<T>(a * cos_value - b * sin_value);
    out[base + half + pair] = fromFloat<T>(b * cos_value + a * sin_value);
}

template <typename T>
__global__ void attentionScoresKernel(float *scores,
                                      const T *q,
                                      const T *k,
                                      size_t total_scores,
                                      size_t q_len,
                                      size_t kv_len,
                                      size_t num_heads,
                                      size_t num_kv_heads,
                                      size_t head_dim,
                                      float scale) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total_scores) {
        return;
    }
    const size_t key_pos = i % kv_len;
    const size_t row = i / kv_len;
    const size_t head = row % num_heads;
    const size_t query_pos = row / num_heads;
    const size_t last_allowed = query_pos + kv_len - q_len;
    if (key_pos > last_allowed) {
        scores[i] = -CUDART_INF_F;
        return;
    }
    const size_t kv_head = head / (num_heads / num_kv_heads);
    const size_t q_base = (query_pos * num_heads + head) * head_dim;
    const size_t k_base = (key_pos * num_kv_heads + kv_head) * head_dim;
    float dot = 0.0f;
    for (size_t d = 0; d < head_dim; ++d) {
        dot += toFloat(q[q_base + d]) * toFloat(k[k_base + d]);
    }
    scores[i] = dot * scale;
}

__global__ void softmaxKernel(float *scores, size_t kv_len) {
    __shared__ float partial[BLOCK_SIZE];
    const size_t base = static_cast<size_t>(blockIdx.x) * kv_len;
    float max_value = -CUDART_INF_F;
    for (size_t i = threadIdx.x; i < kv_len; i += blockDim.x) {
        max_value = fmaxf(max_value, scores[base + i]);
    }
    partial[threadIdx.x] = max_value;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float row_max = partial[0];

    float sum = 0.0f;
    for (size_t i = threadIdx.x; i < kv_len; i += blockDim.x) {
        const float value = expf(scores[base + i] - row_max);
        scores[base + i] = value;
        sum += value;
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inverse_sum = 1.0f / partial[0];
    for (size_t i = threadIdx.x; i < kv_len; i += blockDim.x) {
        scores[base + i] *= inverse_sum;
    }
}

template <typename T>
__global__ void attentionValuesKernel(T *out,
                                      const float *scores,
                                      const T *v,
                                      size_t output_numel,
                                      size_t kv_len,
                                      size_t num_heads,
                                      size_t num_kv_heads,
                                      size_t head_dim) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= output_numel) {
        return;
    }
    const size_t dim = i % head_dim;
    const size_t row = i / head_dim;
    const size_t head = row % num_heads;
    const size_t kv_head = head / (num_heads / num_kv_heads);
    float value = 0.0f;
    for (size_t key_pos = 0; key_pos < kv_len; ++key_pos) {
        value += scores[row * kv_len + key_pos]
               * toFloat(v[(key_pos * num_kv_heads + kv_head) * head_dim + dim]);
    }
    out[i] = fromFloat<T>(value);
}

template <typename T>
__global__ void swigluKernel(T *out, const T *gate, const T *up, size_t numel) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < numel) {
        const float gate_value = toFloat(gate[i]);
        out[i] = fromFloat<T>(toFloat(up[i]) * gate_value / (1.0f + expf(-gate_value)));
    }
}

template <typename T>
struct TypeTag {
    using type = T;
};

template <typename Function>
void dispatchFloatTypes(llaisysDataType_t dtype, Function &&function) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return function(TypeTag<float>{});
    case LLAISYS_DTYPE_F16:
        return function(TypeTag<__half>{});
    case LLAISYS_DTYPE_BF16:
        return function(TypeTag<__nv_bfloat16>{});
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

class CublasHandleCache {
private:
    cublasHandle_t _handle = nullptr;
    int _device = -1;

public:
    ~CublasHandleCache() {
        if (_handle != nullptr) {
            cublasDestroy(_handle);
        }
    }

    cublasHandle_t get(cudaStream_t stream) {
        int device = 0;
        checkCuda(cudaGetDevice(&device), "cudaGetDevice");
        if (_handle == nullptr || _device != device) {
            if (_handle != nullptr) {
                checkCublas(cublasDestroy(_handle), "cublasDestroy");
            }
            checkCublas(cublasCreate(&_handle), "cublasCreate");
            _device = device;
        }
        checkCublas(cublasSetStream(_handle, stream), "cublasSetStream");
        return _handle;
    }
};

thread_local CublasHandleCache cublas_cache;
} // namespace

void add(std::byte *out, const std::byte *a, const std::byte *b, llaisysDataType_t dtype, size_t numel, llaisysStream_t stream_) {
    const auto stream = reinterpret_cast<cudaStream_t>(stream_);
    dispatchFloatTypes(dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        addKernel<<<gridSize(numel), BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<T *>(out), reinterpret_cast<const T *>(a), reinterpret_cast<const T *>(b), numel);
    });
    checkLaunch("addKernel");
}

void argmax(std::byte *max_idx,
            std::byte *max_val,
            const std::byte *vals,
            llaisysDataType_t dtype,
            size_t numel,
            llaisysStream_t stream_) {
    const auto stream = reinterpret_cast<cudaStream_t>(stream_);
    dispatchFloatTypes(dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        argmaxKernel<<<1, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<T *>(max_val), reinterpret_cast<const T *>(vals), numel);
    });
    checkLaunch("argmaxKernel");
}

void embedding(std::byte *out,
               const std::byte *index,
               const std::byte *weight,
               llaisysDataType_t dtype,
               size_t num_indices,
               size_t num_embeddings,
               size_t embedding_dim,
               llaisysStream_t stream_) {
    const auto stream = reinterpret_cast<cudaStream_t>(stream_);
    const size_t numel = num_indices * embedding_dim;
    dispatchFloatTypes(dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        embeddingKernel<<<gridSize(numel), BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<T *>(out),
            reinterpret_cast<const int64_t *>(index),
            reinterpret_cast<const T *>(weight),
            num_indices,
            num_embeddings,
            embedding_dim);
    });
    checkLaunch("embeddingKernel");
}

void linear(std::byte *out,
            const std::byte *in,
            const std::byte *weight,
            const std::byte *bias,
            llaisysDataType_t dtype,
            size_t m,
            size_t n,
            size_t k,
            llaisysStream_t stream_) {
    CHECK_ARGUMENT(m <= static_cast<size_t>(std::numeric_limits<int>::max())
                       && n <= static_cast<size_t>(std::numeric_limits<int>::max())
                       && k <= static_cast<size_t>(std::numeric_limits<int>::max()),
                   "Linear dimensions exceed cuBLAS limits.");
    const auto stream = reinterpret_cast<cudaStream_t>(stream_);
    const cublasHandle_t handle = cublas_cache.get(stream);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cudaDataType_t data_type;
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        data_type = CUDA_R_32F;
        break;
    case LLAISYS_DTYPE_F16:
        data_type = CUDA_R_16F;
        break;
    case LLAISYS_DTYPE_BF16:
        data_type = CUDA_R_16BF;
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
    checkCublas(
        cublasGemmEx(handle,
                     CUBLAS_OP_T,
                     CUBLAS_OP_N,
                     static_cast<int>(n),
                     static_cast<int>(m),
                     static_cast<int>(k),
                     &alpha,
                     weight,
                     data_type,
                     static_cast<int>(k),
                     in,
                     data_type,
                     static_cast<int>(k),
                     &beta,
                     out,
                     data_type,
                     static_cast<int>(n),
                     CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "cublasGemmEx");

    if (bias != nullptr) {
        const size_t numel = m * n;
        dispatchFloatTypes(dtype, [&](auto tag) {
            using T = typename decltype(tag)::type;
            addBiasKernel<<<gridSize(numel), BLOCK_SIZE, 0, stream>>>(
                reinterpret_cast<T *>(out), reinterpret_cast<const T *>(bias), numel, n);
        });
        checkLaunch("addBiasKernel");
    }
}

void rms_norm(std::byte *out,
              const std::byte *in,
              const std::byte *weight,
              llaisysDataType_t dtype,
              size_t rows,
              size_t hidden,
              float eps,
              llaisysStream_t stream_) {
    CHECK_ARGUMENT(rows <= std::numeric_limits<unsigned int>::max(), "RMSNorm row count is too large.");
    const auto stream = reinterpret_cast<cudaStream_t>(stream_);
    dispatchFloatTypes(dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        rmsNormKernel<<<static_cast<unsigned int>(rows), BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in), reinterpret_cast<const T *>(weight), hidden, eps);
    });
    checkLaunch("rmsNormKernel");
}

void rope(std::byte *out,
          const std::byte *in,
          const std::byte *pos_ids,
          llaisysDataType_t dtype,
          size_t seq_len,
          size_t num_heads,
          size_t head_dim,
          float theta,
          llaisysStream_t stream_) {
    const auto stream = reinterpret_cast<cudaStream_t>(stream_);
    const size_t num_pairs = seq_len * num_heads * head_dim / 2;
    dispatchFloatTypes(dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        ropeKernel<<<gridSize(num_pairs), BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<T *>(out),
            reinterpret_cast<const T *>(in),
            reinterpret_cast<const int64_t *>(pos_ids),
            num_heads,
            head_dim,
            theta,
            num_pairs);
    });
    checkLaunch("ropeKernel");
}

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
                    llaisysStream_t stream_) {
    const auto stream = reinterpret_cast<cudaStream_t>(stream_);
    const size_t rows = q_len * num_heads;
    const size_t total_scores = rows * kv_len;
    float *scores = nullptr;
    checkCuda(cudaMallocAsync(reinterpret_cast<void **>(&scores), total_scores * sizeof(float), stream), "cudaMallocAsync");

    dispatchFloatTypes(dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        attentionScoresKernel<<<gridSize(total_scores), BLOCK_SIZE, 0, stream>>>(
            scores,
            reinterpret_cast<const T *>(q),
            reinterpret_cast<const T *>(k),
            total_scores,
            q_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            scale);
    });
    checkLaunch("attentionScoresKernel");

    CHECK_ARGUMENT(rows <= std::numeric_limits<unsigned int>::max(), "SelfAttention row count is too large.");
    softmaxKernel<<<static_cast<unsigned int>(rows), BLOCK_SIZE, 0, stream>>>(scores, kv_len);
    checkLaunch("softmaxKernel");

    const size_t output_numel = rows * head_dim;
    dispatchFloatTypes(dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        attentionValuesKernel<<<gridSize(output_numel), BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<T *>(out),
            scores,
            reinterpret_cast<const T *>(v),
            output_numel,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim);
    });
    checkLaunch("attentionValuesKernel");
    checkCuda(cudaFreeAsync(scores, stream), "cudaFreeAsync");
}

void swiglu(std::byte *out,
            const std::byte *gate,
            const std::byte *up,
            llaisysDataType_t dtype,
            size_t numel,
            llaisysStream_t stream_) {
    const auto stream = reinterpret_cast<cudaStream_t>(stream_);
    dispatchFloatTypes(dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        swigluKernel<<<gridSize(numel), BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<T *>(out), reinterpret_cast<const T *>(gate), reinterpret_cast<const T *>(up), numel);
    });
    checkLaunch("swigluKernel");
}
} // namespace llaisys::ops::nvidia
