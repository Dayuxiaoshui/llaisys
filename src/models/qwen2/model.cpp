#include "model.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../llaisys/llaisys_tensor.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace llaisys::models {
namespace {
tensor_t unwrap(llaisysTensor_t tensor) {
    return tensor->tensor;
}
} // namespace

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device_type, int device_id)
    : _meta(meta), _device_type(device_type), _device_id(device_id), _key_cache(meta.nlayer), _value_cache(meta.nlayer) {
    CHECK_ARGUMENT(meta.nlayer > 0 && meta.hs > 0 && meta.nh > 0 && meta.nkvh > 0 && meta.dh > 0 && meta.di > 0
                       && meta.maxseq > 0 && meta.voc > 0,
                   "Qwen2 metadata dimensions must be positive.");
    CHECK_ARGUMENT(meta.nh * meta.dh == meta.hs, "Qwen2 hidden size must equal num_heads * head_dim.");
    CHECK_ARGUMENT(meta.nh % meta.nkvh == 0, "Qwen2 query heads must be divisible by KV heads.");
    CHECK_ARGUMENT(meta.dtype == LLAISYS_DTYPE_F32 || meta.dtype == LLAISYS_DTYPE_F16 || meta.dtype == LLAISYS_DTYPE_BF16,
                   "Qwen2 only supports float32, float16, and bfloat16 weights.");
    core::context().setDevice(device_type, device_id);
    createWeights();
}

Qwen2Model::~Qwen2Model() {
    destroyWeights();
}

llaisysTensor_t Qwen2Model::createWeight(const std::vector<size_t> &shape) {
    return new LlaisysTensor{Tensor::create(shape, _meta.dtype, _device_type, _device_id)};
}

void Qwen2Model::createWeights() {
    _weights.in_embed = createWeight({_meta.voc, _meta.hs});
    _weights.out_embed = createWeight({_meta.voc, _meta.hs});
    _weights.out_norm_w = createWeight({_meta.hs});

    const auto allocate_layers = [this](llaisysTensor_t *&handles) {
        handles = new llaisysTensor_t[_meta.nlayer]{};
    };
    allocate_layers(_weights.attn_norm_w);
    allocate_layers(_weights.attn_q_w);
    allocate_layers(_weights.attn_q_b);
    allocate_layers(_weights.attn_k_w);
    allocate_layers(_weights.attn_k_b);
    allocate_layers(_weights.attn_v_w);
    allocate_layers(_weights.attn_v_b);
    allocate_layers(_weights.attn_o_w);
    allocate_layers(_weights.mlp_norm_w);
    allocate_layers(_weights.mlp_gate_w);
    allocate_layers(_weights.mlp_up_w);
    allocate_layers(_weights.mlp_down_w);

    _weight_map.emplace("model.embed_tokens.weight", _weights.in_embed);
    _weight_map.emplace("lm_head.weight", _weights.out_embed);
    _weight_map.emplace("model.norm.weight", _weights.out_norm_w);

    const size_t kv_dim = _meta.nkvh * _meta.dh;
    for (size_t layer = 0; layer < _meta.nlayer; ++layer) {
        _weights.attn_norm_w[layer] = createWeight({_meta.hs});
        _weights.attn_q_w[layer] = createWeight({_meta.hs, _meta.hs});
        _weights.attn_q_b[layer] = createWeight({_meta.hs});
        _weights.attn_k_w[layer] = createWeight({kv_dim, _meta.hs});
        _weights.attn_k_b[layer] = createWeight({kv_dim});
        _weights.attn_v_w[layer] = createWeight({kv_dim, _meta.hs});
        _weights.attn_v_b[layer] = createWeight({kv_dim});
        _weights.attn_o_w[layer] = createWeight({_meta.hs, _meta.hs});
        _weights.mlp_norm_w[layer] = createWeight({_meta.hs});
        _weights.mlp_gate_w[layer] = createWeight({_meta.di, _meta.hs});
        _weights.mlp_up_w[layer] = createWeight({_meta.di, _meta.hs});
        _weights.mlp_down_w[layer] = createWeight({_meta.hs, _meta.di});

        const std::string prefix = "model.layers." + std::to_string(layer) + ".";
        _weight_map.emplace(prefix + "input_layernorm.weight", _weights.attn_norm_w[layer]);
        _weight_map.emplace(prefix + "self_attn.q_proj.weight", _weights.attn_q_w[layer]);
        _weight_map.emplace(prefix + "self_attn.q_proj.bias", _weights.attn_q_b[layer]);
        _weight_map.emplace(prefix + "self_attn.k_proj.weight", _weights.attn_k_w[layer]);
        _weight_map.emplace(prefix + "self_attn.k_proj.bias", _weights.attn_k_b[layer]);
        _weight_map.emplace(prefix + "self_attn.v_proj.weight", _weights.attn_v_w[layer]);
        _weight_map.emplace(prefix + "self_attn.v_proj.bias", _weights.attn_v_b[layer]);
        _weight_map.emplace(prefix + "self_attn.o_proj.weight", _weights.attn_o_w[layer]);
        _weight_map.emplace(prefix + "post_attention_layernorm.weight", _weights.mlp_norm_w[layer]);
        _weight_map.emplace(prefix + "mlp.gate_proj.weight", _weights.mlp_gate_w[layer]);
        _weight_map.emplace(prefix + "mlp.up_proj.weight", _weights.mlp_up_w[layer]);
        _weight_map.emplace(prefix + "mlp.down_proj.weight", _weights.mlp_down_w[layer]);
    }
}

void Qwen2Model::destroyWeights() {
    for (auto &[name, handle] : _weight_map) {
        delete handle;
        handle = nullptr;
    }
    _weight_map.clear();

    delete[] _weights.attn_norm_w;
    delete[] _weights.attn_q_w;
    delete[] _weights.attn_q_b;
    delete[] _weights.attn_k_w;
    delete[] _weights.attn_k_b;
    delete[] _weights.attn_v_w;
    delete[] _weights.attn_v_b;
    delete[] _weights.attn_o_w;
    delete[] _weights.mlp_norm_w;
    delete[] _weights.mlp_gate_w;
    delete[] _weights.mlp_up_w;
    delete[] _weights.mlp_down_w;
    _weights = {};
}

LlaisysQwen2Weights *Qwen2Model::weights() {
    return &_weights;
}

void Qwen2Model::loadWeight(const std::string &name, const void *data, size_t nbytes) {
    const auto it = _weight_map.find(name);
    CHECK_ARGUMENT(it != _weight_map.end(), "Unknown Qwen2 weight: " + name);
    const tensor_t tensor = unwrap(it->second);
    CHECK_ARGUMENT(nbytes == tensor->numel() * tensor->elementSize(), "Qwen2 weight size mismatch: " + name);
    tensor->load(data);
    _loaded_weights.insert(name);
}

void Qwen2Model::ensureCacheCapacity(size_t required_capacity) {
    if (required_capacity <= _cache_capacity) {
        return;
    }
    CHECK_ARGUMENT(required_capacity <= _meta.maxseq, "Qwen2 KV cache exceeds maximum sequence length.");

    size_t new_capacity = std::min<size_t>(_meta.maxseq, std::max<size_t>(256, _cache_capacity));
    while (new_capacity < required_capacity) {
        new_capacity = std::min(_meta.maxseq, new_capacity * 2);
    }

    std::vector<tensor_t> new_keys(_meta.nlayer);
    std::vector<tensor_t> new_values(_meta.nlayer);
    for (size_t layer = 0; layer < _meta.nlayer; ++layer) {
        new_keys[layer] = Tensor::create({new_capacity, _meta.nkvh, _meta.dh}, _meta.dtype, _device_type, _device_id);
        new_values[layer] = Tensor::create({new_capacity, _meta.nkvh, _meta.dh}, _meta.dtype, _device_type, _device_id);
    }

    if (_cache_length > 0) {
        auto &runtime = core::context().runtime();
        const size_t cache_bytes = _cache_length * _meta.nkvh * _meta.dh * utils::dsize(_meta.dtype);
        for (size_t layer = 0; layer < _meta.nlayer; ++layer) {
            runtime.api()->memcpy_async(
                new_keys[layer]->data(), _key_cache[layer]->data(), cache_bytes, LLAISYS_MEMCPY_D2D, runtime.stream());
            runtime.api()->memcpy_async(
                new_values[layer]->data(), _value_cache[layer]->data(), cache_bytes, LLAISYS_MEMCPY_D2D, runtime.stream());
        }
        runtime.synchronize();
    }

    _key_cache = std::move(new_keys);
    _value_cache = std::move(new_values);
    _cache_capacity = new_capacity;
}

void Qwen2Model::reset() {
    _cache_length = 0;
}

int64_t Qwen2Model::infer(const int64_t *token_ids, size_t ntoken) {
    CHECK_ARGUMENT(token_ids != nullptr && ntoken > 0, "Qwen2 inference requires at least one token.");
    CHECK_ARGUMENT(_loaded_weights.size() == _weight_map.size(), "Qwen2 weights are not fully loaded.");
    CHECK_ARGUMENT(_cache_length + ntoken <= _meta.maxseq, "Qwen2 input exceeds maximum sequence length.");
    core::context().setDevice(_device_type, _device_id);
    ensureCacheCapacity(_cache_length + ntoken);

    const auto make_tensor = [this](const std::vector<size_t> &shape, llaisysDataType_t dtype) {
        return Tensor::create(shape, dtype, _device_type, _device_id);
    };
    const auto make_model_tensor = [this, &make_tensor](const std::vector<size_t> &shape) {
        return make_tensor(shape, _meta.dtype);
    };

    auto indices = make_tensor({ntoken}, LLAISYS_DTYPE_I64);
    indices->load(token_ids);
    std::vector<int64_t> positions(ntoken);
    for (size_t i = 0; i < ntoken; ++i) {
        positions[i] = static_cast<int64_t>(_cache_length + i);
    }
    auto pos_ids = make_tensor({ntoken}, LLAISYS_DTYPE_I64);
    pos_ids->load(positions.data());

    auto hidden = make_model_tensor({ntoken, _meta.hs});
    auto norm = make_model_tensor({ntoken, _meta.hs});
    auto q_flat = make_model_tensor({ntoken, _meta.hs});
    const size_t kv_dim = _meta.nkvh * _meta.dh;
    auto k_flat = make_model_tensor({ntoken, kv_dim});
    auto v_flat = make_model_tensor({ntoken, kv_dim});
    auto q_rotated = make_model_tensor({ntoken, _meta.nh, _meta.dh});
    auto k_rotated = make_model_tensor({ntoken, _meta.nkvh, _meta.dh});
    auto attention = make_model_tensor({ntoken, _meta.nh, _meta.dh});
    auto projection = make_model_tensor({ntoken, _meta.hs});
    auto gate = make_model_tensor({ntoken, _meta.di});
    auto up = make_model_tensor({ntoken, _meta.di});
    auto activated = make_model_tensor({ntoken, _meta.di});
    auto down = make_model_tensor({ntoken, _meta.hs});

    ops::embedding(hidden, indices, unwrap(_weights.in_embed));
    const size_t total_length = _cache_length + ntoken;
    auto &runtime = core::context().runtime();
    const size_t appended_cache_bytes = ntoken * kv_dim * utils::dsize(_meta.dtype);

    for (size_t layer = 0; layer < _meta.nlayer; ++layer) {
        ops::rms_norm(norm, hidden, unwrap(_weights.attn_norm_w[layer]), _meta.epsilon);
        ops::linear(q_flat, norm, unwrap(_weights.attn_q_w[layer]), unwrap(_weights.attn_q_b[layer]));
        ops::linear(k_flat, norm, unwrap(_weights.attn_k_w[layer]), unwrap(_weights.attn_k_b[layer]));
        ops::linear(v_flat, norm, unwrap(_weights.attn_v_w[layer]), unwrap(_weights.attn_v_b[layer]));

        auto q = q_flat->view({ntoken, _meta.nh, _meta.dh});
        auto k = k_flat->view({ntoken, _meta.nkvh, _meta.dh});
        auto v = v_flat->view({ntoken, _meta.nkvh, _meta.dh});
        ops::rope(q_rotated, q, pos_ids, _meta.theta);
        ops::rope(k_rotated, k, pos_ids, _meta.theta);

        auto key_destination = _key_cache[layer]->slice(0, _cache_length, total_length);
        auto value_destination = _value_cache[layer]->slice(0, _cache_length, total_length);
        runtime.api()->memcpy_async(
            key_destination->data(), k_rotated->data(), appended_cache_bytes, LLAISYS_MEMCPY_D2D, runtime.stream());
        runtime.api()->memcpy_async(
            value_destination->data(), v->data(), appended_cache_bytes, LLAISYS_MEMCPY_D2D, runtime.stream());

        auto keys = _key_cache[layer]->slice(0, 0, total_length);
        auto values = _value_cache[layer]->slice(0, 0, total_length);
        ops::self_attention(attention, q_rotated, keys, values, 1.0f / std::sqrt(static_cast<float>(_meta.dh)));
        ops::linear(projection, attention->view({ntoken, _meta.hs}), unwrap(_weights.attn_o_w[layer]), nullptr);
        ops::add(hidden, hidden, projection);

        ops::rms_norm(norm, hidden, unwrap(_weights.mlp_norm_w[layer]), _meta.epsilon);
        ops::linear(gate, norm, unwrap(_weights.mlp_gate_w[layer]), nullptr);
        ops::linear(up, norm, unwrap(_weights.mlp_up_w[layer]), nullptr);
        ops::swiglu(activated, gate, up);
        ops::linear(down, activated, unwrap(_weights.mlp_down_w[layer]), nullptr);
        ops::add(hidden, hidden, down);
    }
    _cache_length = total_length;

    ops::rms_norm(norm, hidden, unwrap(_weights.out_norm_w), _meta.epsilon);
    auto last_hidden = norm->slice(0, ntoken - 1, ntoken);
    auto logits = make_model_tensor({1, _meta.voc});
    ops::linear(logits, last_hidden, unwrap(_weights.out_embed), nullptr);
    auto max_index = make_tensor({1}, LLAISYS_DTYPE_I64);
    auto max_value = make_model_tensor({1});
    ops::argmax(max_index, max_value, logits->view({_meta.voc}));

    runtime.synchronize();
    int64_t result = 0;
    runtime.api()->memcpy_sync(&result, max_index->data(), sizeof(result), LLAISYS_MEMCPY_D2H);
    return result;
}
} // namespace llaisys::models
