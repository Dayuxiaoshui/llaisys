#pragma once

#include "llaisys/models/qwen2.h"

#include "../../tensor/tensor.hpp"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace llaisys::models {
class Qwen2Model {
private:
    LlaisysQwen2Meta _meta;
    llaisysDeviceType_t _device_type;
    int _device_id;
    LlaisysQwen2Weights _weights{};
    std::unordered_map<std::string, llaisysTensor_t> _weight_map;
    std::unordered_set<std::string> _loaded_weights;
    std::vector<tensor_t> _key_cache;
    std::vector<tensor_t> _value_cache;
    size_t _cache_length = 0;
    size_t _cache_capacity = 0;

    llaisysTensor_t createWeight(const std::vector<size_t> &shape);
    void createWeights();
    void destroyWeights();
    void ensureCacheCapacity(size_t required_capacity);

public:
    Qwen2Model(const LlaisysQwen2Meta &meta, llaisysDeviceType_t device_type, int device_id);
    ~Qwen2Model();

    Qwen2Model(const Qwen2Model &) = delete;
    Qwen2Model &operator=(const Qwen2Model &) = delete;

    LlaisysQwen2Weights *weights();
    void loadWeight(const std::string &name, const void *data, size_t nbytes);
    void reset();
    int64_t infer(const int64_t *token_ids, size_t ntoken);
};
} // namespace llaisys::models
