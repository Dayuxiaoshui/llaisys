#include "llaisys/models/qwen2.h"

#include "../models/qwen2/model.hpp"
#include "../utils.hpp"

#include <memory>

struct LlaisysQwen2Model {
    std::unique_ptr<llaisys::models::Qwen2Model> impl;
};

__C {
    LlaisysQwen2Model *llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta *meta,
        llaisysDeviceType_t device,
        int *device_ids,
        int ndevice) {
        CHECK_ARGUMENT(meta != nullptr, "Qwen2 metadata must not be null.");
        CHECK_ARGUMENT(device_ids != nullptr && ndevice == 1, "Qwen2 currently supports exactly one device.");
        return new LlaisysQwen2Model{std::make_unique<llaisys::models::Qwen2Model>(*meta, device, device_ids[0])};
    }

    void llaisysQwen2ModelDestroy(LlaisysQwen2Model *model) {
        delete model;
    }

    LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model *model) {
        CHECK_ARGUMENT(model != nullptr, "Qwen2 model must not be null.");
        return model->impl->weights();
    }

    void llaisysQwen2ModelLoadWeight(LlaisysQwen2Model *model, const char *name, const void *data, size_t nbytes) {
        CHECK_ARGUMENT(model != nullptr && name != nullptr && data != nullptr, "Invalid Qwen2 weight load arguments.");
        model->impl->loadWeight(name, data, nbytes);
    }

    void llaisysQwen2ModelReset(LlaisysQwen2Model *model) {
        CHECK_ARGUMENT(model != nullptr, "Qwen2 model must not be null.");
        model->impl->reset();
    }

    int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
        CHECK_ARGUMENT(model != nullptr, "Qwen2 model must not be null.");
        return model->impl->infer(token_ids, ntoken);
    }
}
