#include "../runtime_api.hpp"

#include <mc_runtime.h>

#include <stdexcept>
#include <string>

namespace llaisys::device::metax {

namespace runtime_api {
namespace {
void checkMaca(mcError_t status, const char *operation) {
    if (status != mcSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + mcGetErrorString(status));
    }
}

mcMemcpyKind toMacaMemcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        return mcMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D:
        return mcMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return mcMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
        return mcMemcpyDeviceToDevice;
    default:
        throw std::invalid_argument("Invalid memcpy kind");
    }
}
} // namespace

int getDeviceCount() {
    int count = 0;
    const mcError_t status = mcGetDeviceCount(&count);
    if (status != mcSuccess) {
        // A MACA-enabled build must remain usable on hosts without a working
        // MetaX driver. Other runtime calls still report MACA errors strictly.
        mcGetLastError();
        return 0;
    }
    return count;
}

void setDevice(int device) {
    checkMaca(mcSetDevice(device), "mcSetDevice");
}

void deviceSynchronize() {
    checkMaca(mcDeviceSynchronize(), "mcDeviceSynchronize");
}

llaisysStream_t createStream() {
    mcStream_t stream = nullptr;
    checkMaca(mcStreamCreate(&stream), "mcStreamCreate");
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    if (stream != nullptr) {
        checkMaca(mcStreamDestroy(reinterpret_cast<mcStream_t>(stream)), "mcStreamDestroy");
    }
}
void streamSynchronize(llaisysStream_t stream) {
    checkMaca(mcStreamSynchronize(reinterpret_cast<mcStream_t>(stream)), "mcStreamSynchronize");
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    checkMaca(mcMalloc(&ptr, size), "mcMalloc");
    return ptr;
}

void freeDevice(void *ptr) {
    if (ptr != nullptr) {
        checkMaca(mcFree(ptr), "mcFree");
    }
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    checkMaca(mcMallocHost(&ptr, size), "mcMallocHost");
    return ptr;
}

void freeHost(void *ptr) {
    if (ptr != nullptr) {
        checkMaca(mcFreeHost(ptr), "mcFreeHost");
    }
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    checkMaca(mcMemcpy(dst, src, size, toMacaMemcpyKind(kind)), "mcMemcpy");
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    checkMaca(mcMemcpyAsync(dst, src, size, toMacaMemcpyKind(kind), reinterpret_cast<mcStream_t>(stream)), "mcMemcpyAsync");
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::metax
