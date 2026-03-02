#pragma once

#include <instant_tensor/dl_loader/dl_loader_utils.hpp>

#define cudaStreamDefault 0x00
#define cudaStreamLegacy ((cudaStream_t)0x1)
#define cudaStreamNonBlocking 0x01
#define cudaStreamPerThread ((cudaStream_t)0x2)

#define cudaEventDefault 0x00
#define cudaEventBlockingSync 0x01
#define cudaEventDisableTiming 0x02
#define cudaEventInterprocess 0x04

#define cudaHostRegisterDefault 0x00
#define cudaHostRegisterPortable 0x01
#define cudaHostRegisterMapped 0x02
#define cudaHostRegisterIoMemory 0x04
#define cudaHostRegisterReadOnly 0x08

namespace instanttensor {
namespace cuda_loader {

// Opaque types (match CUDA runtime ABI)
typedef void* cudaStream_t;
typedef void* cudaEvent_t;

enum cudaError_t {
    cudaSuccess = 0,
    // ... other codes; cudaGetErrorString handles unknown
};

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
};

static void* lib_handle = nullptr;

static cudaError_t (*cudaMalloc_fn)(void** devPtr, size_t size) = nullptr;
static cudaError_t (*cudaFree_fn)(void* devPtr) = nullptr;
static cudaError_t (*cudaSetDevice_fn)(int device) = nullptr;
static cudaError_t (*cudaStreamCreateWithFlags_fn)(cudaStream_t* pStream, unsigned int flags) = nullptr;
static cudaError_t (*cudaStreamDestroy_fn)(cudaStream_t stream) = nullptr;
static cudaError_t (*cudaMemcpyAsync_fn)(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) = nullptr;
static cudaError_t (*cudaEventCreateWithFlags_fn)(cudaEvent_t* event, unsigned int flags) = nullptr;
static cudaError_t (*cudaEventRecord_fn)(cudaEvent_t event, cudaStream_t stream) = nullptr;
static cudaError_t (*cudaEventSynchronize_fn)(cudaEvent_t event) = nullptr;
static cudaError_t (*cudaStreamWaitEvent_fn)(cudaStream_t stream, cudaEvent_t event, unsigned int flags) = nullptr;
static cudaError_t (*cudaHostRegister_fn)(void* ptr, size_t size, unsigned int flags) = nullptr;
static cudaError_t (*cudaHostUnregister_fn)(void* ptr) = nullptr;
static const char* (*cudaGetErrorString_fn)(cudaError_t error) = nullptr;

inline bool init() {
    if (lib_handle != nullptr) {
        if (lib_handle == (void*)0x1) return false;
        return true;
    }

    lib_handle = dl_loader_utils::find_loaded_so("cudart");
    if (lib_handle == nullptr) {
        lib_handle = (void*)0x1;
        return false;
    }

    using dl_loader_utils::resolve;
    cudaMalloc_fn = resolve<decltype(cudaMalloc_fn)>(lib_handle, "cudaMalloc");
    cudaFree_fn = resolve<decltype(cudaFree_fn)>(lib_handle, "cudaFree");
    cudaSetDevice_fn = resolve<decltype(cudaSetDevice_fn)>(lib_handle, "cudaSetDevice");
    cudaStreamCreateWithFlags_fn = resolve<decltype(cudaStreamCreateWithFlags_fn)>(lib_handle, "cudaStreamCreateWithFlags");
    cudaStreamDestroy_fn = resolve<decltype(cudaStreamDestroy_fn)>(lib_handle, "cudaStreamDestroy");
    cudaMemcpyAsync_fn = resolve<decltype(cudaMemcpyAsync_fn)>(lib_handle, "cudaMemcpyAsync");
    cudaEventCreateWithFlags_fn = resolve<decltype(cudaEventCreateWithFlags_fn)>(lib_handle, "cudaEventCreateWithFlags");
    cudaEventRecord_fn = resolve<decltype(cudaEventRecord_fn)>(lib_handle, "cudaEventRecord");
    cudaEventSynchronize_fn = resolve<decltype(cudaEventSynchronize_fn)>(lib_handle, "cudaEventSynchronize");
    cudaStreamWaitEvent_fn = resolve<decltype(cudaStreamWaitEvent_fn)>(lib_handle, "cudaStreamWaitEvent");
    cudaHostRegister_fn = resolve<decltype(cudaHostRegister_fn)>(lib_handle, "cudaHostRegister");
    cudaHostUnregister_fn = resolve<decltype(cudaHostUnregister_fn)>(lib_handle, "cudaHostUnregister");
    cudaGetErrorString_fn = resolve<decltype(cudaGetErrorString_fn)>(lib_handle, "cudaGetErrorString");

    return true;
}

// Wrappers with same names as CUDA Runtime API (used by main.cpp)
inline cudaError_t cudaMalloc(void** devPtr, size_t size) {
    return cudaMalloc_fn(devPtr, size);
}
inline cudaError_t cudaFree(void* devPtr) {
    return cudaFree_fn(devPtr);
}
inline cudaError_t cudaSetDevice(int device) {
    return cudaSetDevice_fn(device);
}
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags) {
    return cudaStreamCreateWithFlags_fn(pStream, flags);
}
inline cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    return cudaStreamDestroy_fn(stream);
}
inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0) {
    return cudaMemcpyAsync_fn(dst, src, count, kind, stream);
}
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
    return cudaEventCreateWithFlags_fn(event, flags);
}
inline cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0) {
    return cudaEventRecord_fn(event, stream);
}
inline cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    return cudaEventSynchronize_fn(event);
}
inline cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags = 0) {
    return cudaStreamWaitEvent_fn(stream, event, flags);
}
inline cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags) {
    return cudaHostRegister_fn(ptr, size, flags);
}
inline cudaError_t cudaHostUnregister(void* ptr) {
    return cudaHostUnregister_fn(ptr);
}
inline const char* cudaGetErrorString(cudaError_t error) {
    return cudaGetErrorString_fn(error);
}

} // namespace cuda_loader
}