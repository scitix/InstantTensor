#pragma once

#include <instant_tensor/common.hpp>

#include <cstdlib>
#include <cstring>

#include <instant_tensor/dl_binding/dl_binding_utils.hpp>

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
namespace cuda_binding {

inline bool is_rocm = false;
// Host runtime: buffers live in host memory and copies are synchronous memcpy.
// For devices whose runtime cannot be bound here (Intel XPU); the caller copies
// each yielded host tensor to its device.
inline bool is_host = false;

// Opaque types (match CUDA runtime ABI)
typedef void* cudaStream_t;
typedef void* cudaEvent_t;

enum cudaError_t {
    cudaSuccess = 0,
    cudaErrorNotReady = 600,
    // ... other codes; cudaGetErrorString handles unknown
};

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
};

inline void* lib_handle = nullptr;

inline cudaError_t (*cudaMalloc_fn)(void** devPtr, size_t size) = nullptr;
inline cudaError_t (*cudaFree_fn)(void* devPtr) = nullptr;
inline cudaError_t (*cudaSetDevice_fn)(int device) = nullptr;
inline cudaError_t (*cudaStreamCreateWithFlags_fn)(cudaStream_t* pStream, unsigned int flags) = nullptr;
inline cudaError_t (*cudaStreamDestroy_fn)(cudaStream_t stream) = nullptr;
inline cudaError_t (*cudaMemcpyAsync_fn)(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) = nullptr;
inline cudaError_t (*cudaEventCreateWithFlags_fn)(cudaEvent_t* event, unsigned int flags) = nullptr;
inline cudaError_t (*cudaEventRecord_fn)(cudaEvent_t event, cudaStream_t stream) = nullptr;
inline cudaError_t (*cudaEventQuery_fn)(cudaEvent_t event) = nullptr;
inline cudaError_t (*cudaEventSynchronize_fn)(cudaEvent_t event) = nullptr;
inline cudaError_t (*cudaStreamWaitEvent_fn)(cudaStream_t stream, cudaEvent_t event, unsigned int flags) = nullptr;
inline cudaError_t (*cudaHostRegister_fn)(void* ptr, size_t size, unsigned int flags) = nullptr;
inline cudaError_t (*cudaHostUnregister_fn)(void* ptr) = nullptr;
inline const char* (*cudaGetErrorString_fn)(cudaError_t error) = nullptr;

namespace host_runtime {

constexpr size_t kAlignment = 4096;

inline cudaError_t malloc_(void** ptr, size_t size) {
    // aligned_alloc needs a size that is a nonzero multiple of the alignment;
    // a zero-byte request may otherwise return nullptr.
    size_t rounded = size == 0 ? kAlignment : (size + kAlignment - 1) / kAlignment * kAlignment;
    *ptr = std::aligned_alloc(kAlignment, rounded);
    return *ptr != nullptr ? cudaSuccess : static_cast<cudaError_t>(2);  // cudaErrorMemoryAllocation
}
inline cudaError_t free_(void* ptr) { std::free(ptr); return cudaSuccess; }
inline cudaError_t set_device(int) { return cudaSuccess; }
// Streams and events only need distinct non-null handles: every operation
// completes before it returns, so there is nothing to order or wait for.
inline cudaError_t stream_create(cudaStream_t* stream, unsigned int) {
    *stream = new char;
    return cudaSuccess;
}
inline cudaError_t stream_destroy(cudaStream_t stream) {
    delete static_cast<char*>(stream);
    return cudaSuccess;
}
inline cudaError_t memcpy_(void* dst, const void* src, size_t count, cudaMemcpyKind, cudaStream_t) {
    std::memcpy(dst, src, count);
    return cudaSuccess;
}
inline cudaError_t event_create(cudaEvent_t* event, unsigned int) {
    *event = new char;
    return cudaSuccess;
}
inline cudaError_t event_record(cudaEvent_t, cudaStream_t) { return cudaSuccess; }
inline cudaError_t event_query(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t event_synchronize(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t stream_wait_event(cudaStream_t, cudaEvent_t, unsigned int) { return cudaSuccess; }
inline cudaError_t host_register(void*, size_t, unsigned int) { return cudaSuccess; }
inline cudaError_t host_unregister(void*) { return cudaSuccess; }
inline const char* error_string(cudaError_t error) {
    return error == cudaSuccess ? "success" : "host runtime: allocation failed";
}

} // namespace host_runtime

// Select the host runtime. Must run before init() binds a device runtime;
// returns false if CUDA or HIP is already bound in this process.
inline bool init_host() {
    if (is_host) return true;
    // (void*)0x1 is init()'s "no device runtime found" marker, not a bound runtime.
    if (lib_handle != nullptr && lib_handle != (void*)0x1) return false;
    is_host = true;
    lib_handle = (void*)0x2;
    cudaMalloc_fn = host_runtime::malloc_;
    cudaFree_fn = host_runtime::free_;
    cudaSetDevice_fn = host_runtime::set_device;
    cudaStreamCreateWithFlags_fn = host_runtime::stream_create;
    cudaStreamDestroy_fn = host_runtime::stream_destroy;
    cudaMemcpyAsync_fn = host_runtime::memcpy_;
    cudaEventCreateWithFlags_fn = host_runtime::event_create;
    cudaEventRecord_fn = host_runtime::event_record;
    cudaEventQuery_fn = host_runtime::event_query;
    cudaEventSynchronize_fn = host_runtime::event_synchronize;
    cudaStreamWaitEvent_fn = host_runtime::stream_wait_event;
    cudaHostRegister_fn = host_runtime::host_register;
    cudaHostUnregister_fn = host_runtime::host_unregister;
    cudaGetErrorString_fn = host_runtime::error_string;
    return true;
}

inline bool init() {
    if (lib_handle != nullptr) {
        if (lib_handle == (void*)0x1) return false;
        return true;
    }

    // Try CUDA runtime first
    lib_handle = dl_binding_utils::find_loaded_so("cudart");

    // If CUDA runtime not found, try HIP runtime (for AMD GPUs)
    if (lib_handle == nullptr) {
        lib_handle = dl_binding_utils::find_loaded_so("amdhip64");
        if (lib_handle != nullptr) {
            is_rocm = true;
        }
    }

    if (lib_handle == nullptr) {
        lib_handle = (void*)0x1;
        return false;
    }

    using dl_binding_utils::resolve_required;

    // Try CUDA function names first (cuda* prefix), then HIP names (hip* prefix) as fallback
    // On NVIDIA: libcudart.so has cuda* functions - uses CUDA path
    // On AMD: libamdhip64.so only has hip* functions - uses HIP fallback path
    cudaMalloc_fn = resolve_required<decltype(cudaMalloc_fn)>(lib_handle, {"cudaMalloc", "hipMalloc"});
    cudaFree_fn = resolve_required<decltype(cudaFree_fn)>(lib_handle, {"cudaFree", "hipFree"});
    cudaSetDevice_fn = resolve_required<decltype(cudaSetDevice_fn)>(lib_handle, {"cudaSetDevice", "hipSetDevice"});
    cudaStreamCreateWithFlags_fn = resolve_required<decltype(cudaStreamCreateWithFlags_fn)>(lib_handle, {"cudaStreamCreateWithFlags", "hipStreamCreateWithFlags"});
    cudaStreamDestroy_fn = resolve_required<decltype(cudaStreamDestroy_fn)>(lib_handle, {"cudaStreamDestroy", "hipStreamDestroy"});
    cudaMemcpyAsync_fn = resolve_required<decltype(cudaMemcpyAsync_fn)>(lib_handle, {"cudaMemcpyAsync", "hipMemcpyAsync"});
    cudaEventCreateWithFlags_fn = resolve_required<decltype(cudaEventCreateWithFlags_fn)>(lib_handle, {"cudaEventCreateWithFlags", "hipEventCreateWithFlags"});
    cudaEventRecord_fn = resolve_required<decltype(cudaEventRecord_fn)>(lib_handle, {"cudaEventRecord", "hipEventRecord"});
    cudaEventQuery_fn = resolve_required<decltype(cudaEventQuery_fn)>(lib_handle, {"cudaEventQuery", "hipEventQuery"});
    cudaEventSynchronize_fn = resolve_required<decltype(cudaEventSynchronize_fn)>(lib_handle, {"cudaEventSynchronize", "hipEventSynchronize"});
    cudaStreamWaitEvent_fn = resolve_required<decltype(cudaStreamWaitEvent_fn)>(lib_handle, {"cudaStreamWaitEvent", "hipStreamWaitEvent"});
    cudaHostRegister_fn = resolve_required<decltype(cudaHostRegister_fn)>(lib_handle, {"cudaHostRegister", "hipHostRegister"});
    cudaHostUnregister_fn = resolve_required<decltype(cudaHostUnregister_fn)>(lib_handle, {"cudaHostUnregister", "hipHostUnregister"});
    cudaGetErrorString_fn = resolve_required<decltype(cudaGetErrorString_fn)>(lib_handle, {"cudaGetErrorString", "hipGetErrorString"});

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
inline cudaError_t cudaEventQuery(cudaEvent_t event) {
    return cudaEventQuery_fn(event);
}
inline bool cudaErrorIsNotReady(cudaError_t error) {
    return error == cudaErrorNotReady;
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

} // namespace cuda_binding
}

#define CUDA_CHECK(_call) \
    do { \
        ::instanttensor::cuda_binding::cudaError_t _e = (_call); \
        if (_e != ::instanttensor::cuda_binding::cudaSuccess) { \
            ::instanttensor::print_and_throw(std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + " '" + #_call + "' -> " + std::to_string(_e) + ":" + ::instanttensor::cuda_binding::cudaGetErrorString(_e))); \
        } \
    } while (0)
