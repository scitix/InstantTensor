#pragma once

#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <cstring>
#include <vector>
#include <dlpack/dlpack.h>
#include <instant_tensor/dl_loader/cuda_loader.hpp>

namespace py = pybind11;

namespace instanttensor {

struct Ctx {
    int64_t* shape = nullptr;
    int64_t* strides = nullptr;
    int ndim = 0;
};

static void dl_managed_tensor_deleter(DLManagedTensor* self) {
    if (!self) return;
    Ctx* ctx = reinterpret_cast<Ctx*>(self->manager_ctx);
    if (ctx) {
        delete[] ctx->shape;
        delete[] ctx->strides;
        delete ctx;
    }
    delete self;
}

static DLDataType torch_to_dlpack_dtype(const std::string& s) {
    DLDataType t;
    t.lanes = 1;
    if (s == "bool") {
        t.code = kDLBool; t.bits = 8; return t;
    } 
    else if (s == "float4_e2m1fn_x2") {
        t.code = kDLFloat4_e2m1fn; t.bits = 4; return t;
    }
    // else if (s == "F6_E2M3") {
    //     t.code = kDLFloat6_e2m3fn; t.bits = 6; return t;
    // } 
    // else if (s == "F6_E3M2") {
    //     t.code = kDLFloat6_e3m2fn; t.bits = 6; return t;
    // } 
    else if (s == "uint8") {
        t.code = kDLUInt; t.bits = 8; return t;
    }
    else if (s == "int8") {
        t.code = kDLInt; t.bits = 8; return t;
    } 
    else if (s == "float8_e5m2") { // Is this correct for float8_e5m2fnuz?
        t.code = kDLFloat8_e5m2; t.bits = 8; return t;
    } 
    else if (s == "float8_e5m2fnuz") {
        t.code = kDLFloat8_e5m2fnuz; t.bits = 8; return t;
    }
    else if (s == "float8_e4m3fn") {
        t.code = kDLFloat8_e4m3fn; t.bits = 8; return t;
    }
    else if (s == "float8_e4m3fnuz") {
        t.code = kDLFloat8_e4m3fnuz; t.bits = 8; return t;
    }
    else if (s == "float8_e8m0fnu") {
        t.code = kDLFloat8_e8m0fnu; t.bits = 8; return t;
    } 
    else if (s == "int16") {
        t.code = kDLInt; t.bits = 16; return t;
    } 
    else if (s == "uint16") {
        t.code = kDLUInt; t.bits = 16; return t;
    }
    else if (s == "float16") {
        t.code = kDLFloat; t.bits = 16; return t;
    } 
    else if (s == "bfloat16") {
        t.code = kDLBfloat; t.bits = 16; return t;
    } 
    else if (s == "int32") {
        t.code = kDLInt; t.bits = 32; return t;
    }
    else if (s == "uint32") {
        t.code = kDLUInt; t.bits = 32; return t;
    } 
    else if (s == "float32") {
        t.code = kDLFloat; t.bits = 32; return t;
    } 
    else if (s == "complex64") {
        t.code = kDLComplex; t.bits = 64; return t;
    }
    else if (s == "float64") {
        t.code = kDLFloat; t.bits = 64; return t;
    } 
    else if (s == "int64") {
        t.code = kDLInt; t.bits = 64; return t;
    } 
    else if (s == "uint64") {
        t.code = kDLUInt; t.bits = 64; return t;
    }
    else {
        throw std::runtime_error("Unsupported dtype: " + s);
    }
}

// Detect the device type based on the loaded runtime library
static DLDeviceType get_device_type() {
    static DLDeviceType cached_type = static_cast<DLDeviceType>(-1);
    if (cached_type != static_cast<DLDeviceType>(-1)) {
        return cached_type;
    }

    // Ensure cuda_loader is initialized
    if (!cuda_loader::init()) {
        // Failed to initialize - default to CUDA for backward compatibility
        cached_type = kDLCUDA;
        return cached_type;
    }

    // Check which runtime library is loaded
    void* cudart_handle = cuda_loader::lib_handle;
    if (cudart_handle == nullptr || cudart_handle == (void*)0x1) {
        // Default to CUDA for NVIDIA compatibility
        cached_type = kDLCUDA;
        return cached_type;
    }

    // Detect AMD HIP by checking for HIP-specific symbol
    // This only affects AMD systems - NVIDIA CUDA runtime won't have this symbol
    dlerror(); // Clear any previous error
    void* hip_symbol = dlsym(cudart_handle, "hipGetDeviceCount");
    if (hip_symbol != nullptr) {
        // It's HIP (AMD ROCm) - use kDLROCM device type
        cached_type = kDLROCM;
    } else {
        // It's CUDA (NVIDIA) - use kDLCUDA device type (default)
        cached_type = kDLCUDA;
    }

    return cached_type;
}

// ptr: uint64 (CUDA/HIP device pointer)
// shape: vector<int64_t>
// dtype: string ("float32" etc.)
// device_id: int
static py::object pack_dlpack(uint64_t ptr,
                       const std::vector<int64_t>& shape,
                       const std::string& dtype,
                       int device_id) {
    auto* managed = new DLManagedTensor();
    std::memset(managed, 0, sizeof(DLManagedTensor));
    // ctx is used to free shape/strides etc.
    auto* ctx = new Ctx();
    ctx->ndim = (int)shape.size();
    if (ctx->ndim > 0) {
        ctx->shape = new int64_t[ctx->ndim];
        ctx->strides = new int64_t[ctx->ndim];
        for (int i = 0; i < ctx->ndim; ++i) {
            ctx->shape[i] = shape[i];
        }
        // Before DLPack v1.2, strides can be NULL to indicate contiguous data.
        // This is not allowed in DLPack v1.2 and later. The rationale
        // is to simplify the consumer handling.
        ctx->strides[ctx->ndim - 1] = 1;
        for (int i = ctx->ndim - 2; i >= 0; --i) {
            ctx->strides[i] = ctx->strides[i + 1] * ctx->shape[i + 1];
        }
    } else {
        ctx->shape = nullptr;
        ctx->strides = nullptr;
    }
    managed->dl_tensor.data = reinterpret_cast<void*>(ptr);
    // Use the appropriate device type (kDLCUDA for NVIDIA, kDLROCM for AMD)
    managed->dl_tensor.device = DLDevice{get_device_type(), device_id};
    managed->dl_tensor.ndim = ctx->ndim;
    managed->dl_tensor.dtype = torch_to_dlpack_dtype(dtype);
    managed->dl_tensor.shape = ctx->shape;
    managed->dl_tensor.strides = ctx->strides;
    managed->dl_tensor.byte_offset = 0;
    managed->manager_ctx = ctx;
    managed->deleter = dl_managed_tensor_deleter;
    // Capsule name must be "dltensor".
    // Capsule destructor is set to nullptr here: if the user does not consume the capsule, the wrapper will leak (generally acceptable).
    // Once consumed by torch, torch will call managed->deleter when the Tensor is freed.
    PyObject* cap = PyCapsule_New((void*)managed, "dltensor", nullptr);
    if (!cap) throw std::runtime_error("PyCapsule_New failed");
    return py::reinterpret_steal<py::object>(cap);
}

} // namespace instanttensor
