#ifndef DLPACK_HPP
#define DLPACK_HPP

#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <cstring>
#include <vector>
#include <dlpack/dlpack.h>
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

static inline DLDataType safetensors_to_dlpack_dtype(const std::string& s) {
    DLDataType t;
    t.lanes = 1;
    if (s == "BOOL") {
        t.code = kDLBool; t.bits = 8; return t;
    } 
    else if (s == "F4") {
        t.code = kDLFloat4_e2m1fn; t.bits = 4; return t;
    }
    else if (s == "F6_E2M3") {
        t.code = kDLFloat6_e2m3fn; t.bits = 6; return t;
    } 
    else if (s == "F6_E3M2") {
        t.code = kDLFloat6_e3m2fn; t.bits = 6; return t;
    } 
    else if (s == "U8") {
        t.code = kDLUInt; t.bits = 8; return t;
    }
    else if (s == "I8") {
        t.code = kDLInt; t.bits = 8; return t;
    } 
    else if (s == "F8_E5M2") {
        t.code = kDLFloat8_e5m2; t.bits = 8; return t;
    } 
    else if (s == "F8_E4M3") {
        t.code = kDLFloat8_e4m3; t.bits = 8; return t;
    }
    else if (s == "F8_E8M0") {
        t.code = kDLFloat8_e8m0fnu; t.bits = 8; return t;
    } 
    else if (s == "I16") {
        t.code = kDLInt; t.bits = 16; return t;
    } 
    else if (s == "U16") {
        t.code = kDLUInt; t.bits = 16; return t;
    }
    else if (s == "F16") {
        t.code = kDLFloat; t.bits = 16; return t;
    } 
    else if (s == "BF16") {
        t.code = kDLBfloat; t.bits = 16; return t;
    } 
    else if (s == "I32") {
        t.code = kDLInt; t.bits = 32; return t;
    }
    else if (s == "U32") {
        t.code = kDLUInt; t.bits = 32; return t;
    } 
    else if (s == "F32") {
        t.code = kDLFloat; t.bits = 32; return t;
    } 
    else if (s == "C64") {
        t.code = kDLComplex; t.bits = 64; return t;
    }
    else if (s == "F64") {
        t.code = kDLFloat; t.bits = 64; return t;
    } 
    else if (s == "I64") {
        t.code = kDLInt; t.bits = 64; return t;
    } 
    else if (s == "U64") {
        t.code = kDLUInt; t.bits = 64; return t;
    }
    else {
        throw std::runtime_error("Unsupported dtype: " + s);
    }
}

// ptr: uint64 (CUDA device pointer)
// shape: vector<int64_t>
// dtype: string ("float32" etc.)
// device_id: int
py::object pack_dlpack(uint64_t ptr,
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
    managed->dl_tensor.device = DLDevice{kDLCUDA, device_id};
    managed->dl_tensor.ndim = ctx->ndim;
    managed->dl_tensor.dtype = safetensors_to_dlpack_dtype(dtype);
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

#endif // DLPACK_HPP