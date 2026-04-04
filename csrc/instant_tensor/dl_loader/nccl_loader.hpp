#pragma once

#include <instant_tensor/dl_loader/dl_loader_utils.hpp>
#include <instant_tensor/dl_loader/cuda_loader.hpp>
#include <cstddef>

namespace instanttensor {
namespace nccl_loader {

inline bool is_rocm = false;

// Minimal types matching nccl.h ABI
typedef void* ncclComm_t;
typedef cuda_loader::cudaStream_t cudaStream_t;

enum ncclResult_t {
    ncclSuccess = 0,
};

enum ncclDataType_t {
    ncclInt8 = 0,
};

inline void* lib_handle = nullptr;

inline ncclResult_t (*ncclAllGather_fn)(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) = nullptr;
inline ncclResult_t (*ncclCommUserRank_fn)(const ncclComm_t comm, int* rank) = nullptr;
inline ncclResult_t (*ncclCommCount_fn)(const ncclComm_t comm, int* count) = nullptr;
inline const char* (*ncclGetErrorString_fn)(ncclResult_t result) = nullptr;

inline bool init() {
    if (lib_handle != nullptr) {
        if (lib_handle == (void*)0x1) return false;
        return true;
    }

    // Try NCCL first (NVIDIA)
    lib_handle = dl_loader_utils::find_loaded_so("nccl");

    // If NCCL not found, try RCCL (AMD's NCCL for ROCm)
    if (lib_handle == nullptr) {
        lib_handle = dl_loader_utils::find_loaded_so("rccl");
        if (lib_handle != nullptr) {
            is_rocm = true;
        }
    }

    if (lib_handle == nullptr) {
        lib_handle = (void*)0x1;
        return false;
    }

    using dl_loader_utils::resolve;
    // RCCL provides NCCL-compatible API (same function names and signatures), thus we don't need things like "rcclAllGather"
    ncclAllGather_fn = resolve<decltype(ncclAllGather_fn)>(lib_handle, "ncclAllGather");
    ncclCommUserRank_fn = resolve<decltype(ncclCommUserRank_fn)>(lib_handle, "ncclCommUserRank");
    ncclCommCount_fn = resolve<decltype(ncclCommCount_fn)>(lib_handle, "ncclCommCount");
    ncclGetErrorString_fn = resolve<decltype(ncclGetErrorString_fn)>(lib_handle, "ncclGetErrorString");

    return true;
}

inline ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
    return ncclAllGather_fn(sendbuff, recvbuff, sendcount, datatype, comm, stream);
}
inline ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
    return ncclCommUserRank_fn(comm, rank);
}
inline ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
    return ncclCommCount_fn(comm, count);
}
inline const char* ncclGetErrorString(ncclResult_t result) {
    return ncclGetErrorString_fn(result);
}

} // namespace nccl_loader
} // namespace instanttensor
