#pragma once

#include <instant_tensor/dl_loader/dl_loader_utils.hpp>
#include <cstddef>
#include <cstdint>
#include <unistd.h>

#define CUFILE_ERRSTR(err) \
        (string("cuFile error code ") + std::to_string(err))

namespace instanttensor {
namespace cufile_loader {

// Minimal types matching cufile.h ABI
typedef void* CUfileHandle_t;

enum CUfileOpError {
    CU_FILE_SUCCESS = 0,
};

enum CUresult {
    CUDA_SUCCESS = 0,
};

typedef struct CUfileError {
    CUfileOpError err; // cufile error
    enum CUresult cu_err; // for CUDA-specific errors
} CUfileError_t;

enum CUfileFileHandleType {
    CU_FILE_HANDLE_TYPE_OPAQUE_FD = 1, /* linux based fd    */
    CU_FILE_HANDLE_TYPE_OPAQUE_WIN32 = 2, /* windows based handle */
    CU_FILE_HANDLE_TYPE_USERSPACE_FS  = 3, /* userspace based FS */
};

typedef struct CUfileDescr_t {
    CUfileFileHandleType type; /* type of file being registered */
    union {
        int fd;             /* Linux   */
        void *handle;       /* Windows */
    } handle;
    const void *fs_ops;     /* Unused. Original type is "const CUfileFSOps_t *". file system operation table */
} CUfileDescr_t;

static void* lib_handle = nullptr;

static CUfileError_t (*cuFileDriverOpen_fn)() = nullptr;
static CUfileError_t (*cuFileDriverClose_fn)() = nullptr;
static CUfileError_t (*cuFileHandleRegister_fn)(CUfileHandle_t *fh, CUfileDescr_t *descr) = nullptr;
static void (*cuFileHandleDeregister_fn)(CUfileHandle_t fh) = nullptr;
static CUfileError_t (*cuFileBufRegister_fn)(const void* devPtr_base, size_t size, int flags) = nullptr;
static CUfileError_t (*cuFileBufDeregister_fn)(const void* devPtr_base) = nullptr;
static ssize_t (*cuFileRead_fn)(CUfileHandle_t fh, void *bufPtr_base, size_t size, off_t file_offset, off_t devPtr_offset) = nullptr;

inline bool init() {
    if (lib_handle != nullptr) {
        if (lib_handle == (void*)0x1) return false;
        return true;
    }

    lib_handle = dl_loader_utils::find_loaded_so("cufile");
    if (lib_handle == nullptr) {
        lib_handle = (void*)0x1;
        return false;
    }

    using dl_loader_utils::resolve;
    cuFileDriverOpen_fn = resolve<decltype(cuFileDriverOpen_fn)>(lib_handle, "cuFileDriverOpen");
    cuFileDriverClose_fn = resolve<decltype(cuFileDriverClose_fn)>(lib_handle, "cuFileDriverClose");
    cuFileHandleRegister_fn = resolve<decltype(cuFileHandleRegister_fn)>(lib_handle, "cuFileHandleRegister");
    cuFileHandleDeregister_fn = resolve<decltype(cuFileHandleDeregister_fn)>(lib_handle, "cuFileHandleDeregister");
    cuFileBufRegister_fn = resolve<decltype(cuFileBufRegister_fn)>(lib_handle, "cuFileBufRegister");
    cuFileBufDeregister_fn = resolve<decltype(cuFileBufDeregister_fn)>(lib_handle, "cuFileBufDeregister");
    cuFileRead_fn = resolve<decltype(cuFileRead_fn)>(lib_handle, "cuFileRead");

    return true;
}

inline CUfileError_t cuFileDriverOpen() {
    return cuFileDriverOpen_fn();
}
inline CUfileError_t cuFileDriverClose() {
    return cuFileDriverClose_fn();
}
inline CUfileError_t cuFileHandleRegister(CUfileHandle_t* fh, CUfileDescr_t* descr) {
    return cuFileHandleRegister_fn(fh, descr);
}
inline void cuFileHandleDeregister(CUfileHandle_t fh) {
    cuFileHandleDeregister_fn(fh);
}
inline CUfileError_t cuFileBufRegister(const void* devPtr_base, size_t size, int flags) {
    return cuFileBufRegister_fn(devPtr_base, size, flags);
}
inline CUfileError_t cuFileBufDeregister(const void* devPtr_base) {
    return cuFileBufDeregister_fn(devPtr_base);
}
inline ssize_t cuFileRead(CUfileHandle_t fh, void* bufPtr_base, size_t size, off_t file_offset, off_t devPtr_offset) {
    return cuFileRead_fn(fh, bufPtr_base, size, file_offset, devPtr_offset);
}

} // namespace cufile_loader
} // namespace instanttensor
