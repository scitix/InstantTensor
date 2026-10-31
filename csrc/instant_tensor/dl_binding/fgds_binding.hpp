#pragma once

#include <instant_tensor/dl_binding/dl_binding_utils.hpp>
#include <cstddef>
#include <cstdint>
#include <unistd.h>
#include <atomic>

#define FGDS_ERRSTR(err) \
        (string("fgds error code ") + std::to_string(err))

namespace instanttensor {
namespace fgds_binding {

typedef struct fgds_fileid {
    int fd;
    int deviceID;
} fgds_fileid_t;

// Forward-declared so the function pointer return type compiles without
// pulling in the full C header.
struct fgds_xfer_addr;

inline void* lib_handle = nullptr;
inline std::atomic<int> ref_count{0};

// Function pointers — named with _fn suffix to avoid shadowing the
// wrapper functions below (the original code had infinite recursion
// because the pointer and wrapper shared the same name).
inline int (*fgds_open_fn)(int) = nullptr;
inline int (*fgds_close_fn)(int) = nullptr;
inline int (*fgds_regmem_fn)(int, const void*, size_t, void**) = nullptr;
inline int (*fgds_deregmem_fn)(int, const void*, size_t) = nullptr;
inline ssize_t (*fgds_read_fn)(fgds_fileid_t, void*, off_t, ssize_t, off_t) = nullptr;
inline ssize_t (*fgds_write_fn)(fgds_fileid_t, void*, off_t, ssize_t, off_t) = nullptr;
inline struct fgds_xfer_addr* (*fgds_do_xfer_addr_fn)(int, const void*, off_t, size_t) = nullptr;

inline bool init() {
    if (lib_handle != nullptr) {
        if (lib_handle == (void*)0x1) return false;
        return true;
    }

    // Strategy 1: look for an already-loaded libfgds.so in the process
    // (e.g. loaded by another component or via LD_PRELOAD).
    lib_handle = dl_binding_utils::find_loaded_so("fgds");

    // Strategy 2: try dlopen("libfgds.so") so a separately installed
    // shared library is picked up on demand.
    if (lib_handle == nullptr) {
        lib_handle = dlopen("libfgds.so", RTLD_LAZY | RTLD_LOCAL | RTLD_NODELETE);
    }

    if (lib_handle == nullptr) {
        lib_handle = (void*)0x1;
        return false;
    }

    using dl_binding_utils::resolve;
    fgds_open_fn           = resolve<decltype(fgds_open_fn)>(lib_handle, "fgds_open");
    fgds_close_fn          = resolve<decltype(fgds_close_fn)>(lib_handle, "fgds_close");
    fgds_regmem_fn         = resolve<decltype(fgds_regmem_fn)>(lib_handle, "fgds_regmem");
    fgds_deregmem_fn       = resolve<decltype(fgds_deregmem_fn)>(lib_handle, "fgds_deregmem");
    fgds_read_fn           = resolve<decltype(fgds_read_fn)>(lib_handle, "fgds_read");
    fgds_write_fn          = resolve<decltype(fgds_write_fn)>(lib_handle, "fgds_write");
    fgds_do_xfer_addr_fn   = resolve<decltype(fgds_do_xfer_addr_fn)>(lib_handle, "fgds_do_xfer_addr");

    return true;
}

inline bool is_initialized() {
    return lib_handle != nullptr && lib_handle != (void*)0x1;
}

inline int fgds_open(int device_id) {
    return fgds_open_fn(device_id);
}
inline int fgds_close(int device_id) {
    return fgds_close_fn(device_id);
}
inline int fgds_regmem(int device_id, const void* addr, size_t size, void** target_addr) {
    return fgds_regmem_fn(device_id, addr, size, target_addr);
}
inline int fgds_deregmem(int device_id, const void* addr, size_t size) {
    return fgds_deregmem_fn(device_id, addr, size);
}
inline ssize_t fgds_read(fgds_fileid_t fid, void* bufPtr_base,
                         off_t buf_offset, ssize_t nbyte, off_t file_offset) {
    return fgds_read_fn(fid, bufPtr_base, buf_offset, nbyte, file_offset);
}
inline ssize_t fgds_write(fgds_fileid_t fid, void* bufPtr_base,
                          off_t buf_offset, ssize_t nbyte, off_t file_offset) {
    return fgds_write_fn(fid, bufPtr_base, buf_offset, nbyte, file_offset);
}
inline struct fgds_xfer_addr* fgds_do_xfer_addr(int device_id, const void* buf,
                                                off_t buf_offset, size_t nbyte) {
    return fgds_do_xfer_addr_fn(device_id, buf, buf_offset, nbyte);
}

} // namespace fgds_binding
} // namespace instanttensor
