#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/vfs.h>
#include <linux/magic.h>
#include <libaio.h>

#include <iostream>
#include <cstdlib>
#include <unordered_map>
#include <vector>
#include <string>
#include <thread>
#include <future>
#include <chrono>
#include <any>

#include <instant_tensor/queue.hpp>
#include <instant_tensor/async_executor.hpp>
#include <instant_tensor/dl_binding/cuda_binding.hpp>
#include <instant_tensor/dl_binding/cufile_binding.hpp>
#include <instant_tensor/dl_binding/nccl_binding.hpp>

using namespace instanttensor::cuda_binding;
using namespace instanttensor::cufile_binding;
using namespace instanttensor::nccl_binding;

#define MAX_PREFETCH_CHUNKS 1024
const size_t PAGE_SIZE = sysconf(_SC_PAGESIZE);// typically 4096

namespace py = pybind11;

using std::string;
using std::vector;
using std::optional;
using std::unordered_map;
using std::unique_ptr;
using std::pair;
using std::atomic;
using std::max;
using std::min;

[[noreturn]]
inline void print_and_throw(const std::exception& e) {
    fprintf(stderr, "%s\n", e.what());
    throw e;
}

template <typename T>
T DIV_CEIL(T x, T y) {
    return (x + y - 1) / y;
}

template <typename T>
T ROUND_UP(T x, T y) {
    return DIV_CEIL(x, y) * y;
}

template <typename T>
T ROUND_DOWN(T x, T y) {
    return x / y * y;
}

#define CUDA_CHECK(_call) \
    do { \
        cudaError_t _e = (_call); \
        if (_e != cudaSuccess) { \
            print_and_throw(std::runtime_error(string(__FILE__) + ":" + std::to_string(__LINE__) + " '" + #_call + "' -> " + std::to_string(_e) + ":" + cudaGetErrorString(_e))); \
        } \
    } while (0)

#define CUFILE_CHECK(_call) \
    do { \
        CUfileError_t _e = (_call); \
        if (_e.err != CU_FILE_SUCCESS) { \
            print_and_throw(std::runtime_error(string(__FILE__) + ":" + std::to_string(__LINE__) + " '" + #_call + "' -> " + std::to_string(_e.err) + ":" + CUFILE_ERRSTR(_e.err))); \
        } \
    } while (0)

#define NCCL_CHECK(_call) \
    do { \
        ncclResult_t _e = (_call); \
        if (_e != ncclSuccess) { \
            print_and_throw(std::runtime_error(string(__FILE__) + ":" + std::to_string(__LINE__) + " '" + #_call + "' -> " + std::to_string(_e) + ":" + ncclGetErrorString(_e))); \
        } \
    } while (0)

inline std::optional<string> get_env(const string& name) {
    if (const char* val = std::getenv(name.c_str())) {
        return string(val);
    } else {
        return std::nullopt;
    }
}

namespace instanttensor {

inline bool _determine_use_cufile(){
    bool ret = get_env("INSTANTTENSOR_USE_CUFILE").value_or("0") == "1";
    if (ret) {
        if (!cufile_binding::init()) {
            fprintf(stderr, "cuFile not found, fallback to aio.\n");
            ret = false;
        }
    }
    return ret;
}

inline bool _env_use_cufile(){
    static bool ret = _determine_use_cufile();
    return ret;
}

inline bool _env_use_internal_memory_register(){
    static bool ret = get_env("INSTANTTENSOR_USE_INTERNAL_MEMORY_REGISTER").value_or("0") == "1";
    return ret;
}

inline bool _env_cache_buffer(){
    static bool ret = get_env("INSTANTTENSOR_CACHE_BUFFER").value_or("0") == "1";
    return ret;
}

inline bool _env_debug() {
    static bool ret = get_env("INSTANTTENSOR_DEBUG").value_or("0") == "1";
    return ret;
}

} // namespace instanttensor
