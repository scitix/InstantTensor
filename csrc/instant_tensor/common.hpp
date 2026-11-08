#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <unistd.h>

namespace instanttensor {

inline constexpr size_t MAX_IO_DEPTH = 1024;
inline constexpr size_t MAX_CHUNK_SIZE = 1ULL << 30;
const size_t PAGE_SIZE = sysconf(_SC_PAGESIZE);// typically 4096

using std::string;
using std::vector;
using std::optional;
using std::unordered_map;
using std::unique_ptr;
using std::pair;
using std::atomic;
using std::max;
using std::min;

template <typename Exception>
[[noreturn]]
inline void print_and_throw(const Exception& e) {
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

inline size_t required_buffer_size_for_io(
    size_t chunk_size, size_t io_depth, size_t world_size) {
    return ROUND_UP(chunk_size, PAGE_SIZE) * io_depth * world_size;
}

inline std::optional<string> get_env(const string& name) {
    if (const char* val = std::getenv(name.c_str())) {
        return string(val);
    } else {
        return std::nullopt;
    }
}

inline bool _env_cache_buffer(){
    static bool ret = get_env("INSTANTTENSOR_CACHE_BUFFER").value_or("0") == "1";
    return ret;
}

inline bool _env_debug() {
    static bool ret = get_env("INSTANTTENSOR_DEBUG").value_or("0") == "1";
    return ret;
}

template <typename... Args>
inline void debug_log(const char* fmt, Args... args) {
    if (!_env_debug()) {
        return;
    }
    fprintf(stderr, "[InstantTensor][DEBUG] ");
    if constexpr (sizeof...(args) == 0) {
        fputs(fmt, stderr);
    } else {
        fprintf(stderr, fmt, args...);
    }
    fprintf(stderr, "\n");
    fflush(stderr);
}

} // namespace instanttensor
