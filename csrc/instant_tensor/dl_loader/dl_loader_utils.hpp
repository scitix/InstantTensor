#pragma once

#include <dlfcn.h>
#include <fstream>
#include <string>
#include <stdexcept>

namespace instanttensor {
namespace dl_loader_utils {

/** Find an already-loaded .so in the current process whose path contains \p prefix.
 *  Scans /proc/self/maps and returns a dlopen handle with RTLD_NOLOAD (reuse only).
 *  Returns nullptr if none found.
 */
inline void* find_loaded_so(const char* prefix) {
    std::string so_prefix = std::string("lib") + prefix + ".so";
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) {
        size_t path_start = line.find('/');
        if (path_start == std::string::npos) continue;
        std::string path = line.substr(path_start);
        if (path.find(so_prefix) == std::string::npos) continue;
        while (!path.empty() && (path.back() == ' ' || path.back() == '\t')) path.pop_back();
        void* handle = dlopen(path.c_str(), RTLD_NOLOAD | RTLD_NOW | RTLD_LOCAL);
        if (handle != nullptr) return handle;
    }
    return nullptr;
}

template <typename T>
inline T resolve(void* handle, const char* name) {
    void* sym = dlsym(handle, name);
    if (!sym) {
        throw std::runtime_error(std::string("failed to resolve ") + name + ": " + dlerror());
    }
    return reinterpret_cast<T>(sym);
}

} // namespace dl_loader_utils
} // namespace instanttensor
