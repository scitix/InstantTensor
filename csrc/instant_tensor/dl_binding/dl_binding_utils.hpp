#pragma once

#include <dlfcn.h>
#include <fstream>
#include <string>
#include <stdexcept>

namespace instanttensor {
namespace dl_binding_utils {

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

// Lookup helpers return nullptr when a symbol is unavailable.
template <typename T>
inline T resolve(void* handle, const std::string& name) {
    // Clear any previous error
    dlerror();
    return reinterpret_cast<T>(dlsym(handle, name.c_str()));
}

template <typename T>
inline T resolve(void* handle, std::initializer_list<std::string> names) {
    for (const auto& name : names) {
        if (T sym = resolve<T>(handle, name)) return sym;
    }
    return nullptr;
}

template <typename T>
inline T resolve_global(const std::string& name) {
    return resolve<T>(RTLD_DEFAULT, name);
}

template <typename T>
inline T resolve_required(void* handle, const std::string& name) {
    T sym = resolve<T>(handle, name);
    if (!sym) {
        const char* error = dlerror();
        throw std::runtime_error("failed to resolve " + name + ": " +
                                 (error ? error : "null symbol address"));
    }
    return sym;
}

template <typename T>
inline T resolve_required(void* handle, std::initializer_list<std::string> names) {
    T sym = resolve<T>(handle, names);
    if (!sym) {
        std::string error_message = "failed to resolve any of the following names: ";
        size_t i = 0;
        for(auto& name : names) {
            if (i) {
                error_message += ", ";
            }
            error_message += name;
            i++;
        }
        throw std::runtime_error(error_message);
    }
    return sym;
}

} // namespace dl_binding_utils
} // namespace instanttensor
