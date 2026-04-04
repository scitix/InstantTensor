#pragma once

#include <instant_tensor/common.hpp>
#include <instant_tensor/types.hpp>

namespace instanttensor {

inline bool file_in_memory(const string& filename) {
    struct statfs buf;
    if (statfs(filename.c_str(), &buf)==0) {
        return buf.f_type == TMPFS_MAGIC || buf.f_type == RAMFS_MAGIC;
    }
    return false;
}

class CufileContext {
public:
    CufileContext() {
        auto t0 = std::chrono::high_resolution_clock::now();
        CUFILE_CHECK(cuFileDriverOpen());// This costs a long time (~2s)
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dt = t1 - t0;
        if(_env_debug()) {
            fprintf(stderr, "cuFile Driver Opened in %.2f seconds\n", dt.count());
        }
    }
    ~CufileContext() {
        CUFILE_CHECK(cuFileDriverClose());
        if(_env_debug()) {
            fprintf(stderr, "cuFile Driver Closed\n");
        }
    }
};

class CufileContextInitializer {
    unique_ptr<CufileContext> context;
    std::mutex mutex;
public:
    void initialize() {
        std::lock_guard<std::mutex> lock(mutex);
        if(!context) context = std::make_unique<CufileContext>();
    }
};

class HostBufferCache {// NOTE: do not cache too many entries since get() is O(n)
    std::mutex mutex;
    vector<HostBufferCacheEntry> cached_host_buffers;
public:
    ~HostBufferCache() {
        std::lock_guard<std::mutex> lock(mutex);
        for(auto& entry : cached_host_buffers) {
            entry.deleter(entry.ptr);
        }
    }
    void put(HostBufferCacheEntry entry) {
        std::lock_guard<std::mutex> lock(mutex);
        cached_host_buffers.emplace_back(std::move(entry));
    }
    HostBufferCacheEntry get(size_t size) {
        std::lock_guard<std::mutex> lock(mutex);
        for(size_t i = 0; i < cached_host_buffers.size(); i++) {
            if (cached_host_buffers[i].size >= size) {
                HostBufferCacheEntry entry = std::move(cached_host_buffers[i]);
                cached_host_buffers.erase(cached_host_buffers.begin() + i);
                return entry;
            }
        }
        return HostBufferCacheEntry{NULL, 0, nullptr};
    }
};

class Munmaper {
    std::mutex mutex;
    vector<pair<void*, size_t>> mapped_memory;
public:
    ~Munmaper() noexcept(false) {
        std::lock_guard<std::mutex> lock(mutex);
        for(auto& ptr : mapped_memory) {
            int ret = munmap(ptr.first, ptr.second);
            if (ret != 0) {
                throw std::runtime_error("Failed to munmap: " + std::string(strerror(errno)));
            }
        }
    }
    void add(void* ptr, size_t size) {
        std::lock_guard<std::mutex> lock(mutex);
        mapped_memory.emplace_back(std::make_pair(ptr, size));
    }
};

// Global instances
inline unique_ptr<CufileContextInitializer> cufile_context_initializer = std::make_unique<CufileContextInitializer>();
inline unique_ptr<HostBufferCache> host_buffer_cache = std::make_unique<HostBufferCache>();
inline unique_ptr<Munmaper> munmaper = std::make_unique<Munmaper>();

} // namespace instanttensor
