#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace instanttensor {

struct HostRegistrationRange {
    void* ptr;
    size_t size;
};

struct HostRegistration {
    std::vector<HostRegistrationRange> ranges;
    int whole_buffer_error = 0;
};

template <typename RegisterFn, typename UnregisterFn, typename ClearErrorFn>
HostRegistration register_host_buffer(
    void* ptr,
    size_t size,
    unsigned int flags,
    size_t segment_size,
    RegisterFn&& register_fn,
    UnregisterFn&& unregister_fn,
    ClearErrorFn&& clear_error_fn
) {
    if (ptr == nullptr || size == 0 || segment_size == 0) {
        throw std::invalid_argument("Host registration requires non-empty storage and segments");
    }

    HostRegistration registration;
    int result = register_fn(ptr, size, flags);
    if (result == 0) {
        registration.ranges.push_back({ptr, size});
        return registration;
    }

    registration.whole_buffer_error = result;
    clear_error_fn();
    auto* base = static_cast<char*>(ptr);
    for (size_t offset = 0; offset < size; offset += segment_size) {
        const size_t current_size = std::min(segment_size, size - offset);
        void* current_ptr = base + offset;
        result = register_fn(current_ptr, current_size, flags);
        if (result == 0) {
            registration.ranges.push_back({current_ptr, current_size});
            continue;
        }

        clear_error_fn();
        for (auto it = registration.ranges.rbegin(); it != registration.ranges.rend(); ++it) {
            unregister_fn(it->ptr);
        }
        throw std::runtime_error(
            "Host registration failed for the whole buffer (code "
            + std::to_string(registration.whole_buffer_error)
            + ") and segment at offset " + std::to_string(offset)
            + " (code " + std::to_string(result) + ")"
        );
    }
    return registration;
}

} // namespace instanttensor
