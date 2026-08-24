#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <instant_tensor/host_registration.hpp>

using instanttensor::register_host_buffer;
using instanttensor::allocate_registered_host_buffer;

int main() {
    std::vector<char> storage(1024);
    void* base = storage.data();

    {
        int register_calls = 0;
        int clear_calls = 0;
        auto registration = register_host_buffer(
            base,
            storage.size(),
            0,
            256,
            [&](void*, size_t, unsigned int) {
                ++register_calls;
                return 0;
            },
            [](void*) { return 0; },
            [&]() { ++clear_calls; }
        );
        assert(register_calls == 1);
        assert(clear_calls == 0);
        assert(registration.whole_buffer_error == 0);
        assert(registration.ranges.size() == 1);
    }

    {
        int register_calls = 0;
        int clear_calls = 0;
        auto registration = register_host_buffer(
            base,
            storage.size(),
            0,
            256,
            [&](void*, size_t size, unsigned int) {
                ++register_calls;
                return size > 256 ? 1 : 0;
            },
            [](void*) { return 0; },
            [&]() { ++clear_calls; }
        );
        assert(register_calls == 5);
        assert(clear_calls == 1);
        assert(registration.whole_buffer_error == 1);
        assert(registration.ranges.size() == 4);
    }

    {
        int register_calls = 0;
        int clear_calls = 0;
        std::vector<void*> unregistered;
        bool threw = false;
        try {
            register_host_buffer(
                base,
                storage.size(),
                0,
                256,
                [&](void*, size_t size, unsigned int) {
                    ++register_calls;
                    if (size > 256) {
                        return 1;
                    }
                    return register_calls == 4 ? 2 : 0;
                },
                [&](void* ptr) {
                    unregistered.push_back(ptr);
                    return 0;
                },
                [&]() { ++clear_calls; }
            );
        } catch (const std::runtime_error&) {
            threw = true;
        }
        assert(threw);
        assert(clear_calls == 2);
        assert(unregistered.size() == 2);
        assert(unregistered[0] == static_cast<char*>(base) + 256);
        assert(unregistered[1] == base);
    }

    {
        std::vector<char> runtime_storage(1024);
        int ordinary_free_calls = 0;
        int runtime_alloc_calls = 0;
        auto allocation = allocate_registered_host_buffer(
            storage.size(),
            256,
            0,
            0,
            256,
            [&](size_t, size_t) { return base; },
            [&](void* ptr) {
                assert(ptr == base);
                ++ordinary_free_calls;
            },
            [](void*, size_t, unsigned int) { return 1; },
            [](void*) { return 0; },
            []() {},
            [&](void** ptr, size_t size, unsigned int flags) {
                ++runtime_alloc_calls;
                assert(size == runtime_storage.size());
                assert(flags == 0);
                *ptr = runtime_storage.data();
                return 0;
            }
        );
        assert(allocation.ptr == runtime_storage.data());
        assert(allocation.runtime_allocated);
        assert(allocation.registration.ranges.empty());
        assert(!allocation.registration_failure.empty());
        assert(ordinary_free_calls == 1);
        assert(runtime_alloc_calls == 1);
    }

    {
        int ordinary_free_calls = 0;
        bool threw = false;
        try {
            allocate_registered_host_buffer(
                storage.size(),
                256,
                0,
                0,
                256,
                [&](size_t, size_t) { return base; },
                [&](void* ptr) {
                    assert(ptr == base);
                    ++ordinary_free_calls;
                },
                [](void*, size_t, unsigned int) { return 1; },
                [](void*) { return 0; },
                []() {},
                [](void**, size_t, unsigned int) { return 2; }
            );
        } catch (const std::runtime_error& error) {
            threw = true;
            assert(std::string(error.what()).find(
                "runtime pinned allocation failed (code 2)"
            ) != std::string::npos);
        }
        assert(threw);
        assert(ordinary_free_calls == 1);
    }
}
