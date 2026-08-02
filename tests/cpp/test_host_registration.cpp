#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <instant_tensor/host_registration.hpp>

using instanttensor::register_host_buffer;

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
}
