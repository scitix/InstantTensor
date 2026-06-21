#pragma once

#include <atomic_queue/atomic_queue.h>

#include <cstddef>
#include <limits>
#include <thread>
#include <type_traits>
#include <utility>

namespace instanttensor {

// Fixed-capacity cross-thread queue based on atomic_queue's ring buffer.
// T is constructed once per slot; queue operations move-assign values in and out.
template<typename T, size_t Capacity, bool IsSPSC>
class AtomicQueueAdapter {
    static_assert(Capacity > 0, "queue capacity must be greater than zero");
    static_assert((Capacity & (Capacity - 1)) == 0,
                  "atomic_queue capacity must be a power of two");
    static_assert(Capacity <= std::numeric_limits<unsigned>::max(),
                  "atomic_queue capacity exceeds its unsigned index range");
    static_assert(std::is_nothrow_default_constructible_v<T>,
                  "AtomicQueue2 requires noexcept default construction");
    static_assert(std::is_nothrow_move_constructible_v<T>,
                  "AtomicQueue2 requires noexcept move construction");
    static_assert(std::is_nothrow_move_assignable_v<T>,
                  "AtomicQueue2 requires noexcept move assignment");
    static_assert(std::is_nothrow_destructible_v<T>,
                  "AtomicQueue2 requires noexcept destruction");

    using Queue = atomic_queue::AtomicQueue2<
        T,
        static_cast<unsigned>(Capacity),
        true,   // Spread adjacent slots across cache lines.
        true,   // Use pause instructions while waiting on a claimed slot.
        false,  // Relaxed ticket ordering; each producer remains FIFO.
        IsSPSC>;

    Queue q;

public:
    AtomicQueueAdapter() = default;
    AtomicQueueAdapter(const AtomicQueueAdapter&) = delete;
    AtomicQueueAdapter& operator=(const AtomicQueueAdapter&) = delete;

    void push(const T& value) {
        T copy(value);
        push(std::move(copy));
    }

    void push(T&& value) {
        while (!q.try_push(std::move(value))) {
            std::this_thread::yield();
        }
    }

    void pop(T& result) {
        while (!q.try_pop(result)) {
            std::this_thread::yield();
        }
    }

    void pop() {
        T discarded;
        pop(discarded);
    }

    bool try_push(const T& value) {
        T copy(value);
        return q.try_push(std::move(copy));
    }

    bool try_push(T&& value) {
        return q.try_push(std::move(value));
    }

    bool try_pop(T& result) {
        return q.try_pop(result);
    }

    bool try_pop() {
        T discarded;
        return try_pop(discarded);
    }

    bool empty() const {
        return q.was_empty();
    }
};

template<typename T, size_t Capacity = 1024>
using SPSCQueue = AtomicQueueAdapter<T, Capacity, true>;

template<typename T, size_t Capacity = 1024>
using MPMCQueue = AtomicQueueAdapter<T, Capacity, false>;

template<typename T, size_t Capacity = 1024>
using SPMCQueue = MPMCQueue<T, Capacity>;

template<typename T, size_t Capacity = 1024>
using MPSCQueue = MPMCQueue<T, Capacity>;

} // namespace instanttensor
