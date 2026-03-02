#pragma once
// #include <boost/interprocess/ipc/message_queue.hpp> // although we only need inter-thread communication, we use RPCRequest_queue to avoid polling
// #include <boost/uuid/uuid.hpp>
// #include <boost/uuid/uuid_generators.hpp>
// #include <boost/uuid/uuid_io.hpp>
// #include <string>
#include <boost/lockfree/spsc_queue.hpp>
#include <thread>        // for std::this_thread::yield
#include <cstddef>       // for std::size_t
// namespace bip = boost::interprocess;

namespace instanttensor {

// std::string make_random_string() {
//     auto uuid = boost::uuids::random_generator()();
//     return boost::uuids::to_string(uuid);
// }

// template<typename T, std::size_t Capacity = 1024>
// class Queue { // BUG: this only supports POD types
// public:
//     Queue()
//         : _name(("instanttensor_" + make_random_string()))
//         , _mq(bip::create_only, _name.c_str(), Capacity, sizeof(T))
//     {
//         bip::message_queue::remove(_name.c_str());
//     }
//     ~Queue() {
//         bip::message_queue::remove(_name.c_str());
//     }
//     void push(const T& message) {
//         _mq.send(&message, sizeof(T), 0);
//     }
//     void pop(T& message) {
//         std::size_t recvd_size;
//         unsigned int priority;
//         _mq.receive(&message, sizeof(T), recvd_size, priority);
//         // recvd_size is sizeof(T) and priority is 0
//     }
//     bool full() {
//         return _mq.get_num_msg() >= _mq.get_max_msg();
//     }
//     bool empty() {
//         return _mq.get_num_msg() == 0;
//     }
// private:
//     std::string               _name;
//     bip::message_queue        _mq;
//     Queue(const Queue&)            = delete;
//     Queue& operator=(const Queue&) = delete;
// };


// Cross-thread queue based on cv and mutex, may spin and hang
// This supports non-POD types
// template<typename T>
// class Queue {
//   std::mutex               mu;
//   std::condition_variable  cv;
//   std::deque<T>            dq;
// public:
//   void push(T const& v) {
//     std::lock_guard lk(mu);
//     dq.push_back(v);
//     cv.notify_one();
//   }
//   T pop() {
//     std::unique_lock lk(mu);
//     cv.wait(lk, [&]{ return !dq.empty(); });
//     T v = std::move(dq.front());
//     dq.pop_front();
//     return v;
//   }
//   bool empty() {
//     std::lock_guard lk(mu);
//     return dq.empty();
//   }
// };


// Cross-thread queue based on polling and lock-free atomic operations
// Latency: ~250ns. Baselines: promise+future ~4us, mutex-based queue ~8us, boost::asio::thread_pool.post() ~3us
template<typename T, size_t Capacity = 1024>
class SPSCQueue {
    boost::lockfree::spsc_queue<T, boost::lockfree::capacity<Capacity>> q;
public:
    SPSCQueue() = default;
    void push(T const& v) {
        while (!q.push(v)) {
            std::this_thread::yield();
        }
    }
    void push(T&& v) {
        while (!q.push(std::forward<T>(v))) {
            std::this_thread::yield();
        }
    }
    void pop(T& result) {
        while (!q.pop(result)) {
            std::this_thread::yield();
        }
    }
    void pop() {
        while (!q.pop()) {
            std::this_thread::yield();
        }
    }
    bool try_push(T const& v) {
        return q.push(v);
    }
    bool try_push(T&& v) {
        return q.push(std::forward<T>(v));
    }
    bool try_pop(T& result) {
        return q.pop(result);
    }
    bool try_pop() {
        return q.pop();
    }
    bool empty() {
        return q.empty();
    }
};

}