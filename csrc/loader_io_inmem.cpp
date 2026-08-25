#include <instant_tensor/loader.hpp>

namespace instanttensor {

void Loader::open_file_inmem(FileInfo &f) {
    f.fd = ::open(f.filename.c_str(), O_RDONLY);
    if (f.fd < 0) {
        throw std::runtime_error("Failed to open file: " + f.filename);
    }
    struct stat st;
    if (fstat(f.fd, &st) < 0) { throw std::runtime_error("Failed to fstat file: " + f.filename); }
    f.size = st.st_size;

    f.mapped_memory = mmap(NULL, f.size, PROT_READ, MAP_SHARED, f.fd, 0);
    if (f.mapped_memory == MAP_FAILED) {
        throw std::runtime_error("Failed to mmap file: " + f.filename);
    }
    if (this->use_internal_memory_register) {
        // NOTE: this requires cudaHostRegisterReadOnly since the file is read-only
        CUDA_CHECK(cudaHostRegister(f.mapped_memory, f.size, this->cuda_host_register_flags | cudaHostRegisterReadOnly));
    }
    else {
        this->need_host_buffer = true;
        this->need_worker_threads = true;
        this->need_cuda_thread = true;
    }
}

void Loader::close_file_inmem(FileInfo &f) {
    if (this->use_internal_memory_register) {
        CUDA_CHECK(cudaHostUnregister(f.mapped_memory));
    }
    // NOTE: Since munmap is very slow, we defer it to exit time
    munmaper->add(f.mapped_memory, f.size);
    ::close(f.fd);
}

ChunkRequest Loader::post_read_chunk_inmem(const ChunkIOParams &p) {
    int completion_req_id = -1;
    SingleThreadTaskExecutor *completion_thread = nullptr;

    if (!this->use_internal_memory_register) {
        int memcpy_req_id = EXECUTOR_STOP_REQUEST_ID;
        if(p.rank_size > 0) {
            void *rank_src = (char*)p.file.mapped_memory + p.chunk.file_offset + p.rank_offset;
            void *rank_mid = (char*)this->host_buffer + p.window_offset;
            size_t rank_size = p.rank_size;
            auto memcpy_func = [=]() {
                memcpy(rank_mid, rank_src, rank_size);
            };
            memcpy_req_id = this->next_executor_request_id();
            this->worker_threads->submit(memcpy_req_id, std::move(memcpy_func));
        }

        void *rank_mid = (char*)this->host_buffer + p.window_offset;
        void *rank_dst = p.rank_dst;
        void *all_dst = p.all_dst;
        size_t rank_size = p.rank_size;
        size_t padded_rank_size = p.padded_rank_size;
        cudaEvent_t event = p.event;
        auto cuda_func = [=]() {
            if(memcpy_req_id != EXECUTOR_STOP_REQUEST_ID) {
                this->worker_threads->reap(memcpy_req_id);
            }
            CUDA_CHECK(cudaMemcpyAsync(rank_dst, rank_mid, rank_size, cudaMemcpyHostToDevice, this->cuda_stream));// 240GB/s for 8 GPUs
            CUDA_CHECK(cudaEventRecord(event, this->cuda_stream));
            if(this->world_size > 1) {
                CUDA_CHECK(cudaStreamWaitEvent(this->nccl_stream, event));
                // In-place AllGather
                NCCL_CHECK(ncclAllGather(rank_dst, all_dst, padded_rank_size, ncclInt8, this->group_communicator, this->nccl_stream));// 320GB/s for 8 GPUs
                CUDA_CHECK(cudaEventRecord(event, this->nccl_stream));
            }
        };
        int cuda_req_id = this->next_executor_request_id();
        this->cuda_thread->submit(cuda_req_id, std::move(cuda_func));

        auto wait_func = [=]() mutable {
            this->cuda_thread->reap(cuda_req_id);
            CUDA_CHECK(cudaEventSynchronize(event));
        };
        completion_req_id = this->next_executor_request_id();
        this->wait_thread->submit(completion_req_id, std::move(wait_func));
        completion_thread = this->wait_thread.get();
    }
    else if (this->use_internal_memory_register) {
        void *rank_src = (char*)p.file.mapped_memory + p.chunk.file_offset + p.rank_offset;
        void *rank_dst = p.rank_dst;
        void *all_dst = p.all_dst;
        size_t rank_size = p.rank_size;
        size_t padded_rank_size = p.padded_rank_size;
        cudaEvent_t event = p.event;
        CUDA_CHECK(cudaMemcpyAsync(rank_dst, rank_src, rank_size, cudaMemcpyHostToDevice, this->cuda_stream));// use default stream 0
        CUDA_CHECK(cudaEventRecord(event, this->cuda_stream));
        if(this->world_size > 1) {
            CUDA_CHECK(cudaStreamWaitEvent(this->nccl_stream, event));
            NCCL_CHECK(ncclAllGather(rank_dst, all_dst, padded_rank_size, ncclInt8, this->group_communicator, this->nccl_stream));
            CUDA_CHECK(cudaEventRecord(event, this->nccl_stream));
        }

        auto wait_func = [=]() {
            CUDA_CHECK(cudaEventSynchronize(event));
        };
        completion_req_id = this->next_executor_request_id();
        this->wait_thread->submit(completion_req_id, std::move(wait_func));
        completion_thread = this->wait_thread.get();
    }

    return ChunkRequest{completion_thread, completion_req_id};
}

} // namespace instanttensor
