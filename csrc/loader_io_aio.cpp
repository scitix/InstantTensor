#include <instant_tensor/loader.hpp>

namespace instanttensor {

void Loader::open_file_aio(FileInfo &f) {
    int open_flags = O_RDONLY;
    if (this->backend == Backend::AIO) {// != AIO_BUFFERED
        open_flags |= O_DIRECT;
    }
    f.fd = ::open(f.filename.c_str(), open_flags);
    if (f.fd < 0) {
        throw std::runtime_error("Failed to open file: " + f.filename);
    }
    struct stat st;
    if (fstat(f.fd, &st) < 0) { throw std::runtime_error("Failed to fstat file: " + f.filename); }
    f.size = st.st_size;

    if (this->backend == Backend::AIO_BUFFERED) {
        posix_fadvise(f.fd, 0, 0, POSIX_FADV_SEQUENTIAL);
    }

    this->need_host_buffer = true;
    this->need_cuda_thread = true;
}

void Loader::initialize_aio_context() {
    int ret = io_setup(this->io_depth, &this->aio_ctx);
    if(ret < 0){
        print_and_throw(std::runtime_error("Failed to setup aio: " + std::string(strerror(-ret))));
    }
    this->aio_iocbs.resize(this->io_depth);
    this->aio_iocb_ptrs.resize(this->io_depth);
    for(size_t i = 0; i < this->io_depth; i++) {
        this->aio_iocb_ptrs[i] = &this->aio_iocbs[i];
    }
    this->aio_events.resize(this->io_depth);
}

void Loader::close_file_aio(FileInfo &f) {
    ::close(f.fd);
}

void Loader::destroy_aio_context() {
    int ret = io_destroy(this->aio_ctx);
    if(ret < 0){
        print_and_throw(std::runtime_error("Failed to destroy aio: " + std::string(strerror(-ret))));
    }
}

ChunkRequest Loader::post_read_chunk_aio(const ChunkIOParams &p) {
    chunk_id_t chunk_id = p.chunk_id;
    size_t submit_cnt = 0;
    size_t window_idx = p.window_idx;

    bool unaligned_last_page = p.rank_size % this->rank_alignment != 0;

    if(p.rank_size > 0) {
        size_t rank_size_aligned = ROUND_UP(p.rank_size, this->rank_alignment);
        struct iocb *iocb = this->aio_iocb_ptrs[window_idx];
        size_t file_offset = p.chunk.file_offset + p.rank_offset;
        // NOTE: aio needs the read size aligned to PAGE_SIZE
        io_prep_pread(iocb, p.file.fd, (char*)this->host_buffer + p.window_offset, rank_size_aligned, file_offset);
        iocb->data = reinterpret_cast<void*>(static_cast<uintptr_t>(chunk_id));
        submit_cnt = 1;
    }
    this->chunks[chunk_id].extra_data.unfinished_cnt = submit_cnt;

    // NOTE: This will block at the last page of the file if the file is not page aligned.
    //       So we put the last page into another thread
    auto aio_func = [=]() {
        size_t submitted = 0;
        while(submitted < submit_cnt) {
            int ret = io_submit(this->aio_ctx, submit_cnt - submitted, this->aio_iocb_ptrs.data() + window_idx + submitted);
            if(ret < 0){
                print_and_throw(std::runtime_error("Failed to submit aio: " + std::string(strerror(-ret))));
            }
            submitted += ret;
        }
    };

    int aio_req_id = EXECUTOR_STOP_REQUEST_ID;
    if(submit_cnt > 0) {
        if(unaligned_last_page) {
            aio_req_id = this->next_executor_request_id();
            this->last_page_reader_thread->submit(aio_req_id, std::move(aio_func));
        }
        else {
            aio_func();
        }
    }
    

    void *rank_mid = (char*)this->host_buffer + p.window_offset;
    void *rank_dst = p.rank_dst;
    void *all_dst = p.all_dst;
    size_t rank_size = p.rank_size;
    size_t padded_rank_size = p.padded_rank_size;
    cudaEvent_t event = p.event;
    auto cuda_func = [=]() {
        if(aio_req_id != EXECUTOR_STOP_REQUEST_ID) {
            this->last_page_reader_thread->reap(aio_req_id);
        }
        // disk to host
        size_t &unfinished_cnt = this->chunks[chunk_id].extra_data.unfinished_cnt;
        while(unfinished_cnt > 0) {
            int got = io_getevents(this->aio_ctx, unfinished_cnt, unfinished_cnt, this->aio_events.data(), NULL);
            // got may < min_nr and >= 0 if interrupted
            if(got < 0){
                print_and_throw(std::runtime_error("Failed to get aio events: " + std::string(strerror(-got))));
            }
            for(int i = 0; i < got; i++) {
                if(this->aio_events[i].res < 0) {
                    print_and_throw(std::runtime_error("Failed to get aio events: " + std::string(strerror(-this->aio_events[i].res))));
                }
                chunk_id_t event_chunk_id = static_cast<chunk_id_t>(
                    reinterpret_cast<uintptr_t>(this->aio_events[i].data));
                const Chunk &event_chunk = this->chunks[event_chunk_id];
                size_t padded_world_size = ROUND_UP(event_chunk.size, this->world_chunk_alignment);
                size_t padded_rank_size = padded_world_size / this->world_size;
                size_t logical_size = rank_logical_size(
                    event_chunk.size, padded_rank_size * this->rank, padded_rank_size);
                if(static_cast<size_t>(this->aio_events[i].res) < logical_size) {
                    print_and_throw(std::runtime_error(
                        "Unexpected AIO short read: chunk_id=" + std::to_string(event_chunk_id)
                        + ", bytes_read=" + std::to_string(this->aio_events[i].res)
                        + ", logical_size=" + std::to_string(logical_size)));
                }
                this->chunks[event_chunk_id].extra_data.unfinished_cnt --;
            }
        }

        // host to device
        CUDA_CHECK(cudaMemcpyAsync(rank_dst, rank_mid, rank_size, cudaMemcpyHostToDevice, this->cuda_stream));// 240GB/s for 8 GPUs
        CUDA_CHECK(cudaEventRecord(event, this->cuda_stream));
        if(this->world_size > 1) {
            CUDA_CHECK(cudaStreamWaitEvent(this->nccl_stream, event));
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

    int completion_req_id = this->next_executor_request_id();
    this->wait_thread->submit(completion_req_id, std::move(wait_func));
    SingleThreadTaskExecutor *completion_thread = this->wait_thread.get();

    return ChunkRequest{completion_thread, completion_req_id};
}

} // namespace instanttensor
