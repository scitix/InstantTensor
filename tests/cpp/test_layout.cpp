#include <cassert>
#include <random>
#include <stdexcept>

#include "../../csrc/loader_common.cpp"

using namespace instanttensor;

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void configure(Loader &loader, size_t chunk_size, int world_size, size_t logical_size) {
    loader.world_size = world_size;
    loader.rank_chunk_size = chunk_size;
    loader.rank_alignment = PAGE_SIZE;
    loader.world_chunk_alignment = PAGE_SIZE * world_size;
    loader.world_chunk_size = chunk_size * world_size;
    loader.buffer_size = logical_size + 3 * (loader.first_tensor_alignment +
        loader.rank_alignment + loader.world_chunk_alignment);
}

void verify(const Loader &loader) {
    chunk_id_t previous_watermark = -1;
    chunk_id_t completed = -1;
    for (const auto &tensor : loader.tensors) {
        require(tensor.prefetch_chunk_id >= previous_watermark, "watermarks must be monotonic");
        require(tensor.prefetch_chunk_id < static_cast<chunk_id_t>(loader.chunks.size()), "invalid watermark");
        previous_watermark = tensor.prefetch_chunk_id;
        if (tensor.size == 0) {
            require(tensor.first_chunk_id == tensor.last_chunk_id && tensor.last_chunk_id >= -1 &&
                    tensor.last_chunk_id < static_cast<chunk_id_t>(loader.chunks.size()), "invalid empty tensor anchor");
            require(tensor.last_chunk_id <= completed, "empty tensor waits beyond the completed prefix");
        } else {
            require(tensor.first_chunk_id >= 0 && tensor.first_chunk_id <= tensor.last_chunk_id,
                    "invalid tensor chunk range");
            require(tensor.last_chunk_id <= tensor.prefetch_chunk_id, "watermark prevents tensor completion");
            const auto &first = loader.chunks[tensor.first_chunk_id];
            const auto &last = loader.chunks[tensor.last_chunk_id];
            require(first.device_buffer_offset <= tensor.device_buffer_offset &&
                    tensor.device_buffer_offset < first.device_buffer_offset + first.size, "incorrect first chunk");
            size_t end = tensor.device_buffer_offset + tensor.size;
            require(last.device_buffer_offset < end && end <= last.device_buffer_offset + last.size,
                    "incorrect last chunk");
            require(end <= loader.buffer_size, "tensor exceeds allocation");
        }

        // On advancing to this tensor, only the preceding tensors are guaranteed complete.
        // Check the whole permitted prefetch window, without relying on an IO-depth limit.
        for (chunk_id_t a = completed + 1; a <= tensor.prefetch_chunk_id; ++a) {
            const auto &left = loader.chunks[a];
            size_t left_end = left.device_buffer_offset + ROUND_UP(left.size, loader.world_chunk_alignment);
            require(left_end <= loader.buffer_size, "chunk exceeds allocation");
            for (chunk_id_t b = a + 1; b <= tensor.prefetch_chunk_id; ++b) {
                const auto &right = loader.chunks[b];
                size_t right_end = right.device_buffer_offset + ROUND_UP(right.size, loader.world_chunk_alignment);
                require(left_end <= right.device_buffer_offset || right_end <= left.device_buffer_offset,
                        "permitted in-flight chunks overlap");
            }
        }
        completed = std::max(completed, tensor.last_chunk_id);
    }

    for (size_t id = 0; id < loader.chunks.size(); ++id) {
        const auto &chunk = loader.chunks[id];
        bool has_payload = false;
        for (const auto &tensor : loader.tensors) {
            if (tensor.size == 0 || tensor.first_chunk_id > static_cast<chunk_id_t>(id) ||
                tensor.last_chunk_id < static_cast<chunk_id_t>(id)) continue;
            require(tensor.file_index == chunk.file_index, "tensor crosses files");
            require(static_cast<ssize_t>(tensor.device_buffer_offset) - static_cast<ssize_t>(tensor.file_offset) ==
                    static_cast<ssize_t>(chunk.device_buffer_offset) - static_cast<ssize_t>(chunk.file_offset),
                    "noncontiguous tensor placement");
            has_payload |= tensor.device_buffer_offset < chunk.device_buffer_offset + chunk.size &&
                chunk.device_buffer_offset < tensor.device_buffer_offset + tensor.size;
        }
        require(has_payload, "chunk contains only header or reread prefix");
    }
}

void test_header_only_chunk() {
    Loader loader(nullptr, nullptr);
    configure(loader, 4096, 2, 65536);
    loader.compute_layout({{0, 4096}, {0, 53248}, {1, 5120}, {1, 70656}});
    require(loader.chunks.size() == 15, "header-only chunk was emitted");
    require(loader.tensors[1].first_chunk_id == 6, "incorrect first chunk after header wrap");
    require(loader.chunks[6].device_buffer_offset == 16, "header chunk was not skipped");
    verify(loader);
}

void test_first_chunk_prefix() {
    Loader loader(nullptr, nullptr);
    configure(loader, 8192, 4, 273380);
    loader.compute_layout({
        {0,5832}, {0,48981}, {0,90554}, {0,142964}, {0,217929},
        {1,5592}, {1,10616}, {1,45020}, {1,71367}, {1,166175}, {1,228380},
        {2,6264}, {2,38695}, {2,69314}, {2,115248}, {2,228971}, {2,243621},
        {3,5296}, {3,83058}, {3,180764}, {3,208846},
        {4,4488}, {4,13883}, {4,102719}, {4,115287}, {4,190737}, {4,220644},
        {5,6800}, {5,18114},
    });
    require(loader.tensors[9].first_chunk_id == 15, "incorrect prefix first chunk");
    require(loader.tensors[9].prefetch_chunk_id == 22, "watermark still allows chunk 23 to overwrite the prefix");
    verify(loader);
}

void test_boundaries_and_empty_tensors() {
    Loader loader(nullptr, nullptr);
    configure(loader, 4096, 2, 65536);
    loader.compute_layout({{0,4096}, {0,4096}, {0,12288}, {0,12288}, {0,20480}, {0,20480}});
    require(loader.chunks.size() == 2, "empty tensor created an extra chunk");
    require(loader.tensors[1].first_chunk_id == 0 && loader.tensors[1].last_chunk_id == 0, "first full chunk");
    require(loader.tensors[3].first_chunk_id == 1 && loader.tensors[3].last_chunk_id == 1, "second full chunk");
    require(loader.tensors[0].last_chunk_id == -1, "leading empty tensor anchor");
    require(loader.tensors[2].last_chunk_id == 0, "middle empty tensor anchor");
    require(loader.tensors[4].last_chunk_id == 1, "trailing empty tensor anchor");
    verify(loader);

    Loader empty(nullptr, nullptr);
    configure(empty, 4096, 2, 65536);
    empty.compute_layout({{0,4096}, {0,4096}, {1,5120}, {1,5120}});
    require(empty.chunks.empty(), "all-empty input performs IO");
    verify(empty);

    Loader shared(nullptr, nullptr);
    configure(shared, 4096, 2, 65536);
    shared.compute_layout({{0,5120}, {0,5121}, {0,5121}, {0,10239}, {0,51200}, {0,51200}});
    require(shared.tensors[0].first_chunk_id == shared.tensors[2].first_chunk_id, "small tensors should share a chunk");
    require(shared.tensors[3].last_chunk_id > shared.tensors[3].first_chunk_id, "large tensor must span chunks");
    require(shared.tensors[1].last_chunk_id == -1, "empty tensor before any emission");
    require(shared.tensors[4].last_chunk_id == shared.tensors[3].last_chunk_id - 1,
            "empty tensor after a partial chunk must anchor to a previously emitted chunk");
    verify(shared);
}

void test_random_layouts() {
    std::mt19937 random(73);
    for (int trial = 0; trial < 1000; ++trial) {
        size_t ranks = 1 + random() % 4;
        size_t chunk = PAGE_SIZE * (1 + random() % 8);
        size_t world_chunk = chunk * ranks;
        vector<size_t> sizes = {0, 1, PAGE_SIZE - 1, PAGE_SIZE, PAGE_SIZE + 1,
                              world_chunk - 1, world_chunk, world_chunk + 1, 3 * world_chunk + 13};
        vector<pair<size_t, size_t>> offsets;
        size_t files = 1 + random() % 4;
        for (size_t file = 0; file < files; ++file) {
            size_t offset = ROUND_UP(PAGE_SIZE + random() % PAGE_SIZE, size_t(8));
            offsets.emplace_back(file, offset);
            size_t count = 1 + random() % 10;
            for (size_t t = 0; t < count; ++t) {
                offset += sizes[random() % sizes.size()];
                offsets.emplace_back(file, offset);
            }
        }
        Loader loader(nullptr, nullptr);
        configure(loader, chunk, ranks, 10 * world_chunk + 64);
        loader.compute_layout(offsets);
        try {
            verify(loader);
        } catch (...) {
            fprintf(stderr, "layout trial %d failed, ranks=%zu chunk=%zu\n", trial, ranks, chunk);
            throw;
        }
    }
}
}

int main() {
    test_header_only_chunk();
    test_first_chunk_prefix();
    test_boundaries_and_empty_tensors();
    test_random_layouts();
    puts("layout regressions and 1000 randomized layouts passed");
}
