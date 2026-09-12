import unittest
from unittest import mock

import torch

import instanttensor._impl as impl


class ChunkLimitTest(unittest.TestCase):
    def test_chunk_size_is_limited_after_page_alignment(self):
        loader = impl.safe_open.__new__(impl.safe_open)
        loader.filename = ["model.safetensors"]
        loader.world_size = 1
        loader.process_group = None
        loader.device = torch.device("cuda:0")

        config = impl._resolve_open_config(
            buffer_size=None,
            chunk_size=impl.MAX_CHUNK_SIZE + 1,
            concurrency=1,
            io_depth=1,
            max_free_mem_usage=1.0,
            backend=impl.Backend.URING,
        )

        with mock.patch.object(impl, "file_in_memory", return_value=False), \
             mock.patch.object(impl, "select_backend", return_value=impl.Backend.URING):
            with self.assertRaisesRegex(ValueError, "after page alignment"):
                loader._determine_io_params(config)


if __name__ == "__main__":
    unittest.main()
