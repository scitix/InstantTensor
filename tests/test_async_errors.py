import json
import os
import tempfile
import unittest

import torch

from instanttensor import Backend, safe_open


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class AsyncErrorPropagationTest(unittest.TestCase):
    def make_file(self, path):
        metadata = {
            "weight": {
                "dtype": "I8",
                "shape": [4096],
                "data_offsets": [0, 4096],
            }
        }
        header = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
        with open(path, "wb") as f:
            f.write(len(header).to_bytes(8, "little"))
            f.write(header)
            f.write(b"x" * 4096)
        return 8 + len(header)

    def test_async_read_errors_reach_python(self):
        for backend in (Backend.AIO, Backend.URING):
            with self.subTest(backend=backend.name), tempfile.TemporaryDirectory() as directory:
                path = os.path.join(directory, "truncated.safetensors")
                header_end = self.make_file(path)
                loader = safe_open(
                    path,
                    framework="pt",
                    device=0,
                    backend=backend,
                    load_now=False,
                )
                with open(path, "r+b") as f:
                    f.truncate(header_end)

                with self.assertRaisesRegex(RuntimeError, "short read"):
                    with loader as opened:
                        next(opened.tensors())


if __name__ == "__main__":
    unittest.main()
