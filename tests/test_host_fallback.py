import io
import json
import os
import struct
import sys
import tempfile
import unittest
import warnings
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

import instanttensor._impl as impl
from instanttensor import Backend, safe_open

MiB = 1 << 20


def write_file(path, tensors, payload=None):
    """Write a safetensors file. ``tensors`` maps name -> (dtype, shape, nbytes),
    laid out contiguously in insertion order. ``payload`` (bytes) fills the data
    section; without it the payload is sparse (metadata-only tests)."""
    header = {}
    offset = 0
    for name, (dtype, shape, nbytes) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    encoded = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(encoded)) + encoded)
        if payload is None:
            f.truncate(f.tell() + offset)
        else:
            assert len(payload) == offset
            f.write(payload)
    return 8 + len(encoded)


def open_meta(paths, *, cuda_free, fraction=1.0, host_fallback=None, buffer_size=None, load_now=False):
    real_open = open

    def open_file(name, *args, **kwargs):
        if str(name) == "/proc/meminfo":
            return io.StringIO("MemAvailable: 28311552 kB\n")
        return real_open(name, *args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(sys, "platform", "linux"))
        stack.enter_context(mock.patch.object(
            torch.cuda, "get_device_properties",
            return_value=SimpleNamespace(is_integrated=False, managed_memory=True, unified_addressing=True)))
        stack.enter_context(mock.patch.object(torch.cuda, "mem_get_info", return_value=(cuda_free, cuda_free)))
        stack.enter_context(mock.patch("builtins.open", side_effect=open_file))
        return safe_open([str(p) for p in paths], "pt", "cuda:0", backend=Backend.MMAP,
                         concurrency=1, chunk_size=MiB, io_depth=1, buffer_size=buffer_size,
                         max_free_mem_usage=fraction, host_fallback=host_fallback, load_now=load_now)


class HostFallbackTest(unittest.TestCase):
    def setUp(self):
        env = {key: value for key, value in os.environ.items() if not key.startswith("INSTANTTENSOR_")}
        patch = mock.patch.dict(os.environ, env, clear=True)
        patch.start()
        self.addCleanup(patch.stop)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.dir = Path(directory.name)
        # Sorted file order puts "big" before "small".
        self.big = self.dir / "big.safetensors"
        self.small = self.dir / "small.safetensors"
        write_file(self.big, {"embed": ("U8", [100 * MiB], 100 * MiB)})
        write_file(self.small, {"a": ("U8", [2 * MiB], 2 * MiB), "b": ("U8", [3 * MiB], 3 * MiB)})

    def test_file_with_oversized_tensor_is_served_from_host(self):
        with self.assertWarnsRegex(RuntimeWarning, "big.safetensors") as caught:
            loader = open_meta([self.big, self.small], cuda_free=64 * MiB, host_fallback=True)
        self.assertEqual(loader._device_memory_budget, 64 * MiB)
        self.assertEqual(loader.host_files, [0])
        self.assertEqual(loader.native_filename, [str(self.small)])
        # Public views still cover every tensor, in file order.
        self.assertEqual(loader.keys(), ["embed", "a", "b"])
        self.assertEqual(loader.offset_keys(), ["embed", "a", "b"])
        self.assertEqual(loader.get_tensor_metadata("embed"), (torch.uint8, torch.Size([100 * MiB])))
        self.assertEqual(loader.total_tensor_size, 105 * MiB)
        # Only native tensors size the ring buffer.
        self.assertEqual(loader.tensor_sizes, [2 * MiB, 3 * MiB])
        self.assertLessEqual(loader.buffer_size, loader._device_memory_budget)
        self.assertEqual([src[0] for src in loader._tensor_sources], ["host", "native", "native"])
        self.assertEqual([src[1] for src in loader._tensor_sources if src[0] == "native"], [0, 1])
        # Native offsets are re-indexed against the native file list.
        self.assertTrue(all(f_idx == 0 for f_idx, _ in loader.tensor_offsets))
        self.assertIn("host memory", str(caught.warning))

    def test_default_fails_and_names_the_tensor(self):
        with self.assertRaisesRegex(RuntimeError, "Tensor 'embed' in .*big.safetensors.*INSTANTTENSOR_HOST_FALLBACK=1"):
            open_meta([self.big, self.small], cuda_free=64 * MiB)
        with self.assertRaisesRegex(RuntimeError, "Tensor 'embed'"):
            open_meta([self.big, self.small], cuda_free=64 * MiB, host_fallback=False)

    def test_environment_variable_controls_fallback(self):
        with mock.patch.dict(os.environ, {"INSTANTTENSOR_HOST_FALLBACK": "0"}):
            with self.assertRaisesRegex(RuntimeError, "Tensor 'embed'"):
                open_meta([self.big], cuda_free=64 * MiB)
        with mock.patch.dict(os.environ, {"INSTANTTENSOR_HOST_FALLBACK": "2"}):
            with self.assertRaisesRegex(ValueError, "INSTANTTENSOR_HOST_FALLBACK"):
                open_meta([self.big], cuda_free=64 * MiB)
        with mock.patch.dict(os.environ, {"INSTANTTENSOR_HOST_FALLBACK": "1"}):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.assertEqual(open_meta([self.big], cuda_free=64 * MiB).host_files, [0])

    def test_nothing_changes_when_everything_fits(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loader = open_meta([self.small], cuda_free=64 * MiB)
        self.assertEqual(loader.host_files, [])
        self.assertEqual(loader.native_filename, [str(self.small)])
        self.assertEqual(loader.tensor_sizes, [2 * MiB, 3 * MiB])
        self.assertEqual([src[0] for src in loader._tensor_sources], ["native", "native"])
        self.assertFalse([w for w in caught if issubclass(w.category, RuntimeWarning)])

    def test_automatic_buffer_is_clamped_to_budget(self):
        path = self.dir / "even.safetensors"
        write_file(path, {f"t{i}": ("U8", [40 * MiB], 40 * MiB) for i in range(3)})
        # Recommendation is t[i] + 2 * t[i+1] = 120 MiB; budget 64 MiB still holds the largest tensor.
        with self.assertWarnsRegex(RuntimeWarning, "fit the device memory budget"):
            loader = open_meta([path], cuda_free=64 * MiB)
        self.assertEqual(loader.buffer_size, 64 * MiB)
        self.assertEqual(loader.host_files, [])

    def test_explicit_oversized_buffer_still_fails(self):
        path = self.dir / "even.safetensors"
        write_file(path, {f"t{i}": ("U8", [40 * MiB], 40 * MiB) for i in range(3)})
        with self.assertRaisesRegex(RuntimeError, "exceeds device memory budget"):
            open_meta([path], cuda_free=64 * MiB, buffer_size=100 * MiB)

    def test_native_loader_is_not_opened_when_all_files_are_host(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loader = open_meta([self.big], cuda_free=64 * MiB, host_fallback=True)
        self.assertEqual(loader.native_filename, [])
        with mock.patch.object(impl._C, "open") as native_open, \
             mock.patch.object(impl._C, "close") as native_close:
            with loader:
                pass
        native_open.assert_not_called()
        native_close.assert_not_called()
        self.assertIsNone(loader.loader_handle)


class ReadHostTensorTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.dir = Path(directory.name)

    def test_roundtrip_dtypes_and_chunking(self):
        a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        b = torch.randn(5, 7).to(torch.bfloat16)
        payload = a.contiguous().view(torch.uint8).numpy().tobytes() + b.contiguous().view(torch.uint8).numpy().tobytes()
        path = self.dir / "w.safetensors"
        header_size = write_file(path, {
            "a": ("F32", [3, 4], a.numel() * 4),
            "b": ("BF16", [5, 7], b.numel() * 2),
        }, payload=payload)
        with mock.patch.object(impl, "HOST_READ_CHUNK_SIZE", 7):  # force many partial reads
            got_a = impl.read_host_tensor(str(path), header_size, [0, a.numel() * 4], torch.float32, [3, 4])
            got_b = impl.read_host_tensor(str(path), header_size, [a.numel() * 4, a.numel() * 4 + b.numel() * 2], torch.bfloat16, [5, 7])
        self.assertEqual(got_a.device.type, "cpu")
        self.assertTrue(torch.equal(got_a, a))
        self.assertTrue(torch.equal(got_b, b))
        self.assertEqual(got_b.dtype, torch.bfloat16)

    def test_truncated_file_is_reported(self):
        path = self.dir / "t.safetensors"
        header_size = write_file(path, {"a": ("U8", [16], 16)}, payload=bytes(16))
        with open(path, "r+b") as f:
            f.truncate(header_size + 8)
        with self.assertRaisesRegex(ValueError, "Unexpected end of file"):
            impl.read_host_tensor(str(path), header_size, [0, 16], torch.uint8, [16])

    def test_tensors_yields_host_tensors_without_a_device_loader(self):
        payload = bytes(range(256)) * (3 * MiB // 256)
        path = self.dir / "big.safetensors"
        write_file(path, {"embed": ("U8", [3 * MiB], 3 * MiB)}, payload=payload)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loader = open_meta([path], cuda_free=2 * MiB, host_fallback=True)  # budget 2 MiB < 3 MiB tensor
        with mock.patch.object(impl._C, "open") as native_open, \
             mock.patch.object(torch.cuda, "current_stream") as current_stream:
            with loader as f:
                tensors = list(f.tensors())
        native_open.assert_not_called()
        current_stream.assert_not_called()
        self.assertEqual([name for name, _ in tensors], ["embed"])
        tensor = tensors[0][1]
        self.assertEqual(tensor.device.type, "cpu")
        self.assertEqual(tensor.shape, torch.Size([3 * MiB]))
        self.assertEqual(tensor[:256].tolist(), list(range(256)))


if __name__ == "__main__":
    unittest.main()
