import io
import json
import os
import re
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
CUDA0 = torch.device("cuda:0")


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


def open_meta(paths, *, cuda_free, config=None, buffer_size=None, copy=True):
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
                         max_free_mem_usage=1.0, tensor_distribution_config=config,
                         load_now=False, copy=copy)


def clean_env(test):
    env = {key: value for key, value in os.environ.items() if not key.startswith("INSTANTTENSOR_")}
    patch = mock.patch.dict(os.environ, env, clear=True)
    patch.start()
    test.addCleanup(patch.stop)


class ParseConfigTest(unittest.TestCase):
    def test_dict_and_pair_list_are_equivalent_and_ordered(self):
        as_dict = impl.parse_tensor_distribution_config(
            {r"\.engram\.": {"device": "cpu"}, "": {}}, CUDA0)
        as_list = impl.parse_tensor_distribution_config(
            [[r"\.engram\.", {"device": "cpu"}], ["", {}]], CUDA0)
        for rules in (as_dict, as_list):
            self.assertEqual([(r.pattern, d) for r, d in rules], [(r"\.engram\.", "cpu"), ("", None)])

    def test_first_match_wins_and_unmatched_keeps_default(self):
        rules = impl.parse_tensor_distribution_config(
            [("embed", {}), ("engram", {"device": "cpu"})], CUDA0)
        self.assertIsNone(impl.resolve_tensor_device("layers.1.engram.embed.weight", rules))
        self.assertEqual(impl.resolve_tensor_device("layers.1.engram.proj.weight", rules), "cpu")
        self.assertIsNone(impl.resolve_tensor_device("lm_head.weight", rules))

    def test_device_values(self):
        for device, expected in (("cpu", "cpu"), (torch.device("cpu"), "cpu"), ("cuda:0", None),
                                 ("cuda", None), (CUDA0, None), (None, None)):
            with self.subTest(device=device):
                rules = impl.parse_tensor_distribution_config({"x": {"device": device}}, CUDA0)
                self.assertEqual(rules[0][1], expected)

    def test_invalid_configs_are_rejected(self):
        cases = [
            ("not a config", "dict or a list"),
            ([("only-pattern",)], "pairs"),
            ({1: {}}, "must be a str"),
            ({"(": {}}, "Invalid tensor_distribution_config pattern"),
            ({"x": "cpu"}, "must be a dict"),
            ({"x": {"dtype": "bf16"}}, "Unsupported keys"),
            ({"x": {"device": "cuda:1"}}, "Unsupported device"),
            ({"x": {"device": "not-a-device"}}, "Invalid device"),
        ]
        for config, message in cases:
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, message):
                impl.parse_tensor_distribution_config(config, CUDA0)

    def test_environment_variable(self):
        clean_env(self)
        with mock.patch.dict(os.environ, {"INSTANTTENSOR_TENSOR_DISTRIBUTION_CONFIG": '{"\\\\.engram\\\\.": {"device": "cpu"}}'}):
            self.assertEqual(impl.env_tensor_distribution_config(), {r"\.engram\.": {"device": "cpu"}})
        with mock.patch.dict(os.environ, {"INSTANTTENSOR_TENSOR_DISTRIBUTION_CONFIG": '[["a", {"device": "cpu"}]]'}):
            self.assertEqual(impl.env_tensor_distribution_config(), [["a", {"device": "cpu"}]])
        with mock.patch.dict(os.environ, {"INSTANTTENSOR_TENSOR_DISTRIBUTION_CONFIG": "{oops"}):
            with self.assertRaisesRegex(ValueError, "not valid JSON"):
                impl.env_tensor_distribution_config()
        with mock.patch.dict(os.environ, {"INSTANTTENSOR_TENSOR_DISTRIBUTION_CONFIG": "  "}):
            self.assertIsNone(impl.env_tensor_distribution_config())


class RoutingTest(unittest.TestCase):
    def setUp(self):
        clean_env(self)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.dir = Path(directory.name)
        # Sorted file order puts "a_mixed" before "b_small".
        self.mixed = self.dir / "a_mixed.safetensors"
        self.small = self.dir / "b_small.safetensors"
        self.mixed_header = write_file(self.mixed, {
            "w0": ("U8", [2 * MiB], 2 * MiB),
            "layers.1.engram.embed.weight": ("U8", [100 * MiB], 100 * MiB),
            "w2": ("U8", [3 * MiB], 3 * MiB),
            "w3": ("U8", [1 * MiB], 1 * MiB),
        })
        self.small_header = write_file(self.small, {"s0": ("U8", [4 * MiB], 4 * MiB)})
        self.config = {r"\.engram\.": {"device": "cpu"}}

    def test_cpu_tensor_splits_its_file_into_native_runs(self):
        loader = open_meta([self.mixed, self.small], cuda_free=64 * MiB, config=self.config)
        self.assertEqual(loader.cpu_tensor_names, ["layers.1.engram.embed.weight"])
        # mixed file -> two runs ([w0], [w2, w3]), small file -> one run
        self.assertEqual(loader.native_filename, [str(self.mixed), str(self.mixed), str(self.small)])
        h, s = self.mixed_header, self.small_header
        self.assertEqual(loader.tensor_offsets, [
            (0, h + 0), (0, h + 2 * MiB),
            (1, h + 102 * MiB), (1, h + 105 * MiB), (1, h + 106 * MiB),
            (2, s + 0), (2, s + 4 * MiB),
        ])
        self.assertEqual(loader.tensor_sizes, [2 * MiB, 3 * MiB, 1 * MiB, 4 * MiB])
        self.assertEqual([src[0] for src in loader._tensor_sources],
                         ["native", "cpu", "native", "native", "native"])
        self.assertEqual([src[1] for src in loader._tensor_sources if src[0] == "native"], [0, 1, 2, 3])
        # Public views still cover every tensor, in file order.
        self.assertEqual(loader.keys(), ["w0", "layers.1.engram.embed.weight", "w2", "w3", "s0"])
        self.assertEqual(loader.total_tensor_size, 110 * MiB)
        self.assertLessEqual(loader.buffer_size, loader._device_memory_budget)

    def test_offsets_match_unrouted_layout_when_nothing_is_routed(self):
        loader = open_meta([self.small], cuda_free=64 * MiB, config={"nomatch": {"device": "cpu"}})
        self.assertEqual(loader.native_filename, [str(self.small)])
        self.assertEqual(loader.tensor_offsets, [(0, self.small_header), (0, self.small_header + 4 * MiB)])
        self.assertEqual(loader.cpu_tensor_names, [])

    def test_oversized_device_tensor_error_names_it_and_suggests_a_rule(self):
        with self.assertRaises(RuntimeError) as raised:
            open_meta([self.mixed, self.small], cuda_free=64 * MiB)
        message = str(raised.exception)
        self.assertIn("'layers.1.engram.embed.weight'", message)
        self.assertIn("tensor_distribution_config=", message)
        suggested = re.search(r"tensor_distribution_config=(\{.*\})\.$", message).group(1)
        rules = impl.parse_tensor_distribution_config(eval(suggested), CUDA0)
        self.assertEqual(impl.resolve_tensor_device("layers.1.engram.embed.weight", rules), "cpu")
        self.assertIsNone(impl.resolve_tensor_device("w0", rules))

    def test_environment_config_is_used_and_argument_wins(self):
        with mock.patch.dict(os.environ, {"INSTANTTENSOR_TENSOR_DISTRIBUTION_CONFIG": json.dumps(self.config)}):
            loader = open_meta([self.mixed], cuda_free=64 * MiB)
            self.assertEqual(loader.cpu_tensor_names, ["layers.1.engram.embed.weight"])
            loader = open_meta([self.mixed], cuda_free=128 * MiB, config={"w0": {"device": "cpu"}})
            self.assertEqual(loader.cpu_tensor_names, ["w0"])

    def test_automatic_buffer_is_clamped_to_budget(self):
        path = self.dir / "even.safetensors"
        write_file(path, {f"t{i}": ("U8", [40 * MiB], 40 * MiB) for i in range(3)})
        # Recommendation is t[i] + 2 * t[i+1] = 120 MiB; budget 64 MiB still holds the largest tensor.
        with self.assertWarnsRegex(RuntimeWarning, "fit the device memory budget"):
            loader = open_meta([path], cuda_free=64 * MiB)
        self.assertEqual(loader.buffer_size, 64 * MiB)

    def test_explicit_oversized_buffer_still_fails(self):
        path = self.dir / "even.safetensors"
        write_file(path, {f"t{i}": ("U8", [40 * MiB], 40 * MiB) for i in range(3)})
        with self.assertRaisesRegex(RuntimeError, "exceeds device memory budget"):
            open_meta([path], cuda_free=64 * MiB, buffer_size=100 * MiB)

    def test_native_loader_is_not_opened_when_every_tensor_is_cpu(self):
        loader = open_meta([self.mixed], cuda_free=64 * MiB, config={"": {"device": "cpu"}})
        self.assertEqual(loader.native_filename, [])
        self.assertEqual(loader.tensor_offsets, [])
        with mock.patch.object(impl._C, "open") as native_open, \
             mock.patch.object(impl._C, "close") as native_close:
            with loader:
                pass
        native_open.assert_not_called()
        native_close.assert_not_called()


class CPUReadTest(unittest.TestCase):
    def setUp(self):
        clean_env(self)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.dir = Path(directory.name)
        self.a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        self.b = torch.randn(5, 7).to(torch.bfloat16)
        payload = (self.a.contiguous().view(torch.uint8).numpy().tobytes()
                   + self.b.contiguous().view(torch.uint8).numpy().tobytes()
                   + bytes(0))
        self.path = self.dir / "w.safetensors"
        write_file(self.path, {
            "a": ("F32", [3, 4], self.a.numel() * 4),
            "b": ("BF16", [5, 7], self.b.numel() * 2),
            "empty": ("F32", [0], 0),
        }, payload=payload)

    def read_all(self, copy):
        loader = open_meta([self.path], cuda_free=64 * MiB, config={"": {"device": "cpu"}}, copy=copy)
        with mock.patch.object(impl._C, "open") as native_open, \
             mock.patch.object(torch.cuda, "current_stream") as current_stream:
            with loader as f:
                tensors = dict(f.tensors())
        native_open.assert_not_called()
        current_stream.assert_not_called()
        return tensors

    def test_roundtrip_with_copy(self):
        tensors = self.read_all(copy=True)
        self.assertEqual(list(tensors), ["a", "b", "empty"])
        self.assertTrue(torch.equal(tensors["a"], self.a))
        self.assertTrue(torch.equal(tensors["b"], self.b))
        self.assertEqual(tensors["b"].dtype, torch.bfloat16)
        self.assertEqual(tensors["empty"].shape, torch.Size([0]))
        for tensor in tensors.values():
            self.assertEqual(tensor.device.type, "cpu")
        # Owned memory: writing must not affect a second read of the file.
        tensors["a"].fill_(-1)
        self.assertTrue(torch.equal(self.read_all(copy=True)["a"], self.a))

    def test_zero_copy_views_outlive_the_context_and_never_touch_the_file(self):
        before = self.path.read_bytes()
        tensors = self.read_all(copy=False)
        self.assertTrue(torch.equal(tensors["a"], self.a))
        self.assertTrue(torch.equal(tensors["b"], self.b))
        tensors["a"].fill_(-1)  # copy-on-write mapping: allowed, but private
        self.assertTrue(torch.equal(tensors["b"], self.b))
        self.assertEqual(self.path.read_bytes(), before)

    def test_read_cpu_tensor_clones_misaligned_views(self):
        import mmap
        data = bytes(1) + self.a.contiguous().view(torch.uint8).numpy().tobytes()
        path = self.dir / "raw.bin"
        path.write_bytes(data)
        with open(path, "rb") as f:
            mapped = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
        tensor = impl.read_cpu_tensor(mapped, 0, [1, 1 + self.a.numel() * 4], torch.float32, [3, 4], copy=False)
        self.assertTrue(torch.equal(tensor, self.a))
        self.assertEqual(tensor.data_ptr() % 4, 0)


if __name__ == "__main__":
    unittest.main()
