import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import instanttensor._impl as impl
from instanttensor._impl import read_safetensors_metadata


def write_safetensors(path, metadata, payload=b""):
    header = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    path.write_bytes(len(header).to_bytes(8, "little") + header + payload)


class MetadataValidationTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_metadata_accepts_file_covering_last_tensor(self):
        path = self.temp_path / "valid.safetensors"
        write_safetensors(
            path,
            {"weight": {"dtype": "I8", "shape": [4], "data_offsets": [0, 4]}},
            payload=b"data",
        )

        file_metadata, tensor_metadata, header_size = read_safetensors_metadata(str(path))

        self.assertIsNone(file_metadata)
        self.assertEqual(tensor_metadata["weight"]["data_offsets"], [0, 4])
        self.assertEqual(header_size + 4, path.stat().st_size)

    def test_metadata_rejects_file_shorter_than_length_field(self):
        path = self.temp_path / "short-length.safetensors"
        path.write_bytes(b"\x00" * 7)

        with self.assertRaisesRegex(ValueError, "8-byte metadata length field"):
            read_safetensors_metadata(str(path))

    def test_metadata_rejects_truncated_header(self):
        path = self.temp_path / "short-header.safetensors"
        path.write_bytes((10).to_bytes(8, "little") + b"{}")

        with self.assertRaisesRegex(ValueError, "metadata header size"):
            read_safetensors_metadata(str(path))

    def test_metadata_rejects_truncated_tensor_payload(self):
        path = self.temp_path / "short-payload.safetensors"
        write_safetensors(
            path,
            {"weight": {"dtype": "I8", "shape": [4], "data_offsets": [0, 4]}},
            payload=b"abc",
        )

        with self.assertRaisesRegex(ValueError, "last tensor payload end"):
            read_safetensors_metadata(str(path))

    def test_metadata_rejects_invalid_last_tensor_offsets(self):
        path = self.temp_path / "invalid-offsets.safetensors"
        write_safetensors(
            path,
            {"weight": {"dtype": "I8", "shape": [1], "data_offsets": [2, 1]}},
        )

        with self.assertRaisesRegex(ValueError, "invalid data_offsets"):
            read_safetensors_metadata(str(path))


class _FakeProcessGroup:
    """Runs all_gather_object between ranks that are threads of one process."""

    def __init__(self, world_size):
        self.slots = [None] * world_size
        self.barrier = threading.Barrier(world_size, timeout=30)

    def all_gather_object(self, gathered, obj, rank):
        self.slots[rank] = obj
        self.barrier.wait()
        gathered[:] = self.slots
        self.barrier.wait()


class ReadMetadataTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.files = []
        for i in range(5):
            path = Path(self.temp_dir.name) / f"model-{i}.safetensors"
            size = i + 1
            write_safetensors(
                path,
                {f"weight{i}": {"dtype": "I8", "shape": [size], "data_offsets": [0, size]}},
                payload=b"x" * size,
            )
            self.files.append(str(path))
        self.expected = [read_safetensors_metadata(f) for f in self.files]

    def tearDown(self):
        self.temp_dir.cleanup()

    def read_as_group(self, world_size):
        group = _FakeProcessGroup(world_size)
        results = [None] * world_size
        reads = []

        def counting_read(filename):
            reads.append(filename)
            return read_safetensors_metadata(filename)

        def read_on_rank(rank):
            loader = impl.safe_open.__new__(impl.safe_open)
            loader.filename = self.files
            loader.world_size = world_size
            loader.rank = rank
            # The fake group identifies the calling rank by its process_group.
            loader.process_group = rank
            results[rank] = loader._read_metadata()

        with mock.patch.object(impl, "read_safetensors_metadata", side_effect=counting_read), \
             mock.patch.object(impl.dist, "all_gather_object", side_effect=group.all_gather_object):
            threads = [threading.Thread(target=read_on_rank, args=(rank,)) for rank in range(world_size)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=30)
        return results, reads

    def test_single_process_reads_every_header_in_file_order(self):
        results, reads = self.read_as_group(1)

        self.assertEqual(results[0], self.expected)
        self.assertCountEqual(reads, self.files)

    def test_group_reads_each_header_once_and_every_rank_gets_all_in_order(self):
        # 8 ranks for 5 files leaves some ranks without a file to read.
        for world_size in (2, 3, 5, 8):
            with self.subTest(world_size=world_size):
                results, reads = self.read_as_group(world_size)

                self.assertCountEqual(reads, self.files)
                for rank_results in results:
                    self.assertEqual(rank_results, self.expected)
