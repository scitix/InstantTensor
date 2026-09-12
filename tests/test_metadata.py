import json
import tempfile
import unittest
from pathlib import Path

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
