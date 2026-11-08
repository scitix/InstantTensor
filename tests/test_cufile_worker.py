import pathlib
import shutil
import subprocess
import tempfile
import unittest


class CufileWorkerTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("g++"), "g++ required")
    def test_complete_read_in_worker(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            executable = pathlib.Path(directory) / "test_cufile_worker"
            includes = [
                root / "csrc",
                root / "csrc/third_party/atomic_queue/include",
            ]
            build = subprocess.run([
                "g++", "-std=c++17", "-pthread", "-ffunction-sections",
                "-fdata-sections", "-Wl,--gc-sections",
                *(f"-I{path}" for path in includes),
                str(root / "tests/cpp/test_cufile_worker.cpp"),
                "-ldl", "-o", str(executable),
            ], capture_output=True, text=True, timeout=120)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            subprocess.run([str(executable)], check=True, timeout=30)
