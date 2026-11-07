import contextlib
from datetime import timedelta
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch
import torch.distributed as dist

from instanttensor import cgroup, resources


BOOT = "12345678-1234-5678-9abc-def012345678"
OTHER_BOOT = "22345678-1234-5678-9abc-def012345678"


def entry(inode, host=BOOT):
    return cgroup.CgroupResources(
        cgroup.CgroupKey(host, 0, 29, inode), f"/cgroup-{inode}",
        None, None,
    )


class CgroupTest(unittest.TestCase):
    def test_mount_mapping_and_escaping(self):
        mounts = cgroup._mounts(
            "10 1 0:29 /job /cg\\040sub rw - cgroup2 cgroup rw\n"
            "11 1 0:29 / /cg rw - cgroup2 cgroup rw\n"
            "12 1 0:29 /.. /bad rw - cgroup2 cgroup rw\n"
        )
        self.assertEqual([m["id"] for m in mounts], [11, 10])
        self.assertEqual(cgroup._local_path(mounts[1], "/job/worker"), "/cg sub/worker")
        self.assertIsNone(cgroup._local_path(mounts[1], "/job-other/worker"))
        self.assertEqual(cgroup._v2_path("1:cpu:/old\n0::/job\n"), "/job")
        for text in ("1:memory:/old\n", "0::/../hidden\n"):
            with self.assertRaises(ValueError):
                cgroup._v2_path(text)

    def test_resource_files_and_missing_controllers(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            path = Path(directory)
            fd = os.open(path, os.O_PATH | os.O_DIRECTORY)
            stack.callback(os.close, fd)
            key = entry(10).key
            empty = cgroup._resources(fd, key, "/")
            self.assertIsNone(empty.avail_cpus)
            self.assertIsNone(empty.avail_mem)
            for name, value in {
                "cpu.max": "150000 100000\n", "cpuset.cpus.effective": "0-2,4,8-9\n",
                "memory.max": "4096\n", "memory.current": "3072\n",
                "memory.stat": "anon 1024\nfile 2048\ninactive_file 1024\n",
            }.items():
                (path / name).write_text(value)
            resources = cgroup._resources(fd, key, "/job")
            self.assertEqual(resources.avail_cpus, 2)
            self.assertEqual(resources.avail_mem, 2048)
            (path / "memory.current").write_text("8192\n")
            self.assertEqual(cgroup._resources(fd, key, "/job").avail_mem, 0)
            (path / "cpu.max").write_text("max 100000\n")
            (path / "memory.max").write_text("max\n")
            resources = cgroup._resources(fd, key, "/job")
            self.assertEqual(resources.avail_cpus, 6)
            self.assertIsNone(resources.avail_mem)
            (path / "cpu.max").write_text("800000 100000\n")
            self.assertEqual(cgroup._resources(fd, key, "/job").avail_cpus, 6)
            (path / "memory.max").write_text("4096\n")
            (path / "memory.current").unlink()
            with self.assertRaises(FileNotFoundError):
                cgroup._resources(fd, key, "/job")

    def test_permission_errors_are_not_unlimited(self):
        with mock.patch.object(cgroup, "_read_at", side_effect=PermissionError("denied")):
            with self.assertRaises(PermissionError):
                cgroup._resources(0, entry(1).key, "/")

    def test_query_includes_unlimited_parents_and_visible_root(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "worker").mkdir()
            (path / "memory.max").write_text("4096\n")
            (path / "memory.current").write_text("1024\n")
            (path / "memory.stat").write_text("inactive_file 512\n")
            fd = os.open(path, os.O_PATH | os.O_DIRECTORY)
            try:
                info = dict(line.split(":", 1) for line in cgroup._read(f"/proc/self/fdinfo/{fd}").splitlines())
                st = os.fstat(fd)
            finally:
                os.close(fd)
            mount = {"id": int(info["mnt_id"]), "device": f"{os.major(st.st_dev)}:{os.minor(st.st_dev)}",
                     "point": directory, "root_parts": ("job",)}
            original_read = cgroup._read

            def read(filename):
                if filename == "/proc/self/cgroup":
                    return "0::/job/worker\n"
                return original_read(filename)

            with mock.patch.object(cgroup, "_read", side_effect=read), \
                 mock.patch.object(cgroup, "_mounts", return_value=[mount]):
                snapshot = cgroup.query_resources(host_boot_id=BOOT)
            self.assertEqual([e.path for e in snapshot.entries], ["/job/worker", "/job"])
            self.assertIsNone(snapshot.entries[0].avail_mem)
            self.assertEqual(snapshot.entries[1].avail_mem, 3584)
            self.assertFalse(snapshot.ancestors_complete)
            self.assertEqual(snapshot.entries[-1].key.inode, st.st_ino)

    def test_local_count_and_lossless_key_encoding(self):
        leaf, root = entry((1 << 64) - 1), entry(1)
        self.assertEqual(cgroup._decode_key(cgroup._encode_key(leaf.key)), leaf.key)
        snapshot = cgroup.CgroupSnapshot((leaf, root, root), False)
        with mock.patch.object(dist, "all_reduce") as reduce, \
             mock.patch.object(dist, "all_gather_into_tensor") as gather:
            usage = cgroup.count_process_group_users(snapshot)
        reduce.assert_not_called()
        gather.assert_not_called()
        self.assertEqual(usage.process_counts, {leaf.key: 1, root.key: 1})
        self.assertEqual(usage.ancestors_complete_by_rank, (False,))

    def test_distributed_cpu_counts(self):
        self.run_distributed("gloo")

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "two GPUs required")
    def test_distributed_cuda_counts(self):
        self.run_distributed("nccl")

    def run_distributed(self, backend):
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PYTHONPATH=str(root))
        result = subprocess.run([
            sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
            __file__, "--distributed", backend,
        ], cwd=root, env=env, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


def exercise_distributed(backend):
    rank = int(os.environ["LOCAL_RANK"])
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(backend, timeout=timedelta(seconds=20))
    try:
        root, parent = entry(1), entry(10)
        leaf = entry(100 + rank)
        extra = entry((1 << 64) - 1)
        entries = (leaf, extra, parent, root) if rank == 0 else (leaf, parent, root)
        snapshot = cgroup.CgroupSnapshot(entries, True)
        with mock.patch.object(dist, "all_reduce", wraps=dist.all_reduce) as reduce, \
             mock.patch.object(dist, "all_gather_into_tensor", wraps=dist.all_gather_into_tensor) as gather:
            usage = cgroup.count_process_group_users(snapshot, dist.group.WORLD)
        assert reduce.call_count == 1
        assert gather.call_count == 1
        assert usage.process_counts == {entry(100).key: 1, entry(101).key: 1,
                                        extra.key: 1, parent.key: 2, root.key: 2}, usage
        assert usage.ancestors_complete_by_rank == (True, True)

        # Different hosts may have identical device/inode numbers and paths.
        node_root = entry(1, BOOT if rank == 0 else OTHER_BOOT)
        usage = cgroup.count_process_group_users(cgroup.CgroupSnapshot((node_root,), True), dist.group.WORLD)
        assert usage.process_counts == {entry(1).key: 1, entry(1, OTHER_BOOT).key: 1}, usage

        # A hidden ancestor is not inferred from strings or counted as observed.
        snapshot = cgroup.CgroupSnapshot((leaf, parent, root) if rank == 0 else (leaf, parent), rank == 0)
        usage = cgroup.count_process_group_users(snapshot, dist.group.WORLD)
        assert usage.process_counts[parent.key] == 2
        assert usage.process_counts[root.key] == 1
        assert usage.ancestors_complete_by_rank == (True, False)

        snapshot = cgroup.query_resources()
        usage = cgroup.count_process_group_users(snapshot, dist.group.WORLD)
        assert all(usage.process_counts[e.key] == 2 for e in snapshot.entries), usage

        # Exercise the resource record exchange with unequal affinity/ancestry lengths.
        local = resources.RankResources(BOOT, 16, frozenset(range(4 if rank == 0 else 8)),
            1000 if rank == 0 else 800,
            cgroup.CgroupSnapshot((cgroup.CgroupResources(parent.key, "/", 7, 901),)
                                 + ((leaf,) if rank == 0 else ()), True), 1000, False, 1.0)
        with mock.patch.object(resources, "query_rank_resources", return_value=local):
            budget = resources.negotiate_resources(dist.group.WORLD, device=None, max_free_mem_usage=1.0)
        assert budget == resources.ResourceBudget(2, 400, 1000), budget

        # Membership uses group ranks, including a subgroup without global rank 0.
        subgroup = dist.new_group([1], backend=backend)
        if rank == 1:
            usage = cgroup.count_process_group_users(snapshot, subgroup)
            assert all(count == 1 for count in usage.process_counts.values()), usage
            with mock.patch.object(resources, "query_rank_resources", return_value=local):
                budget = resources.negotiate_resources(subgroup, device=None, max_free_mem_usage=1.0)
            assert budget == resources.ResourceBudget(7, 800, 1000), budget
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    if "--distributed" in sys.argv:
        exercise_distributed(sys.argv[-1])
    else:
        unittest.main()
