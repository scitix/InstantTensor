from dataclasses import replace
import unittest
from unittest import mock

from instanttensor._internal import cgroup, resources


BOOT = "12345678-1234-5678-9abc-def012345678"
OTHER_BOOT = "22345678-1234-5678-9abc-def012345678"


def entry(inode, cpus=None, memory=None, host=BOOT):
    return cgroup.CgroupResources(cgroup.CgroupKey(host, 0, 29, inode), "/",
                                  cpus, memory)


def rank(*, host=BOOT, cpus=16, affinity=range(16), memory=1000, entries=(),
         device_memory=1000, integrated=False, fraction=1.0):
    return resources.RankResources(host, cpus, frozenset(affinity), memory,
                                   cgroup.CgroupSnapshot(tuple(entries), True),
                                   device_memory, integrated, fraction)


class ResourceAllocationTest(unittest.TestCase):
    def test_shared_parents_and_private_leaves(self):
        root, parent = entry(1, 20, 1600), entry(2, 7, 901)
        ranks = [rank(entries=(entry(3, 6, 800), parent, root)),
                 rank(entries=(entry(4, 2, 600), parent, root))]
        self.assertEqual(resources.allocate_resources(ranks),
                         (resources.ResourceBudget(3, 450, 1000), resources.ResourceBudget(2, 450, 1000)))

    def test_partial_affinity_overlap(self):
        # rank 0: 1 + 1/2 + 1/2 = 2; rank 1: 1/2 + 1/2 + 1 + 1 = 3.
        ranks = [rank(affinity=(0, 1, 2)), rank(affinity=(1, 2, 3, 4))]
        self.assertEqual([b.avail_cpus for b in resources.allocate_resources(ranks)], [2, 3])

    def test_affinity_uses_exact_floor(self):
        ranks = [rank(affinity=range(12))] * 6
        self.assertEqual([b.avail_cpus for b in resources.allocate_resources(ranks)], [2] * 6)

    def test_disjoint_affinity_and_host_limit(self):
        ranks = [rank(cpus=5, affinity=range(4)), rank(cpus=5, affinity=range(4, 8))]
        self.assertEqual([b.avail_cpus for b in resources.allocate_resources(ranks)], [2, 2])

    def test_different_hosts_do_not_share_cores_or_cgroups(self):
        ranks = [rank(entries=(entry(1, 7, 901),)),
                 rank(host=OTHER_BOOT, entries=(entry(1, 3, 400, OTHER_BOOT),))]
        self.assertEqual(resources.allocate_resources(ranks),
                         (resources.ResourceBudget(7, 901, 1000), resources.ResourceBudget(3, 400, 1000)))

    def test_minimums_unlimited_and_duplicate_keys(self):
        root = entry(1, 1, 1)
        ranks = [rank(entries=(root, root)), rank(entries=(root, entry(2)))]
        self.assertEqual(resources.allocate_resources(ranks), (resources.ResourceBudget(1, 0, 1000),) * 2)
        self.assertEqual(resources.allocate_resources([rank(affinity=(), memory=0)]),
                         (resources.ResourceBudget(1, 0, 1000),))

    def test_independent_cpu_and_memory_minimum(self):
        ranks = [rank(cpus=4, memory=1000), rank(cpus=16, memory=400)]
        with mock.patch.object(resources, "query_rank_resources", return_value=ranks[0]), \
             mock.patch.object(cgroup, "gather_records", return_value=[resources.encode_rank(r) for r in ranks]):
            self.assertEqual(resources.negotiate_resources(object(), device=None, max_free_mem_usage=1.0), resources.ResourceBudget(2, 200, 1000))

    def test_record_roundtrip(self):
        local = rank(affinity=(1, 1000), entries=(entry((1 << 64) - 1, 3, 0), entry(1)))
        local = replace(local, device_free_mem=123456789, max_free_mem_usage=0.1)
        decoded = resources.decode_rank(resources.encode_rank(local))
        self.assertEqual(decoded, replace(local, cgroups=replace(local.cgroups,
            entries=tuple(replace(e, path="") for e in local.cgroups.entries))))

    def test_local_negotiation_does_not_communicate(self):
        with mock.patch.object(resources, "query_rank_resources", return_value=rank()), \
             mock.patch("torch.distributed.all_reduce") as reduce, \
             mock.patch("torch.distributed.all_gather_into_tensor") as gather:
            self.assertEqual(resources.negotiate_resources(None, device=None, max_free_mem_usage=1.0), resources.ResourceBudget(16, 1000, 1000))
        reduce.assert_not_called()
        gather.assert_not_called()

    def test_query_uses_host_limits_when_v2_is_unavailable(self):
        with mock.patch.object(cgroup, "get_host_boot_id", return_value=BOOT), \
             mock.patch.object(cgroup, "query_resources", side_effect=ValueError("No v2")), \
             mock.patch.object(resources.os, "cpu_count", return_value=8), \
             mock.patch.object(resources.os, "sched_getaffinity", return_value={1, 2}), \
             mock.patch.object(resources, "host_available_bytes", return_value=500), \
             mock.patch.object(resources.torch.cuda, "get_device_properties", return_value=mock.Mock(is_integrated=False)), \
             mock.patch.object(resources.torch.cuda, "mem_get_info", return_value=(1000, 2000)):
            local = resources.query_rank_resources(None, max_free_mem_usage=1.0)
        self.assertFalse(local.cgroups.ancestors_complete)
        self.assertEqual(resources.allocate_resources([local]), (resources.ResourceBudget(2, 500, 1000),))

    def test_fraction_applies_to_host_and_device_minimums(self):
        ranks = [rank(memory=1000, device_memory=1000, fraction=0.5),
                 rank(memory=20, device_memory=400, fraction=0.5)]
        with mock.patch.object(resources, "query_rank_resources", return_value=ranks[0]), \
             mock.patch.object(cgroup, "gather_records", return_value=[resources.encode_rank(r) for r in ranks]):
            budget = resources.negotiate_resources(object(), device=None, max_free_mem_usage=1.0)
        self.assertEqual(budget, resources.ResourceBudget(8, 5, 200))

    def test_integrated_device_splits_scaled_host_memory(self):
        ranks = [rank(memory=2000, integrated=True, fraction=0.5),
                 rank(memory=414, integrated=True, device_memory=0, fraction=0.5)]
        with mock.patch.object(resources, "query_rank_resources", return_value=ranks[0]), \
             mock.patch.object(cgroup, "gather_records", return_value=[resources.encode_rank(r) for r in ranks]):
            budget = resources.negotiate_resources(object(), device=None, max_free_mem_usage=1.0)
        # 414 / 2 host users -> 207; floor(207 * 0.5) / 2 -> 51 each.
        self.assertEqual(budget, resources.ResourceBudget(8, 51, 51))

    def test_query_sets_gpu_fields_and_fraction(self):
        for integrated in (False, True):
            with self.subTest(integrated=integrated), \
                 mock.patch.object(cgroup, "get_host_boot_id", return_value=BOOT), \
                 mock.patch.object(cgroup, "query_resources", return_value=cgroup.CgroupSnapshot((), True)), \
                 mock.patch.object(resources, "host_available_bytes", return_value=500), \
                 mock.patch.object(resources.torch.cuda, "get_device_properties", return_value=mock.Mock(is_integrated=integrated)) as properties, \
                 mock.patch.object(resources.torch.cuda, "mem_get_info", return_value=(1000, 2000)) as query:
                local = resources.query_rank_resources("cuda:1", max_free_mem_usage=0.25)
            properties.assert_called_once_with("cuda:1")
            query.assert_called_once_with("cuda:1")
            self.assertEqual(local.device_free_mem, 1000)
            self.assertEqual(local.is_integrated, integrated)
            self.assertEqual(local.max_free_mem_usage, 0.25)

    def test_rank_configuration_must_match(self):
        for field, value in (("max_free_mem_usage", 0.5), ("is_integrated", True)):
            ranks = [rank(), replace(rank(), **{field: value})]
            with self.subTest(field=field), \
                 mock.patch.object(resources, "query_rank_resources", return_value=ranks[0]), \
                 mock.patch.object(cgroup, "gather_records", return_value=[resources.encode_rank(r) for r in ranks]), \
                 self.assertRaisesRegex(ValueError, field + " must be identical"):
                resources.negotiate_resources(object(), device=None, max_free_mem_usage=1.0)

    def test_remote_invalid_fraction_rejects_after_exchange(self):
        for fraction in (0.0, -0.1, 1.1, float("nan"), float("inf")):
            ranks = [rank(), replace(rank(), max_free_mem_usage=fraction)]
            with self.subTest(fraction=fraction), \
                 mock.patch.object(resources, "query_rank_resources", return_value=ranks[0]), \
                 mock.patch.object(cgroup, "gather_records", return_value=[resources.encode_rank(r) for r in ranks]), \
                 self.assertRaisesRegex(ValueError, "max_free_mem_usage"):
                resources.negotiate_resources(object(), device=None, max_free_mem_usage=1.0)


if __name__ == "__main__":
    unittest.main()
