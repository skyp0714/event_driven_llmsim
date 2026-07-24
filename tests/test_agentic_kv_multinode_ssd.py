import unittest

from serving.core.agentic_kv import (
    AgenticKVConfig,
    AgenticKVManager,
    IdleKVEntry,
    KVLocation,
    SSDRecord,
)


class FakeMemory:
    def __init__(self):
        self.npu_used = 0
        self.cpu_used = 0


class FakeScheduler:
    def __init__(self, instance_id, node_id):
        self.instance_id = instance_id
        self.node_id = node_id
        self.num_npus = 4
        self.pd_type = None
        self.enable_prefix_caching = False
        self.memory = FakeMemory()


def make_entry(session_id, instance_id, total_bytes=800):
    return IdleKVEntry(
        session_id=session_id,
        instance_id=instance_id,
        tokens=8,
        block_tokens=8,
        per_rank_bytes=total_bytes // 4,
        total_bytes=total_bytes,
        location=KVLocation.HBM,
        tier_since_ns=0,
        last_access_ns=0,
    )


class MultiNodeSSDTest(unittest.TestCase):
    def make_manager(self, **overrides):
        config = {
            "policy": "tiered",
            "ssd_capacity_gb": 0.000001,
            "ssd_num_devices": 1,
        }
        config.update(overrides)
        return AgenticKVManager(
            [FakeScheduler(0, 0), FakeScheduler(1, 1)],
            AgenticKVConfig(**config),
        )

    def test_media_queues_are_node_local(self):
        manager = self.make_manager()

        node0_write = manager._transfer_resources("cpu_to_ssd", 0)
        node1_write = manager._transfer_resources("cpu_to_ssd", 1)
        node0_read = manager._transfer_resources("ssd_to_cpu_stage", 0)
        node1_read = manager._transfer_resources("ssd_to_cpu_stage", 1)

        self.assertIn("node:0:ssd-pool:write", node0_write)
        self.assertIn("node:1:ssd-pool:write", node1_write)
        self.assertIn("node:0:ssd-pool:read", node0_read)
        self.assertIn("node:1:ssd-pool:read", node1_read)
        self.assertTrue(set(node0_write).isdisjoint(set(node1_write)))
        self.assertTrue(set(node0_read).isdisjoint(set(node1_read)))

    def test_capacity_eviction_cannot_cross_nodes(self):
        manager = self.make_manager()
        node0 = make_entry("node0-old", 0)
        node1 = make_entry("node1-old", 1)
        newcomer = make_entry("node1-new", 1, total_bytes=400)
        node0.location = KVLocation.SSD
        node1.location = KVLocation.SSD
        manager.entries = {
            node0.session_id: node0,
            node1.session_id: node1,
            newcomer.session_id: newcomer,
        }
        manager.ssd_records = {
            node0.session_id: SSDRecord(
                tokens=8, block_tokens=8, bytes=800,
                last_access_ns=0, accounted_until_ns=0, node_id=0),
            node1.session_id: SSDRecord(
                tokens=8, block_tokens=8, bytes=800,
                last_access_ns=0, accounted_until_ns=0, node_id=1),
        }
        manager.ssd_used_bytes = 1600

        self.assertTrue(
            manager._ensure_ssd_capacity(
                newcomer.session_id, 400, 1, node_id=1))

        self.assertIn(node0.session_id, manager.ssd_records)
        self.assertNotIn(node1.session_id, manager.ssd_records)
        self.assertEqual(node0.location, KVLocation.SSD)
        self.assertEqual(node1.location, KVLocation.DROPPED)
        self.assertEqual(manager._ssd_used_bytes_on_node(0), 800)
        self.assertEqual(manager._ssd_used_bytes_on_node(1), 0)

    def test_each_node_can_reserve_its_full_pool(self):
        manager = self.make_manager(policy="hbm_ssd_direct")
        left = make_entry("left", 0, total_bytes=1000)
        right = make_entry("right", 1, total_bytes=1000)
        manager.entries = {left.session_id: left, right.session_id: right}

        self.assertTrue(manager._reserve_direct_ssd_capacity(left, 0))
        self.assertTrue(manager._reserve_direct_ssd_capacity(right, 0))
        self.assertEqual(manager._ssd_reserved_bytes(node_id=0), 1000)
        self.assertEqual(manager._ssd_reserved_bytes(node_id=1), 1000)
        self.assertEqual(manager._ssd_reserved_bytes(), 2000)

        manager._release_direct_ssd_capacity(left, 1)
        manager._release_direct_ssd_capacity(right, 1)
        self.assertEqual(manager._ssd_reserved_bytes(), 0)
        self.assertEqual(
            manager._direct_ssd_capacity_reservation_nodes, {})

    def test_report_counts_devices_and_capacity_per_host(self):
        manager = self.make_manager(
            ssd_capacity_gb=3840,
            ssd_num_devices=8,
        )

        summary = manager.summary(simulated_duration_ns=1)
        ssd = summary["ssd"]
        devices = summary["storage"]["devices"]

        self.assertEqual(ssd["node_count"], 2)
        self.assertEqual(ssd["devices_per_node"], 8)
        self.assertEqual(ssd["num_devices"], 16)
        self.assertEqual(ssd["capacity_bytes_per_node"], 30_720_000_000_000)
        self.assertEqual(ssd["capacity_bytes"], 61_440_000_000_000)
        self.assertEqual(len(devices), 16)
        self.assertEqual(
            {device["node_id"] for device in devices}, {0, 1})
        self.assertEqual(
            summary["storage"]["distribution"],
            "balanced_across_node_local_pools_assumption",
        )


if __name__ == "__main__":
    unittest.main()
