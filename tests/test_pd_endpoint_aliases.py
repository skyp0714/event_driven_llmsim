import tempfile
import unittest
from pathlib import Path

import yaml

from serving.core.config_builder import (
    _compute_pd_endpoint_aliases,
    _create_network_config,
)


def _instance(instance_id, role, *, ranks=2, node_id=0):
    return {
        "instance_id": instance_id,
        "node_id": node_id,
        "model_name": "model",
        "num_npus": ranks,
        "tp_size": ranks,
        "pp_size": 1,
        "pd_type": role,
    }


class PDEndpointAliasTest(unittest.TestCase):
    def test_tp8_pair_maps_eight_virtual_receivers(self):
        instances = [
            _instance(0, "prefill", ranks=8),
            _instance(1, "decode", ranks=8),
        ]

        aliases, mapped, unresolved = _compute_pd_endpoint_aliases(instances)

        self.assertEqual(len(aliases), 24)
        self.assertEqual(aliases[8:16], list(range(16, 24)))
        self.assertEqual(mapped, dict(zip(range(8, 16), range(16, 24))))
        self.assertEqual(unresolved, [])

    def test_unique_pd_pair_aliases_shadow_ranks_to_decode(self):
        instances = [
            _instance(0, "prefill"),
            _instance(1, "decode"),
        ]

        aliases, mapped, unresolved = _compute_pd_endpoint_aliases(instances)

        # P ranks 0-1, virtual receivers 2-3, physical D ranks 4-5.
        self.assertEqual(aliases, [0, 1, 4, 5, 4, 5])
        self.assertEqual(mapped, {2: 4, 3: 5})
        self.assertEqual(unresolved, [])

    def test_ambiguous_decode_replicas_keep_identity_mapping(self):
        instances = [
            _instance(0, "prefill", ranks=1),
            _instance(1, "decode", ranks=1),
            _instance(2, "decode", ranks=1),
        ]

        aliases, mapped, unresolved = _compute_pd_endpoint_aliases(instances)

        self.assertEqual(aliases, [0, 1, 2, 3])
        self.assertEqual(mapped, {})
        self.assertEqual(unresolved, [0])

    def test_two_nodes_alias_each_prefill_to_its_local_decode(self):
        instances = [
            _instance(0, "prefill", ranks=4, node_id=0),
            _instance(1, "decode", ranks=4, node_id=0),
            _instance(2, "prefill", ranks=4, node_id=1),
            _instance(3, "decode", ranks=4, node_id=1),
        ]

        aliases, mapped, unresolved = _compute_pd_endpoint_aliases(instances)

        self.assertEqual(len(aliases), 24)
        self.assertEqual(
            mapped,
            {
                **dict(zip(range(4, 8), range(8, 12))),
                **dict(zip(range(16, 20), range(20, 24))),
            },
        )
        self.assertEqual(aliases[4:8], list(range(8, 12)))
        self.assertEqual(aliases[16:20], list(range(20, 24)))
        self.assertEqual(unresolved, [])

    def test_network_yaml_carries_aliases_and_cluster_metadata(self):
        instances = [
            _instance(0, "prefill"),
            _instance(1, "decode"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "network.yml"

            metadata = _create_network_config(path, instances, 16, 20_000)
            network = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertEqual(network["endpoint_aliases"], [0, 1, 4, 5, 4, 5])
        self.assertEqual(
            metadata,
            {
                "logical_to_physical": {2: 4, 3: 5},
                "unresolved_prefill_instances": [],
                "contention_mode": "decode-endpoint-alias",
            },
        )


if __name__ == "__main__":
    unittest.main()
