import math
import tempfile
import unittest
from pathlib import Path

import yaml

from serving.core.config_builder import (
    _ASTRA_GBPS_TO_BYTES_PER_NS,
    _link_bw_for_astra,
    build_cluster_config,
)


class NetworkBandwidthUnitTest(unittest.TestCase):
    def test_omitted_unit_preserves_legacy_astra_value(self):
        self.assertEqual(_link_bw_for_astra(450), 450.0)
        self.assertEqual(
            _link_bw_for_astra([900, 100]), [900.0, 100.0])

    def test_decimal_unit_compensates_for_astra_binary_conversion(self):
        generated = _link_bw_for_astra(450, "decimal_GBps")

        self.assertTrue(math.isclose(
            generated * _ASTRA_GBPS_TO_BYTES_PER_NS,
            450.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ))

    def test_unknown_unit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported link_bw_unit"):
            _link_bw_for_astra(450, "GBps")

    def test_paper_cluster_keeps_physical_and_generated_values_auditable(self):
        repo_root = Path(__file__).resolve().parents[1]
        astra_sim = repo_root / "astra-sim"
        config_path = (
            repo_root / "configs" / "cluster" /
            "single_node_qwen3_1m_pd_p4d4_h100.json"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cluster = build_cluster_config(
                str(astra_sim), str(config_path), inputs_root=tmpdir)
            network = yaml.safe_load(
                (Path(tmpdir) / "network" / "network.yml").read_text(
                    encoding="utf-8"))

        self.assertEqual(cluster["link_bw"], 450)
        self.assertEqual(cluster["link_bw_unit"], "decimal_GBps")
        self.assertEqual(cluster["link_bw_unit_effective"], "decimal_GBps")
        self.assertTrue(math.isclose(
            cluster["astra_link_bw"] * _ASTRA_GBPS_TO_BYTES_PER_NS,
            450.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ))
        self.assertEqual(len(network["bandwidth"]), 2)
        for generated in network["bandwidth"]:
            self.assertTrue(math.isclose(
                generated * _ASTRA_GBPS_TO_BYTES_PER_NS,
                450.0,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ))


if __name__ == "__main__":
    unittest.main()
