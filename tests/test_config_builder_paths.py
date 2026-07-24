import os
from pathlib import Path
import unittest

from serving.core.config_builder import _resolve_cluster_config_path


class ConfigBuilderPathTests(unittest.TestCase):
    def test_repo_relative_path_moves_out_of_astra_working_directory(self):
        self.assertEqual(
            _resolve_cluster_config_path(
                "configs/cluster/single_node_single_instance.json"),
            os.path.join(
                "..", "configs/cluster/single_node_single_instance.json"),
        )

    def test_absolute_path_is_preserved(self):
        path = Path("/tmp/cold-kv-cluster.json")
        self.assertEqual(_resolve_cluster_config_path(path), str(path))


if __name__ == "__main__":
    unittest.main()
