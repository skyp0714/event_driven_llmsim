import json
from pathlib import Path
import tempfile
import unittest

from scripts.astra_operation_conformance_runner import (
    AstraConformanceError,
    _network_yaml,
    _parse_cycle_records,
    _strict_json,
    _write_configs,
)


class AstraOperationConformanceRunnerTests(unittest.TestCase):
    def test_cycle_parser_is_strict_and_typed(self):
        output = (
            "[workload] sys[7] iteration 0 finished, 243681 cycles, "
            "exposed communication 243678 cycles.\n"
        )
        self.assertEqual(
            _parse_cycle_records(output),
            [{
                "sys": 7,
                "iteration": 0,
                "total_cycles": 243_681,
                "exposed_communication_cycles": 243_678,
            }],
        )
        with self.assertRaisesRegex(
                AstraConformanceError, "no iteration cycle record"):
            _parse_cycle_records("All Request Has Been Exited\n")

    def test_multidimensional_probe_requires_fully_connected_axes(self):
        with self.assertRaisesRegex(
                AstraConformanceError, "requires FullyConnected"):
            _network_yaml(
                dimensions=(4, 2),
                topology=("Ring", "Ring"),
            )

    def test_config_bundle_is_deterministic_and_auditable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, first = _write_configs(
                root,
                dimensions=(4, 2),
                topology=("FullyConnected", "FullyConnected"),
            )
            _, second = _write_configs(
                root,
                dimensions=(4, 2),
                topology=("FullyConnected", "FullyConnected"),
            )
            self.assertEqual(first, second)
            self.assertEqual(
                first["network"]["dimensions"], [4, 2])
            self.assertEqual(
                first["network"]["topology"],
                ["FullyConnected", "FullyConnected"],
            )
            self.assertEqual(
                first["system"]["collective_implementation"],
                ["ring", "ring"],
            )
            for section in ("network", "system", "memory"):
                self.assertRegex(
                    first[section]["sha256"], r"^[0-9a-f]{64}$")

    def test_report_serializer_rejects_non_finite_values(self):
        self.assertEqual(
            json.loads(_strict_json({"value": 1.25})),
            {"value": 1.25},
        )
        with self.assertRaises(ValueError):
            _strict_json({"value": float("nan")})


if __name__ == "__main__":
    unittest.main()
