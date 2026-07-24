import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from serving.core.graph_generator import generate_graph
from serving.core.utils import formatter, header


REPO_ROOT = Path(__file__).resolve().parents[1]
ASTRA_ROOT = REPO_ROOT / "astra-sim"
CHAKRA_ROOT = (
    ASTRA_ROOT / "extern" / "graph_frontend" / "chakra"
)


def _write_colocated_trace(inputs_root):
    trace = (
        Path(inputs_root) / "trace" / "H100" / "test" / "model"
        / "instance3_batch7.txt"
    )
    trace.parent.mkdir(parents=True, exist_ok=True)
    trace.write_text(
        "COLOCATED\t\tmodel_parallel_NPU_group: 1\n"
        "2\n"
        + header()
        + formatter(
            "embedding_0",
            10,
            "REMOTE:0",
            4,
            "LOCAL",
            8,
            "LOCAL",
            16,
            "NONE",
            0,
            "NONE",
        )
        + formatter(
            "sampler_1",
            20,
            "LOCAL",
            16,
            "LOCAL",
            0,
            "REMOTE:0",
            4,
            "NONE",
            0,
            "NONE",
        ),
        encoding="utf-8",
    )
    return trace


class GraphGeneratorTests(unittest.TestCase):
    def test_in_process_converter_matches_cli_bytes(self):
        batch = SimpleNamespace(
            model="test/model",
            batch_id=7,
        )
        with TemporaryDirectory() as direct_root, TemporaryDirectory() as cli_root:
            direct_trace = _write_colocated_trace(direct_root)
            cli_trace = _write_colocated_trace(cli_root)

            with (
                patch(
                    "serving.core.graph_generator.os.getcwd",
                    return_value=str(ASTRA_ROOT),
                ),
                patch(
                    "serving.core.graph_generator._llm_converter_type",
                    None,
                ),
            ):
                generate_graph(
                    batch,
                    "H100",
                    4,
                    instance_id=3,
                    npu_offset=8,
                    enable_local_offloading=True,
                    inputs_root=direct_root,
                    cleanup_trace=False,
                )

            cli_output = Path(cli_root) / "cli" / "llm"
            cli_output.parent.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            search_roots = [
                str(CHAKRA_ROOT / "build" / "lib"),
                str(CHAKRA_ROOT),
            ]
            existing = environment.get("PYTHONPATH")
            if existing:
                search_roots.append(existing)
            environment["PYTHONPATH"] = os.pathsep.join(search_roots)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "chakra.src.converter.converter",
                    "--log-filename",
                    str(Path(cli_root) / "converter.log"),
                    "LLM",
                    "--input",
                    str(cli_trace),
                    "--output",
                    str(cli_output),
                    "--num-npus",
                    "4",
                    "--npu-offset",
                    "8",
                    "--local-offloading",
                ],
                cwd=CHAKRA_ROOT,
                env=environment,
                text=True,
                check=True,
            )

            direct_output = (
                Path(direct_root) / "workload" / "H100" / "test" / "model"
                / "instance3_batch7" / "llm"
            )
            for npu_id in range(8, 12):
                with self.subTest(npu_id=npu_id):
                    self.assertEqual(
                        Path(f"{direct_output}.{npu_id}.et").read_bytes(),
                        Path(f"{cli_output}.{npu_id}.et").read_bytes(),
                    )
            self.assertTrue(direct_trace.is_file())

    def test_fresh_converter_per_graph_and_cleanup_after_success(self):
        instances = []

        class FakeConverter:
            def __init__(self, *args):
                self.args = args
                self.convert_calls = 0
                instances.append(self)

            def convert(self):
                self.convert_calls += 1

        with TemporaryDirectory() as inputs_root:
            trace = Path(inputs_root) / "trace" / "event_handler.txt"
            trace.parent.mkdir(parents=True)
            with (
                patch(
                    "serving.core.graph_generator.os.getcwd",
                    return_value=str(ASTRA_ROOT),
                ),
                patch(
                    "serving.core.graph_generator._load_llm_converter",
                    return_value=FakeConverter,
                ),
            ):
                for _ in range(2):
                    trace.write_text("trace", encoding="utf-8")
                    generate_graph(
                        None,
                        None,
                        4,
                        npu_offset=8,
                        enable_local_offloading=True,
                        event=True,
                        inputs_root=inputs_root,
                        cleanup_trace=True,
                    )
                    self.assertFalse(trace.exists())

        self.assertEqual(len(instances), 2)
        self.assertIsNot(instances[0], instances[1])
        for converter in instances:
            self.assertEqual(converter.convert_calls, 1)
            self.assertEqual(converter.args[2:], (4, 8, True))

    def test_failed_conversion_propagates_and_keeps_trace(self):
        failure = RuntimeError("conversion failed")

        class FailingConverter:
            def __init__(self, *args):
                pass

            def convert(self):
                raise failure

        with TemporaryDirectory() as inputs_root:
            trace = Path(inputs_root) / "trace" / "event_handler.txt"
            trace.parent.mkdir(parents=True)
            trace.write_text("trace", encoding="utf-8")
            with (
                patch(
                    "serving.core.graph_generator.os.getcwd",
                    return_value=str(ASTRA_ROOT),
                ),
                patch(
                    "serving.core.graph_generator._load_llm_converter",
                    return_value=FailingConverter,
                ),
                self.assertRaises(RuntimeError) as raised,
            ):
                generate_graph(
                    None,
                    None,
                    4,
                    event=True,
                    inputs_root=inputs_root,
                    cleanup_trace=True,
                )

            self.assertIs(raised.exception, failure)
            self.assertTrue(trace.is_file())


if __name__ == "__main__":
    unittest.main()
