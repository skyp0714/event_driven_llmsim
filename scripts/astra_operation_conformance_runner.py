#!/usr/bin/env python3
"""Run mandatory Chakra/ASTRA tests and emit four cycle probes as JSON."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


EXPECTED_PROTOBUF_VERSION = "7.35.0"
TEST_MODULE = "tests.test_astra_operation_conformance"
OPTIONAL_CLASSES = (
    "OptionalChakraConverterConformanceTests",
    "OptionalAstraBackendConformanceTests",
)
ASTRA_BINARY = Path(
    "astra-sim/build/astra_analytical/build/bin/"
    "AstraSim_Analytical_Congestion_Aware"
)
CHAKRA_ROOT = Path("astra-sim/extern/graph_frontend/chakra")
_CYCLE_RECORD = re.compile(
    r"sys\[(\d+)\] iteration (\d+) finished, (\d+) cycles, "
    r"exposed communication (\d+) cycles\."
)


class AstraConformanceError(RuntimeError):
    """Raised when mandatory conformance or an ASTRA probe fails."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _flatten(item)
        else:
            yield item


def _run_unittests() -> dict[str, int]:
    suite = unittest.defaultTestLoader.loadTestsFromName(TEST_MODULE)
    test_ids = tuple(test.id() for test in _flatten(suite))
    missing = [
        class_name for class_name in OPTIONAL_CLASSES
        if not any(
            test_id.startswith(f"{TEST_MODULE}.{class_name}.")
            for test_id in test_ids
        )
    ]
    if missing:
        raise AstraConformanceError(
            f"mandatory optional test classes were not collected: {missing}"
        )

    result = unittest.TextTestRunner(
        stream=sys.stderr, verbosity=2
    ).run(suite)
    if result.skipped:
        detail = "; ".join(
            f"{test.id()}: {reason}" for test, reason in result.skipped
        )
        raise AstraConformanceError(
            "optional Chakra/ASTRA tests skipped: " + detail
        )
    if not result.wasSuccessful():
        raise AstraConformanceError(
            "Chakra/ASTRA conformance unittest suite failed"
        )
    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skips": len(result.skipped),
    }


def _strict_json(value, *, indent=None) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        indent=indent,
        separators=None if indent else (",", ":"),
        sort_keys=True,
    )


def _network_yaml(dimensions, topology) -> str:
    dimensions = tuple(dimensions)
    topology = tuple(topology)
    if not dimensions or len(dimensions) != len(topology):
        raise AstraConformanceError(
            "network dimensions and topology must align"
        )
    if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in dimensions
    ):
        raise AstraConformanceError(
            "network dimensions must be positive integers"
        )
    if len(dimensions) > 1 and any(
            value != "FullyConnected" for value in topology
    ):
        raise AstraConformanceError(
            "congestion-aware ASTRA requires FullyConnected in every "
            "multi-dimensional topology axis"
        )
    return (
        f"topology: [ {', '.join(topology)} ]\n"
        f"npus_count: [ {', '.join(map(str, dimensions))} ]\n"
        f"bandwidth: [ {', '.join('50.0' for _ in dimensions)} ]\n"
        f"latency: [ {', '.join('1.0' for _ in dimensions)} ]\n"
    )


def _write_configs(root: Path, *, dimensions, topology):
    dimension_count = len(dimensions)
    collective = ["ring"] * dimension_count
    network_text = _network_yaml(dimensions, topology)
    system = {
        "scheduling-policy": "LIFO",
        "endpoint-delay": 0,
        "active-chunks-per-dimension": 1,
        "preferred-dataset-splits": 1,
        "all-reduce-implementation": collective,
        "all-gather-implementation": collective,
        "reduce-scatter-implementation": collective,
        "all-to-all-implementation": collective,
        "collective-optimization": "localBWAware",
        "local-mem-bw": 3_350,
        "boost-mode": 0,
    }
    memory = {
        "remote_mem": {
            "memory-type": "PER_NODE_MEMORY_EXPANSION",
            "mem-bw": 256,
            "mem-latency": 0,
            "num-devices": 1,
        },
        "hbf_mem": {},
    }
    system_text = _strict_json(system)
    memory_text = _strict_json(memory)
    paths = {
        "network": root / "network.yml",
        "system": root / "system.json",
        "memory": root / "memory.json",
    }
    paths["network"].write_text(network_text, encoding="utf-8")
    paths["system"].write_text(system_text, encoding="utf-8")
    paths["memory"].write_text(memory_text, encoding="utf-8")
    audit = {
        "network": {
            "topology": list(topology),
            "dimensions": list(dimensions),
            "bandwidth_config_values": [50.0] * dimension_count,
            "latency_config_values": [1.0] * dimension_count,
            "sha256": _sha256_bytes(network_text.encode()),
        },
        "system": {
            "collective_implementation": collective,
            "sha256": _sha256_bytes(system_text.encode()),
        },
        "memory": {
            "sha256": _sha256_bytes(memory_text.encode()),
        },
    }
    return paths, audit


def _parse_cycle_records(output: str) -> list[dict[str, int]]:
    records = [
        {
            "sys": int(system_id),
            "iteration": int(iteration),
            "total_cycles": int(total),
            "exposed_communication_cycles": int(communication),
        }
        for system_id, iteration, total, communication
        in _CYCLE_RECORD.findall(output)
    ]
    if not records:
        raise AstraConformanceError(
            "ASTRA output contained no iteration cycle record"
        )
    return records


def _run_probe(
        binary, converter, *, case, artifact, parameters,
        dimensions, topology,
):
    with tempfile.TemporaryDirectory(
            prefix=f"astra-operation-probe-{case}-") as temporary:
        root = Path(temporary)
        trace = root / "trace.txt"
        graph = root / "graph"
        trace.write_text(artifact.text, encoding="utf-8")
        converter(
            str(trace), str(graph), num_npus=artifact.num_npus
        ).convert()
        paths, config = _write_configs(
            root, dimensions=dimensions, topology=topology
        )
        process = subprocess.run(
            [
                str(binary),
                f"--workload-configuration={graph}",
                f"--system-configuration={paths['system']}",
                f"--network-configuration={paths['network']}",
                f"--memory-configuration={paths['memory']}",
                "--start-npu-ids=0",
                f"--end-npu-ids={artifact.num_npus - 1}",
            ],
            input="exit\n",
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    if process.returncode:
        raise AstraConformanceError(
            f"ASTRA probe {case} failed ({process.returncode}): "
            f"{process.stderr[-2000:]}"
        )
    if "All Request Has Been Exited" not in process.stdout:
        raise AstraConformanceError(
            f"ASTRA probe {case} did not fully exit"
        )
    records = _parse_cycle_records(process.stdout)
    if not any(
            record["sys"] == artifact.num_npus - 1
            and record["iteration"] == 0
            for record in records):
        raise AstraConformanceError(
            f"ASTRA probe {case} did not report its final endpoint"
        )
    return {
        "case": case,
        "artifact_parameters": parameters,
        "num_npus": artifact.num_npus,
        "trace_sha256": _sha256_bytes(artifact.text.encode()),
        "config": config,
        "cycle_records": records,
        "cycle_equality_claimed": False,
    }


def _load_runtime(repo_root: Path):
    import google.protobuf

    observed = google.protobuf.__version__
    if observed != EXPECTED_PROTOBUF_VERSION:
        raise AstraConformanceError(
            "protobuf runtime must match Chakra gencode exactly: "
            f"expected={EXPECTED_PROTOBUF_VERSION}, observed={observed}"
        )
    chakra = repo_root / CHAKRA_ROOT
    for path in (chakra / "build" / "lib", chakra):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from chakra.src.converter.llm_converter import LLMConverter
    from serving.core.astra_operation_conformance import (
        build_bulk_kv_transfer_microtrace,
        build_collective_microtrace,
        build_hbf_media_microtrace,
    )
    return (
        observed,
        LLMConverter,
        build_bulk_kv_transfer_microtrace,
        build_collective_microtrace,
        build_hbf_media_microtrace,
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    (
        protobuf_version,
        converter,
        build_bulk,
        build_collective,
        build_hbf,
    ) = _load_runtime(repo_root)
    binary = repo_root / ASTRA_BINARY
    if not binary.is_file():
        raise AstraConformanceError(
            f"ASTRA binary is missing: {binary}; run scripts/compile.sh"
        )
    unittest_summary = _run_unittests()
    probes = [
        _run_probe(
            binary, converter,
            case="p2d_tp4_1024",
            artifact=build_bulk(
                direction="p_to_d", tp_size=4, token_count=1_024
            ),
            parameters={
                "direction": "p_to_d",
                "tp_size": 4,
                "token_count": 1_024,
            },
            dimensions=(8,),
            topology=("Ring",),
        ),
        _run_probe(
            binary, converter,
            case="hbf_read_tp8",
            artifact=build_hbf(
                operation="read",
                tp_size=8,
                runtime_ns=20_001,
                tensor_bytes_per_rank=65_536,
            ),
            parameters={
                "operation": "read",
                "tp_size": 8,
                "runtime_ns": 20_001,
                "tensor_bytes_per_rank": 65_536,
            },
            dimensions=(8,),
            topology=("Ring",),
        ),
        _run_probe(
            binary, converter,
            case="collective_tp4x2",
            artifact=build_collective(
                tp_size=4, total_tokens=1_024, replicas=2
            ),
            parameters={
                "tp_size": 4,
                "total_tokens": 1_024,
                "replicas": 2,
            },
            dimensions=(4, 2),
            topology=("FullyConnected", "FullyConnected"),
        ),
        _run_probe(
            binary, converter,
            case="collective_tp8",
            artifact=build_collective(
                tp_size=8, total_tokens=1_024, replicas=1
            ),
            parameters={
                "tp_size": 8,
                "total_tokens": 1_024,
                "replicas": 1,
            },
            dimensions=(8,),
            topology=("Ring",),
        ),
    ]
    report = {
        "schema_version": 1,
        "claim_scope": (
            "Chakra/ASTRA operation conformance only; analytical-to-ASTRA "
            "cycle equality is not claimed"
        ),
        "cycle_equality_claimed": False,
        "protobuf": {
            "runtime_version": protobuf_version,
            "required_exact_version": EXPECTED_PROTOBUF_VERSION,
        },
        "astra_binary": {
            "path": ASTRA_BINARY.as_posix(),
            "sha256": _sha256_file(binary),
            "size_bytes": binary.stat().st_size,
        },
        "unittest": unittest_summary,
        "probes": probes,
    }
    print(_strict_json(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        AstraConformanceError,
        OSError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"ASTRA conformance failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
