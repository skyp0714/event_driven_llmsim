import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from serving.core.astra_operation_conformance import (
    HBFMediaOperation,
    KVTransferDirection,
    TRACE_HEADER,
    build_bulk_kv_transfer_microtrace,
    build_collective_microtrace,
    build_hbf_media_microtrace,
    parse_microtrace,
    qwen_kv_bytes_per_rank,
)
from serving.core.gpu_pd_latency import (
    P4D4GPUHardware,
    P4D4LatencyModel,
)
from serving.core.h100_kernel_calibrated_prompt import (
    BF16_BYTES,
    QWEN_EXPERTS,
    QWEN_HIDDEN_SIZE,
    QWEN_LAYERS,
)
from serving.core.hbf_full_model_latency import (
    HBFModelBatchShape,
    HBFParallelLayout,
    HBFServerHardware,
    build_full_model_hbf_latency,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CHAKRA_ROOT = (
    REPO_ROOT / "astra-sim" / "extern" / "graph_frontend" / "chakra"
)
ASTRA_BINARY = (
    REPO_ROOT / "astra-sim" / "build" / "astra_analytical" / "build"
    / "bin" / "AstraSim_Analytical_Congestion_Aware"
)


def _load_chakra_bindings():
    """Load optional generated protobuf bindings without affecting pure tests."""

    search_roots = (
        CHAKRA_ROOT / "build" / "lib",
        CHAKRA_ROOT,
    )
    for candidate in reversed(search_roots):
        root = str(candidate)
        if candidate.is_dir() and root not in sys.path:
            sys.path.insert(0, root)
    try:
        from chakra.schema.protobuf.et_def_pb2 import GlobalMetadata, Node
        from chakra.src.converter.llm_converter import LLMConverter
        from chakra.src.third_party.utils.protolib import decodeMessage
    except Exception as exc:
        return None, (
            f"Chakra converter/protobuf unavailable: "
            f"{type(exc).__name__}: {exc}"
        )
    return {
        "GlobalMetadata": GlobalMetadata,
        "Node": Node,
        "LLMConverter": LLMConverter,
        "decodeMessage": decodeMessage,
    }, None


def _node_attributes(node):
    return {attribute.name: attribute for attribute in node.attr}


class PureAstraOperationConformanceTests(unittest.TestCase):
    def test_tp4_and_tp8_collective_bytes_and_scope_are_exact(self):
        total_tokens = 1_024
        cases = (
            (4, 2, "1,0", (True, False)),
            (8, 1, "1", (True,)),
        )
        for tp_size, replicas, suffix, involved_dim in cases:
            with self.subTest(tp_size=tp_size, replicas=replicas):
                artifact = build_collective_microtrace(
                    tp_size=tp_size,
                    total_tokens=total_tokens,
                    replicas=replicas,
                )
                parsed = parse_microtrace(artifact.text)
                hidden_total = (
                    total_tokens * QWEN_HIDDEN_SIZE * BF16_BYTES)
                dispatch_local = (
                    (total_tokens // tp_size)
                    * (QWEN_HIDDEN_SIZE + QWEN_EXPERTS)
                    * BF16_BYTES
                )

                self.assertEqual(parsed.execution_type, "COLOCATED")
                self.assertEqual(parsed.model_parallel_groups, 1)
                self.assertEqual(artifact.num_npus, 8)
                self.assertEqual(artifact.replicas, replicas)
                self.assertEqual(
                    [row.comm_type for row in parsed.rows],
                    [
                        f"ALLREDUCE:{suffix}",
                        f"ALLGATHER:{suffix}",
                        f"REDUCESCATTER:{suffix}",
                    ],
                )
                self.assertEqual(
                    [row.comm_size for row in parsed.rows],
                    [hidden_total, dispatch_local, hidden_total],
                )
                self.assertEqual(
                    [item.payload_semantics
                     for item in artifact.operations],
                    [
                        "total_activation_buffer",
                        "per_rank_local_chunk",
                        "pre_scatter_total_buffer",
                    ],
                )
                self.assertTrue(all(
                    item.involved_dim == involved_dim
                    for item in artifact.operations
                ))
                self.assertFalse(artifact.cycle_equality_claimed)

    def test_collective_wire_bytes_match_both_analytical_models(self):
        shape = HBFModelBatchShape(
            total_tokens=128,
            prefill_q=(128,),
            prefill_hbf_k=(0,),
            prefill_lpddr_k=(0,),
            lm_head_sequences=1,
        )
        for tp_size, layout_key in ((4, "tp4"), (8, "tp8")):
            with self.subTest(tp_size=tp_size):
                artifact = build_collective_microtrace(
                    tp_size=tp_size, total_tokens=shape.total_tokens)
                expected_per_layer = sum(
                    item.ring_wire_bytes_per_rank
                    for item in artifact.operations
                )
                model = build_full_model_hbf_latency(
                    repo_root=REPO_ROOT,
                    hardware=HBFServerHardware(),
                    layout=HBFParallelLayout.for_key(layout_key),
                )
                latency = model.batch_latency(shape)
                self.assertEqual(
                    latency.collective_bytes_per_rank,
                    QWEN_LAYERS * expected_per_layer,
                )

        gpu_model = P4D4LatencyModel(
            repo_root=REPO_ROOT,
            hardware=P4D4GPUHardware(),
        )
        gpu_latency = gpu_model.batch_latency(shape)
        tp4 = build_collective_microtrace(
            tp_size=4, total_tokens=shape.total_tokens)
        self.assertEqual(
            gpu_latency.collective_bytes_per_rank,
            QWEN_LAYERS * sum(
                item.ring_wire_bytes_per_rank
                for item in tp4.operations
            ),
        )

    def test_small_batch_allgather_uses_one_local_token_chunk(self):
        artifact = build_collective_microtrace(
            tp_size=8, total_tokens=3)
        allgather = artifact.operations[1]
        self.assertEqual(
            allgather.comm_size,
            (QWEN_HIDDEN_SIZE + QWEN_EXPERTS) * BF16_BYTES,
        )

    def test_p_to_d_is_one_bulk_copy_strictly_after_sampler(self):
        artifact = build_bulk_kv_transfer_microtrace(
            direction=KVTransferDirection.P_TO_D,
            tp_size=4,
            token_count=65_537,
        )
        parsed = parse_microtrace(artifact.text)
        contract = artifact.contract

        self.assertEqual(parsed.model_parallel_groups, 2)
        self.assertEqual(artifact.num_npus, 8)
        self.assertEqual(
            [row.name for row in parsed.rows],
            [
                "prefill_model_compute_complete",
                "sampler_ttft_boundary",
                "decode_kv_publish_commit",
            ],
        )
        self.assertEqual(contract.issue_phase, "strictly_after_ttft")
        self.assertEqual(contract.bulk_copy_count, 1)
        self.assertFalse(contract.qkv_streaming)
        self.assertFalse(contract.uses_prefill_converter)
        self.assertNotIn("PREFILL", artifact.text)
        self.assertFalse(any(
            "qkv_proj" in row.name for row in parsed.rows))
        self.assertEqual(
            parsed.rows[1].output_size, contract.bytes_per_rank)
        self.assertEqual(
            parsed.rows[2].input_size, contract.bytes_per_rank)
        self.assertEqual(
            contract.aggregate_bytes,
            contract.bytes_per_rank * contract.tp_size,
        )
        handoff = P4D4LatencyModel(
            repo_root=REPO_ROOT,
            hardware=P4D4GPUHardware(),
        ).handoff_latency(contract.token_count)
        self.assertEqual(
            handoff.bytes_per_rank, contract.bytes_per_rank)
        self.assertEqual(
            handoff.aggregate_bytes, contract.aggregate_bytes)
        self.assertFalse(artifact.cycle_equality_claimed)

    def test_d_to_p_bulk_copy_gates_resume_compute(self):
        artifact = build_bulk_kv_transfer_microtrace(
            direction="d_to_p",
            tp_size=4,
            token_count=1_000_000,
        )
        parsed = parse_microtrace(artifact.text)
        contract = artifact.contract

        self.assertEqual(
            [row.name for row in parsed.rows],
            [
                "decode_kv_resident_ready",
                "decode_to_prefill_bulk_source",
                "resume_prefill_compute_gate",
            ],
        )
        self.assertEqual(
            contract.issue_phase, "before_resume_prefill_compute")
        source_aliases = contract.logical_rank_aliases[:4]
        destination_aliases = contract.logical_rank_aliases[4:]
        self.assertEqual(
            [(item.logical_rank, item.role, item.role_rank)
             for item in source_aliases],
            [(rank, "decode", rank) for rank in range(4)],
        )
        self.assertEqual(
            [(item.logical_rank, item.role, item.role_rank)
             for item in destination_aliases],
            [(4 + rank, "prefill", rank) for rank in range(4)],
        )
        self.assertTrue(contract.requires_source_first_endpoint_alias)

    def test_tp8_kv_contract_includes_physical_head_replication(self):
        tp4_per_token = qwen_kv_bytes_per_rank(
            tp_size=4, token_count=1)
        tp8_per_token = qwen_kv_bytes_per_rank(
            tp_size=8, token_count=1)
        hardware = P4D4GPUHardware()

        self.assertEqual(
            tp4_per_token, hardware.kv_bytes_per_token_per_rank)
        self.assertEqual(tp8_per_token, tp4_per_token)
        self.assertEqual(
            tp8_per_token * 8,
            2 * tp4_per_token * 4,
        )

    def test_hbf_read_and_write_use_explicit_whole_gang_resources(self):
        for tp_size in (4, 8):
            for operation in (
                    HBFMediaOperation.READ, HBFMediaOperation.WRITE):
                with self.subTest(
                        tp_size=tp_size, operation=operation.value):
                    artifact = build_hbf_media_microtrace(
                        operation=operation,
                        tp_size=tp_size,
                        runtime_ns=20_001,
                        tensor_bytes_per_rank=123_456,
                    )
                    parsed = parse_microtrace(artifact.text)
                    descriptor = json.loads(
                        parsed.rows[1].misc)["hbf"]
                    stage = descriptor["stages"][0]

                    self.assertEqual(
                        descriptor["expected_participants"], tp_size)
                    self.assertEqual(
                        stage["tensor_bytes"], 123_456 * tp_size)
                    self.assertEqual(
                        stage["runtime_ns"], 20_001)
                    self.assertEqual(
                        stage["resources"],
                        [
                            f"hbf-card:{rank}:{operation.value}"
                            for rank in range(tp_size)
                        ],
                    )
                    self.assertEqual(
                        tuple(stage["resources"]),
                        artifact.contract.resources,
                    )
                    self.assertTrue(
                        artifact.contract.whole_gang_resource_semantics)
                    self.assertFalse(artifact.cycle_equality_claimed)

    def test_invalid_scope_direction_and_zero_bytes_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "tp_size"):
            build_collective_microtrace(
                tp_size=2, total_tokens=128)
        with self.assertRaisesRegex(ValueError, "only TP4"):
            build_collective_microtrace(
                tp_size=8, total_tokens=128, replicas=2)
        with self.assertRaisesRegex(ValueError, "direction"):
            build_bulk_kv_transfer_microtrace(
                direction="sideways",
                tp_size=4,
                token_count=1,
            )
        with self.assertRaisesRegex(ValueError, "token_count"):
            build_bulk_kv_transfer_microtrace(
                direction="p_to_d",
                tp_size=4,
                token_count=0,
            )
        with self.assertRaisesRegex(ValueError, "tensor_bytes_per_rank"):
            build_hbf_media_microtrace(
                operation="read",
                tp_size=4,
                runtime_ns=1,
                tensor_bytes_per_rank=0,
            )

    def test_parser_rejects_prefill_and_declared_count_mismatch(self):
        trace = (
            "PREFILL\t\tmodel_parallel_NPU_group: 1\n"
            "1\n"
            f"{TRACE_HEADER}\n"
            "probe 1 REMOTE:0 1 LOCAL 0 REMOTE:0 1 NONE 0 NONE\n"
        )
        with self.assertRaisesRegex(ValueError, "COLOCATED"):
            parse_microtrace(trace)

        valid = build_collective_microtrace(
            tp_size=4, total_tokens=1).text
        corrupted = valid.replace("\n3\n", "\n4\n", 1)
        with self.assertRaisesRegex(ValueError, "row count mismatch"):
            parse_microtrace(corrupted)


class OptionalChakraConverterConformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bindings, reason = _load_chakra_bindings()
        if cls.bindings is None:
            raise unittest.SkipTest(
                f"{reason}; pure text conformance tests still ran")

    @classmethod
    def _convert(cls, root, artifact):
        trace = root / "trace.txt"
        output = root / "graph"
        trace.write_text(artifact.text, encoding="utf-8")
        cls.bindings["LLMConverter"](
            str(trace),
            str(output),
            num_npus=artifact.num_npus,
        ).convert()
        return output

    @classmethod
    def _read_nodes(cls, path):
        result = []
        with path.open("rb") as source:
            metadata = cls.bindings["GlobalMetadata"]()
            if not cls.bindings["decodeMessage"](source, metadata):
                raise AssertionError("missing Chakra global metadata")
            while True:
                node = cls.bindings["Node"]()
                if not cls.bindings["decodeMessage"](source, node):
                    break
                result.append(node)
        return result

    def test_tp4_and_tp8_collectives_preserve_bytes_and_scope(self):
        cases = (
            (4, 2, [True, False]),
            (8, 1, [True]),
        )
        for tp_size, replicas, involved_dim in cases:
            with self.subTest(tp_size=tp_size, replicas=replicas):
                artifact = build_collective_microtrace(
                    tp_size=tp_size,
                    total_tokens=1_024,
                    replicas=replicas,
                )
                with tempfile.TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    output = self._convert(root, artifact)
                    for rank in range(artifact.num_npus):
                        nodes = self._read_nodes(
                            Path(f"{output}.{rank}.et"))
                        collectives = [
                            node for node in nodes
                            if node.name.startswith("COMM_COLL_NODE_")
                        ]
                        self.assertEqual(len(collectives), 3)
                        for node, contract in zip(
                                collectives, artifact.operations):
                            attrs = _node_attributes(node)
                            self.assertEqual(
                                attrs["comm_size"].int64_val,
                                contract.comm_size,
                            )
                            self.assertEqual(
                                list(attrs["involved_dim"].bool_list.values),
                                involved_dim,
                            )

    def test_bulk_transfers_become_one_send_recv_after_source_boundary(self):
        for direction in (
                KVTransferDirection.D_TO_P,
                KVTransferDirection.P_TO_D):
            with self.subTest(direction=direction.value):
                artifact = build_bulk_kv_transfer_microtrace(
                    direction=direction,
                    tp_size=4,
                    token_count=4_097,
                )
                with tempfile.TemporaryDirectory() as tmpdir:
                    root = Path(tmpdir)
                    output = self._convert(root, artifact)
                    source_nodes = self._read_nodes(
                        Path(f"{output}.0.et"))
                    destination_nodes = self._read_nodes(
                        Path(f"{output}.4.et"))

                source_by_name = {
                    node.name: node for node in source_nodes}
                destination_by_name = {
                    node.name: node for node in destination_nodes}
                send_nodes = [
                    node for node in source_nodes
                    if node.name.startswith("COMM_SEND_NODE_")
                ]
                recv_nodes = [
                    node for node in destination_nodes
                    if node.name.startswith("COMM_RECV_NODE_")
                ]
                self.assertEqual(len(send_nodes), 1)
                self.assertEqual(len(recv_nodes), 1)
                source = source_by_name[
                    f"COMP_NODE_{artifact.contract.source_boundary}"]
                destination = destination_by_name[
                    f"COMP_NODE_{artifact.contract.destination_gate}"]
                send = send_nodes[0]
                recv = recv_nodes[0]
                self.assertEqual(list(send.data_deps), [source.id])
                self.assertEqual(list(destination.data_deps), [recv.id])
                self.assertEqual(
                    _node_attributes(send)["comm_size"].int64_val,
                    artifact.contract.bytes_per_rank,
                )
                self.assertEqual(
                    _node_attributes(recv)["comm_size"].int64_val,
                    artifact.contract.bytes_per_rank,
                )
                self.assertFalse(any(
                    "kv_proj" in node.name
                    for node in source_nodes + destination_nodes
                ))

    def test_hbf_whole_gang_metadata_is_identical_on_every_rank(self):
        artifact = build_hbf_media_microtrace(
            operation="write",
            tp_size=8,
            runtime_ns=20_001,
            tensor_bytes_per_rank=65_536,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = self._convert(root, artifact)
            observed = []
            for rank in range(8):
                nodes = self._read_nodes(
                    Path(f"{output}.{rank}.et"))
                stages = [
                    node for node in nodes
                    if node.name.startswith("HBF_STAGE_")
                ]
                self.assertEqual(len(stages), 1)
                attrs = _node_attributes(stages[0])
                observed.append((
                    attrs["tensor_size"].uint64_val,
                    attrs["hbf_resources"].string_val,
                    attrs["hbf_gang_id"].string_val,
                    attrs["hbf_expected_participants"].uint32_val,
                ))

        self.assertTrue(all(item == observed[0] for item in observed))
        self.assertEqual(
            observed[0][0], artifact.contract.aggregate_tensor_bytes)
        self.assertEqual(
            observed[0][1], ";".join(artifact.contract.resources))
        self.assertEqual(observed[0][3], 8)


class OptionalAstraBackendConformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bindings, reason = _load_chakra_bindings()
        if cls.bindings is None:
            raise unittest.SkipTest(
                f"{reason}; backend conformance requires converted ET files")
        if not ASTRA_BINARY.is_file():
            raise unittest.SkipTest(
                "analytical congestion-aware ASTRA-Sim binary is not built")

    @staticmethod
    def _write_configs(root, num_npus):
        (root / "network.yml").write_text(
            "topology: [ Ring ]\n"
            f"npus_count: [ {num_npus} ]\n"
            "bandwidth: [ 50.0 ]\n"
            "latency: [ 1.0 ]\n",
            encoding="utf-8",
        )
        (root / "system.json").write_text(json.dumps({
            "scheduling-policy": "LIFO",
            "endpoint-delay": 0,
            "active-chunks-per-dimension": 1,
            "preferred-dataset-splits": 1,
            "all-reduce-implementation": ["ring"],
            "all-gather-implementation": ["ring"],
            "reduce-scatter-implementation": ["ring"],
            "all-to-all-implementation": ["ring"],
            "collective-optimization": "localBWAware",
            "local-mem-bw": 3_350,
            "boost-mode": 0,
        }), encoding="utf-8")
        (root / "memory.json").write_text(json.dumps({
            "remote_mem": {
                "memory-type": "PER_NODE_MEMORY_EXPANSION",
                "mem-bw": 256,
                "mem-latency": 0,
                "num-devices": 1,
            },
            "hbf_mem": {},
        }), encoding="utf-8")

    def test_bulk_p_to_d_graph_is_consumed_without_cycle_equality_claim(self):
        artifact = build_bulk_kv_transfer_microtrace(
            direction="p_to_d",
            tp_size=4,
            token_count=1_024,
        )
        self.assertFalse(artifact.cycle_equality_claimed)
        with tempfile.TemporaryDirectory(
                prefix="astra-operation-conformance-") as tmpdir:
            root = Path(tmpdir)
            trace = root / "trace.txt"
            graph = root / "graph"
            trace.write_text(artifact.text, encoding="utf-8")
            self.bindings["LLMConverter"](
                str(trace),
                str(graph),
                num_npus=artifact.num_npus,
            ).convert()
            self._write_configs(root, artifact.num_npus)
            process = subprocess.run(
                [
                    str(ASTRA_BINARY),
                    f"--workload-configuration={graph}",
                    f"--system-configuration={root / 'system.json'}",
                    f"--network-configuration={root / 'network.yml'}",
                    f"--memory-configuration={root / 'memory.json'}",
                    "--start-npu-ids=0",
                    f"--end-npu-ids={artifact.num_npus - 1}",
                ],
                input="exit\n",
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )

        if process.returncode != 0:
            self.fail(
                f"ASTRA-Sim rejected the conformance graph "
                f"(returncode={process.returncode}):\n"
                f"stdout:\n{process.stdout}\n"
                f"stderr:\n{process.stderr}")
        self.assertRegex(
            process.stdout,
            re.compile(
                rf"sys\[{artifact.num_npus - 1}\] iteration 0 finished"
            ),
        )

    def test_hbf_gang_graph_is_consumed_by_astra_backend(self):
        artifact = build_hbf_media_microtrace(
            operation="read",
            tp_size=8,
            runtime_ns=20_001,
            tensor_bytes_per_rank=65_536,
        )
        self.assertFalse(artifact.cycle_equality_claimed)
        with tempfile.TemporaryDirectory(
                prefix="astra-hbf-conformance-") as tmpdir:
            root = Path(tmpdir)
            trace = root / "trace.txt"
            graph = root / "graph"
            trace.write_text(artifact.text, encoding="utf-8")
            self.bindings["LLMConverter"](
                str(trace),
                str(graph),
                num_npus=artifact.num_npus,
            ).convert()
            self._write_configs(root, artifact.num_npus)
            process = subprocess.run(
                [
                    str(ASTRA_BINARY),
                    f"--workload-configuration={graph}",
                    f"--system-configuration={root / 'system.json'}",
                    f"--network-configuration={root / 'network.yml'}",
                    f"--memory-configuration={root / 'memory.json'}",
                    "--start-npu-ids=0",
                    f"--end-npu-ids={artifact.num_npus - 1}",
                ],
                input="exit\n",
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )

        if process.returncode != 0:
            self.fail(
                f"ASTRA-Sim rejected the HBF conformance graph "
                f"(returncode={process.returncode}):\n"
                f"stdout:\n{process.stdout}\n"
                f"stderr:\n{process.stderr}")
        self.assertRegex(
            process.stdout,
            re.compile(
                rf"sys\[{artifact.num_npus - 1}\] iteration 0 finished"
            ),
        )


if __name__ == "__main__":
    unittest.main()
