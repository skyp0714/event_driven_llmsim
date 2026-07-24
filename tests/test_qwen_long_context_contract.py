import copy
import unittest
from unittest import mock

from serving.core.h100_kernel_calibrated_prompt import (
    QWEN_LONG_CONTEXT_MODE,
    QWEN_NATIVE_MAX_POSITION_EMBEDDINGS,
    QWEN_OFFICIAL_CONFIG_1M_REVISION,
    QWEN_OFFICIAL_CONFIG_1M_SOURCE,
    QWEN_VALIDATED_RUNTIME_MAX_MODEL_LEN,
    qwen_long_context_experiment_contract,
)
from serving.core.online_latency_model import (
    OnlineLatencyModelError,
    TARGET_MODEL,
    validate_runtime_context_contract,
)
from serving.core.utils import get_config
from serving.core.scheduler import Scheduler


class QwenLongContextContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = get_config(TARGET_MODEL)

    def test_checked_in_model_config_matches_pinned_contract(self):
        contract = self.config["long_context_experiment"]
        self.assertEqual(contract, qwen_long_context_experiment_contract())
        self.assertEqual(contract["mode"], QWEN_LONG_CONTEXT_MODE)
        self.assertEqual(
            contract["native_max_position_embeddings"],
            QWEN_NATIVE_MAX_POSITION_EMBEDDINGS,
        )
        self.assertEqual(
            contract["validated_runtime_max_model_len"],
            QWEN_VALIDATED_RUNTIME_MAX_MODEL_LEN,
        )
        official = contract["official_config_1m"]
        self.assertEqual(
            official["revision"], QWEN_OFFICIAL_CONFIG_1M_REVISION
        )
        self.assertEqual(official["source"], QWEN_OFFICIAL_CONFIG_1M_SOURCE)
        self.assertTrue(
            official["dual_chunk_attention_config"][
                "sparse_attention_enabled"
            ]
        )
        sensitivity = contract["experiment_override"]
        self.assertFalse(sensitivity["sparse_attention_enabled"])
        self.assertEqual(
            sensitivity["attention_execution_model"],
            "standard_full_attention_roofline_extrapolation",
        )
        self.assertFalse(
            sensitivity["dca_or_minference_kernel_latency_calibrated"]
        )
        interpretation = contract["paper_interpretation"]
        self.assertIn("relative cache-policy", interpretation[
            "primary_claim_scope"
        ])
        self.assertEqual(
            interpretation["absolute_1m_latency"], "sensitivity_only"
        )

    def test_native_window_does_not_require_an_extension_contract(self):
        native_only = {
            "max_position_embeddings": (
                QWEN_NATIVE_MAX_POSITION_EMBEDDINGS
            )
        }
        self.assertIsNone(validate_runtime_context_contract(
            config=native_only,
            max_model_len=QWEN_NATIVE_MAX_POSITION_EMBEDDINGS,
            model="fixture/native-only",
        ))

    def test_above_native_requires_explicit_mode(self):
        with self.assertRaisesRegex(
                OnlineLatencyModelError, "explicit long_context_experiment"):
            validate_runtime_context_contract(
                config={
                    "max_position_embeddings": (
                        QWEN_NATIVE_MAX_POSITION_EMBEDDINGS
                    )
                },
                max_model_len=QWEN_NATIVE_MAX_POSITION_EMBEDDINGS + 1,
                model=TARGET_MODEL,
            )

    def test_declared_qwen_ceiling_is_accepted_and_enforced(self):
        contract = validate_runtime_context_contract(
            config=self.config,
            max_model_len=QWEN_VALIDATED_RUNTIME_MAX_MODEL_LEN,
            model=TARGET_MODEL,
        )
        self.assertEqual(contract["mode"], QWEN_LONG_CONTEXT_MODE)
        with self.assertRaisesRegex(
                OnlineLatencyModelError, "exceeds.*runtime ceiling"):
            validate_runtime_context_contract(
                config=self.config,
                max_model_len=QWEN_VALIDATED_RUNTIME_MAX_MODEL_LEN + 1,
                model=TARGET_MODEL,
            )

    def test_qwen_contract_drift_is_rejected(self):
        drifted = copy.deepcopy(self.config)
        drifted["long_context_experiment"]["experiment_override"][
            "sparse_attention_enabled"
        ] = True
        with self.assertRaisesRegex(
                OnlineLatencyModelError, "does not match.*provider contract"):
            validate_runtime_context_contract(
                config=drifted,
                max_model_len=QWEN_NATIVE_MAX_POSITION_EMBEDDINGS + 1,
                model=TARGET_MODEL,
            )

    def test_declared_native_window_must_match_model_config(self):
        drifted = copy.deepcopy(self.config)
        drifted["long_context_experiment"][
            "native_max_position_embeddings"
        ] -= 1
        with self.assertRaisesRegex(
                OnlineLatencyModelError, "native_max_position_embeddings"):
            validate_runtime_context_contract(
                config=drifted,
                max_model_len=QWEN_NATIVE_MAX_POSITION_EMBEDDINGS + 1,
                model="fixture/non-pinned-model",
            )

    def test_scheduler_fails_before_allocating_memory_above_ceiling(self):
        with mock.patch(
                "serving.core.scheduler.MemoryModel") as memory_model:
            with self.assertRaisesRegex(
                    OnlineLatencyModelError, "exceeds.*runtime ceiling"):
                Scheduler(
                    model=TARGET_MODEL,
                    node_id=0,
                    instance_id=0,
                    max_num_seqs=1,
                    max_num_batched_tokens=1,
                    num_npus=4,
                    tp_size=4,
                    pp_size=1,
                    npu_mem=80,
                    cpu_mem=512,
                    start_npu=0,
                    pd_type="decode",
                    fp=2,
                    block_size=16,
                    req_num=0,
                    prioritize_prefill=False,
                    enable_prefix_caching=False,
                    enable_prefix_sharing=False,
                    prefix_pool=None,
                    prefix_storage="None",
                    max_model_len=(
                        QWEN_VALIDATED_RUNTIME_MAX_MODEL_LEN + 1
                    ),
                )
        memory_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
