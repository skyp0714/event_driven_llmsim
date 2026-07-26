import dataclasses
from pathlib import Path
import random
import unittest
from unittest import mock

from serving.core.hbf_full_model_latency import (
    HBFModelBatchShape,
    build_full_model_hbf_latency,
    load_hbf_server_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    REPO_ROOT / "configs" / "wakekv_hbf"
    / "full_model_8card_server.json"
)


def _representative_shapes() -> tuple[HBFModelBatchShape, ...]:
    return (
        HBFModelBatchShape(
            total_tokens=1,
            decode_hbf_k=(1,),
            decode_lpddr_k=(0,),
        ),
        HBFModelBatchShape(
            total_tokens=8,
            decode_hbf_k=(500_000,) * 8,
            decode_lpddr_k=(64,) * 8,
            lm_head_sequences=8,
        ),
        HBFModelBatchShape(
            total_tokens=128,
            decode_hbf_k=tuple(
                7_891 * index + 1 for index in range(128)),
            decode_lpddr_k=tuple(
                257 * (index % 17) for index in range(128)),
            lm_head_sequences=128,
        ),
        HBFModelBatchShape(
            total_tokens=4_096,
            prefill_q=(4_096,),
            prefill_hbf_k=(0,),
            prefill_lpddr_k=(0,),
        ),
        HBFModelBatchShape(
            total_tokens=513,
            prefill_q=(513,),
            prefill_hbf_k=(877_777,),
            prefill_lpddr_k=(4_095,),
        ),
        HBFModelBatchShape(
            total_tokens=2_073,
            prefill_q=(1_024, 1_023),
            prefill_hbf_k=(311_111, 400_000),
            prefill_lpddr_k=(511, 1_023),
            decode_hbf_k=(700_001, 800_003),
            decode_lpddr_k=(17, 19),
            lm_head_sequences=4,
        ),
        HBFModelBatchShape(
            total_tokens=2_048,
            prefill_q=(1,),
            prefill_hbf_k=(999_999,),
            prefill_lpddr_k=(9_999,),
            decode_hbf_k=(1_000_000,) * 7,
            decode_lpddr_k=(10_000,) * 7,
            lm_head_sequences=8,
        ),
        HBFModelBatchShape(
            total_tokens=8_192,
            prefill_q=(127, 128, 129),
            prefill_hbf_k=(1, 2, 3),
            prefill_lpddr_k=(4, 5, 6),
            decode_hbf_k=(17, 18, 19),
            decode_lpddr_k=(20, 21, 22),
            lm_head_sequences=6,
        ),
    )


def _random_shapes(seed: int, count: int) -> tuple[HBFModelBatchShape, ...]:
    rng = random.Random(seed)
    shapes = []
    for _ in range(count):
        prefill_count = rng.randrange(4)
        decode_count = rng.randrange(33)
        if prefill_count == 0 and decode_count == 0:
            decode_count = 1

        prefill_q = []
        prefill_hbf = []
        prefill_lpddr = []
        for _ in range(prefill_count):
            q = rng.randint(1, 8_192)
            hbf = rng.randint(0, 1_010_000 - q)
            lpddr = rng.randint(0, 1_010_000 - q - hbf)
            prefill_q.append(q)
            prefill_hbf.append(hbf)
            prefill_lpddr.append(lpddr)

        decode_hbf = []
        decode_lpddr = []
        for _ in range(decode_count):
            total_k = rng.randint(1, 1_010_000)
            lpddr = rng.randint(0, total_k)
            decode_hbf.append(total_k - lpddr)
            decode_lpddr.append(lpddr)

        real_tokens = sum(prefill_q) + decode_count
        shapes.append(HBFModelBatchShape(
            total_tokens=real_tokens + rng.randint(0, 2_048),
            prefill_q=tuple(prefill_q),
            prefill_hbf_k=tuple(prefill_hbf),
            prefill_lpddr_k=tuple(prefill_lpddr),
            decode_hbf_k=tuple(decode_hbf),
            decode_lpddr_k=tuple(decode_lpddr),
            lm_head_sequences=max(
                1, prefill_count + decode_count + rng.randrange(3)),
        ))
    return tuple(shapes)


class FullModelHBFLatencyFastPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hardware, cls.layouts = load_hbf_server_config(CONFIG)
        cls.hardware_variants = (
            hardware,
            dataclasses.replace(
                hardware,
                lpddr_bandwidth_gbps_per_card=51.2,
            ),
            dataclasses.replace(
                hardware,
                hbf_read_bandwidth_gbps_per_card=837.5,
                hbf_read_latency_us=19.0,
                npu_peak_tflops_per_card=247.375,
                intra_fabric_bandwidth_gbps_per_card=25.0,
                intra_fabric_fixed_latency_us=11.0,
            ),
        )
        cls.calibration_bands = ("fast", "central", "slow")
        cls.shapes = (
            _representative_shapes()
            + _random_shapes(seed=0x5EED, count=32)
        )

    def test_aggregate_matches_expanded_plan_exactly(self):
        comparisons = 0
        for hardware_index, hardware in enumerate(
                self.hardware_variants):
            for band in self.calibration_bands:
                for layout_key, layout in self.layouts.items():
                    with self.subTest(
                            hardware=hardware_index,
                            band=band,
                            layout=layout_key):
                        model = build_full_model_hbf_latency(
                            repo_root=REPO_ROOT,
                            hardware=hardware,
                            layout=layout,
                            band=band,
                        )
                        for shape_index, shape in enumerate(self.shapes):
                            with self.subTest(shape=shape_index):
                                fast = model.batch_latency(shape)
                                expanded = (
                                    model._batch_latency_from_execution_plan(
                                        shape))
                                self.assertEqual(fast, expanded)
                                self.assertEqual(
                                    fast.as_dict(), expanded.as_dict())
                                comparisons += 1
        self.assertEqual(
            comparisons,
            (
                len(self.hardware_variants)
                * len(self.calibration_bands)
                * len(self.layouts)
                * len(self.shapes)
            ),
        )

    def test_fast_path_does_not_materialize_execution_plan(self):
        hardware = self.hardware_variants[0]
        model = build_full_model_hbf_latency(
            repo_root=REPO_ROOT,
            hardware=hardware,
            layout=self.layouts["tp8_context"],
        )
        shape = HBFModelBatchShape(
            total_tokens=37,
            prefill_q=(31,),
            prefill_hbf_k=(123_457,),
            prefill_lpddr_k=(17,),
            decode_hbf_k=(456_789,) * 6,
            decode_lpddr_k=(33,) * 6,
            lm_head_sequences=7,
        )
        model.batch_latency.cache_clear()
        with mock.patch.object(
                model,
                "batch_execution_plan",
                side_effect=AssertionError(
                    "aggregate path expanded the diagnostic plan"),
        ):
            result = model.batch_latency(shape)
        self.assertGreater(result.total_ns, 0)

    def test_diagnostic_plan_remains_available_and_self_consistent(self):
        hardware = self.hardware_variants[0]
        for layout_key, layout in self.layouts.items():
            with self.subTest(layout=layout_key):
                model = build_full_model_hbf_latency(
                    repo_root=REPO_ROOT,
                    hardware=hardware,
                    layout=layout,
                )
                shape = self.shapes[5]
                plan = model.batch_execution_plan(shape)
                expanded = model._batch_latency_from_execution_plan(shape)
                self.assertEqual(plan.total_ns, expanded.total_ns)
                self.assertEqual(
                    len(plan.context_attention_rank_executions),
                    2 if layout_key == "tp8_context" else 0,
                )
                self.assertGreater(len(plan.operations), 48)


if __name__ == "__main__":
    unittest.main()
