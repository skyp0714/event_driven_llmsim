import unittest
import warnings
from pathlib import Path

from serving.core.gpu_pd_hbf_prefill import (
    HBF_PREFILL_CARD_COUNT,
    HBFPrefillLatencyAdapter,
    SingleHBFPrefillTieredSystem,
    build_hetero_system_from_config,
    hbf_home_capacity_bytes,
    load_hetero_hbf_prefill_config,
    virtual_hbf_tier_hardware,
)
from serving.core.gpu_pd_latency import (
    P4D4LatencyModel,
    load_p4d4_gpu_config,
)
from serving.core.gpu_pd_single_system import (
    SingleFiniteHBMTieredBaseline,
)
from serving.core.hbf_full_model_latency import (
    FullModelHBFLatencyModel,
    HBFModelBatchShape,
    HBFParallelLayout,
)
from serving.core.hbf_comparison_workload import (
    CallSpec,
    ScheduledSession,
    SessionSpec,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_CONFIG = (
    REPO_ROOT / "configs" / "wakekv_hbf" / "p4d4_gpu_server.json")
HETERO_CONFIG = (
    REPO_ROOT / "configs" / "wakekv_hbf"
    / "hetero_p4hbf_d4hbm_server.json")

ENGINE = dict(
    max_num_batched_tokens=131_072,
    max_num_seqs=128,
    p_max_num_seqs=32,
    d_max_num_seqs=128,
    max_prefill_chunk_tokens=131_072,
    validate_every_event=True,
)


def scheduled(offer, session_id, arrival_ns, calls):
    specs = []
    for index, (inp, out, prefix, tool) in enumerate(calls):
        specs.append(CallSpec(
            session_id=session_id,
            source_index=offer,
            call_index=index,
            input_tokens=inp,
            output_tokens=out,
            tool_duration_ns=tool,
            cached_prefix_tokens=prefix,
            fresh_input_tokens=inp - prefix,
            lineage_status=None,
            inter_turn_gap_type=None,
        ))
    session = SessionSpec(
        source_index=offer,
        session_id=session_id,
        source_arrival_time_ns=arrival_ns,
        source_session_identity_sha256=None,
        calls=tuple(specs),
    )
    return ScheduledSession(
        offer_index=offer,
        session=session,
        arrival_time_ns=arrival_ns,
        unit_interarrival=0.0 if offer == 0 else 1.0,
        unit_arrival_time=float(offer),
    )


class HBFPrefillAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.gpu_model = P4D4LatencyModel(
            repo_root=REPO_ROOT,
            hardware=load_p4d4_gpu_config(GPU_CONFIG),
        )
        _, cls.hbf_hardware, _ = load_hetero_hbf_prefill_config(
            HETERO_CONFIG)
        cls.hbf_model = FullModelHBFLatencyModel(
            base_provider=cls.gpu_model.provider,
            hardware=cls.hbf_hardware,
            layout=HBFParallelLayout.for_key("tp4"),
        )
        cls.adapter = HBFPrefillLatencyAdapter(cls.hbf_model)

    def shapes(self):
        for tokens in (256, 2_048, 8_000):
            yield HBFModelBatchShape(
                total_tokens=tokens,
                prefill_q=(tokens,),
                prefill_hbf_k=(tokens,),
                prefill_lpddr_k=(0,),
                lm_head_sequences=1,
            )

    def test_adapter_preserves_totals(self):
        for shape in self.shapes():
            raw = self.hbf_model.batch_latency(shape)
            adapted = self.adapter.batch_latency(shape)
            self.assertEqual(adapted.total_ns, raw.total_ns)
            self.assertEqual(
                adapted.comp_ns + adapted.collective_ns,
                adapted.total_ns)
            self.assertEqual(
                adapted.collective_ns, raw.collective_ns)

    def test_phase_decomposition_is_exact(self):
        for shape in self.shapes():
            phases = self.adapter.batch_phase_latency(shape)
            reconstructed = (
                phases.prologue_ns
                + phases.layer_count * phases.layer_ns
                + phases.epilogue_ns
            )
            self.assertEqual(
                reconstructed,
                self.adapter.batch_latency(shape).total_ns)
            self.assertGreater(phases.layer_ns, 0)

    def test_collectives_match_nvlink_gpu_model(self):
        # The prefill cards sit on the same NVLink fabric as the decode
        # GPUs, so a TP4 collective must cost the same on both models.
        for shape in self.shapes():
            gpu = self.gpu_model.batch_latency(shape)
            hbf = self.adapter.batch_latency(shape)
            self.assertEqual(
                hbf.tp_allreduce_ns, gpu.tp_allreduce_ns)

    def test_striped_layout_is_rejected(self):
        striped = FullModelHBFLatencyModel(
            base_provider=self.gpu_model.provider,
            hardware=self.hbf_hardware,
            layout=HBFParallelLayout.for_key("tp8_context"),
        )
        with self.assertRaises(ValueError):
            HBFPrefillLatencyAdapter(striped)


class VirtualTierTests(unittest.TestCase):
    def test_hbf_home_capacity_is_one_replica(self):
        _, hbf, _ = load_hetero_hbf_prefill_config(HETERO_CONFIG)
        self.assertEqual(
            hbf_home_capacity_bytes(hbf),
            HBF_PREFILL_CARD_COUNT
            * hbf.hbf_capacity_bytes_per_card)

    def test_virtual_tier_overrides_cpu_fields_only(self):
        gpu, hbf, _ = load_hetero_hbf_prefill_config(HETERO_CONFIG)
        virtual = virtual_hbf_tier_hardware(gpu, hbf)
        self.assertEqual(
            virtual.cpu_memory_capacity_bytes,
            hbf_home_capacity_bytes(hbf))
        self.assertEqual(virtual.ssd_device_count, gpu.ssd_device_count)
        self.assertEqual(
            virtual.nvlink_bandwidth_gbps_per_gpu,
            gpu.nvlink_bandwidth_gbps_per_gpu)


class HeteroSystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")

    def work(self):
        return [
            scheduled(0, "sA", 0, [
                (8_000, 200, 0, 7_200_000_000_000),
                (9_000, 300, 8_200, 0),
            ]),
            scheduled(1, "sB", 1_000_000, [(4_000, 100, 0, 0)]),
        ]

    def test_system_runs_and_reports(self):
        system = build_hetero_system_from_config(
            repo_root=REPO_ROOT, config_path=HETERO_CONFIG, **ENGINE)
        completed = system.run(self.work())
        self.assertEqual(len(completed), 3)
        report = system.report()
        self.assertEqual(
            report["mode"], "single_hetero_hbf_prefill_p4d4")
        pool_report = system.node.pool.report()
        self.assertIsNotNone(pool_report["p_latency_model"])
        self.assertEqual(
            pool_report["p_latency_model"]["kind"],
            "hbf_prefill_p4_adapter")

    def test_prefill_uses_hbf_model_decode_uses_gpu_model(self):
        system = build_hetero_system_from_config(
            repo_root=REPO_ROOT, config_path=HETERO_CONFIG, **ENGINE)
        system.run(self.work())
        pool = system.node.pool
        p_batches = [
            batch for batch in pool.batch_history
            if batch.stage == "p"
        ]
        d_batches = [
            batch for batch in pool.batch_history
            if batch.stage == "d"
        ]
        self.assertTrue(p_batches and d_batches)
        for batch in p_batches:
            self.assertEqual(
                batch.latency.total_ns,
                pool.p_model.batch_latency(batch.shape).total_ns)
        for batch in d_batches:
            self.assertEqual(
                batch.latency.total_ns,
                pool.model.batch_latency(batch.shape).total_ns)

    def test_run_until_is_resumable_on_single_systems(self):
        system = build_hetero_system_from_config(
            repo_root=REPO_ROOT, config_path=HETERO_CONFIG, **ENGINE)
        audit = system.run_until(
            1_000_000_000, scheduled_sessions=self.work())
        self.assertFalse(audit.system_finished)
        completed = system.run()
        self.assertEqual(len(completed), 3)

    def test_baseline_run_until_also_works(self):
        system = SingleFiniteHBMTieredBaseline(
            repo_root=REPO_ROOT,
            hardware=load_p4d4_gpu_config(GPU_CONFIG),
            policy="cpu_ssd",
            **ENGINE)
        audit = system.run_until(
            1_000_000_000, scheduled_sessions=self.work())
        self.assertFalse(audit.system_finished)
        completed = system.run()
        self.assertEqual(len(completed), 3)

    def test_hbf_resume_avoids_ssd_restore_cost(self):
        # sA idles for two hours with a 30.2k-token context.  sC's large
        # prompt lands during the gap and pressures the shrunken D tier,
        # demoting sA (the LRU victim).  The baseline's CPU tier is far
        # too small for the context, so its resume restores from SSD;
        # the hetero system's HBF home holds the context and its resume
        # must come back faster despite the slower HBF-card prefill.
        hardware = load_p4d4_gpu_config(GPU_CONFIG)
        per_rank_token = hardware.kv_bytes_per_token_per_rank
        aggregate_token = per_rank_token * hardware.tp_size
        tight_d = 38_400 * per_rank_token
        # Just one retained context: sC's demotion cascades sA to SSD.
        tight_cpu = 40_000 * aggregate_token

        def pressured_work():
            return [
                scheduled(0, "sA", 0, [
                    (30_000, 200, 0, 7_200_000_000_000),
                    (31_000, 300, 30_200, 0),
                ]),
                scheduled(1, "sC", 100_000_000_000, [
                    (20_000, 200, 0, 0),
                ]),
                scheduled(2, "sD", 200_000_000_000, [
                    (25_000, 200, 0, 0),
                ]),
            ]

        baseline = SingleFiniteHBMTieredBaseline(
            repo_root=REPO_ROOT,
            hardware=hardware,
            policy="cpu_ssd",
            d_capacity_bytes_per_rank=tight_d,
            cpu_capacity_bytes=tight_cpu,
            **ENGINE)
        hetero = build_hetero_system_from_config(
            repo_root=REPO_ROOT, config_path=HETERO_CONFIG,
            d_capacity_bytes_per_rank=tight_d, **ENGINE)

        def resume_ttft(system):
            done = system.run(pressured_work())
            resume = [
                call for call in done
                if call.key.session_id == "sA"
                and call.key.sub_request_index == 1
            ]
            self.assertEqual(len(resume), 1)
            return resume[0].first_token_ns - resume[0].release_ns

        baseline_ttft = resume_ttft(baseline)
        hetero_ttft = resume_ttft(hetero)
        self.assertLess(hetero_ttft, baseline_ttft)


if __name__ == "__main__":
    unittest.main()


class HandoffDeferredReservationTests(unittest.TestCase):
    """The deferred policy detaches first tokens from D-slot waits."""

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter("ignore")
        cls.hardware = load_p4d4_gpu_config(GPU_CONFIG)

    def work(self):
        # Three simultaneous prompts whose final contexts each nearly
        # fill the shrunken D tier: decode capacity admits one at a
        # time, but prefill capacity holds all three.
        return [
            scheduled(i, f"s{i}", 0, [(8_000, 200, 0, 0)])
            for i in range(3)
        ]

    def build(self, d_reservation_policy):
        tight_d = 12_000 * self.hardware.kv_bytes_per_token_per_rank
        return SingleFiniteHBMTieredBaseline(
            repo_root=REPO_ROOT,
            hardware=self.hardware,
            policy="cpu_ssd",
            d_capacity_bytes_per_rank=tight_d,
            **ENGINE,
            d_reservation_policy=d_reservation_policy,
        )

    def test_deferred_policy_cuts_ttft_under_d_pressure(self):
        from serving.core.gpu_pd_tier_lifecycle import (
            D_RESERVATION_FINAL_UPFRONT,
            D_RESERVATION_HANDOFF_DEFERRED,
        )

        def ttfts(system):
            done = system.run(self.work())
            self.assertEqual(len(done), 3)
            return sorted(
                (c.first_token_ns - c.release_ns) / 1e6 for c in done)

        upfront = self.build(D_RESERVATION_FINAL_UPFRONT)
        deferred = self.build(D_RESERVATION_HANDOFF_DEFERRED)
        upfront_ttfts = ttfts(upfront)
        deferred_ttfts = ttfts(deferred)
        # Upfront: the third prompt waits for two full decodes before
        # its prefill may even start.  Deferred: all three prefill
        # back-to-back; only the handoffs serialize on D.
        self.assertLess(deferred_ttfts[-1], upfront_ttfts[-1] * 0.5)
        pool = deferred.node.pool
        self.assertGreaterEqual(pool.metrics.gated_handoffs, 1)
        self.assertEqual(len(pool.gated_handoff_request_ids), 0)
        self.assertGreaterEqual(
            deferred.node.metrics.gated_handoff_releases, 1)

    def test_deferred_policy_supported_on_hetero_system(self):
        from serving.core.gpu_pd_tier_lifecycle import (
            D_RESERVATION_HANDOFF_DEFERRED)

        system = build_hetero_system_from_config(
            repo_root=REPO_ROOT, config_path=HETERO_CONFIG,
            d_reservation_policy=D_RESERVATION_HANDOFF_DEFERRED,
            **ENGINE)
        done = system.run([
            scheduled(0, "sA", 0, [
                (8_000, 200, 0, 7_200_000_000_000),
                (9_000, 300, 8_200, 0),
            ]),
            scheduled(1, "sB", 1_000_000, [(4_000, 100, 0, 0)]),
        ])
        self.assertEqual(len(done), 3)
