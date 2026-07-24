from pathlib import Path
from types import SimpleNamespace
import unittest

from serving.core.hbf_full_model_latency import (
    HBFParallelLayout,
    HBFServerHardware,
    qwen_logical_kv_bytes_per_token,
    qwen_model_weight_bytes_per_rank,
)
from serving.core.hbf_full_model_lifecycle import (
    FullModelHBFLifecycle,
    PerGroupCapacityLedger,
    PlacementState,
    hbf_kv_range_card_bytes,
)
from serving.core.hbf_full_model_pool import (
    FullModelHBFServingPool,
    HBFRequestState,
    derive_lpddr_workspace_bytes,
)
from serving.core.hbf_online_adapter import (
    FullModelHBFOnlineAdapter,
    GPUHBMEventKind,
    ONLINE_HBF_ADAPTER_SCHEMA,
    OnlineHBFCallState,
    OnlineHBFExecution,
)
from serving.core.router import Router


REPO_ROOT = Path(__file__).resolve().parents[1]


def raw_request(
        request_id, session_id, call_index, arrival_ns,
        input_tokens, output_tokens, prefix_reuse_tokens,
        has_successor):
    return {
        "index": request_id,
        "session_id": session_id,
        "sub_request_index": call_index,
        "arrival_time_ns": arrival_ns,
        "input_toks": input_tokens,
        "output_toks": input_tokens + output_tokens,
        "prefix_reuse_toks": prefix_reuse_tokens,
        "prefix_reuse_source": "test",
        "wakekv_has_successor": has_successor,
        "source_session_id": f"source-{session_id}",
        "session_template_index": 3,
        "session_epoch": 2,
    }


def build_adapter(
        *, ledger_capacity_bytes=None,
        gpu_resume_mode="sticky_reuse",
        hardware=None,
        prefill_drain_tail_tokens=None,
        prefill_drain_min_tokens=4096):
    hardware = hardware or HBFServerHardware()
    layout = HBFParallelLayout.for_key("tp4")
    workspace = derive_lpddr_workspace_bytes(
        layout,
        max_num_batched_tokens=16,
        max_num_seqs=4,
    )
    maximum = (
        hardware.lpddr_capacity_bytes_per_card - workspace)
    capacity = (
        maximum
        if ledger_capacity_bytes is None
        else ledger_capacity_bytes
    )
    ledger = PerGroupCapacityLedger(
        group_count=layout.replicas,
        capacity_bytes=capacity,
    )
    lifecycle = FullModelHBFLifecycle(
        hardware=hardware,
        layout=layout,
        lpddr_ledger=ledger,
        execution_backend="external_astra",
        server_id=7,
        astra_chunk_bytes=1024 * 1024,
    )
    pool = FullModelHBFServingPool(
        repo_root=REPO_ROOT,
        hardware=hardware,
        layout=layout,
        lpddr_ledger=ledger,
        placement_resolver=lifecycle.placement_snapshot,
        max_num_batched_tokens=16,
        max_num_seqs=4,
        max_prefill_chunk_tokens=16,
        prefill_drain_tail_tokens=prefill_drain_tail_tokens,
        prefill_drain_min_tokens=prefill_drain_min_tokens,
        execution_backend="external_astra",
        server_id=7,
    )
    return FullModelHBFOnlineAdapter(
        lifecycle=lifecycle,
        pool=pool,
        gpu_resume_mode=gpu_resume_mode,
    )


def complete_job(adapter, job, *, not_before_ns=None):
    _, _, stages = job.controller_arguments()
    completion_ns = (
        job.arrival_ns
        + sum(int(stage["runtime_ns"]) for stage in stages)
        + 1
    )
    if not_before_ns is not None:
        completion_ns = max(completion_ns, not_before_ns)
    result = adapter.complete_astra_dispatch(
        job_id=job.job_id,
        arrival_ns=job.arrival_ns,
        completion_ns=completion_ns,
        stage_count=job.stage_count,
    )
    return completion_ns, result


def prepare_hbf_ready(
        adapter, *, request_id, session_id,
        input_tokens=4, output_tokens=2,
        arrival_ns=0, completion_ns=None,
        gpu_instance_id=0):
    if completion_ns is None:
        completion_ns = arrival_ns + 1
    adapter.offer_raw_requests((
        raw_request(
            request_id, session_id, 0, arrival_ns,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            prefix_reuse_tokens=0,
            has_successor=True,
        ),
    ), now_ns=arrival_ns)
    materialized_tokens = input_tokens + output_tokens - 1
    adapter.complete_native_gpu_request(
        request_id,
        completion_ns=completion_ns,
        materialized_tokens=materialized_tokens,
        gpu_instance_id=gpu_instance_id,
    )
    adapter.pop_gpu_hbm_events()
    migration, = adapter.drain_astra_dispatches()
    ready_ns, _ = complete_job(adapter, migration)
    adapter.pop_gpu_hbm_events()
    return ready_ns, materialized_tokens


class DummyScheduler:
    pd_type = "colocated"
    instance_id = 0


class FullModelHBFOnlineAdapterTests(unittest.TestCase):
    def test_active_prefill_drain_defers_decode_and_preserves_admitted_hit(
            self):
        adapter = build_adapter(
            prefill_drain_tail_tokens=2,
            prefill_drain_min_tokens=1,
        )
        ready_ns, cached_tokens = prepare_hbf_ready(
            adapter,
            request_id=100,
            session_id="session-active-drain",
        )
        resume = raw_request(
            101, "session-active-drain", 1, ready_ns + 1,
            input_tokens=12,
            output_tokens=3,
            prefix_reuse_tokens=cached_tokens,
            has_successor=False,
        )
        adapter.offer_raw_requests(
            (resume,), now_ns=resume["arrival_time_ns"])
        prefill, = adapter.drain_astra_dispatches()
        prefill_done, completion = complete_job(adapter, prefill)
        self.assertEqual(completion.router_completions, ())

        request = adapter.pool.requests[101]
        self.assertEqual(request.state, HBFRequestState.PREFILL_DRAIN)
        self.assertFalse(request.prefill_drain_claimed)
        self.assertEqual(request.generated_tokens, 1)
        self.assertEqual(request.cached_tokens, cached_tokens)
        self.assertEqual(request.published_tokens, cached_tokens)
        self.assertEqual(
            adapter.report()[
                "pending_prefill_drain_request_by_session"],
            {"session-active-drain": 101},
        )
        # The pool callback only records an intent. Lifecycle ownership
        # begins after all same-time callbacks and arrivals are fenced.
        self.assertEqual(adapter.drain_astra_dispatches(), ())

        adapter.flush_admissions(prefill_done)
        self.assertTrue(request.prefill_drain_claimed)
        self.assertEqual(
            request.admitted_hbf_prefix_tokens, cached_tokens)
        self.assertEqual(request.admitted_lpddr_prefix_tokens, 0)
        self.assertEqual(request.cached_tokens, cached_tokens)
        self.assertEqual(request.published_tokens, resume["input_toks"])
        drain, = adapter.drain_astra_dispatches()
        self.assertEqual(
            drain.source_name, adapter.lifecycle_source_name)
        drain_done, _ = complete_job(adapter, drain)

        self.assertEqual(request.state, HBFRequestState.DECODE)
        self.assertIsNone(request.prefill_drain_job_id)
        self.assertEqual(
            request.active_lpddr_tokens,
            adapter.pool.prefill_drain_tail_tokens,
        )
        # Callback release is deliberately defer_schedule=True.
        self.assertEqual(adapter.drain_astra_dispatches(), ())
        with self.assertRaisesRegex(
                RuntimeError, "already completed|duplicate"):
            adapter.complete_astra_dispatch(
                job_id=drain.job_id,
                arrival_ns=drain.arrival_ns,
                completion_ns=drain_done,
                stage_count=drain.stage_count,
            )

        adapter.flush_admissions(drain_done)
        while (
            adapter.calls[101].state
            != OnlineHBFCallState.COMPLETE
        ):
            jobs = adapter.drain_astra_dispatches()
            self.assertTrue(jobs)
            for job in jobs:
                complete_job(adapter, job)
            if (
                adapter.calls[101].state
                != OnlineHBFCallState.COMPLETE
            ):
                adapter.flush_admissions(adapter.current_ns)

        proxy, = adapter.pop_router_completions()
        self.assertEqual(proxy.hbf_prefix_tokens, cached_tokens)
        self.assertEqual(proxy.lpddr_prefix_tokens, 0)
        materialized = proxy.materialize_request(
            model="Qwen/Qwen3-30B-A3B-Instruct-2507",
            instance_id=0,
        )
        self.assertEqual(
            materialized.agentic_kv_hit_tokens, cached_tokens)
        self.assertEqual(
            materialized.agentic_kv_fresh_prompt_tokens,
            resume["input_toks"] - cached_tokens,
        )
        report = adapter.report()
        self.assertEqual(
            report["pending_prefill_drain_request_by_session"], {})
        self.assertEqual(
            report["active_prefill_drain_request_by_job"], {})
        self.assertEqual(
            report[
                "waiting_prefill_drain_append_jobs_by_session"], {})
        self.assertFalse(adapter.has_pending())
        adapter.assert_invariants()

    def test_active_prefill_drain_capacity_fallback_releases_gate(
            self):
        layout = HBFParallelLayout.for_key("tp4")
        initial_tokens = 5
        initial_card_bytes = hbf_kv_range_card_bytes(
            layout=layout,
            card_ids=tuple(range(layout.tp_size)),
            kv_bytes_per_token=qwen_logical_kv_bytes_per_token(),
            token_start=0,
            token_count=initial_tokens,
        )
        hardware = HBFServerHardware(
            hbf_capacity_bytes_per_card=(
                qwen_model_weight_bytes_per_rank(layout.tp_size)
                + max(initial_card_bytes.values())
            ),
        )
        adapter = build_adapter(
            hardware=hardware,
            prefill_drain_tail_tokens=2,
            prefill_drain_min_tokens=1,
        )
        ready_ns, cached_tokens = prepare_hbf_ready(
            adapter,
            request_id=110,
            session_id="session-drain-capacity",
        )
        resume = raw_request(
            111, "session-drain-capacity", 1, ready_ns + 1,
            input_tokens=12,
            output_tokens=2,
            prefix_reuse_tokens=cached_tokens,
            has_successor=False,
        )
        adapter.offer_raw_requests(
            (resume,), now_ns=resume["arrival_time_ns"])
        prefill, = adapter.drain_astra_dispatches()
        prefill_done, _ = complete_job(adapter, prefill)
        self.assertEqual(
            adapter.pool.requests[111].state,
            HBFRequestState.PREFILL_DRAIN,
        )

        adapter.flush_admissions(prefill_done)
        request = adapter.pool.requests[111]
        self.assertEqual(request.state, HBFRequestState.DECODE)
        self.assertGreater(
            request.active_lpddr_tokens,
            adapter.pool.prefill_drain_tail_tokens,
        )
        jobs = adapter.drain_astra_dispatches()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(
            jobs[0].source_name, adapter.pool_source_name)
        _, final = complete_job(adapter, jobs[0])
        proxy, = final.router_completions
        self.assertEqual(proxy.id, 111)
        self.assertEqual(
            adapter.pool.metrics.prefill_drain_fallbacks, 1)
        self.assertEqual(
            adapter.lifecycle.metrics
            .active_prefill_drain_capacity_fallback,
            1,
        )
        self.assertEqual(adapter.pop_router_completions(), [proxy])
        self.assertFalse(adapter.has_pending())
        adapter.assert_invariants()

    def test_prefill_drain_waits_for_prior_append_then_retries(self):
        adapter = build_adapter(
            prefill_drain_tail_tokens=2,
            prefill_drain_min_tokens=1,
        )
        ready_ns, cached_tokens = prepare_hbf_ready(
            adapter,
            request_id=120,
            session_id="session-drain-wait",
        )
        predecessor = raw_request(
            121, "session-drain-wait", 1, ready_ns + 1,
            input_tokens=6,
            output_tokens=1,
            prefix_reuse_tokens=cached_tokens,
            has_successor=True,
        )
        adapter.offer_raw_requests(
            (predecessor,),
            now_ns=predecessor["arrival_time_ns"],
        )
        predecessor_model, = adapter.drain_astra_dispatches()
        predecessor_done, predecessor_completion = complete_job(
            adapter, predecessor_model)
        predecessor_proxy, = (
            predecessor_completion.router_completions)
        self.assertEqual(
            adapter.pop_router_completions(),
            [predecessor_proxy],
        )
        prior_append, = adapter.drain_astra_dispatches()
        self.assertEqual(
            prior_append.source_name,
            adapter.lifecycle_source_name,
        )
        prior_numeric_job_id, = adapter.lifecycle.sessions[
            "session-drain-wait"].append_job_ids

        terminal = raw_request(
            122, "session-drain-wait", 2, predecessor_done,
            input_tokens=12,
            output_tokens=2,
            prefix_reuse_tokens=6,
            has_successor=False,
        )
        decision, = adapter.offer_raw_requests(
            (terminal,), now_ns=predecessor_done)
        self.assertEqual(decision.route_reason, "hbf_append_inflight")
        terminal_prefill, = adapter.drain_astra_dispatches()
        terminal_prefill_done, _ = complete_job(
            adapter, terminal_prefill)
        adapter.flush_admissions(terminal_prefill_done)

        request = adapter.pool.requests[122]
        self.assertTrue(request.prefill_drain_claimed)
        self.assertIsNone(request.prefill_drain_job_id)
        waiting = adapter.report()[
            "waiting_prefill_drain_append_jobs_by_session"]
        self.assertEqual(
            waiting,
            {"session-drain-wait": [prior_numeric_job_id]},
        )
        self.assertEqual(adapter.drain_astra_dispatches(), ())

        prior_done, _ = complete_job(
            adapter,
            prior_append,
            not_before_ns=terminal_prefill_done,
        )
        self.assertIsNotNone(request.prefill_drain_job_id)
        self.assertEqual(
            adapter.report()[
                "waiting_prefill_drain_append_jobs_by_session"],
            {},
        )
        retry_drain, = adapter.drain_astra_dispatches()
        self.assertEqual(
            retry_drain.source_name,
            adapter.lifecycle_source_name,
        )
        retry_done, _ = complete_job(adapter, retry_drain)
        self.assertEqual(request.state, HBFRequestState.DECODE)
        self.assertEqual(adapter.drain_astra_dispatches(), ())

        adapter.flush_admissions(max(prior_done, retry_done))
        decode, = adapter.drain_astra_dispatches()
        _, final = complete_job(adapter, decode)
        proxy, = final.router_completions
        self.assertEqual(proxy.id, 122)
        self.assertEqual(
            adapter.lifecycle.metrics
            .active_prefill_drain_wait_existing_append,
            1,
        )
        self.assertEqual(
            adapter.lifecycle.metrics.active_prefill_drain_started,
            1,
        )
        self.assertEqual(adapter.pop_router_completions(), [proxy])
        self.assertFalse(adapter.has_pending())
        adapter.assert_invariants()

    def test_prefill_drain_callback_rejects_pool_job_mismatch(self):
        adapter = build_adapter(
            prefill_drain_tail_tokens=2,
            prefill_drain_min_tokens=1,
        )
        ready_ns, cached_tokens = prepare_hbf_ready(
            adapter,
            request_id=130,
            session_id="session-drain-mismatch",
        )
        resume = raw_request(
            131, "session-drain-mismatch", 1, ready_ns + 1,
            input_tokens=12,
            output_tokens=2,
            prefix_reuse_tokens=cached_tokens,
            has_successor=False,
        )
        adapter.offer_raw_requests(
            (resume,), now_ns=resume["arrival_time_ns"])
        prefill, = adapter.drain_astra_dispatches()
        prefill_done, _ = complete_job(adapter, prefill)
        adapter.flush_admissions(prefill_done)
        drain, = adapter.drain_astra_dispatches()
        request = adapter.pool.requests[131]
        active_job_id = request.prefill_drain_job_id
        self.assertIsNotNone(active_job_id)
        adapter.pool.clear_prefill_drain_job(
            request.request_id,
            job_id=active_job_id,
        )

        _, _, stages = drain.controller_arguments()
        drain_done = (
            drain.arrival_ns
            + sum(int(stage["runtime_ns"]) for stage in stages)
            + 1
        )
        with self.assertRaisesRegex(
                RuntimeError,
                "active prefill-drain callback identity mismatch"):
            adapter.complete_astra_dispatch(
                job_id=drain.job_id,
                arrival_ns=drain.arrival_ns,
                completion_ns=drain_done,
                stage_count=drain.stage_count,
            )

    def test_tied_prior_append_satisfies_unclaimed_prefill_drain(
            self):
        adapter = build_adapter(
            prefill_drain_tail_tokens=1,
            prefill_drain_min_tokens=1,
        )
        ready_ns, cached_tokens = prepare_hbf_ready(
            adapter,
            request_id=140,
            session_id="session-drain-tie",
        )
        predecessor = raw_request(
            141, "session-drain-tie", 1, ready_ns + 1,
            input_tokens=6,
            output_tokens=1,
            prefix_reuse_tokens=cached_tokens,
            has_successor=True,
        )
        adapter.offer_raw_requests(
            (predecessor,),
            now_ns=predecessor["arrival_time_ns"],
        )
        predecessor_model, = adapter.drain_astra_dispatches()
        predecessor_done, predecessor_completion = complete_job(
            adapter, predecessor_model)
        predecessor_proxy, = (
            predecessor_completion.router_completions)
        self.assertEqual(
            adapter.pop_router_completions(),
            [predecessor_proxy],
        )
        prior_append, = adapter.drain_astra_dispatches()

        terminal = raw_request(
            142, "session-drain-tie", 2, predecessor_done,
            input_tokens=7,
            output_tokens=2,
            prefix_reuse_tokens=6,
            has_successor=False,
        )
        adapter.offer_raw_requests(
            (terminal,), now_ns=predecessor_done)
        terminal_prefill, = adapter.drain_astra_dispatches()
        projections = []
        for job in (prior_append, terminal_prefill):
            _, _, stages = job.controller_arguments()
            projections.append(
                job.arrival_ns
                + sum(int(stage["runtime_ns"]) for stage in stages)
                + 1
            )
        tied_ns = max(projections)

        # The model callback gates the request first. The tied predecessor
        # append then publishes before the barrier is allowed to claim it.
        adapter.complete_astra_dispatch(
            job_id=terminal_prefill.job_id,
            arrival_ns=terminal_prefill.arrival_ns,
            completion_ns=tied_ns,
            stage_count=terminal_prefill.stage_count,
        )
        request = adapter.pool.requests[142]
        self.assertEqual(request.state, HBFRequestState.PREFILL_DRAIN)
        self.assertFalse(request.prefill_drain_claimed)
        adapter.complete_astra_dispatch(
            job_id=prior_append.job_id,
            arrival_ns=prior_append.arrival_ns,
            completion_ns=tied_ns,
            stage_count=prior_append.stage_count,
        )
        self.assertEqual(request.hbf_prefix_tokens, 6)
        self.assertEqual(request.lpddr_prefix_tokens, 0)
        self.assertEqual(adapter.drain_astra_dispatches(), ())

        adapter.flush_admissions(tied_ns)
        self.assertEqual(request.state, HBFRequestState.DECODE)
        self.assertFalse(request.prefill_drain_claimed)
        self.assertEqual(
            adapter.lifecycle.metrics.active_prefill_drain_satisfied,
            1,
        )
        self.assertEqual(
            adapter.lifecycle.metrics.active_prefill_drain_started,
            0,
        )
        decode, = adapter.drain_astra_dispatches()
        self.assertEqual(decode.source_name, adapter.pool_source_name)
        _, final = complete_job(adapter, decode)
        proxy, = final.router_completions
        self.assertEqual(adapter.pop_router_completions(), [proxy])
        self.assertFalse(adapter.has_pending())
        adapter.assert_invariants()

    def test_tied_newcomer_cannot_launch_ahead_of_released_decode(
            self):
        adapter = build_adapter(
            prefill_drain_tail_tokens=1,
            prefill_drain_min_tokens=1,
        )
        ready_ns = 0
        cached_by_session = {}
        for request_id, session_id in (
                (150, "session-drain-priority"),
                (151, "session-balance-only"),
                (152, "session-tied-newcomer")):
            ready_ns, cached_tokens = prepare_hbf_ready(
                adapter,
                request_id=request_id,
                session_id=session_id,
                arrival_ns=ready_ns,
                completion_ns=ready_ns + 1,
            )
            cached_by_session[session_id] = cached_tokens
        self.assertEqual(
            adapter.lifecycle.sessions[
                "session-drain-priority"].group_id,
            adapter.lifecycle.sessions[
                "session-tied-newcomer"].group_id,
        )

        predecessor = raw_request(
            153, "session-drain-priority", 1, ready_ns + 1,
            input_tokens=6,
            output_tokens=1,
            prefix_reuse_tokens=cached_by_session[
                "session-drain-priority"],
            has_successor=True,
        )
        adapter.offer_raw_requests(
            (predecessor,),
            now_ns=predecessor["arrival_time_ns"],
        )
        predecessor_model, = adapter.drain_astra_dispatches()
        predecessor_done, predecessor_completion = complete_job(
            adapter, predecessor_model)
        predecessor_proxy, = (
            predecessor_completion.router_completions)
        self.assertEqual(
            adapter.pop_router_completions(),
            [predecessor_proxy],
        )
        prior_append, = adapter.drain_astra_dispatches()

        terminal = raw_request(
            154, "session-drain-priority", 2, predecessor_done,
            input_tokens=7,
            output_tokens=2,
            prefix_reuse_tokens=6,
            has_successor=False,
        )
        adapter.offer_raw_requests(
            (terminal,), now_ns=predecessor_done)
        terminal_prefill, = adapter.drain_astra_dispatches()
        projected_completions = []
        for job in (prior_append, terminal_prefill):
            _, _, stages = job.controller_arguments()
            projected_completions.append(
                job.arrival_ns
                + sum(int(stage["runtime_ns"]) for stage in stages)
                + 1
            )
        tied_ns = max(projected_completions)
        adapter.complete_astra_dispatch(
            job_id=terminal_prefill.job_id,
            arrival_ns=terminal_prefill.arrival_ns,
            completion_ns=tied_ns,
            stage_count=terminal_prefill.stage_count,
        )
        adapter.complete_astra_dispatch(
            job_id=prior_append.job_id,
            arrival_ns=prior_append.arrival_ns,
            completion_ns=tied_ns,
            stage_count=prior_append.stage_count,
        )

        newcomer = raw_request(
            155, "session-tied-newcomer", 1, tied_ns,
            input_tokens=6,
            output_tokens=2,
            prefix_reuse_tokens=cached_by_session[
                "session-tied-newcomer"],
            has_successor=False,
        )
        decision = adapter.offer_raw_request(
            newcomer, now_ns=tied_ns)
        self.assertTrue(decision.divert_to_hbf)
        self.assertEqual(
            decision.hbf_request.group_id,
            adapter.pool.requests[154].group_id,
        )
        adapter.flush_admissions(tied_ns)

        worker = adapter.pool.workers[
            decision.hbf_request.group_id]
        self.assertIsNotNone(worker.inflight)
        inflight_request_ids = {
            item.request_id for item in worker.inflight.items
        }
        self.assertIn(154, inflight_request_ids)
        jobs = adapter.drain_astra_dispatches()
        self.assertTrue(jobs)
        while any(
                adapter.calls[request_id].state
                != OnlineHBFCallState.COMPLETE
                for request_id in (154, 155)):
            self.assertTrue(jobs)
            for job in jobs:
                complete_job(adapter, job)
            if any(
                    adapter.calls[request_id].state
                    != OnlineHBFCallState.COMPLETE
                    for request_id in (154, 155)):
                adapter.flush_admissions(adapter.current_ns)
                jobs = adapter.drain_astra_dispatches()

        proxies = sorted(
            adapter.pop_router_completions(),
            key=lambda proxy: proxy.id,
        )
        self.assertEqual(
            [proxy.id for proxy in proxies], [154, 155])
        adapter.censor_completed_successor(
            151, now_ns=adapter.current_ns)
        self.assertFalse(adapter.has_pending())
        adapter.assert_invariants()

    def test_gpu_hbm_pressure_reclaim_emits_audited_idle_release(self):
        hardware = HBFServerHardware(
            hbf_capacity_bytes_per_card=(
                qwen_model_weight_bytes_per_rank(4) + 1),
        )
        adapter = build_adapter(
            hardware=hardware,
            gpu_resume_mode="recompute",
        )
        for request_id, session_id in (
                (1, "z-session"),
                (2, "a-session")):
            adapter.offer_raw_request(
                raw_request(
                    request_id, session_id, 0, 0,
                    input_tokens=4,
                    output_tokens=2,
                    prefix_reuse_tokens=0,
                    has_successor=True,
                ),
                now_ns=10,
            )
            self.assertIsNone(adapter.complete_native_gpu_request(
                request_id,
                completion_ns=10,
                materialized_tokens=5,
                gpu_instance_id=3,
            ))
        adapter.pop_gpu_hbm_events()

        audit = adapter.reclaim_gpu_ready_for_hbm_pressure(
            gpu_instance_id=3,
            now_ns=20,
        )

        self.assertEqual(audit["session_id"], "a-session")
        self.assertEqual(audit["gpu_instance_id"], 3)
        self.assertEqual(audit["owner_request_id"], 2)
        self.assertGreater(audit["per_rank_bytes"], 0)
        release, = adapter.pop_gpu_hbm_events()
        self.assertEqual(release.kind, GPUHBMEventKind.IDLE_RELEASE)
        self.assertEqual(release.session_id, "a-session")
        self.assertEqual(
            release.reason, "pd_decode_hbm_pressure_reclaim")
        self.assertEqual(
            adapter.lifecycle.sessions["a-session"].state,
            PlacementState.EVICTED,
        )
        self.assertEqual(
            adapter.lifecycle.sessions["z-session"].state,
            PlacementState.GPU_READY,
        )
        self.assertIsNone(adapter.reclaim_gpu_ready_for_hbm_pressure(
            gpu_instance_id=4,
            now_ns=20,
        ))
        report = adapter.report()
        self.assertEqual(
            report["metrics"]["gpu_ready_hbm_pressure_reclaims"], 1)
        self.assertEqual(
            report["gpu_ready_hbm_pressure_reclaim_audits"],
            [audit],
        )

        decision = adapter.offer_raw_request(
            raw_request(
                3, "a-session", 1, 21,
                input_tokens=6,
                output_tokens=1,
                prefix_reuse_tokens=5,
                has_successor=False,
            ),
            now_ns=21,
        )
        self.assertEqual(
            decision.execution, OnlineHBFExecution.GPU_RECOMPUTE)
        adapter.assert_invariants()

    def test_same_time_barrier_defers_hbf_append_and_can_censor_successor(self):
        adapter = build_adapter()
        adapter.offer_raw_requests((
            raw_request(
                1, "session-deferred", 0, 0,
                input_tokens=4,
                output_tokens=2,
                prefix_reuse_tokens=0,
                has_successor=True,
            ),
        ), now_ns=0)
        adapter.complete_native_gpu_request(
            1,
            completion_ns=1,
            materialized_tokens=5,
            gpu_instance_id=0,
        )
        adapter.pop_gpu_hbm_events()
        migration, = adapter.drain_astra_dispatches()
        migration_done, _ = complete_job(adapter, migration)
        adapter.pop_gpu_hbm_events()

        resume = raw_request(
            2, "session-deferred", 1, migration_done + 1,
            input_tokens=6,
            output_tokens=2,
            prefix_reuse_tokens=5,
            has_successor=True,
        )
        adapter.offer_raw_requests(
            (resume,), now_ns=resume["arrival_time_ns"])
        proxy = None
        completion_ns = None
        while proxy is None:
            model, = adapter.drain_astra_dispatches()
            _, _, stages = model.controller_arguments()
            completion_ns = (
                model.arrival_ns
                + sum(int(stage["runtime_ns"]) for stage in stages)
                + 1
            )
            result = adapter.complete_astra_dispatch(
                job_id=model.job_id,
                arrival_ns=model.arrival_ns,
                completion_ns=completion_ns,
                stage_count=model.stage_count,
                defer_turn_finalization=True,
            )
            if result.router_completions:
                proxy, = result.router_completions
            else:
                adapter.flush_admissions(completion_ns)

        self.assertEqual(
            adapter.calls[2].state, OnlineHBFCallState.HBF_ACTIVE)
        self.assertEqual(
            adapter.lifecycle.sessions["session-deferred"].state,
            PlacementState.HBF_ACTIVE,
        )
        self.assertEqual(
            adapter.report()["pending_hbf_turn_finalization_count"], 1)
        self.assertTrue(adapter.has_pending())
        with self.assertRaisesRegex(
                RuntimeError, "same-time barrier"):
            adapter.flush_admissions(completion_ns)
        with self.assertRaisesRegex(
                RuntimeError, "same-time barrier"):
            adapter.offer_raw_request(
                raw_request(
                    3, "other-session", 0, completion_ns,
                    2, 1, 0, False),
                now_ns=completion_ns,
            )
        self.assertEqual(adapter.drain_astra_dispatches(), ())

        self.assertIsNone(adapter.finalize_deferred_hbf_completion(
            proxy,
            completion_ns=completion_ns,
            publish_successor=False,
        ))
        self.assertTrue(adapter.calls[2].successor_censored)
        self.assertEqual(
            adapter.lifecycle.sessions["session-deferred"].state,
            PlacementState.ENDED,
        )
        self.assertEqual(adapter.pop_router_completions(), [proxy])
        self.assertFalse(adapter.has_pending())
        adapter.assert_invariants()

    def test_gpu_completion_cutoff_does_not_launch_migration(self):
        adapter = build_adapter()
        adapter.offer_raw_requests((
            raw_request(
                3, "session-gpu-cutoff", 0, 0,
                input_tokens=4,
                output_tokens=2,
                prefix_reuse_tokens=0,
                has_successor=True,
            ),
        ), now_ns=0)

        self.assertIsNone(adapter.complete_native_gpu_request(
            3,
            completion_ns=10,
            materialized_tokens=5,
            gpu_instance_id=0,
            publish_successor=False,
        ))
        self.assertEqual(adapter.pop_gpu_hbm_events(), [])
        self.assertEqual(adapter.drain_astra_dispatches(), ())
        self.assertTrue(adapter.calls[3].successor_censored)
        self.assertEqual(
            adapter.lifecycle.sessions["session-gpu-cutoff"].state,
            PlacementState.ENDED,
        )
        self.assertFalse(adapter.has_pending())
        adapter.assert_invariants()

    def test_gpu_migration_hbf_resume_append_and_router_proxy(self):
        adapter = build_adapter()
        first = raw_request(
            10, "session-a", 0, 0,
            input_tokens=4,
            output_tokens=2,
            prefix_reuse_tokens=0,
            has_successor=True,
        )
        decision = adapter.offer_raw_request(first, now_ns=0)
        self.assertEqual(
            decision.execution,
            OnlineHBFExecution.GPU_FIRST_TURN,
        )
        self.assertFalse(decision.divert_to_hbf)
        self.assertEqual(adapter.flush_admissions(0), 0)

        migration = adapter.complete_native_gpu_request(
            SimpleNamespace(
                id=10,
                session_id="session-a",
                num_computed_tokens=5,
                instance_id=2,
            ),
            completion_ns=100,
        )
        self.assertIsNotNone(migration)
        retain, = adapter.pop_gpu_hbm_events()
        self.assertEqual(retain.kind, GPUHBMEventKind.TURN_RETAIN)
        self.assertEqual(retain.gpu_instance_id, 2)
        self.assertEqual(retain.token_count, 5)
        self.assertEqual(retain.accounted_tokens_per_rank, 16)
        self.assertGreater(retain.per_rank_bytes, 0)

        migration_dispatch, = adapter.drain_astra_dispatches()
        migration_done, completion = complete_job(
            adapter, migration_dispatch)
        self.assertEqual(
            completion.multiplexed.source_name,
            adapter.lifecycle_source_name,
        )
        self.assertEqual(
            adapter.lifecycle.sessions["session-a"].state,
            PlacementState.HBF_READY,
        )
        release, = adapter.pop_gpu_hbm_events()
        self.assertEqual(
            release.kind, GPUHBMEventKind.MIGRATION_RELEASE)
        self.assertEqual(release.token_count, 5)

        resume = raw_request(
            11, "session-a", 1, migration_done + 7,
            input_tokens=6,
            output_tokens=1,
            prefix_reuse_tokens=5,
            has_successor=True,
        )
        hbf = adapter.offer_raw_request(
            resume, now_ns=resume["arrival_time_ns"])
        self.assertTrue(hbf.divert_to_hbf)
        self.assertEqual(
            adapter.calls[11].state,
            OnlineHBFCallState.HBF_STAGED,
        )
        with self.assertRaisesRegex(
                RuntimeError, "flush HBF admissions"):
            adapter.drain_astra_dispatches()
        self.assertEqual(
            adapter.flush_admissions(resume["arrival_time_ns"]), 1)

        model_dispatch, = adapter.drain_astra_dispatches()
        model_done, model_completion = complete_job(
            adapter, model_dispatch)
        proxy, = model_completion.router_completions
        self.assertEqual(proxy.id, 11)
        self.assertEqual(proxy.session_id, "session-a")
        self.assertEqual(proxy.num_computed_tokens, 6)
        self.assertEqual(proxy.generated_tokens, 1)
        self.assertEqual(
            proxy.ttft,
            model_done - resume["arrival_time_ns"],
        )
        self.assertEqual(proxy.tpot, 0)
        self.assertEqual(
            Router._prefix_reuse(
                {}, {"input_toks": 6}, proxy),
            (6, "estimated"),
        )
        self.assertEqual(
            adapter.pop_router_completions(), [proxy])

        append_dispatch, = adapter.drain_astra_dispatches()
        append_done, append_completion = complete_job(
            adapter, append_dispatch)
        self.assertEqual(
            append_completion.multiplexed.source_name,
            adapter.lifecycle_source_name,
        )
        placement = adapter.lifecycle.sessions["session-a"]
        self.assertEqual(placement.state, PlacementState.HBF_READY)
        self.assertEqual(placement.committed_hbf_tokens, 6)
        self.assertEqual(placement.lpddr_tokens, 0)

        terminal = raw_request(
            12, "session-a", 2, append_done + 3,
            input_tokens=6,
            output_tokens=1,
            prefix_reuse_tokens=6,
            has_successor=False,
        )
        terminal_decision, = adapter.offer_raw_requests(
            (terminal,),
            now_ns=terminal["arrival_time_ns"],
        )
        self.assertTrue(terminal_decision.divert_to_hbf)
        terminal_dispatch, = adapter.drain_astra_dispatches()
        _, terminal_completion = complete_job(
            adapter, terminal_dispatch)
        terminal_proxy, = terminal_completion.router_completions
        self.assertEqual(terminal_proxy.id, 12)
        self.assertEqual(
            adapter.pop_router_completions(), [terminal_proxy])
        self.assertEqual(
            adapter.lifecycle.sessions["session-a"].state,
            PlacementState.ENDED,
        )
        self.assertFalse(adapter.has_pending())
        adapter.assert_invariants()

        report = adapter.report()
        self.assertEqual(
            report["schema"], ONLINE_HBF_ADAPTER_SCHEMA)
        self.assertEqual(report["metrics"]["gpu_completions"], 1)
        self.assertEqual(report["metrics"]["hbf_completions"], 2)
        self.assertEqual(
            report["execution_counts"]["hbf_ready"], 2)
        self.assertIn("router_arrival_hook",
                      report["integration_contract"])

    def test_migration_inflight_resume_stays_on_gpu(self):
        adapter = build_adapter()
        adapter.offer_raw_requests((
            raw_request(
                1, "session-b", 0, 0, 4, 2, 0, True),
        ), now_ns=0)
        adapter.complete_native_gpu_request(
            1,
            completion_ns=10,
            materialized_tokens=5,
            gpu_instance_id=1,
        )
        adapter.pop_gpu_hbm_events()

        resume = raw_request(
            2, "session-b", 1, 11, 5, 1, 5, True)
        decision = adapter.offer_raw_request(resume, now_ns=11)
        self.assertEqual(
            decision.execution,
            OnlineHBFExecution.GPU_MIGRATION_INFLIGHT,
        )
        self.assertTrue(decision.run_on_gpu)
        self.assertFalse(decision.force_gpu_recompute)
        self.assertTrue(decision.migration_inflight)
        self.assertEqual(decision.required_gpu_instance_id, 1)
        self.assertEqual(
            adapter.lifecycle.sessions["session-b"].state,
            PlacementState.GPU_ACTIVE,
        )
        claim, = adapter.pop_gpu_hbm_events()
        self.assertEqual(claim.kind, GPUHBMEventKind.RESUME_CLAIM)
        self.assertEqual(claim.gpu_instance_id, 1)
        self.assertEqual(
            claim.token_count, decision.gpu_prefix_reuse_tokens)
        decorated = adapter.decorate_gpu_metadata(decision, resume)
        self.assertEqual(decorated["hbf_gpu_required_instance_id"], 1)
        self.assertEqual(decorated["agentic_kv_owner_instance_id"], 1)
        self.assertEqual(
            decorated["prefix_reuse_toks"],
            decision.gpu_prefix_reuse_tokens,
        )
        with self.assertRaisesRegex(
                RuntimeError, "changed retained-KV ownership"):
            adapter.bind_native_gpu_request(
                2, gpu_instance_id=2)

        stale_dispatch, = adapter.drain_astra_dispatches()
        _, stale_completion = complete_job(adapter, stale_dispatch)
        self.assertEqual(
            stale_completion.multiplexed.source_name,
            adapter.lifecycle_source_name,
        )
        self.assertEqual(
            adapter.lifecycle.metrics.migrations_stale, 1)
        self.assertEqual(adapter.pop_gpu_hbm_events(), [])

        adapter.complete_native_gpu_request(
            2,
            completion_ns=adapter.current_ns + 1,
            materialized_tokens=5,
        )
        self.assertEqual(
            adapter.lifecycle.sessions["session-b"].state,
            PlacementState.MIGRATING,
        )
        adapter.assert_invariants()

    def test_pd_resume_mode_recomputes_without_claiming_decode_hbm(self):
        adapter = build_adapter(gpu_resume_mode="recompute")
        adapter.offer_raw_requests((
            raw_request(
                80, "session-pd", 0, 0, 17, 1, 0, True),
        ), now_ns=0)
        adapter.complete_native_gpu_request(
            80,
            completion_ns=1,
            materialized_tokens=17,
            gpu_instance_id=1,
        )
        retain, = adapter.pop_gpu_hbm_events()
        self.assertEqual(retain.kind, GPUHBMEventKind.TURN_RETAIN)

        resume = raw_request(
            81, "session-pd", 1, 2, 17, 1, 17, False)
        decision = adapter.offer_raw_request(resume, now_ns=2)
        self.assertEqual(
            decision.execution,
            OnlineHBFExecution.GPU_RECOMPUTE,
        )
        self.assertEqual(decision.gpu_prefix_reuse_tokens, 0)
        self.assertIsNone(decision.required_gpu_instance_id)
        self.assertTrue(
            decision.route_reason.startswith(
                "gpu_resume_recompute:"))
        release, = adapter.pop_gpu_hbm_events()
        self.assertEqual(release.kind, GPUHBMEventKind.IDLE_RELEASE)
        self.assertEqual(release.gpu_instance_id, 1)
        decorated = adapter.decorate_gpu_metadata(decision, resume)
        self.assertEqual(decorated["agentic_kv_hit_tokens"], 0)
        self.assertEqual(
            decorated["agentic_kv_recompute_tokens"], 17)
        self.assertIsNone(
            decorated["agentic_kv_owner_instance_id"])
        self.assertIsNone(
            decorated["hbf_gpu_required_instance_id"])

    def test_measurement_cutoff_censors_dispatched_gpu_without_migration(self):
        adapter = build_adapter(gpu_resume_mode="recompute")
        adapter.offer_raw_requests((
            raw_request(
                90, "session-cutoff", 0, 0, 17, 2, 0, True),
        ), now_ns=0)
        adapter.censor_active_native_gpu_request(90, now_ns=11)
        call = adapter.calls[90]
        self.assertEqual(call.state, OnlineHBFCallState.COMPLETE)
        self.assertTrue(call.successor_censored)
        self.assertEqual(call.completion_ns, 11)
        self.assertEqual(
            adapter.lifecycle.sessions["session-cutoff"].state,
            PlacementState.ENDED,
        )
        self.assertEqual(adapter.pop_gpu_hbm_events(), [])
        self.assertEqual(adapter.drain_astra_dispatches(), ())
        self.assertEqual(
            adapter.metrics.censored_active_gpu_requests, 1)
        adapter.assert_invariants()

    def test_measurement_cutoff_distinguishes_queued_gpu_from_astra_work(self):
        adapter = build_adapter(gpu_resume_mode="recompute")
        adapter.offer_raw_requests((
            raw_request(
                91, "session-queued", 0, 0, 17, 2, 0, True),
        ), now_ns=0)
        request = SimpleNamespace(
            id=91,
            session_id="session-queued",
            instance_id=0,
        )

        self.assertTrue(adapter.has_pending())
        self.assertTrue(adapter.has_pending_native_gpu_requests())
        self.assertFalse(adapter.has_pending_astra_dispatches())
        audit = adapter.validate_queued_native_gpu_request(
            request, now_ns=11)
        self.assertEqual(audit["execution"], "gpu_first_turn")
        result = adapter.censor_queued_native_gpu_request(
            request, now_ns=11)
        self.assertIsNone(result["retained_gpu_owner_removed"])
        self.assertFalse(adapter.has_pending_native_gpu_requests())
        self.assertFalse(adapter.has_pending())
        self.assertEqual(
            adapter.metrics.censored_queued_gpu_requests, 1)
        adapter.assert_invariants()

    def test_queued_colocated_resume_censor_removes_sticky_owner(self):
        adapter = build_adapter(gpu_resume_mode="sticky_reuse")
        adapter.offer_raw_requests((
            raw_request(
                92, "session-sticky-cutoff", 0, 0, 17, 1, 0, True),
        ), now_ns=0)
        adapter.complete_native_gpu_request(
            92,
            completion_ns=5,
            materialized_tokens=17,
            gpu_instance_id=3,
        )
        adapter.pop_gpu_hbm_events()
        decision = adapter.offer_raw_request(
            raw_request(
                93, "session-sticky-cutoff", 1, 6, 17, 1, 17, False),
            now_ns=6,
        )
        self.assertEqual(decision.required_gpu_instance_id, 3)
        adapter.pop_gpu_hbm_events()
        request = SimpleNamespace(
            id=93,
            session_id="session-sticky-cutoff",
            instance_id=3,
        )

        result = adapter.censor_queued_native_gpu_request(
            request, now_ns=7)
        self.assertEqual(result["retained_gpu_owner_removed"], 3)
        self.assertFalse(adapter.has_pending_native_gpu_requests())
        self.assertTrue(adapter.has_pending_astra_dispatches())
        self.assertEqual(
            adapter.lifecycle.sessions[
                "session-sticky-cutoff"].state,
            PlacementState.ENDED,
        )
        migration, = adapter.drain_astra_dispatches()
        complete_job(adapter, migration)
        self.assertFalse(adapter.has_pending())
        adapter.assert_invariants()

    def test_tied_pool_completion_before_append_callback_is_safe(self):
        adapter = build_adapter()
        adapter.offer_raw_requests((
            raw_request(
                40, "session-tie", 0, 0, 2, 1, 0, True),
        ), now_ns=0)
        adapter.complete_native_gpu_request(
            40, completion_ns=1, materialized_tokens=2,
            gpu_instance_id=3)
        adapter.pop_gpu_hbm_events()
        migration, = adapter.drain_astra_dispatches()
        migration_done, _ = complete_job(adapter, migration)
        adapter.pop_gpu_hbm_events()

        first_resume = raw_request(
            41, "session-tie", 1, migration_done + 1,
            3, 1, 2, True)
        adapter.offer_raw_requests(
            (first_resume,), now_ns=first_resume["arrival_time_ns"])
        first_model, = adapter.drain_astra_dispatches()
        first_done, first_completion = complete_job(
            adapter, first_model)
        first_proxy, = first_completion.router_completions
        self.assertEqual(
            adapter.pop_router_completions(), [first_proxy])

        # The predecessor append remains ASTRA-owned while its zero-gap
        # successor starts from the same logical HBF+LPDDR lineage.
        terminal = raw_request(
            42, "session-tie", 2, first_done,
            3, 1, 3, False)
        terminal_decision, = adapter.offer_raw_requests(
            (terminal,), now_ns=first_done)
        self.assertEqual(
            terminal_decision.route_reason,
            "hbf_append_inflight",
        )
        jobs = adapter.drain_astra_dispatches()
        self.assertEqual(len(jobs), 2)
        pool_job, = (
            job for job in jobs
            if job.source_name == adapter.pool_source_name
        )
        append_job, = (
            job for job in jobs
            if job.source_name == adapter.lifecycle_source_name
        )
        all_runtimes = []
        for job in jobs:
            _, _, stages = job.controller_arguments()
            all_runtimes.append(sum(
                int(stage["runtime_ns"]) for stage in stages))
        tied_completion_ns = (
            max(job.arrival_ns for job in jobs)
            + max(all_runtimes)
            + 1
        )

        # ASTRA is allowed to report either callback first at an exact tie.
        pool_result = adapter.complete_astra_dispatch(
            job_id=pool_job.job_id,
            arrival_ns=pool_job.arrival_ns,
            completion_ns=tied_completion_ns,
            stage_count=pool_job.stage_count,
        )
        terminal_proxy, = pool_result.router_completions
        self.assertEqual(
            adapter.pop_router_completions(), [terminal_proxy])
        self.assertEqual(
            adapter.lifecycle.sessions["session-tie"].state,
            PlacementState.ENDED,
        )
        adapter.complete_astra_dispatch(
            job_id=append_job.job_id,
            arrival_ns=append_job.arrival_ns,
            completion_ns=tied_completion_ns,
            stage_count=append_job.stage_count,
        )
        self.assertEqual(
            adapter.lifecycle.report()["group_reserved_per_card_bytes"],
            {0: 0, 1: 0},
        )
        self.assertFalse(adapter.has_pending())
        adapter.assert_invariants()

    def test_append_callback_refreshes_inflight_successor_placement(self):
        adapter = build_adapter()
        adapter.offer_raw_requests((
            raw_request(
                50, "session-append-first", 0, 0, 2, 1, 0, True),
        ), now_ns=0)
        adapter.complete_native_gpu_request(
            50,
            completion_ns=1,
            materialized_tokens=2,
            gpu_instance_id=3,
        )
        adapter.pop_gpu_hbm_events()
        migration, = adapter.drain_astra_dispatches()
        migration_done, _ = complete_job(adapter, migration)
        adapter.pop_gpu_hbm_events()

        first_resume = raw_request(
            51, "session-append-first", 1, migration_done + 1,
            3, 1, 2, True)
        adapter.offer_raw_requests(
            (first_resume,), now_ns=first_resume["arrival_time_ns"])
        first_model, = adapter.drain_astra_dispatches()
        first_done, first_completion = complete_job(
            adapter, first_model)
        first_proxy, = first_completion.router_completions
        self.assertEqual(
            adapter.pop_router_completions(), [first_proxy])

        terminal = raw_request(
            52, "session-append-first", 2, first_done + 1,
            3, 1, 3, False)
        decision, = adapter.offer_raw_requests(
            (terminal,), now_ns=terminal["arrival_time_ns"])
        self.assertEqual(decision.route_reason, "hbf_append_inflight")
        jobs = adapter.drain_astra_dispatches()
        self.assertEqual(len(jobs), 2)
        append_job, = (
            job for job in jobs
            if job.source_name == adapter.lifecycle_source_name
        )
        pool_job, = (
            job for job in jobs
            if job.source_name == adapter.pool_source_name
        )

        append_done, append_completion = complete_job(
            adapter, append_job)
        self.assertEqual(
            append_completion.multiplexed.source_name,
            adapter.lifecycle_source_name,
        )
        active = adapter.pool.requests[52]
        self.assertEqual(active.hbf_prefix_tokens, 3)
        self.assertEqual(active.lpddr_prefix_tokens, 0)
        self.assertEqual(
            adapter.lifecycle.lpddr_ledger.owner_card_bytes(
                adapter.lifecycle.lpddr_owner(
                    "session-append-first")),
            {},
        )
        self.assertTrue(adapter.pool.has_pending_external_dispatches())

        _, _, stages = pool_job.controller_arguments()
        pool_done = max(
            append_done + 1,
            pool_job.arrival_ns
            + sum(int(stage["runtime_ns"]) for stage in stages)
            + 1,
        )
        pool_completion = adapter.complete_astra_dispatch(
            job_id=pool_job.job_id,
            arrival_ns=pool_job.arrival_ns,
            completion_ns=pool_done,
            stage_count=pool_job.stage_count,
        )
        terminal_proxy, = pool_completion.router_completions
        self.assertEqual(
            adapter.pop_router_completions(), [terminal_proxy])
        self.assertFalse(adapter.has_pending())
        adapter.assert_invariants()

    def test_full_gpu_prefix_hit_adopts_only_input_minus_one_blocks(self):
        adapter = build_adapter()
        adapter.offer_raw_requests((
            raw_request(
                60, "session-block", 0, 0, 17, 1, 0, True),
        ), now_ns=0)
        adapter.complete_native_gpu_request(
            60,
            completion_ns=1,
            materialized_tokens=17,
            gpu_instance_id=6,
        )
        retained, = adapter.pop_gpu_hbm_events()
        self.assertEqual(retained.accounted_tokens_per_rank, 32)

        resume = raw_request(
            61, "session-block", 1, 2, 17, 1, 17, True)
        decision = adapter.offer_raw_request(resume, now_ns=2)
        self.assertEqual(
            decision.execution,
            OnlineHBFExecution.GPU_MIGRATION_INFLIGHT,
        )
        self.assertEqual(decision.operational_prefix_reuse_tokens, 17)
        self.assertEqual(decision.gpu_prefix_reuse_tokens, 16)
        adopted, = adapter.pop_gpu_hbm_events()
        self.assertEqual(adopted.token_count, 16)
        self.assertEqual(adopted.accounted_tokens_per_rank, 16)
        decorated = adapter.decorate_gpu_metadata(decision, resume)
        self.assertEqual(decorated["prefix_reuse_toks"], 16)
        self.assertEqual(decorated["agentic_kv_hit_tokens"], 16)

    def test_lpddr_finish_capacity_fallback_forces_gpu_recompute(self):
        adapter = build_adapter(ledger_capacity_bytes=1)
        adapter.offer_raw_requests((
            raw_request(
                20, "session-c", 0, 0, 4, 2, 0, True),
        ), now_ns=0)
        adapter.complete_native_gpu_request(
            20, completion_ns=1, materialized_tokens=5,
            gpu_instance_id=0)
        adapter.pop_gpu_hbm_events()
        migration, = adapter.drain_astra_dispatches()
        migration_done, _ = complete_job(adapter, migration)
        adapter.pop_gpu_hbm_events()

        resume = raw_request(
            21, "session-c", 1, migration_done + 1,
            input_tokens=6,
            output_tokens=2,
            prefix_reuse_tokens=5,
            has_successor=False,
        )
        decision = adapter.offer_raw_request(
            resume, now_ns=resume["arrival_time_ns"])
        self.assertEqual(
            decision.execution,
            OnlineHBFExecution.GPU_RECOMPUTE,
        )
        self.assertTrue(decision.force_gpu_recompute)
        self.assertEqual(decision.gpu_prefix_reuse_tokens, 0)
        metadata = adapter.decorate_gpu_metadata(decision, resume)
        self.assertEqual(
            metadata["agentic_kv_residency_at_return"],
            PlacementState.HBF_READY.value,
        )
        self.assertEqual(metadata["agentic_kv_source"], "dropped")
        self.assertEqual(
            metadata["hbf_online_execution"],
            OnlineHBFExecution.GPU_RECOMPUTE.value,
        )
        self.assertEqual(
            adapter.lifecycle.sessions["session-c"].state,
            PlacementState.GPU_ACTIVE,
        )
        self.assertFalse(adapter.has_pending_astra_dispatches())

    def test_router_notify_accepts_hbf_completion_proxy(self):
        adapter = build_adapter()
        adapter.offer_raw_requests((
            raw_request(
                30, "session-router", 0, 0, 2, 1, 0, True),
        ), now_ns=0)
        adapter.complete_native_gpu_request(
            30, completion_ns=1, materialized_tokens=2,
            gpu_instance_id=4)
        adapter.pop_gpu_hbm_events()
        migration, = adapter.drain_astra_dispatches()
        migration_done, _ = complete_job(adapter, migration)
        adapter.pop_gpu_hbm_events()
        resume = raw_request(
            31, "session-router", 1, migration_done + 1,
            3, 1, 2, True)
        adapter.offer_raw_requests(
            (resume,), now_ns=resume["arrival_time_ns"])
        model, = adapter.drain_astra_dispatches()
        model_done, result = complete_job(adapter, model)
        proxy, = result.router_completions

        router = Router(
            num_instances=1,
            schedulers=[DummyScheduler()],
            req_num=2,
        )
        router._pending_requests = []
        router._pending_idx = 0
        router._request_to_session = {
            proxy.id: ("session-router", 1)}
        router._deferred_sessions = {
            "session-router": {
                "sub_requests": [
                    {"input_toks": 2, "output_toks": 1},
                    {"input_toks": 3, "output_toks": 1,
                     "tool_duration_ns": 9},
                    {"input_toks": 3, "output_toks": 1},
                ],
                "next_index": 2,
                "id_base": 30,
                "source_session_id": "source-session-router",
                "template_index": 0,
                "epoch": 0,
                "offered_time_ns": 0,
                "admission_time_ns": 0,
                "admission_queue_wait_ns": 0,
            },
        }
        router.notify_request_completed(proxy, model_done)
        next_row, = router._pending_requests
        self.assertEqual(next_row["index"], 32)
        self.assertEqual(next_row["arrival_time_ns"], model_done + 9)
        self.assertEqual(next_row["prefix_reuse_toks"], 3)
        self.assertEqual(next_row["prefix_reuse_source"], "estimated")

    def test_next_wakeup_never_invents_external_completion(self):
        adapter = build_adapter()
        self.assertEqual(
            adapter.next_wakeup_ns(
                10,
                router_arrival_ns=30,
                extra_candidates=(25, 40),
            ),
            25,
        )
        self.assertIsNone(adapter.next_wakeup_ns(10))
        contract = adapter.integration_contract()
        self.assertIn("GPUHBMOwnershipEvent",
                      contract["gpu_hbm_hook"])

    def test_measurement_censor_ends_session_but_drains_stale_migration(self):
        adapter = build_adapter()
        adapter.offer_raw_requests((
            raw_request(
                50, "session-censor", 0, 0, 2, 1, 0, True),
        ), now_ns=0)
        adapter.complete_native_gpu_request(
            50, completion_ns=5, materialized_tokens=2,
            gpu_instance_id=5)
        retain, = adapter.pop_gpu_hbm_events()
        self.assertEqual(retain.kind, GPUHBMEventKind.TURN_RETAIN)
        self.assertIsNone(adapter.censor_completed_successor(
            50, now_ns=5))
        release, = adapter.pop_gpu_hbm_events()
        self.assertEqual(release.kind, GPUHBMEventKind.IDLE_RELEASE)
        self.assertEqual(release.gpu_instance_id, 5)
        self.assertEqual(
            adapter.lifecycle.sessions["session-censor"].state,
            PlacementState.ENDED,
        )
        self.assertTrue(adapter.has_pending_astra_dispatches())

        migration, = adapter.drain_astra_dispatches()
        _, _ = complete_job(adapter, migration)
        self.assertEqual(
            adapter.lifecycle.metrics.migrations_stale, 1)
        self.assertEqual(
            adapter.lifecycle.report()["group_reserved_per_card_bytes"],
            {0: 0, 1: 0},
        )
        self.assertFalse(adapter.has_pending())

    def test_constructor_rejects_independent_lpddr_ledgers(self):
        hardware = HBFServerHardware()
        layout = HBFParallelLayout.for_key("tp4")
        lifecycle = FullModelHBFLifecycle(
            hardware=hardware,
            layout=layout,
            execution_backend="external_astra",
        )
        pool = FullModelHBFServingPool(
            repo_root=REPO_ROOT,
            hardware=hardware,
            layout=layout,
            max_num_batched_tokens=16,
            max_num_seqs=4,
            max_prefill_chunk_tokens=16,
            execution_backend="external_astra",
        )
        with self.assertRaisesRegex(ValueError, "share one LPDDR"):
            FullModelHBFOnlineAdapter(
                lifecycle=lifecycle,
                pool=pool,
            )

    def test_hbf_completion_materializes_native_request_metrics(self):
        adapter = build_adapter()
        adapter.offer_raw_requests((
            raw_request(
                70, "session-report", 0, 0, 2, 1, 0, True),
        ), now_ns=0)
        adapter.complete_native_gpu_request(
            70, completion_ns=1, materialized_tokens=2,
            gpu_instance_id=0)
        adapter.pop_gpu_hbm_events()
        migration, = adapter.drain_astra_dispatches()
        migration_done, _ = complete_job(adapter, migration)
        adapter.pop_gpu_hbm_events()

        resume = raw_request(
            71, "session-report", 1, migration_done + 7,
            3, 3, 2, False)
        resume.update({
            "source_session_id": "trace-session",
            "session_template_index": 4,
            "session_epoch": 2,
            "session_offered_time_ns": 3,
            "session_admission_time_ns": 5,
            "session_admission_queue_wait_ns": 2,
            "prefix_reuse_source": "previous_output",
            "return_gap_type": "tool",
            "return_gap_source": "trace",
            "return_gap_ns": 7,
        })
        adapter.offer_raw_requests(
            (resume,), now_ns=resume["arrival_time_ns"])
        while (
            adapter.calls[71].state
            != OnlineHBFCallState.COMPLETE
        ):
            # The live integration reaches a same-time control barrier after
            # every callback.  That barrier flushes deferred decode
            # scheduling even when no new Router arrival is present.
            adapter.flush_admissions(adapter.current_ns)
            dispatches = adapter.drain_astra_dispatches()
            if not dispatches:
                self.fail(
                    "active HBF request produced no ASTRA dispatch after "
                    "the same-time scheduling barrier")
            for dispatch in dispatches:
                complete_job(adapter, dispatch)
        proxy, = adapter.pop_router_completions()

        request = proxy.materialize_request(
            model="Qwen/Qwen3-30B-A3B-Instruct-2507",
            instance_id=2,
        )
        self.assertEqual(request.id, 71)
        self.assertEqual(request.instance_id, 2)
        self.assertEqual(request.session_id, "session-report")
        self.assertEqual(request.source_session_id, "trace-session")
        self.assertEqual(request.session_template_index, 4)
        self.assertEqual(request.session_epoch, 2)
        self.assertEqual(request.generated_tokens, 3)
        self.assertEqual(request.num_computed_tokens, 5)
        self.assertEqual(request.agentic_kv_source, "hbf")
        self.assertEqual(request.agentic_kv_hit_tokens, 2)
        self.assertEqual(request.agentic_kv_recompute_tokens, 0)
        self.assertEqual(
            request.first_schedule_eligibility_time_ns,
            proxy.admission_ns,
        )
        self.assertEqual(
            request.scheduler_queue_wait_ns,
            request.first_schedule_time_ns - proxy.admission_ns,
        )
        self.assertEqual(
            request.queuing_delay,
            request.first_schedule_time_ns - request.arrival,
        )
        self.assertEqual(len(request.itl), 2)
        self.assertEqual(sum(request.itl), request.latency - request.ttft)
        self.assertEqual(
            request.tpot,
            (request.latency - request.ttft) // 2,
        )
        self.assertEqual(
            request.requested_output_tokens,
            request.generated_tokens,
        )


if __name__ == "__main__":
    unittest.main()
