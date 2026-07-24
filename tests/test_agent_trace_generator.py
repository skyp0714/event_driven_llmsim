import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from workloads.generators import agent_traces


class CharacterTokenizer:
    name_or_path = "test-character-tokenizer"

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [ord(character) for character in text]}


def _otel_span(
    span_id,
    start,
    end,
    input_tokens,
    output_tokens,
    messages,
):
    return {
        "span_id": span_id,
        "name": "chat test/model",
        "start_time": start,
        "end_time": end,
        "attributes": {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": "test/model",
            "gen_ai.usage.input_tokens": input_tokens,
            "gen_ai.usage.output_tokens": output_tokens,
            "gen_ai.input.messages": json.dumps(messages),
        },
    }


class AgentTraceGeneratorTests(unittest.TestCase):
    def test_tracelab_parallel_tools_use_interval_union_or_max(self):
        rows = [
            {
                "session_id": "trace-session",
                "round_index": 0,
                "model": "claude-test",
                "provider": "claude",
                "input_tokens_total": 100,
                "prefix_tokens": 0,
                "newly_append_tokens": 100,
                "output_tokens": 10,
                "tools": [
                    {
                        "emitted_at": "2026-01-01T00:00:00Z",
                        "result_at": "2026-01-01T00:00:03Z",
                        "tool_wall_latency_ms": 3000,
                    },
                    {
                        "emitted_at": "2026-01-01T00:00:01Z",
                        "result_at": "2026-01-01T00:00:05Z",
                        "tool_wall_latency_ms": 4000,
                    },
                ],
            },
            {
                "session_id": "trace-session",
                "round_index": 1,
                "model": "claude-test",
                "provider": "claude",
                "input_tokens_total": 120,
                "prefix_tokens": 100,
                "newly_append_tokens": 20,
                "output_tokens": 8,
                "tools": [],
            },
        ]
        union = agent_traces.convert_tracelab_session(rows, tool_wait_mode="union")
        maximum = agent_traces.convert_tracelab_session(rows, tool_wait_mode="max")

        self.assertEqual(
            union["sub_requests"][0]["tool_duration_ns"], 5_000_000_000
        )
        self.assertEqual(
            maximum["sub_requests"][0]["tool_duration_ns"], 4_000_000_000
        )
        self.assertEqual(
            union["sub_requests"][0]["tool_wait_source"],
            "tracelab_tool_union_fallback",
        )
        second = union["sub_requests"][1]
        self.assertEqual(second["input_toks"], 120)
        self.assertEqual(second["prefix_reuse_toks"], 110)
        self.assertEqual(second["prefix_reuse_source"], "estimated")
        self.assertEqual(second["observed_provider_hit_toks"], 100)
        self.assertEqual(second["tool_duration_ns"], 0)

    def test_tracelab_preserves_human_gap_without_tool_records(self):
        rows = [
            {
                "session_id": "human-gap",
                "round_index": 0,
                "input_tokens_total": 10,
                "prefix_tokens": 0,
                "newly_append_tokens": 10,
                "output_tokens": 2,
                "timing_events": [
                    {"event_type": "user_message", "timestamp": "2026-01-01T00:00:00Z"},
                    {"event_type": "text", "timestamp": "2026-01-01T00:00:10Z"},
                ],
                "tools": [],
            },
            {
                "session_id": "human-gap",
                "round_index": 1,
                "input_tokens_total": 20,
                "prefix_tokens": 10,
                "newly_append_tokens": 10,
                "output_tokens": 2,
                "timing_events": [
                    {"event_type": "user_message", "timestamp": "2026-01-01T01:00:10Z"},
                    {"event_type": "text", "timestamp": "2026-01-01T01:00:11Z"},
                ],
                "tools": [],
            },
        ]
        session = agent_traces.convert_tracelab_session(rows)
        first = session["sub_requests"][0]

        self.assertEqual(first["tool_duration_ns"], 3_600_000_000_000)
        self.assertEqual(first["tool_wait_source"], "tracelab_event_boundary")
        self.assertEqual(first["inter_turn_gap_type"], "human")

    def test_tracelab_event_boundary_excludes_streaming_overlap(self):
        rows = [
            {
                "session_id": "streaming-tool",
                "round_index": 0,
                "input_tokens_total": 10,
                "prefix_tokens": 0,
                "newly_append_tokens": 10,
                "output_tokens": 2,
                "timing_events": [
                    {"event_type": "user_message", "timestamp": "2026-01-01T00:00:00Z"},
                    {"event_type": "tool_call", "timestamp": "2026-01-01T00:00:03Z"},
                    {"event_type": "text", "timestamp": "2026-01-01T00:00:05Z"},
                ],
                "tools": [{
                    "tool_call_id": "tool-1",
                    "emitted_at": "2026-01-01T00:00:03Z",
                    "result_at": "2026-01-01T00:00:20Z",
                }],
            },
            {
                "session_id": "streaming-tool",
                "round_index": 1,
                "input_tokens_total": 20,
                "prefix_tokens": 10,
                "newly_append_tokens": 10,
                "output_tokens": 2,
                "timing_events": [
                    {
                        "event_type": "tool_result",
                        "tool_call_id": "tool-1",
                        "timestamp": "2026-01-01T00:00:20Z",
                    },
                    {"event_type": "text", "timestamp": "2026-01-01T00:00:21Z"},
                ],
                "tools": [],
            },
        ]
        session = agent_traces.convert_tracelab_session(rows)
        first = session["sub_requests"][0]

        self.assertEqual(first["tool_duration_ns"], 15_000_000_000)
        self.assertEqual(first["inter_turn_gap_type"], "tool")

    def test_tracelab_preserves_raw_zero_append_before_replay_normalization(self):
        rows = [{
            "session_id": "zero-append",
            "round_index": 0,
            "input_tokens_total": 1,
            "prefix_tokens": 1,
            "newly_append_tokens": 0,
            "output_tokens": 1,
            "tools": [],
        }]
        audit = agent_traces.ConversionAudit()

        request = agent_traces.convert_tracelab_session(
            rows, audit=audit)["sub_requests"][0]

        self.assertEqual(request["raw_newly_append_toks"], 0)
        self.assertEqual(request["newly_append_toks"], 1)
        self.assertEqual(
            audit.warning_counts["tracelab_zero_append_promoted_to_one"], 1)

    def test_lmcache_shifts_next_pre_gap_and_marks_estimate_without_tokenizer(self):
        rows = [
            {
                "session_id": "lm-session",
                "model": "source-model",
                "input": [{"role": "user", "content": "fix it"}],
                "output_length": 12,
                "pre_gap": 0.0,
            },
            {
                "session_id": "lm-session",
                "model": "source-model",
                "input": [
                    {"role": "user", "content": "fix it"},
                    {"role": "assistant", "content": "checking"},
                    {"role": "tool", "content": "result"},
                ],
                "output_length": 7,
                "pre_gap": 1.25,
            },
        ]
        session = agent_traces.convert_lmcache_session(rows)
        first, second = session["sub_requests"]

        self.assertEqual(first["tool_duration_ns"], 1_250_000_000)
        self.assertEqual(second["tool_duration_ns"], 0)
        self.assertEqual(first["prefix_reuse_source"], "estimated")
        self.assertGreater(second["prefix_reuse_toks"], 0)
        self.assertEqual(
            second["output_token_count_source"], "reported_source_tokenizer"
        )

    def test_lmcache_target_tokenizer_produces_exact_prefix_and_ids(self):
        rows = [
            {
                "session_id": "lm-exact",
                "model": "source-model",
                "input": [{"role": "user", "content": "abc"}],
                "output_length": 4,
                "pre_gap": 0,
            },
            {
                "session_id": "lm-exact",
                "model": "source-model",
                "input": [
                    {"role": "user", "content": "abc"},
                    {"role": "assistant", "content": "def"},
                ],
                "output_length": 5,
                "pre_gap": 0.5,
            },
        ]
        session = agent_traces.convert_lmcache_session(
            rows,
            tokenizer=CharacterTokenizer(),
            target_tokenizer="target-model",
        )
        first, second = session["sub_requests"]

        self.assertEqual(first["prefix_reuse_source"], "exact")
        self.assertEqual(second["prefix_reuse_source"], "exact")
        self.assertEqual(len(second["input_tok_ids"]), second["input_toks"])
        self.assertEqual(
            second["prefix_reuse_toks"], len(first["input_tok_ids"]) + 4
        )
        self.assertEqual(second["prefix_lineage_scope"], "completed_context")
        self.assertTrue(session["trace_metadata"]["mixed_tokenizers"])

    def test_lmcache_without_replayed_output_uses_input_only_lcp(self):
        rows = [
            {
                "session_id": "lm-input-only",
                "model": "source-model",
                "input": [{"role": "user", "content": "abc"}],
                "output_length": 4,
                "pre_gap": 0,
            },
            {
                "session_id": "lm-input-only",
                "model": "source-model",
                "input": [
                    {"role": "user", "content": "abc"},
                    {"role": "user", "content": "retry after error"},
                ],
                "output_length": 5,
                "pre_gap": 0.5,
            },
        ]
        session = agent_traces.convert_lmcache_session(
            rows,
            tokenizer=CharacterTokenizer(),
            target_tokenizer="target-model",
        )
        first, second = session["sub_requests"]

        self.assertEqual(second["prefix_reuse_toks"], len(first["input_tok_ids"]))
        self.assertEqual(
            second["prefix_lineage_scope"],
            "input_only_no_replayed_output",
        )

    def test_lmcache_message_prefix_divergence_is_audited(self):
        rows = [
            {
                "session_id": "lm-diverged",
                "model": "source-model",
                "input": [{"role": "system", "content": "old"}],
                "output_length": 4,
                "pre_gap": 0,
            },
            {
                "session_id": "lm-diverged",
                "model": "source-model",
                "input": [
                    {"role": "system", "content": "new"},
                    {"role": "assistant", "content": "answer"},
                ],
                "output_length": 5,
                "pre_gap": 0.5,
            },
        ]
        audit = agent_traces.ConversionAudit()
        session = agent_traces.convert_lmcache_session(
            rows,
            tokenizer=CharacterTokenizer(),
            target_tokenizer="target-model",
            audit=audit,
        )

        self.assertEqual(
            session["sub_requests"][1]["prefix_lineage_scope"],
            "input_only_message_prefix_diverged",
        )
        self.assertEqual(audit.warning_counts["lmcache_message_prefix_diverged"], 1)

    def test_exgentic_uses_gap_between_llm_spans(self):
        first_messages = [{"role": "user", "content": "task"}]
        second_messages = first_messages + [
            {"role": "assistant", "content": "call tool"},
            {"role": "tool", "content": "done"},
        ]
        row = {
            "session_id": "otel-session",
            "models": ["test/model"],
            "benchmark": "swebench",
            "spans": [
                _otel_span(
                    "span-2",
                    "2026-01-01T00:00:05Z",
                    "2026-01-01T00:00:07Z",
                    130,
                    9,
                    second_messages,
                ),
                _otel_span(
                    "span-1",
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:00:02Z",
                    100,
                    11,
                    first_messages,
                ),
            ],
        }
        session = agent_traces.convert_exgentic_session(row)
        first, second = session["sub_requests"]

        self.assertEqual(first["source_span_id"], "span-1")
        self.assertEqual(first["tool_duration_ns"], 3_000_000_000)
        self.assertEqual(second["tool_duration_ns"], 0)
        self.assertEqual(first["prefix_reuse_source"], "estimated")
        self.assertGreater(second["prefix_reuse_toks"], 0)

    def test_poisson_arrivals_and_manifest_are_deterministic(self):
        rows = [
            {
                "session_id": f"trace-{index}",
                "round_index": 0,
                "input_tokens_total": 10,
                "prefix_tokens": 0,
                "newly_append_tokens": 10,
                "output_tokens": 2,
                "tools": [],
            }
            for index in range(3)
        ]
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            source = directory_path / "source.jsonl"
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            outputs = []
            for output_name in ("one.jsonl", "two.jsonl"):
                output = directory_path / output_name
                args = argparse.Namespace(
                    source_format="tracelab",
                    source=str(source),
                    output=str(output),
                    sps=2.0,
                    seed=17,
                    first_arrival_sec=0.0,
                    max_sessions=0,
                    max_source_rows=0,
                    split="train",
                    hf_config=None,
                    source_revision=None,
                    tokenizer=None,
                    tokenizer_revision=None,
                    trust_remote_code=False,
                    tool_wait_mode="union",
                    tracelab_reuse_mode="eligible",
                    manifest_output=None,
                    strict=True,
                )
                self.assertEqual(agent_traces.run(args), 0)
                emitted = [json.loads(line) for line in output.read_text().splitlines()]
                outputs.append(emitted)
                manifest = json.loads(
                    Path(str(output) + ".manifest.json").read_text()
                )
                self.assertEqual(manifest["validation"]["status"], "passed")
                self.assertEqual(manifest["summary"]["sessions_emitted"], 3)
                self.assertEqual(
                    manifest["source"]["sha256"],
                    hashlib.sha256(source.read_bytes()).hexdigest(),
                )
                self.assertEqual(
                    manifest["source"]["size_bytes"], source.stat().st_size
                )
                self.assertEqual(
                    len(manifest["converter"]["module_sha256"]), 64
                )
                self.assertIn("git_commit", manifest["converter"])
                self.assertEqual(
                    manifest["converter"]["arguments"]["tool_wait_mode"],
                    "union",
                )
                self.assertTrue(
                    manifest["converter"]["arguments"]["strict"]
                )
                self.assertEqual(
                    manifest["summary"]["prefix_reuse_source_counts"],
                    {"estimated": 3},
                )
            arrivals_one = [row["arrival_time_ns"] for row in outputs[0]]
            arrivals_two = [row["arrival_time_ns"] for row in outputs[1]]
            self.assertEqual(arrivals_one, arrivals_two)
            self.assertEqual(arrivals_one[0], 0)
            self.assertEqual(arrivals_one, sorted(arrivals_one))

    def test_remote_source_revision_is_passed_to_huggingface(self):
        calls = []

        def load_dataset(*args, **kwargs):
            calls.append((args, kwargs))
            return [{"row": 1}]

        datasets = SimpleNamespace(load_dataset=load_dataset)
        with patch.dict(sys.modules, {"datasets": datasets}):
            rows = list(agent_traces._load_source_rows(
                "owner/dataset",
                split="train",
                hf_config="subset",
                revision="deadbeef",
                max_rows=0,
            ))

        self.assertEqual(rows, [{"row": 1}])
        self.assertEqual(calls, [(('owner/dataset', 'subset'), {
            "split": "train",
            "streaming": True,
            "revision": "deadbeef",
        })])

    def test_tokenizer_revision_is_passed_to_huggingface(self):
        calls = []

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                calls.append((args, kwargs))
                return "tokenizer"

        transformers = SimpleNamespace(AutoTokenizer=AutoTokenizer)
        with patch.dict(sys.modules, {"transformers": transformers}):
            tokenizer = agent_traces._load_tokenizer(
                "owner/tokenizer",
                False,
                revision="cafebabe",
            )

        self.assertEqual(tokenizer, "tokenizer")
        self.assertEqual(calls, [(('owner/tokenizer',), {
            "use_fast": True,
            "trust_remote_code": False,
            "revision": "cafebabe",
        })])

    def test_local_json_array_and_wrapped_json_are_supported(self):
        rows = [{"session_id": "one", "spans": []}]
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            array_path = directory_path / "array.json"
            wrapped_path = directory_path / "wrapped.json"
            array_path.write_text(json.dumps(rows), encoding="utf-8")
            wrapped_path.write_text(json.dumps({"traces": rows}), encoding="utf-8")

            self.assertEqual(
                list(agent_traces._load_local_rows(array_path)), rows
            )
            self.assertEqual(
                list(agent_traces._load_local_rows(wrapped_path)), rows
            )

    def test_noncontiguous_tracelab_rows_are_spooled_into_one_session(self):
        rows = []
        for session_id, round_index in (("a", 0), ("b", 0), ("a", 1)):
            rows.append({
                "session_id": session_id,
                "round_index": round_index,
                "input_tokens_total": 10 + round_index,
                "prefix_tokens": 10 if round_index else 0,
                "newly_append_tokens": 1 if round_index else 10,
                "output_tokens": 2,
                "tools": [],
            })
        audit = agent_traces.ConversionAudit()
        sessions = list(agent_traces.convert_rows(
            rows, "tracelab", audit=audit, strict=True))
        self.assertEqual([session["session_id"] for session in sessions], ["a", "b"])
        self.assertEqual(len(sessions[0]["sub_requests"]), 2)
        self.assertEqual(audit.source_rows, 3)

    def test_tracelab_composite_identity_disambiguates_raw_session_collision(self):
        rows = []
        for project in ("project-a", "project-b"):
            rows.append({
                "provider": "claude",
                "project": project,
                "session_id": "same-session",
                "round_index": 0,
                "input_tokens_total": 10,
                "prefix_tokens": 0,
                "newly_append_tokens": 10,
                "output_tokens": 2,
                "tools": [],
            })
        audit = agent_traces.ConversionAudit()
        sessions = list(agent_traces.convert_rows(
            rows, "tracelab", audit=audit, strict=True))

        self.assertEqual(len(sessions), 2)
        self.assertEqual(len({session["session_id"] for session in sessions}), 2)
        self.assertTrue(all(
            session["session_id"].startswith("same-session@")
            for session in sessions
        ))
        self.assertEqual(
            audit.warning_counts["source_session_id_collision_disambiguated"], 2)

    def test_tracelab_round_ties_keep_global_ingest_order(self):
        rows = []
        for input_tokens in (11, 12):
            rows.append({
                "provider": "codex",
                "project": "project-a",
                "session_id": "tie-session",
                "round_index": 3,
                "input_tokens_total": input_tokens,
                "prefix_tokens": 10,
                "newly_append_tokens": input_tokens - 10,
                "output_tokens": 2,
                "tools": [],
            })
        session = list(agent_traces.convert_rows(
            rows, "tracelab", strict=True))[0]

        self.assertEqual(
            [request["source_ingest_seq"] for request in session["sub_requests"]],
            [0, 1],
        )
        self.assertEqual(
            [request["input_toks"] for request in session["sub_requests"]],
            [11, 12],
        )

    def test_tracelab_eligible_reuse_is_independent_of_observed_miss(self):
        rows = [
            {
                "session_id": "provider-miss",
                "round_index": 0,
                "input_tokens_total": 100,
                "prefix_tokens": 0,
                "newly_append_tokens": 100,
                "output_tokens": 10,
                "tools": [],
            },
            {
                "session_id": "provider-miss",
                "round_index": 1,
                "input_tokens_total": 120,
                "prefix_tokens": 0,
                "newly_append_tokens": 120,
                "output_tokens": 5,
                "tools": [],
            },
        ]
        eligible = agent_traces.convert_tracelab_session(rows)
        observed = agent_traces.convert_tracelab_session(
            rows, reuse_mode="observed")

        self.assertEqual(
            eligible["sub_requests"][1]["policy_independent_reuse_toks"], 110)
        self.assertEqual(eligible["sub_requests"][1]["prefix_reuse_toks"], 110)
        self.assertEqual(
            eligible["sub_requests"][1]["observed_provider_hit_toks"], 0)
        self.assertEqual(observed["sub_requests"][1]["prefix_reuse_toks"], 0)
        self.assertEqual(observed["sub_requests"][1]["prefix_reuse_source"], "reported")

    def test_tracelab_context_reset_breaks_reuse_lineage(self):
        base = {
            "session_id": "lineage-reset",
            "prefix_tokens": 0,
            "output_tokens": 5,
            "tools": [],
        }
        rows = [
            dict(base, round_index=0, input_tokens_total=100, newly_append_tokens=100),
            dict(base, round_index=1, input_tokens_total=40, newly_append_tokens=40),
            dict(
                base,
                round_index=2,
                input_tokens_total=50,
                newly_append_tokens=10,
                timing_events=[{
                    "event_type": "context_compacted",
                    "source": "runtime.compaction",
                }],
            ),
            dict(
                base,
                round_index=3,
                input_tokens_total=60,
                newly_append_tokens=10,
                timing_events=[{
                    "event_type": "context_reset",
                    "source": "runtime",
                }],
            ),
            dict(base, round_index=5, input_tokens_total=70, newly_append_tokens=10),
        ]
        session = agent_traces.convert_tracelab_session(rows)
        requests = session["sub_requests"]

        self.assertEqual(requests[1]["prefix_reuse_toks"], 0)
        self.assertEqual(requests[1]["lineage_status"], "context_shrink")
        self.assertEqual(requests[2]["prefix_reuse_toks"], 0)
        self.assertEqual(requests[2]["lineage_status"], "explicit_compaction")
        self.assertEqual(requests[3]["prefix_reuse_toks"], 0)
        self.assertEqual(requests[3]["lineage_status"], "explicit_context_reset")
        self.assertEqual(requests[4]["prefix_reuse_toks"], 0)
        self.assertEqual(requests[4]["lineage_status"], "round_gap")

    def test_validation_rejects_nonzero_final_wait(self):
        session = {
            "session_id": "invalid",
            "arrival_time_ns": 0,
            "sub_requests": [
                {
                    "input_toks": 10,
                    "output_toks": 2,
                    "tool_duration_ns": 1,
                    "prefix_reuse_toks": 0,
                    "prefix_reuse_source": "reported",
                }
            ],
        }
        errors = agent_traces.validate_session(session)
        self.assertIn("the final sub-request must have tool_duration_ns=0", errors)


if __name__ == "__main__":
    unittest.main()
