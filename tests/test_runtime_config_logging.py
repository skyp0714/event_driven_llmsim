from argparse import Namespace
from io import StringIO
import unittest

from rich.console import Console

from serving import __main__ as serving_main
from serving.core import logger as serving_logger


class RuntimeConfigLoggingTest(unittest.TestCase):
    def _args(self):
        return Namespace(
            cluster_config="cluster.json",
            run_id="run",
            inputs_root="inputs",
            dataset="workload.jsonl",
            num_req=0,
            max_model_len=None,
            max_num_seqs=128,
            max_num_batched_tokens=2048,
            long_prefill_token_threshold=0,
            block_size=16,
            dtype="bfloat16",
            kv_cache_dtype="auto",
            request_routing_policy="LOAD",
            expert_routing_policy="BALANCED",
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            prefix_storage="None",
            enable_prefix_sharing=False,
            enable_local_offloading=False,
            enable_attn_offloading=False,
            enable_sub_batch_interleaving=False,
            enable_block_copy=True,
            prioritize_prefill=False,
            latency_model=None,
            latency_model_band="central",
            network_backend="analytical",
            log_interval=1.0,
            log_level="WARNING",
        )

    def _instances(self):
        common = {
            "model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "max_model_len": 1_010_000,
            "max_num_batched_tokens": 131_072,
            "long_prefill_token_threshold": 131_072,
            "block_size": 16,
            "dtype": "bfloat16",
            "kv_cache_dtype": "auto",
            "enable_chunked_prefill": True,
            "enable_prefix_caching": False,
            "latency_model": "h100-qwen3-tp4-kernel-calibrated",
        }
        return [
            {**common, "instance_id": 0, "pd_type": "prefill",
             "max_num_seqs": 32},
            {**common, "instance_id": 1, "pd_type": "decode",
             "max_num_seqs": 128},
        ]

    def test_cluster_values_override_cli_fallbacks(self):
        runtime = serving_main._build_instance_runtime_configs(
            self._instances(), self._args(), {"bfloat16": 16})

        self.assertEqual(
            [config["max_num_batched_tokens"] for config in runtime],
            [131_072, 131_072],
        )
        self.assertEqual(
            [config["long_prefill_token_threshold"] for config in runtime],
            [131_072, 131_072],
        )
        self.assertEqual(
            [config["max_num_seqs"] for config in runtime],
            [32, 128],
        )

    def test_startup_log_shows_effective_instance_values(self):
        instances = self._instances()
        runtime = serving_main._build_instance_runtime_configs(
            instances, self._args(), {"bfloat16": 16})
        output = StringIO()
        previous_console = serving_logger._console
        serving_logger._console = Console(
            file=output, force_terminal=False, width=240)
        try:
            serving_logger.print_input_config(
                self._args(), instances=instances,
                instance_runtime_configs=runtime)
        finally:
            serving_logger._console = previous_console

        rendered = output.getvalue()
        self.assertNotIn("Max batched tokens", rendered)
        self.assertIn("Effective per-instance runtime", rendered)
        self.assertIn("Instance 0 (prefill)", rendered)
        self.assertIn("Instance 1 (decode)", rendered)
        self.assertEqual(
            rendered.count("max_num_batched_tokens=131072"), 2)
        self.assertEqual(
            rendered.count("long_prefill_token_threshold=131072"), 2)
        self.assertIn("max_num_seqs=32", rendered)
        self.assertIn("max_num_seqs=128", rendered)

    def test_effective_logging_arguments_are_paired(self):
        with self.assertRaisesRegex(ValueError, "supplied together"):
            serving_logger.print_input_config(
                self._args(), instances=self._instances())


if __name__ == "__main__":
    unittest.main()
