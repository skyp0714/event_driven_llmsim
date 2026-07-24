import dataclasses
from pathlib import Path
import unittest

from serving.core.hbf_mirrored_p4d4_config import (
    DECODE_ROLE,
    GIB,
    MIN_PREFILL_WORKSPACE_BYTES_PER_CARD,
    PREFILL_ROLE,
    MirroredP4D4Config,
    MirroredRoleConfig,
    load_mirrored_p4d4_config,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    REPO_ROOT
    / "configs/wakekv_hbf/full_model_mirrored_p4d4_server.json"
)


class MirroredP4D4ConfigTests(unittest.TestCase):
    def test_checked_in_contract_exposes_physical_role_limits(self):
        config = load_mirrored_p4d4_config(CONFIG)

        self.assertEqual(config.prefill.card_ids, (0, 1, 2, 3))
        self.assertEqual(config.decode.card_ids, (4, 5, 6, 7))
        self.assertEqual(config.prefill.workspace_bytes_per_card, 16 * GIB)
        self.assertEqual(config.decode.workspace_bytes_per_card, 16 * GIB)
        self.assertEqual(config.prefill.max_num_seqs, 128)
        self.assertEqual(config.decode.max_num_seqs, 128)
        self.assertEqual(
            config.storage.write_chunk_bytes_per_card, 64 * 1024 ** 2)
        self.assertEqual(config.pcie_handoff.prefill_root_id, 0)
        self.assertEqual(config.pcie_handoff.decode_root_id, 1)

    def test_max_active_kv_is_exact_tp4_per_card_geometry(self):
        config = load_mirrored_p4d4_config(CONFIG)

        self.assertEqual(config.prefill.max_active_kv_tokens, 131_072)
        self.assertEqual(
            config.prefill.max_active_kv_bytes_per_card,
            3 * GIB,
        )
        report = config.report()
        self.assertEqual(
            report["roles"][PREFILL_ROLE][
                "minimum_lpddr_contract_bytes_per_card"],
            19 * GIB,
        )
        self.assertEqual(
            report["roles"][DECODE_ROLE][
                "minimum_lpddr_contract_bytes_per_card"],
            19 * GIB,
        )

    def test_prefill_workspace_has_conservative_lower_bound(self):
        config = load_mirrored_p4d4_config(CONFIG)
        too_small = dataclasses.replace(
            config.prefill,
            workspace_bytes_per_card=(
                MIN_PREFILL_WORKSPACE_BYTES_PER_CARD - 1),
        )
        with self.assertRaisesRegex(ValueError, "at least 10 GiB"):
            too_small.validate(role=PREFILL_ROLE)

    def test_role_lpddr_must_fit_workspace_and_max_active_kv(self):
        config = load_mirrored_p4d4_config(CONFIG)
        exact = dataclasses.replace(
            config.prefill,
            lpddr_capacity_bytes_per_card=(
                config.prefill.workspace_bytes_per_card
                + config.prefill.max_active_kv_bytes_per_card
            ),
        )
        exact.validate(role=PREFILL_ROLE)

        undersized = dataclasses.replace(
            exact,
            lpddr_capacity_bytes_per_card=(
                exact.lpddr_capacity_bytes_per_card - 1),
        )
        with self.assertRaisesRegex(
                ValueError, "workspace plus max_active_kv_tokens"):
            undersized.validate(role=PREFILL_ROLE)

    def test_p_and_d_lpddr_are_independently_parameterized(self):
        config = load_mirrored_p4d4_config(CONFIG)
        varied = MirroredP4D4Config(
            hardware=config.hardware,
            prefill=dataclasses.replace(
                config.prefill,
                lpddr_capacity_bytes_per_card=80 * GIB,
            ),
            decode=dataclasses.replace(
                config.decode,
                lpddr_capacity_bytes_per_card=48 * GIB,
            ),
            pcie_handoff=config.pcie_handoff,
            storage=config.storage,
        )
        varied.validate()
        self.assertEqual(
            varied.prefill.lpddr_capacity_bytes_per_card, 80 * GIB)
        self.assertEqual(
            varied.decode.lpddr_capacity_bytes_per_card, 48 * GIB)

    def test_unique_capacity_is_one_tp4_role_not_eight_cards(self):
        config = load_mirrored_p4d4_config(CONFIG)
        per_role = (
            config.usable_hbf_bytes_per_card
            * config.hardware.cards_per_role
        )
        self.assertEqual(config.unique_logical_capacity_bytes, per_role)
        self.assertEqual(
            config.physical_persistent_capacity_bytes, 2 * per_role)

    def test_overlapping_role_cards_fail_closed(self):
        config = load_mirrored_p4d4_config(CONFIG)
        overlap = dataclasses.replace(
            config,
            decode=dataclasses.replace(
                config.decode, card_ids=(3, 4, 5, 6)),
        )
        with self.assertRaisesRegex(
                ValueError, "disjoint and cover all cards|overlap"):
            overlap.validate()

    def test_role_rejects_sequence_limit_above_calibration(self):
        role = MirroredRoleConfig(card_ids=(0, 1, 2, 3))
        invalid = dataclasses.replace(role, max_num_seqs=129)
        with self.assertRaisesRegex(ValueError, "calibrated support"):
            invalid.validate(role=PREFILL_ROLE)


if __name__ == "__main__":
    unittest.main()
