#!/usr/bin/env python3
"""Unit tests for the determine_verdict function in scripts/eval_candidate.py.

Tests the SRCH-006 heavy position anti-masking guard and TOOL-004 regression guard
relaxation & simplification policy alongside existing verdict logic.
"""

import os
import sys
import unittest

# Add scripts/ to path so we can import eval_candidate
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from eval_candidate import (
    HEAVY_POSITIONS,
    HEAVY_REGRESSION_THRESHOLD_PCT,
    SIMPLIFICATION_THRESHOLD_PCT,
    determine_verdict,
    generate_markdown_table,
    generate_text_table,
)


def _make_summary(total_node_delta_pct):
    """Helper: create a minimal summary dict with the given aggregate node delta."""
    return {
        "total_candidate_nodes": 34_000_000_000,
        "total_baseline_nodes": 34_400_000_000,
        "total_node_delta_pct": total_node_delta_pct,
        "total_candidate_time": 100.0,
        "total_baseline_time": 105.0,
        "total_time_delta_pct": -4.76,
    }


def _make_comparison(**overrides):
    """Helper: create a comparison dict with per-position node deltas.

    By default, all positions are neutral (0.0% delta).
    Pass keyword overrides as position_name=delta_pct (e.g., ffo55=+1.5).
    """
    # Map short names to FFO names
    name_map = {
        "ffo40": "FFO #40", "ffo41": "FFO #41", "ffo42": "FFO #42",
        "ffo43": "FFO #43", "ffo44": "FFO #44", "ffo45": "FFO #45",
        "ffo46": "FFO #46", "ffo47": "FFO #47", "ffo48": "FFO #48",
        "ffo49": "FFO #49", "ffo50": "FFO #50", "ffo51": "FFO #51",
        "ffo52": "FFO #52", "ffo53": "FFO #53", "ffo54": "FFO #54",
        "ffo55": "FFO #55", "ffo56": "FFO #56", "ffo57": "FFO #57",
        "ffo59": "FFO #59",
    }
    comparison = {}
    for short, ffo_name in name_map.items():
        delta = overrides.get(short, 0.0)
        comparison[ffo_name] = {
            "candidate_nodes": 1_000_000,
            "baseline_nodes": 1_000_000,
            "node_delta_pct": delta,
            "candidate_time": 1.0,
            "baseline_time": 1.0,
            "time_delta_pct": 0.0,
        }
    return comparison


class TestDetermineVerdictBasic(unittest.TestCase):
    """Tests for basic verdict logic."""

    def test_timeout_takes_priority(self):
        verdict, reason = determine_verdict(
            "full", True, True, _make_summary(-2.0),
            timed_out_positions=["FFO #55"]
        )
        self.assertEqual(verdict, "REJECT_TIMEOUT")

    def test_correctness_failure(self):
        verdict, _ = determine_verdict("full", False, True, _make_summary(-2.0))
        self.assertEqual(verdict, "REJECT_CORRECTNESS")

        verdict, _ = determine_verdict("full", True, False, _make_summary(-2.0))
        self.assertEqual(verdict, "REJECT_CORRECTNESS")

    def test_no_baseline(self):
        verdict, _ = determine_verdict("full", True, True, None)
        self.assertEqual(verdict, "ACCEPT_NO_BASELINE")

    def test_aggregate_regression(self):
        verdict, reason = determine_verdict("full", True, True, _make_summary(+0.8))
        self.assertEqual(verdict, "REJECT_REGRESSION")
        self.assertIn("+0.80%", reason)

    def test_screen_needs_full(self):
        verdict, _ = determine_verdict("screen", True, True, _make_summary(-1.0))
        self.assertEqual(verdict, "NEEDS_FULL")

    def test_full_accept(self):
        verdict, _ = determine_verdict("full", True, True, _make_summary(-1.0))
        self.assertEqual(verdict, "ACCEPT")

    def test_neutral(self):
        verdict, _ = determine_verdict("full", True, True, _make_summary(0.1))
        self.assertEqual(verdict, "NEUTRAL")

    def test_neutral_boundary(self):
        verdict, _ = determine_verdict("full", True, True, _make_summary(0.5))
        self.assertEqual(verdict, "NEUTRAL")

        verdict, _ = determine_verdict("full", True, True, _make_summary(-0.5))
        self.assertEqual(verdict, "NEUTRAL")


class TestHeavyPositionGuard(unittest.TestCase):
    """Tests for heavy position anti-masking guard with relaxed 3.0% threshold (TOOL-004)."""

    def test_heavy_regression_rejects_despite_aggregate_improvement(self):
        """Core anti-masking scenario: aggregate is great (-2.0%) but FFO #55 regressed +3.5% (> 3.0%)."""
        comparison = _make_comparison(ffo55=3.5)
        verdict, reason = determine_verdict(
            "full", True, True, _make_summary(-2.0),
            comparison=comparison
        )
        self.assertEqual(verdict, "REJECT_REGRESSION")
        self.assertIn("FFO #55", reason)
        self.assertIn("anti-masking guard", reason)
        self.assertIn("+3.50%", reason)
        self.assertIn("> +3.0%", reason)

    def test_multiple_heavy_regressions(self):
        """Multiple heavy positions regressing > 3.0% should all be listed."""
        comparison = _make_comparison(ffo53=3.5, ffo55=4.0, ffo57=5.0)
        verdict, reason = determine_verdict(
            "full", True, True, _make_summary(-2.0),
            comparison=comparison
        )
        self.assertEqual(verdict, "REJECT_REGRESSION")
        self.assertIn("FFO #53", reason)
        self.assertIn("FFO #55", reason)
        self.assertIn("FFO #57", reason)

    def test_heavy_position_within_threshold_passes(self):
        """Heavy positions at or below default threshold (3.0%) should not trigger guard."""
        comparison = _make_comparison(ffo55=2.5, ffo53=3.0)  # 3.0 is not > 3.0
        verdict, _ = determine_verdict(
            "full", True, True, _make_summary(-2.0),
            comparison=comparison
        )
        self.assertEqual(verdict, "ACCEPT")

    def test_relaxed_threshold_allows_minor_variance(self):
        """Verify that regressions between 1.0% and 3.0% pass under the new TOOL-004 default."""
        comparison = _make_comparison(ffo55=1.8, ffo57=2.5)
        verdict, _ = determine_verdict(
            "full", True, True, _make_summary(-1.5),
            comparison=comparison
        )
        self.assertEqual(verdict, "ACCEPT")

    def test_non_heavy_position_regression_ignored_by_guard(self):
        """Non-heavy positions (e.g. FFO #45) regressing > 3.0% should NOT trigger the heavy guard."""
        comparison = _make_comparison(ffo45=5.0)
        verdict, _ = determine_verdict(
            "full", True, True, _make_summary(-2.0),
            comparison=comparison
        )
        # Should pass because FFO #45 is not a heavy position
        self.assertEqual(verdict, "ACCEPT")

    def test_heavy_guard_applies_in_screen_mode(self):
        """Heavy guard should trigger even in screen mode when regression > 3.0%."""
        comparison = _make_comparison(ffo55=3.5)
        verdict, reason = determine_verdict(
            "screen", True, True, _make_summary(-2.0),
            comparison=comparison
        )
        self.assertEqual(verdict, "REJECT_REGRESSION")
        self.assertIn("anti-masking guard", reason)

    def test_custom_heavy_threshold(self):
        """Custom heavy_threshold should override the default."""
        comparison = _make_comparison(ffo55=2.5)

        # With lower custom threshold (2.0%), this should reject
        verdict, _ = determine_verdict(
            "full", True, True, _make_summary(-2.0),
            comparison=comparison, heavy_threshold=2.0
        )
        self.assertEqual(verdict, "REJECT_REGRESSION")

        # With higher custom threshold (4.0%), this should accept
        verdict, _ = determine_verdict(
            "full", True, True, _make_summary(-2.0),
            comparison=comparison, heavy_threshold=4.0
        )
        self.assertEqual(verdict, "ACCEPT")

    def test_no_comparison_skips_guard(self):
        """When comparison is None, heavy guard is skipped (backward compat)."""
        verdict, _ = determine_verdict(
            "full", True, True, _make_summary(-2.0),
            comparison=None
        )
        self.assertEqual(verdict, "ACCEPT")

    def test_timeout_still_takes_priority_over_heavy_guard(self):
        """Timeout should still win over heavy position regression."""
        comparison = _make_comparison(ffo55=5.0)
        verdict, _ = determine_verdict(
            "full", True, True, _make_summary(-2.0),
            timed_out_positions=["FFO #55"],
            comparison=comparison
        )
        self.assertEqual(verdict, "REJECT_TIMEOUT")

    def test_correctness_still_takes_priority_over_heavy_guard(self):
        """Correctness failure should still win over heavy position regression."""
        comparison = _make_comparison(ffo55=5.0)
        verdict, _ = determine_verdict(
            "full", False, True, _make_summary(-2.0),
            comparison=comparison
        )
        self.assertEqual(verdict, "REJECT_CORRECTNESS")

    def test_heavy_guard_before_aggregate_regression(self):
        """When both aggregate and heavy regress, heavy guard message should appear (checked first)."""
        comparison = _make_comparison(ffo55=3.5)
        verdict, reason = determine_verdict(
            "full", True, True, _make_summary(+1.0),
            comparison=comparison
        )
        self.assertEqual(verdict, "REJECT_REGRESSION")
        # Heavy guard fires first, so reason should mention anti-masking
        self.assertIn("anti-masking guard", reason)

    def test_heavy_positions_set_correct(self):
        """Verify the HEAVY_POSITIONS constant contains the expected positions."""
        self.assertEqual(HEAVY_POSITIONS, {"FFO #53", "FFO #54", "FFO #55", "FFO #57"})

    def test_default_threshold(self):
        """Verify the default threshold constant is 3.0% (relaxed from 1.0% by TOOL-004)."""
        self.assertEqual(HEAVY_REGRESSION_THRESHOLD_PCT, 3.0)


class TestSimplificationMode(unittest.TestCase):
    """Tests for TOOL-004 simplification mode and relaxed policy."""

    def test_simplification_threshold_constant(self):
        """Verify the simplification threshold constant is 5.0%."""
        self.assertEqual(SIMPLIFICATION_THRESHOLD_PCT, 5.0)

    def test_simplification_mode_allows_heavy_regression_up_to_5_pct(self):
        """In simplification mode, heavy position regressions up to 5.0% should pass."""
        comparison = _make_comparison(ffo55=4.0)

        # In standard mode (threshold=3.0%), +4.0% regresses
        verdict_std, _ = determine_verdict(
            "full", True, True, _make_summary(-1.0),
            comparison=comparison, simplification_mode=False
        )
        self.assertEqual(verdict_std, "REJECT_REGRESSION")

        # In simplification mode (threshold=5.0%), +4.0% passes
        verdict_simp, _ = determine_verdict(
            "full", True, True, _make_summary(-1.0),
            comparison=comparison, simplification_mode=True
        )
        self.assertEqual(verdict_simp, "ACCEPT")

    def test_simplification_mode_rejects_heavy_regression_above_5_pct(self):
        """In simplification mode, heavy position regression > 5.0% must still reject."""
        comparison = _make_comparison(ffo55=5.5)
        verdict, reason = determine_verdict(
            "full", True, True, _make_summary(-1.0),
            comparison=comparison, simplification_mode=True
        )
        self.assertEqual(verdict, "REJECT_REGRESSION")
        self.assertIn("> +5.0%", reason)
        self.assertIn("FFO #55", reason)
        self.assertIn("anti-masking guard", reason)

    def test_simplification_mode_accepts_neutral_aggregate(self):
        """Simplification mode accepts net neutral aggregate nodes (e.g. 0.0%, within +/-0.5%)."""
        summary = _make_summary(0.0)

        # Standard mode returns NEUTRAL
        verdict_std, _ = determine_verdict("full", True, True, summary, simplification_mode=False)
        self.assertEqual(verdict_std, "NEUTRAL")

        # Simplification mode returns ACCEPT
        verdict_simp, reason = determine_verdict("full", True, True, summary, simplification_mode=True)
        self.assertEqual(verdict_simp, "ACCEPT")
        self.assertIn("Simplification accepted", reason)

    def test_simplification_mode_accepts_slight_reduction(self):
        """Simplification mode accepts small node reductions (e.g. -0.2%) as ACCEPT."""
        summary = _make_summary(-0.2)
        verdict, reason = determine_verdict("full", True, True, summary, simplification_mode=True)
        self.assertEqual(verdict, "ACCEPT")
        self.assertIn("Simplification accepted", reason)

    def test_simplification_mode_needs_full_in_screen_mode(self):
        """In screen mode with simplification, neutral results should return NEEDS_FULL."""
        summary = _make_summary(0.1)
        verdict, reason = determine_verdict("screen", True, True, summary, simplification_mode=True)
        self.assertEqual(verdict, "NEEDS_FULL")
        self.assertIn("Proceed to --mode full", reason)

    def test_simplification_mode_rejects_aggregate_regression_over_half_pct(self):
        """Even in simplification mode, aggregate regression > +0.5% must reject."""
        summary = _make_summary(+0.8)
        verdict, reason = determine_verdict("full", True, True, summary, simplification_mode=True)
        self.assertEqual(verdict, "REJECT_REGRESSION")
        self.assertIn("+0.80%", reason)

    def test_simplification_mode_preserves_correctness_invariant(self):
        """Correctness failure must still reject in simplification mode."""
        verdict, _ = determine_verdict("full", False, True, _make_summary(-1.0), simplification_mode=True)
        self.assertEqual(verdict, "REJECT_CORRECTNESS")

        verdict, _ = determine_verdict("full", True, False, _make_summary(-1.0), simplification_mode=True)
        self.assertEqual(verdict, "REJECT_CORRECTNESS")

    def test_simplification_mode_preserves_timeout_invariant(self):
        """Timeout must still reject in simplification mode."""
        verdict, _ = determine_verdict(
            "full", True, True, _make_summary(-1.0),
            timed_out_positions=["FFO #55"], simplification_mode=True
        )
        self.assertEqual(verdict, "REJECT_TIMEOUT")

    def test_simplification_mode_with_custom_heavy_threshold(self):
        """Explicit heavy_threshold overrides the 5.0% default in simplification mode."""
        comparison = _make_comparison(ffo55=4.5)
        verdict, reason = determine_verdict(
            "full", True, True, _make_summary(-1.0),
            comparison=comparison, heavy_threshold=4.0, simplification_mode=True
        )
        self.assertEqual(verdict, "REJECT_REGRESSION")
        self.assertIn("> +4.0%", reason)

    def test_early_halt_respects_simplification_threshold(self):
        """Early halt message correctly reflects simplification heavy threshold."""
        verdict, reason = determine_verdict(
            "full", True, True, _make_summary(-1.0),
            early_halted=("FFO #55", 5.2), simplification_mode=True
        )
        self.assertEqual(verdict, "REJECT_REGRESSION")
        self.assertIn("> +5.0% limit", reason)


class TestTableGeneration(unittest.TestCase):
    """Tests for text and markdown table warning tags with configurable thresholds."""

    def test_text_table_heavy_tag(self):
        """Text table displays [HEAVY!] only when position exceeds threshold."""
        comparison = _make_comparison(ffo55=2.0)
        summary = _make_summary(-1.0)

        # At default 3.0% threshold: no [HEAVY!] tag for +2.0%
        text_def = generate_text_table(comparison, summary, "full", 1, 22, heavy_threshold=3.0)
        self.assertNotIn("[HEAVY!]", text_def)

        # At 1.0% threshold: [HEAVY!] tag present
        text_1pct = generate_text_table(comparison, summary, "full", 1, 22, heavy_threshold=1.0)
        self.assertIn("[HEAVY!]", text_1pct)

    def test_markdown_table_heavy_warning(self):
        """Markdown table displays warning emoji only when position exceeds threshold."""
        comparison = _make_comparison(ffo55=2.0)
        summary = _make_summary(-1.0)

        # At default 3.0% threshold: no warning emoji
        md_def = generate_markdown_table(comparison, summary, "full", 1, 22, heavy_threshold=3.0)
        self.assertNotIn("⚠️", md_def)

        # At 1.0% threshold: warning emoji present
        md_1pct = generate_markdown_table(comparison, summary, "full", 1, 22, heavy_threshold=1.0)
        self.assertIn("⚠️", md_1pct)


if __name__ == "__main__":
    unittest.main()
