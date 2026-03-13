"""
Unit tests for universal_solve and its helpers.

Run with:  pytest empirical-validation/tests/test_universal_solve.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make empirical-validation directory importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimizers.universal_solve import (
    MU8_ORDER,
    enumerate_all,
    execute,
    proven_and_cheap,
    retry_next,
    universal_solve,
)
from optimizers.mu8_cycle_optimizer import lean_coherence, PHASE_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# 1. Path enumeration
# ─────────────────────────────────────────────────────────────────────────────

class TestEnumerateAll:
    """enumerate_all returns well-formed, deterministic candidates."""

    PROBLEM = "mu8_identity"

    def test_returns_list(self):
        paths = enumerate_all(self.PROBLEM)
        assert isinstance(paths, list)

    def test_non_empty(self):
        paths = enumerate_all(self.PROBLEM)
        assert len(paths) > 0

    def test_expected_count(self):
        # Should return max(MU8_ORDER, len(PHASE_NAMES) * 2) = 10 candidates.
        paths = enumerate_all(self.PROBLEM)
        expected = max(MU8_ORDER, len(PHASE_NAMES) * 2)
        assert len(paths) == expected

    def test_path_keys(self):
        paths = enumerate_all(self.PROBLEM)
        for p in paths:
            assert "id" in p
            assert "steps" in p
            assert "cost" in p
            assert "proven" in p

    def test_unique_ids(self):
        paths = enumerate_all(self.PROBLEM)
        ids = [p["id"] for p in paths]
        assert len(ids) == len(set(ids))

    def test_steps_non_empty(self):
        paths = enumerate_all(self.PROBLEM)
        for p in paths:
            assert len(p["steps"]) >= 1

    def test_steps_drawn_from_phase_names(self):
        paths = enumerate_all(self.PROBLEM)
        for p in paths:
            for step in p["steps"]:
                assert step in PHASE_NAMES

    def test_cost_positive(self):
        paths = enumerate_all(self.PROBLEM)
        for p in paths:
            assert p["cost"] > 0.0

    def test_proven_field_is_bool(self):
        paths = enumerate_all(self.PROBLEM)
        for p in paths:
            assert isinstance(p["proven"], bool)

    def test_at_least_one_proven(self):
        """At least one candidate must be proven (complete μ⁸ revolution)."""
        paths = enumerate_all(self.PROBLEM)
        proven = [p for p in paths if p["proven"]]
        assert len(proven) >= 1

    def test_proven_step_count_divisible_by_mu8(self):
        paths = enumerate_all(self.PROBLEM)
        for p in paths:
            if p["proven"]:
                assert len(p["steps"]) % MU8_ORDER == 0

    def test_deterministic_same_problem(self):
        """Same problem → same paths in same order."""
        paths1 = enumerate_all(self.PROBLEM)
        paths2 = enumerate_all(self.PROBLEM)
        assert paths1 == paths2

    def test_different_problems_different_paths(self):
        """Different problems → different path ordering (by cost at minimum)."""
        paths_a = enumerate_all("problem_alpha")
        paths_b = enumerate_all("problem_beta")
        # Not necessarily all different, but the first IDs should differ.
        costs_a = [p["cost"] for p in paths_a]
        costs_b = [p["cost"] for p in paths_b]
        # They should differ in at least one cost value (different seeds).
        assert costs_a != costs_b

    def test_toy_problem_candidates(self):
        """Toy problem: check the first candidate has expected structure."""
        paths = enumerate_all("toy")
        first = paths[0]
        assert first["id"] == "p0"
        assert isinstance(first["steps"], list)
        assert first["cost"] > 0.0

    def test_non_string_problem(self):
        """enumerate_all accepts non-string problems (uses str() internally)."""
        paths_int = enumerate_all(42)
        paths_dict = enumerate_all({"key": "val"})
        assert len(paths_int) > 0
        assert len(paths_dict) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 2. proven_and_cheap scoring
# ─────────────────────────────────────────────────────────────────────────────

class TestProvenAndCheap:
    """proven_and_cheap scores paths correctly."""

    def _make_path(self, proven: bool, cost: float, pid: str = "px") -> dict:
        return {"id": pid, "steps": [], "cost": cost, "proven": proven}

    def test_proven_scores_higher_than_unproven_same_cost(self):
        proven = self._make_path(True, 8.0)
        unproven = self._make_path(False, 8.0)
        assert proven_and_cheap(proven) > proven_and_cheap(unproven)

    def test_cheaper_proven_beats_expensive_proven(self):
        cheap = self._make_path(True, 1.0)
        expensive = self._make_path(True, 100.0)
        assert proven_and_cheap(cheap) > proven_and_cheap(expensive)

    def test_cheaper_unproven_beats_expensive_unproven(self):
        cheap = self._make_path(False, 1.0)
        expensive = self._make_path(False, 100.0)
        assert proven_and_cheap(cheap) > proven_and_cheap(expensive)

    def test_returns_float(self):
        p = self._make_path(True, 8.0)
        assert isinstance(proven_and_cheap(p), float)

    def test_proof_bonus_equals_lean_coherence_at_one(self):
        """Proof bonus = lean_coherence(1.0) = 1.0."""
        # With zero cost the score for a proven path should equal 1.0.
        p = self._make_path(True, 0.0)
        expected = lean_coherence(1.0)  # = 1.0
        assert abs(proven_and_cheap(p) - expected) < 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# 3. execute (default implementation)
# ─────────────────────────────────────────────────────────────────────────────

class TestExecute:
    """Default execute() returns True for proven paths, False otherwise."""

    def test_proven_path_succeeds(self):
        p = {"id": "p0", "steps": ["coherence"] * 8, "cost": 8.0, "proven": True}
        assert execute(p) is True

    def test_unproven_path_fails(self):
        p = {"id": "p1", "steps": ["coherence"], "cost": 1.0, "proven": False}
        assert execute(p) is False


# ─────────────────────────────────────────────────────────────────────────────
# 4. universal_solve — best candidate selected
# ─────────────────────────────────────────────────────────────────────────────

class TestUniversalSolve:
    """universal_solve selects the best candidate by proven_and_cheap."""

    def test_returns_dict_on_success(self):
        result = universal_solve("mu8_identity")
        assert result is not None
        assert isinstance(result, dict)

    def test_result_is_proven(self):
        """Default executor: only proven paths succeed, so the result is proven."""
        result = universal_solve("mu8_identity")
        assert result is not None
        assert result["proven"] is True

    def test_result_has_required_keys(self):
        result = universal_solve("mu8_identity")
        assert result is not None
        for key in ("id", "steps", "cost", "proven"):
            assert key in result

    def test_returns_best_among_all_succeed(self):
        """When all paths succeed, the returned path is the highest-scored."""
        def always_succeed(path):
            return True

        paths = enumerate_all("any_problem")
        best = max(paths, key=proven_and_cheap)
        result = universal_solve("any_problem", _execute=always_succeed)
        assert result is not None
        assert result["id"] == best["id"]

    def test_returns_none_when_all_fail(self):
        """When all paths fail, universal_solve returns None."""
        def always_fail(path):
            return False

        result = universal_solve("hard_problem", _execute=always_fail)
        assert result is None

    def test_skips_failed_candidates(self):
        """universal_solve moves on when a candidate fails and finds the next."""
        paths = enumerate_all("skip_test")
        ranked = sorted(paths, key=proven_and_cheap, reverse=True)
        best_id = ranked[0]["id"]
        second_id = ranked[1]["id"] if len(ranked) > 1 else None

        # Fail only the best candidate, succeed on everything else.
        def fail_best(path):
            return path["id"] != best_id

        result = universal_solve("skip_test", _execute=fail_best)
        assert result is not None
        if second_id is not None:
            assert result["id"] == second_id

    def test_deterministic_same_problem(self):
        """Same problem always produces the same winning path."""
        r1 = universal_solve("repeat_problem")
        r2 = universal_solve("repeat_problem")
        assert r1 == r2

    def test_custom_executor_called(self):
        """Custom _execute is invoked (not the default execute)."""
        called_with = []

        def capture(path):
            called_with.append(path["id"])
            return path.get("proven", False)

        universal_solve("capture_test", _execute=capture)
        assert len(called_with) > 0

    def test_toy_problem_end_to_end(self):
        """Full end-to-end: toy problem returns a valid result."""
        result = universal_solve("toy")
        assert result is not None
        assert result["proven"] is True
        assert len(result["steps"]) % MU8_ORDER == 0


# ─────────────────────────────────────────────────────────────────────────────
# 5. retry_next — fallback behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestRetryNext:
    """retry_next skips the failed path and returns the next successful one."""

    def test_skips_failed_path(self):
        paths = enumerate_all("retry_problem")
        ranked = sorted(paths, key=proven_and_cheap, reverse=True)
        first = ranked[0]

        result = retry_next("retry_problem", first)
        # Result must differ from the failed path (or be None if nothing else works).
        if result is not None:
            assert result["id"] != first["id"]

    def test_returns_none_when_nothing_left(self):
        """If every other path also fails, retry_next returns None."""
        def always_fail(path):
            return False

        paths = enumerate_all("none_left")
        ranked = sorted(paths, key=proven_and_cheap, reverse=True)
        result = retry_next(
            "none_left", ranked[0], _execute=always_fail
        )
        assert result is None

    def test_fallback_path_is_proven(self):
        """When the best path fails, retry_next should find a proven fallback."""
        paths = enumerate_all("fallback_test")
        ranked = sorted(paths, key=proven_and_cheap, reverse=True)
        first = ranked[0]

        # Fail only the top-ranked candidate; accept all others.
        def fail_first(path):
            return path["id"] != first["id"]

        result = retry_next("fallback_test", first, _execute=fail_first)
        # There should be another proven candidate available.
        assert result is not None

    def test_retry_next_custom_executor(self):
        """retry_next passes custom executor to inner execute calls."""
        executed = []

        def tracker(path):
            executed.append(path["id"])
            return path.get("proven", False)

        paths = enumerate_all("track_retry")
        ranked = sorted(paths, key=proven_and_cheap, reverse=True)
        retry_next("track_retry", ranked[0], _execute=tracker)
        assert len(executed) > 0
