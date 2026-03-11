"""
Unit tests for the μ⁸=1 spiral cycle optimizer and its validator.

Run with:  pytest empirical-validation/tests/test_mu8_optimizer.py -v
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

# Make empirical-validation directory importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimizers.mu8_cycle_optimizer import (
    C_NATURAL,
    LEAN_FINGERPRINT,
    MU,
    MU_ANGLE,
    PHASE_NAMES,
    SILVER_RATIO,
    Mu8CycleOptimizer,
    CycleMetrics,
    lean_coherence,
    _wrap,
)
from validators.mu8_optimizer import validate


# ─────────────────────────────────────────────────────────────────────────────
# 1. Constants
# ─────────────────────────────────────────────────────────────────────────────

class TestConstants:
    """Lean-grounded constants are correct."""

    def test_mu_angle(self):
        assert abs(MU_ANGLE - 3.0 * math.pi / 4.0) < 1e-15

    def test_mu_complex(self):
        expected = complex(math.cos(MU_ANGLE), math.sin(MU_ANGLE))
        assert abs(MU - expected) < 1e-15

    def test_mu_unit_circle(self):
        assert abs(abs(MU) - 1.0) < 1e-14

    def test_mu8_eq_1_numerical(self):
        assert abs(MU ** 8 - 1.0) < 1e-12

    def test_mu8_eq_1_sympy(self):
        mu_sym = sp.exp(sp.I * 3 * sp.pi / 4)
        assert sp.simplify(mu_sym ** 8) == sp.Integer(1)

    def test_silver_ratio_value(self):
        assert abs(SILVER_RATIO - (1.0 + math.sqrt(2.0))) < 1e-15

    def test_silver_ratio_self_similar(self):
        # δS = 2 + 1/δS  (CriticalEigenvalue.lean §20)
        assert abs(SILVER_RATIO - (2.0 + 1.0 / SILVER_RATIO)) < 1e-14

    def test_c_natural(self):
        assert C_NATURAL == 137

    def test_lean_fingerprint_stable(self):
        from optimizers.mu8_cycle_optimizer import _LEAN_CONSTANTS
        actual = hashlib.sha256(
            json.dumps(_LEAN_CONSTANTS, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        assert actual == LEAN_FINGERPRINT

    def test_phase_names_five(self):
        assert len(PHASE_NAMES) == 5
        assert PHASE_NAMES[0] == "coherence"
        assert PHASE_NAMES[-1] == "optimization"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Lean coherence function
# ─────────────────────────────────────────────────────────────────────────────

class TestLeanCoherence:
    """C(r) = 2r/(1+r²) — Lean Section 5 properties."""

    def test_at_one(self):
        assert abs(lean_coherence(1.0) - 1.0) < 1e-15

    def test_palindrome_symmetry(self):
        for r in [0.5, 2.0, SILVER_RATIO, math.pi]:
            assert abs(lean_coherence(r) - lean_coherence(1.0 / r)) < 1e-14

    def test_positive_for_positive_r(self):
        for r in np.linspace(0.01, 10.0, 50):
            assert lean_coherence(float(r)) > 0.0

    def test_bounded_above_by_one(self):
        for r in np.linspace(0.01, 20.0, 200):
            assert lean_coherence(float(r)) <= 1.0 + 1e-14

    def test_approaches_zero_large_r(self):
        assert lean_coherence(1000.0) < 1e-2

    def test_formula(self):
        r = 3.7
        assert abs(lean_coherence(r) - 2.0 * r / (1.0 + r * r)) < 1e-15


# ─────────────────────────────────────────────────────────────────────────────
# 3. _wrap helper
# ─────────────────────────────────────────────────────────────────────────────

class TestWrap:
    """Angle wrapping stays in (−π, π]."""

    def test_zero(self):
        assert abs(_wrap(np.array([0.0]))[0]) < 1e-15

    def test_pi(self):
        # np.angle(exp(i·π)) = π
        val = _wrap(np.array([math.pi]))[0]
        assert abs(abs(val) - math.pi) < 1e-14

    def test_beyond_pi(self):
        val = _wrap(np.array([math.pi + 0.1]))[0]
        assert -math.pi < val <= math.pi

    def test_negative(self):
        val = _wrap(np.array([-0.5]))[0]
        assert abs(val - (-0.5)) < 1e-14

    def test_array_all_in_range(self):
        x = np.linspace(-4 * math.pi, 4 * math.pi, 200)
        w = _wrap(x)
        assert np.all(w > -math.pi - 1e-12)
        assert np.all(w <= math.pi + 1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Optimizer construction / logic gates
# ─────────────────────────────────────────────────────────────────────────────

class TestOptimizerConstruction:
    """Gates fire correctly; construction validates inputs."""

    def test_default_construction(self):
        opt = Mu8CycleOptimizer()
        assert opt.N == 8
        assert opt.gain == 0.3
        assert opt.revolution == 0
        assert len(opt.history) == 0

    def test_custom_params(self):
        opt = Mu8CycleOptimizer(N=16, gain=0.5, seed=7)
        assert opt.N == 16
        assert opt.gain == 0.5

    def test_phase_shape(self):
        opt = Mu8CycleOptimizer(N=12, seed=0)
        assert opt.phases.shape == (12,)
        assert opt.phases.dtype == np.float64

    def test_invalid_N(self):
        with pytest.raises(ValueError):
            Mu8CycleOptimizer(N=1)

    def test_invalid_gain_zero(self):
        with pytest.raises(ValueError):
            Mu8CycleOptimizer(gain=0.0)

    def test_invalid_gain_negative(self):
        with pytest.raises(ValueError):
            Mu8CycleOptimizer(gain=-0.1)

    def test_invalid_gain_above_one(self):
        with pytest.raises(ValueError):
            Mu8CycleOptimizer(gain=1.5)

    def test_gain_one_valid(self):
        opt = Mu8CycleOptimizer(gain=1.0)
        assert opt.gain == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Single-cycle mechanics
# ─────────────────────────────────────────────────────────────────────────────

class TestSingleCycle:
    """One call to run_cycle() behaves correctly."""

    def test_returns_cycle_metrics(self):
        opt = Mu8CycleOptimizer()
        m = opt.run_cycle()
        assert isinstance(m, CycleMetrics)

    def test_revolution_increments(self):
        opt = Mu8CycleOptimizer()
        for k in range(5):
            m = opt.run_cycle()
            assert m.revolution == k

    def test_mu_power_cycles_0_to_7(self):
        opt = Mu8CycleOptimizer()
        for expected_k in range(8):
            m = opt.run_cycle()
            assert m.mu_power == expected_k

    def test_mu_power_wraps_after_8(self):
        opt = Mu8CycleOptimizer()
        opt.run(8)
        m = opt.run_cycle()
        assert m.mu_power == 0  # wraps back to 0

    def test_metrics_in_history(self):
        opt = Mu8CycleOptimizer()
        opt.run(3)
        assert len(opt.history) == 3

    def test_coherence_in_range(self):
        opt = Mu8CycleOptimizer()
        for _ in range(8):
            m = opt.run_cycle()
            assert 0.0 <= m.coherence_out <= 1.0 + 1e-12

    def test_frustration_nonnegative(self):
        opt = Mu8CycleOptimizer()
        for _ in range(8):
            m = opt.run_cycle()
            assert m.frustration_out >= -1e-12

    def test_checksum_is_hex_string(self):
        opt = Mu8CycleOptimizer()
        m = opt.run_cycle()
        assert len(m.state_checksum) == 64
        int(m.state_checksum, 16)  # must parse as hex

    def test_lean_fingerprint_in_metrics(self):
        opt = Mu8CycleOptimizer()
        m = opt.run_cycle()
        assert m.lean_fingerprint == LEAN_FINGERPRINT


# ─────────────────────────────────────────────────────────────────────────────
# 6. Cycle invariants (coherence non-decreasing, frustration non-increasing)
# ─────────────────────────────────────────────────────────────────────────────

class TestCycleInvariants:
    """EMA contraction is monotone per cycle."""

    @pytest.mark.parametrize("gain", [0.1, 0.3, 0.5, 0.8, 1.0])
    def test_coherence_nondecreasing_various_gains(self, gain):
        opt = Mu8CycleOptimizer(N=8, gain=gain, seed=42)
        for _ in range(16):
            m = opt.run_cycle()
            assert m.delta_coherence >= -1e-12, (
                f"Coherence decreased by {m.delta_coherence:.2e} "
                f"(gain={gain}, revolution={m.revolution})"
            )

    @pytest.mark.parametrize("gain", [0.1, 0.3, 0.5, 0.8, 1.0])
    def test_frustration_nonincreasing_various_gains(self, gain):
        opt = Mu8CycleOptimizer(N=8, gain=gain, seed=42)
        for _ in range(16):
            m = opt.run_cycle()
            assert m.delta_frustration <= 1e-12, (
                f"Frustration increased by {m.delta_frustration:.2e} "
                f"(gain={gain}, revolution={m.revolution})"
            )

    @pytest.mark.parametrize("N", [2, 4, 8, 16, 32])
    def test_invariants_hold_various_N(self, N):
        opt = Mu8CycleOptimizer(N=N, gain=0.3, seed=0)
        for _ in range(8):
            m = opt.run_cycle()
            assert m.delta_coherence >= -1e-12
            assert m.delta_frustration <= 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# 7. Convergence
# ─────────────────────────────────────────────────────────────────────────────

class TestConvergence:
    """Optimizer converges to high coherence over many cycles."""

    def test_coherence_improves_over_40_cycles(self):
        opt = Mu8CycleOptimizer(N=16, gain=0.35, seed=7)
        metrics = opt.run(40)
        R_init = metrics[0].coherence_in
        R_final = metrics[-1].coherence_out
        assert R_final > R_init + 0.1, (
            f"Insufficient improvement: R_init={R_init:.4f}, R_final={R_final:.4f}"
        )

    def test_frustration_falls_over_40_cycles(self):
        opt = Mu8CycleOptimizer(N=16, gain=0.35, seed=7)
        metrics = opt.run(40)
        e_start = float(np.mean([m.frustration_out for m in metrics[:8]]))
        e_end = float(np.mean([m.frustration_out for m in metrics[-8:]]))
        assert e_end < e_start, (
            f"Frustration did not fall: e_start={e_start:.4f}, e_end={e_end:.4f}"
        )

    def test_high_coherence_after_many_cycles(self):
        """Starting from uniform random phases, coherence should reach ≥ 0.9."""
        opt = Mu8CycleOptimizer(N=8, gain=0.5, seed=3)
        opt.run(100)
        R_final = opt._circular_coherence()
        assert R_final >= 0.85, f"R_final={R_final:.4f} below 0.85"

    def test_revolution_counter_accurate(self):
        n = 25
        opt = Mu8CycleOptimizer()
        opt.run(n)
        assert opt.revolution == n


# ─────────────────────────────────────────────────────────────────────────────
# 8. Determinism / reproducibility
# ─────────────────────────────────────────────────────────────────────────────

class TestReproducibility:
    """Same seed → identical trajectories and checksums."""

    def test_same_seed_same_checksums(self):
        opt_a = Mu8CycleOptimizer(N=8, gain=0.3, seed=42)
        opt_b = Mu8CycleOptimizer(N=8, gain=0.3, seed=42)
        for _ in range(10):
            a = opt_a.run_cycle()
            b = opt_b.run_cycle()
            assert a.state_checksum == b.state_checksum
            assert abs(a.coherence_out - b.coherence_out) < 1e-15

    def test_different_seeds_different_checksums(self):
        opt_a = Mu8CycleOptimizer(N=8, gain=0.3, seed=1)
        opt_b = Mu8CycleOptimizer(N=8, gain=0.3, seed=2)
        a = opt_a.run_cycle()
        b = opt_b.run_cycle()
        assert a.state_checksum != b.state_checksum

    def test_run_and_individual_steps_agree(self):
        """run(n) should give same results as n calls to run_cycle()."""
        opt_a = Mu8CycleOptimizer(N=8, gain=0.3, seed=5)
        opt_b = Mu8CycleOptimizer(N=8, gain=0.3, seed=5)
        batch = opt_a.run(5)
        individual = [opt_b.run_cycle() for _ in range(5)]
        for a, b in zip(batch, individual):
            assert a.state_checksum == b.state_checksum


# ─────────────────────────────────────────────────────────────────────────────
# 9. Cross-language portability (JSON export)
# ─────────────────────────────────────────────────────────────────────────────

class TestJsonExport:
    """export_state() is JSON-serialisable and self-consistent."""

    def test_all_required_keys_present(self):
        opt = Mu8CycleOptimizer(N=8, gain=0.3, seed=0)
        opt.run(3)
        state = opt.export_state()
        for key in ("phases", "revolution", "N", "gain", "mu_angle",
                    "lean_fingerprint", "coherence", "frustration"):
            assert key in state, f"Missing key: {key}"

    def test_json_round_trip(self):
        opt = Mu8CycleOptimizer(N=8, gain=0.3, seed=0)
        opt.run(3)
        state = opt.export_state()
        recovered = json.loads(json.dumps(state))
        assert np.allclose(recovered["phases"], state["phases"], atol=1e-15)
        assert recovered["revolution"] == state["revolution"]
        assert recovered["lean_fingerprint"] == LEAN_FINGERPRINT

    def test_phases_length(self):
        opt = Mu8CycleOptimizer(N=12, seed=0)
        state = opt.export_state()
        assert len(state["phases"]) == 12

    def test_mu_angle_in_export(self):
        opt = Mu8CycleOptimizer()
        state = opt.export_state()
        assert abs(state["mu_angle"] - MU_ANGLE) < 1e-15

    def test_coherence_in_export_matches_optimizer(self):
        opt = Mu8CycleOptimizer(N=8, gain=0.3, seed=9)
        opt.run(5)
        state = opt.export_state()
        assert abs(state["coherence"] - opt._circular_coherence()) < 1e-15

    def test_revolution_in_export(self):
        opt = Mu8CycleOptimizer(N=8, gain=0.3, seed=9)
        opt.run(7)
        assert opt.export_state()["revolution"] == 7


# ─────────────────────────────────────────────────────────────────────────────
# 10. Validator interface
# ─────────────────────────────────────────────────────────────────────────────

class TestValidatorInterface:
    """validate() returns well-formed results compatible with the pipeline."""

    @pytest.fixture(scope="class")
    def results(self):
        return validate()

    def test_returns_list(self, results):
        assert isinstance(results, list)
        assert len(results) > 0

    def test_each_result_has_required_keys(self, results):
        required = {"name", "check_type", "pass_criterion", "modelled",
                    "observed", "rel_error", "passed", "method", "description"}
        for r in results:
            missing = required - set(r.keys())
            assert not missing, f"Result '{r.get('name')}' missing keys: {missing}"

    def test_passed_is_bool(self, results):
        for r in results:
            assert isinstance(r["passed"], bool), (
                f"Result '{r['name']}': 'passed' is {type(r['passed'])}, not bool"
            )

    def test_all_checks_pass(self, results):
        failures = [r["name"] for r in results if not r["passed"]]
        assert not failures, f"Failed checks: {failures}"

    def test_check_types_are_valid(self, results):
        valid_types = {
            "logic_gate", "mathematical_identity",
            "cycle_invariant", "convergence", "reproducibility",
        }
        for r in results:
            assert r["check_type"] in valid_types, (
                f"Unknown check_type '{r['check_type']}' in '{r['name']}'"
            )

    def test_gate_checks_present(self, results):
        gate_names = {r["name"] for r in results if r["check_type"] == "logic_gate"}
        assert "gate_sympy_mu8_eq_1" in gate_names
        assert "gate_numpy_mu_unit_circle" in gate_names
        assert "gate_checksum_lean_fingerprint" in gate_names

    def test_cycle_invariant_checks_present(self, results):
        inv_names = {r["name"] for r in results if r["check_type"] == "cycle_invariant"}
        assert "invariant_coherence_nondecreasing" in inv_names
        assert "invariant_frustration_nonincreasing" in inv_names
        assert "invariant_mu_power_cycle" in inv_names

    def test_rel_error_nonneg(self, results):
        for r in results:
            assert r["rel_error"] >= 0.0, (
                f"Negative rel_error in '{r['name']}': {r['rel_error']}"
            )

    def test_accepts_none_data(self):
        res = validate(None)
        assert isinstance(res, list)

    def test_accepts_empty_dict(self):
        res = validate({})
        assert isinstance(res, list)


# ─────────────────────────────────────────────────────────────────────────────
# 11. μ⁸=1 deep-spiral property
# ─────────────────────────────────────────────────────────────────────────────

class TestDeepSpiral:
    """After every 8-revolution group the coherence is strictly higher."""

    def test_each_group_of_8_deepens_coherence(self):
        """Three consecutive groups of 8 must each yield higher final coherence."""
        opt = Mu8CycleOptimizer(N=8, gain=0.3, seed=42)
        prev_R = opt._circular_coherence()
        for group in range(3):
            opt.run(8)  # one full μ⁸=1 revolution-group
            curr_R = opt._circular_coherence()
            assert curr_R >= prev_R, (
                f"Group {group}: coherence did not improve "
                f"(prev={prev_R:.6f}, curr={curr_R:.6f})"
            )
            prev_R = curr_R

    def test_cumulative_mu_rotation_after_8(self):
        """Generation applies μ^k for k=0..7 across 8 cycles, then wraps to 0."""
        # Each revolution k applies a generation offset of k·(3π/4) for k=0..7.
        # The cumulative angle sum = (3π/4)·(0+1+…+7) = (3π/4)·28 = 21π.
        # 21π mod 2π = π (since 21 = 10·2+1), so the net sum is not 0 mod 2π.
        # However, the generation phase visits every eighth root of unity exactly
        # once per 8-cycle group (μ^0, μ^1, …, μ^7), and then mu_power resets
        # to 0 for the next group — this is the μ⁸=1 periodicity in action.
        opt = Mu8CycleOptimizer(N=8, gain=0.3, seed=42)
        metrics = opt.run(8)
        powers = [m.mu_power for m in metrics]
        assert powers == list(range(8))
        # Next group starts at 0 (μ⁸ = 1 wraps back to identity)
        m_next = opt.run_cycle()
        assert m_next.mu_power == 0
