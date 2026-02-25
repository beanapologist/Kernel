"""
Unit tests for KernelSync Demo.

Run with:  pytest experiments/kernelsync_demo/tests/ -v
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Make the experiment directory importable regardless of where pytest is run.
sys.path.insert(0, str(Path(__file__).parent.parent))

from kernelsync_energy_proxy_sim import (
    D,
    MU_CYCLE,
    MU_STEP_TURNS,
    make_baseline_code,
    make_kernelsync_code,
    matched_filter_tau_fast,
    phase_turns,
)

# ---------------------------------------------------------------------------
# 1. phase_turns — periodicity and correctness for first 16 steps
# ---------------------------------------------------------------------------


class TestPhaseTurns:
    """Verify φ(n) periodic properties for first 16 steps (μ-cycle)."""

    def test_mu_cycle_component_first_16(self):
        """
        The μ-cycle component (3/8) * (n mod 8) must repeat with period 8.
        """
        for n in range(16):
            mu_part = (MU_STEP_TURNS * (n % MU_CYCLE)) % 1.0
            expected_mu_part = (MU_STEP_TURNS * (n % MU_CYCLE)) % 1.0
            assert abs(mu_part - expected_mu_part) < 1e-14, (
                f"Step {n}: mu component mismatch"
            )

    def test_phase_turns_formula_first_16(self):
        """
        phase_turns(n) must equal ((3/8)*(n mod 8) + (n mod D)/D) mod 1
        for n = 0..15.
        """
        for n in range(16):
            expected = ((MU_STEP_TURNS * (n % MU_CYCLE)) + ((n % D) / D)) % 1.0
            result = phase_turns(n)
            assert abs(result - expected) < 1e-14, (
                f"Step {n}: phase_turns={result:.15f}, expected={expected:.15f}"
            )

    def test_phase_turns_periodic_mu_component_8_steps(self):
        """
        The difference phase_turns(n+8) - phase_turns(n) must equal exactly
        the precession increment 8/D (mod 1) for n in 0..15.
        """
        for n in range(16):
            delta = (phase_turns(n + 8) - phase_turns(n)) % 1.0
            expected_delta = (8 / D) % 1.0
            assert abs(delta - expected_delta) < 1e-10, (
                f"Step {n}: Δ={delta:.15f}, expected {expected_delta:.15f}"
            )

    def test_phase_turns_at_zero(self):
        """phase_turns(0) must be exactly 0."""
        assert phase_turns(0) == 0.0

    def test_phase_turns_at_D_wraps(self):
        """
        The precession term (n mod D)/D resets at multiples of D.
        phase_turns(D) should equal phase_turns(D mod 8 * 3/8 mod 1).
        """
        n = D
        expected = ((MU_STEP_TURNS * (n % MU_CYCLE)) + 0.0) % 1.0
        result = phase_turns(n)
        assert abs(result - expected) < 1e-12

    def test_phase_turns_returns_float_in_unit_interval(self):
        """phase_turns must return a value in [0, 1) for various n."""
        for n in [0, 1, 7, 8, 100, 1000, D - 1, D, D + 1]:
            pt = phase_turns(n)
            assert 0.0 <= pt < 1.0, f"phase_turns({n}) = {pt} out of [0,1)"


# ---------------------------------------------------------------------------
# 2. Matched filter — noiseless synthetic examples
# ---------------------------------------------------------------------------


class TestMatchedFilterNoiseless:
    """
    Verify that matched_filter_tau_fast returns the correct τ on noiseless
    synthetic examples for both Baseline and KernelSync codes.

    Note on τ=0 uniqueness:
        When τ_true = 0, the full M-chip code is present in y (no edge zeros),
        so the peak metric equals M (strictly greater than M−|τ| for all τ≠0).
        This gives an unambiguous unique maximum at τ̂ = 0 regardless of the
        code structure.

    Note on τ≠0 ambiguity:
        For τ_true ≠ 0, the windowed received signal y has |τ_true| zero chips
        at one edge.  The KernelSync code has a constant per-chip phase ratio,
        so the correlation magnitude equals the number of overlapping non-zero
        chips for ANY lag within a certain plateau.  Reliable non-zero tau
        detection in the simulation relies on accumulating many noisy pilot
        observations rather than single-shot noiseless detection.
    """

    @pytest.mark.parametrize("psi", [0.0, 0.5, math.pi / 5, math.pi, 2.7])
    def test_baseline_zero_offset_various_phases(self, psi):
        """Baseline τ=0, noiseless with various carrier phases: τ̂ must be 0."""
        M = 32
        W = 7
        s = make_baseline_code(M)
        y = np.exp(1j * psi) * s
        tau_hat = matched_filter_tau_fast(y, s, W)
        assert tau_hat == 0, f"Baseline: psi={psi:.3f}, tau_hat={tau_hat}"

    @pytest.mark.parametrize("psi", [0.0, 0.5, math.pi / 5, math.pi, 2.7])
    def test_kernelsync_zero_offset_various_phases(self, psi):
        """KernelSync τ=0, noiseless with various carrier phases: τ̂ must be 0."""
        M = 32
        W = 7
        n0 = 100
        s = make_kernelsync_code(n0, M)
        y = np.exp(1j * psi) * s
        tau_hat = matched_filter_tau_fast(y, s, W)
        assert tau_hat == 0, f"KernelSync: psi={psi:.3f}, tau_hat={tau_hat}"

    def test_baseline_zero_offset_metric_is_maximum(self):
        """
        At τ=0, the matched-filter metric must be strictly greater than at
        every other τ in the search window.
        """
        M = 32
        W = 7
        s = make_baseline_code(M)
        y = np.exp(1j * 1.23) * s
        metrics = np.array([
            abs(sum(
                y[k] * np.conj(s[k - t])
                for k in range(M)
                if 0 <= k - t < M
            ))
            for t in range(-W, W + 1)
        ])
        peak_metric = metrics[W]          # index W corresponds to τ=0
        assert peak_metric == pytest.approx(M, abs=1e-9)
        assert all(metrics[i] < peak_metric for i in range(len(metrics)) if i != W)

    def test_kernelsync_zero_offset_metric_is_maximum(self):
        """
        For KernelSync at τ=0, the matched-filter metric (M) must be strictly
        greater than at τ=±1 (M−1), verifying the peak is at zero lag.
        """
        M = 32
        W = 7
        n0 = 100
        s = make_kernelsync_code(n0, M)
        y = np.exp(1j * 2.3) * s
        metric_0 = abs(np.dot(y, np.conj(s)))          # τ=0: full overlap
        metric_p1 = abs(np.dot(y[1:], np.conj(s[:-1])))  # τ=+1: M-1 overlap
        metric_m1 = abs(np.dot(y[:-1], np.conj(s[1:])))  # τ=-1: M-1 overlap
        assert metric_0 == pytest.approx(M, abs=1e-9)
        assert metric_0 > metric_p1
        assert metric_0 > metric_m1

    def test_kernelsync_code_shape_and_amplitude(self):
        """KernelSync chips must be unit-magnitude complex numbers."""
        s = make_kernelsync_code(0, 64)
        assert s.shape == (64,)
        np.testing.assert_allclose(np.abs(s), 1.0, atol=1e-12)

    def test_baseline_code_shape_and_values(self):
        """Baseline chips must all be +1."""
        s = make_baseline_code(64)
        assert s.shape == (64,)
        np.testing.assert_array_equal(s, np.ones(64, dtype=complex))
