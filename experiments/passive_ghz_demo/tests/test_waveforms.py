"""
Unit tests for Passive GHz Demo — Drive-only Variant 1.

Run with:  pytest experiments/passive_ghz_demo/tests/
"""

import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

# Make the experiment directory importable regardless of where pytest is run.
sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_waveforms import (
    SAMPLES_PER_STEP,
    D,
    MU_CYCLE,
    MU_INCREMENT_TURNS,
    build_waveforms,
    compute_phase_schedule,
)

# ---------------------------------------------------------------------------
# Constants under test
# ---------------------------------------------------------------------------

FS = 2.4e9
T_STEP = 40e-9
N_STEPS_TEST = 200


# ---------------------------------------------------------------------------
# 1. Sample-count correctness
# ---------------------------------------------------------------------------

class TestSampleCounts:
    def test_samples_per_step_matches_hardware(self):
        """96 samples/step = 2.4 GS/s × 40 ns."""
        expected = round(FS * T_STEP)
        assert expected == 96
        assert SAMPLES_PER_STEP == 96

    def test_total_samples(self):
        """Total samples = N_steps × samples_per_step."""
        data = build_waveforms(n_steps=N_STEPS_TEST, a_drive=0.5)
        assert len(data["I_pal"]) == N_STEPS_TEST * SAMPLES_PER_STEP
        assert len(data["Q_pal"]) == N_STEPS_TEST * SAMPLES_PER_STEP
        assert len(data["I_ctrl"]) == N_STEPS_TEST * SAMPLES_PER_STEP
        assert len(data["Q_ctrl"]) == N_STEPS_TEST * SAMPLES_PER_STEP

    def test_metadata_total_samples(self):
        """Metadata stores the correct total sample count."""
        data = build_waveforms(n_steps=N_STEPS_TEST, a_drive=0.5)
        meta = data["metadata"].item()
        assert meta["total_samples"] == N_STEPS_TEST * SAMPLES_PER_STEP


# ---------------------------------------------------------------------------
# 2. Phase schedule correctness for first 16 steps
# ---------------------------------------------------------------------------

class TestPhaseSchedule:
    """Verify φ(n) = φ_μ(n) + φ_p(n) for both variants over n = 0..15."""

    def _expected_phi_pal(self, n: int) -> float:
        """Reference implementation using exact rational arithmetic."""
        mu_turns = Fraction(3 * (n % MU_CYCLE), 8)
        prec_turns = Fraction(n, D)
        return float((mu_turns + prec_turns) * 2 * math.pi)

    def _expected_phi_ctrl(self, n: int) -> float:
        mu_turns = Fraction(3 * (n % MU_CYCLE), 8)
        return float(mu_turns * 2 * math.pi)

    def test_pal_phase_first_16_steps(self):
        phi_pal, _ = compute_phase_schedule(16)
        for n in range(16):
            expected = self._expected_phi_pal(n)
            assert abs(phi_pal[n] - expected) < 1e-12, (
                f"Step {n}: phi_pal={phi_pal[n]:.12f}, expected={expected:.12f}"
            )

    def test_ctrl_phase_first_16_steps(self):
        _, phi_ctrl = compute_phase_schedule(16)
        for n in range(16):
            expected = self._expected_phi_ctrl(n)
            assert abs(phi_ctrl[n] - expected) < 1e-12, (
                f"Step {n}: phi_ctrl={phi_ctrl[n]:.12f}, expected={expected:.12f}"
            )

    def test_mu_cycle_periodicity_in_ctrl(self):
        """Control waveform must repeat with period MU_CYCLE = 8 steps."""
        _, phi_ctrl = compute_phase_schedule(24)
        for n in range(8, 24):
            diff = phi_ctrl[n] - phi_ctrl[n % MU_CYCLE]
            # Must be a multiple of 2π (i.e. same phase modulo 2π)
            assert abs(diff % (2 * math.pi)) < 1e-12 or abs(diff % (2 * math.pi) - 2 * math.pi) < 1e-12, (
                f"Step {n}: ctrl phase not periodic with period 8"
            )

    def test_pal_differs_from_ctrl_by_precession(self):
        """φ_pal(n) − φ_ctrl(n) must equal 2π·n/D for all n."""
        n_test = 100
        phi_pal, phi_ctrl = compute_phase_schedule(n_test)
        for n in range(n_test):
            expected_prec = 2 * math.pi * n / D
            diff = phi_pal[n] - phi_ctrl[n]
            assert abs(diff - expected_prec) < 1e-12, (
                f"Step {n}: precession diff={diff:.12f}, expected={expected_prec:.12f}"
            )


# ---------------------------------------------------------------------------
# 3. Control waveform has identical φ_μ without φ_p
# ---------------------------------------------------------------------------

class TestControlVariant:
    def test_ctrl_has_no_precession(self):
        """phi_per_step_ctrl must exactly equal the μ-only schedule."""
        n_test = 32
        _, phi_ctrl = compute_phase_schedule(n_test)
        for n in range(n_test):
            expected = float(Fraction(3 * (n % MU_CYCLE), 8) * 2 * math.pi)
            assert abs(phi_ctrl[n] - expected) < 1e-12

    def test_ctrl_iq_amplitude_constant(self):
        """Control I/Q amplitude must be constant = a_drive."""
        a = 0.7
        data = build_waveforms(n_steps=100, a_drive=a)
        amp = np.sqrt(data["I_ctrl"] ** 2 + data["Q_ctrl"] ** 2)
        assert np.allclose(amp, a, atol=1e-10)

    def test_pal_iq_amplitude_constant(self):
        """Palindromic I/Q amplitude must also be constant = a_drive."""
        a = 0.7
        data = build_waveforms(n_steps=100, a_drive=a)
        amp = np.sqrt(data["I_pal"] ** 2 + data["Q_pal"] ** 2)
        assert np.allclose(amp, a, atol=1e-10)


# ---------------------------------------------------------------------------
# 4. Metadata correctness
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_metadata_fields(self):
        data = build_waveforms(n_steps=50, a_drive=0.3)
        meta = data["metadata"].item()
        assert meta["n_steps"] == 50
        assert meta["a_drive"] == pytest.approx(0.3)
        assert meta["samples_per_step"] == SAMPLES_PER_STEP
        assert meta["D"] == D

    def test_invalid_a_drive_raises(self):
        with pytest.raises(ValueError, match="a_drive"):
            build_waveforms(n_steps=10, a_drive=1.5)

    def test_zero_a_drive_raises(self):
        with pytest.raises(ValueError, match="a_drive"):
            build_waveforms(n_steps=10, a_drive=0.0)
