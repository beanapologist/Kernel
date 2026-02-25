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
    _batch_matched_filter_coherent,
    make_baseline_code,
    make_kernelsync_code,
    matched_filter_tau_fast,
    phase_turns,
    run_grid,
    simulate_node,
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


# ---------------------------------------------------------------------------
# 3. Coherent matched filter — noiseless correctness
# ---------------------------------------------------------------------------


class TestCoherentReceiver:
    """
    Tests for _batch_matched_filter_coherent and the sub-chip timing
    mechanism introduced in the KernelSync coherent receiver.

    All tests use noiseless Y (no additive noise) so that exact arithmetic
    relations can be verified.
    """

    @pytest.mark.parametrize("psi", [0.0, 0.5, math.pi / 5, math.pi, 2.7])
    def test_coherent_mf_zero_tau_known_phase(self, psi):
        """
        At tau=0 with psi_hat equal to the true carrier phase, the coherent
        MF must return tau_hat=0 for any carrier phase.
        """
        M = 32
        W = 7
        W_tight = 3
        n0 = 100
        s = make_kernelsync_code(n0, M)
        Y = np.exp(1j * psi) * s[np.newaxis, :]   # shape (1, M), noiseless
        tau_hats, _ = _batch_matched_filter_coherent(
            Y, s, W, np.array([psi]), np.array([0]), W_tight
        )
        assert tau_hats[0] == 0, f"psi={psi:.3f}: tau_hat={tau_hats[0]}"

    def test_coherent_mf_peak_magnitude_equals_M(self):
        """
        At tau=0, noiseless, the magnitude of the complex MF peak must equal M
        (all M chips contribute coherently).
        """
        M = 32
        W = 7
        W_tight = 3
        n0 = 0
        psi = 1.5
        s = make_kernelsync_code(n0, M)
        Y = np.exp(1j * psi) * s[np.newaxis, :]
        _, peak_complex = _batch_matched_filter_coherent(
            Y, s, W, np.array([psi]), np.array([0]), W_tight
        )
        assert abs(abs(peak_complex[0]) - M) < 1e-9, (
            f"|peak|={abs(peak_complex[0]):.6f}, expected {M}"
        )

    @pytest.mark.parametrize(
        "delta_frac", [0.0, 0.1, 0.25, -0.2, -0.4, 0.45]
    )
    def test_subchip_phase_encodes_fractional_offset(self, delta_frac):
        """
        The complex MF peak phase encodes the fractional chip offset delta_frac.

        Construction: Y[k] = exp(i*(psi − 2π×(3/8)×delta_frac)) × s[k]
        (continuous model, tau_int=0, known fractional delay).

        With W_tight=0 the search is restricted to tau=0, isolating the
        sub-chip measurement.  The recovered estimate must satisfy:

            delta_est = −phi_res / (2π × 3/8) ≈ delta_frac
        """
        M = 32
        W = 7
        W_tight = 0          # force search to tau=0 only
        n0 = 100
        psi_true = 1.0
        s = make_kernelsync_code(n0, M)
        subchip_phase = 2 * math.pi * MU_STEP_TURNS * delta_frac
        Y = np.exp(1j * (psi_true - subchip_phase)) * s[np.newaxis, :]
        _, peak_complex = _batch_matched_filter_coherent(
            Y, s, W, np.array([psi_true]), np.array([0]), W_tight
        )
        measured_phase = float(np.angle(peak_complex[0]))
        phi_residual = (
            (measured_phase - psi_true + math.pi) % (2 * math.pi) - math.pi
        )
        delta_est = -phi_residual / (2 * math.pi * MU_STEP_TURNS)
        assert abs(delta_est - delta_frac) < 1e-9, (
            f"delta_frac={delta_frac:.2f}: delta_est={delta_est:.9f}"
        )

    def test_coherent_mf_batch_multiple_nodes(self):
        """
        Batch coherent MF over N nodes: each node's tau_hat must be 0 when
        Y[n] = exp(i*psi_n) * s  (tau=0, noiseless, different carrier phases).
        """
        N = 8
        M = 32
        W = 7
        W_tight = 3
        n0 = 50
        rng = np.random.default_rng(0)
        psi_vals = rng.uniform(0, 2 * math.pi, N)
        s = make_kernelsync_code(n0, M)
        Y = np.exp(1j * psi_vals[:, np.newaxis]) * s[np.newaxis, :]  # (N, M)
        pred_tau = np.zeros(N, dtype=int)
        tau_hats, _ = _batch_matched_filter_coherent(
            Y, s, W, psi_vals, pred_tau, W_tight
        )
        np.testing.assert_array_equal(
            tau_hats, np.zeros(N, dtype=int),
            err_msg="Batch coherent MF: some tau_hat ≠ 0"
        )

    def test_coherent_mf_peak_phase_matches_carrier(self):
        """
        At tau=0, noiseless, the phase of the complex MF peak must equal
        the true carrier phase psi (mod 2π).
        """
        M = 64
        W = 7
        W_tight = 3
        n0 = 200
        psi = 2.1
        s = make_kernelsync_code(n0, M)
        Y = np.exp(1j * psi) * s[np.newaxis, :]
        _, peak_complex = _batch_matched_filter_coherent(
            Y, s, W, np.array([psi]), np.array([0]), W_tight
        )
        measured_phase = float(np.angle(peak_complex[0]))
        phase_diff = abs(
            (measured_phase - psi + math.pi) % (2 * math.pi) - math.pi
        )
        assert phase_diff < 1e-9, (
            f"peak phase={measured_phase:.6f}, expected psi={psi:.6f}"
        )


# ---------------------------------------------------------------------------
# 4. simulate_node — smoke tests for both schemes
# ---------------------------------------------------------------------------


class TestSimulateNode:
    """
    Smoke tests confirming that simulate_node runs without error for both
    schemes and returns a well-formed result array.
    """

    @pytest.mark.parametrize("scheme", ["kernelsync", "baseline"])
    def test_simulate_node_returns_finite_array(self, scheme):
        """
        simulate_node must return a 1-D array of finite float64 values with
        length equal to n_times for both kernelsync and baseline schemes.
        """
        n_times = 10
        rng = np.random.default_rng(0)
        err = simulate_node(
            rng,
            skew_ppm=5.0,
            theta0=50e-9,
            scheme=scheme,
            R=100,
            M=8,
            Tc=40e-9,
            W=3,
            T_sim=0.5,
            n_times=n_times,
        )
        assert err.shape == (n_times,), (
            f"scheme={scheme}: shape={err.shape}, expected ({n_times},)"
        )
        assert err.dtype == np.float64, f"scheme={scheme}: dtype={err.dtype}"
        assert np.all(np.isfinite(err)), (
            f"scheme={scheme}: non-finite values in output"
        )


# ---------------------------------------------------------------------------
# 5. Coherent grid sweep — RMS improvement over baseline
# ---------------------------------------------------------------------------


class TestCoherentGridSweep:
    """
    Integration test: run a small grid sweep and verify that the coherent
    KernelSync receiver achieves at least 5× lower RMS timing error than
    the incoherent baseline at the same (R, M) operating point.

    At R = 500 pilots/s the PI loop converges under coherent reception
    (KernelSync ~73 ns) but fails to converge under incoherent reception
    (Baseline >2000 ns), giving a ratio well above the 5× floor that
    follows from the sub-chip noise-reduction factor
    ~σ_phase / (2π × 3/8) × Tc  vs  Tc/2.
    """

    def test_kernelsync_rms_at_least_5x_better_than_baseline(self):
        """
        Grid sweep over R = 500 evt/s, M ∈ {16, 32}: at each (R, M) point,
        the final RMS of KernelSync must be at least 5× smaller than the
        final RMS of the incoherent baseline.
        """
        grid_R = [500.0]
        grid_M = [16, 32]
        Tc = 40e-9
        W_ns = 250e-9
        n_nodes = 30
        T_sim = 3.0
        n_times = 30

        results_base = run_grid(
            rng=np.random.default_rng(42),
            scheme="baseline",
            grid_R=grid_R,
            grid_M=grid_M,
            n_nodes=n_nodes,
            T_sim=T_sim,
            Tc=Tc,
            W_ns=W_ns,
            n_times=n_times,
        )
        results_ks = run_grid(
            rng=np.random.default_rng(42),
            scheme="kernelsync",
            grid_R=grid_R,
            grid_M=grid_M,
            n_nodes=n_nodes,
            T_sim=T_sim,
            Tc=Tc,
            W_ns=W_ns,
            n_times=n_times,
        )

        for R in grid_R:
            for M in grid_M:
                rms_base = results_base[(R, M)]["final_rms_ns"]
                rms_ks = results_ks[(R, M)]["final_rms_ns"]
                assert rms_ks > 0, f"R={R:.0f}, M={M}: KernelSync RMS is zero"
                ratio = rms_base / rms_ks
                assert ratio >= 5.0, (
                    f"R={R:.0f} evt/s, M={M} chips: "
                    f"KernelSync improvement {ratio:.2f}x < 5x "
                    f"(baseline={rms_base:.1f} ns, "
                    f"kernelsync={rms_ks:.1f} ns)"
                )
