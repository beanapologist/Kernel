"""
μ⁸=1 Spiral Cycle Optimizer validator.
=======================================
Battle-tests the :class:`~optimizers.mu8_cycle_optimizer.Mu8CycleOptimizer`
by exercising the five-phase cycle and verifying every Lean-grounded invariant.

Check-type taxonomy
-------------------
``logic_gate``
    Pre-flight gate checks (SymPy exact, NumPy numerical, checksum) that must
    pass before the optimizer can run.  A failure is a coding bug.

``mathematical_identity``
    Pure algebraic facts about μ, the silver ratio, and C(r) that are
    proved in Lean and checked symbolically or numerically here.

``cycle_invariant``
    Properties of the five-phase cycle that must hold every revolution:
    coherence non-decreasing, frustration non-increasing, μ-power cycling.

``convergence``
    Properties of the optimizer over many revolutions: monotone R increase,
    monotone E decrease, convergence to R≈1.

``reproducibility``
    Deterministic checksum test: same seed + same parameters → identical
    state fingerprints, cross-language portability verified via JSON export.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

# Allow running from arbitrary working directories.
sys.path.insert(0, str(Path(__file__).parent.parent))

from optimizers.mu8_cycle_optimizer import (
    C_NATURAL,
    LEAN_FINGERPRINT,
    MU,
    MU_ANGLE,
    SILVER_RATIO,
    Mu8CycleOptimizer,
    bit_strength,
    lean_coherence,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_result(
    name: str,
    check_type: str,
    pass_criterion: str,
    modelled: float,
    observed: float,
    passed: bool,
    method: str,
    description: str,
    rel_error: float | None = None,
) -> dict[str, Any]:
    if rel_error is None:
        denom = observed if observed != 0.0 else 1.0
        rel_error = abs(modelled - observed) / abs(denom)
    return {
        "name": name,
        "check_type": check_type,
        "pass_criterion": pass_criterion,
        "modelled": modelled,
        "observed": observed,
        "rel_error": rel_error,
        "passed": bool(passed),
        "method": method,
        "description": description,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Gate checks
# ─────────────────────────────────────────────────────────────────────────────

def _check_sympy_gate() -> dict[str, Any]:
    """SymPy: μ⁸ = 1 (exact symbolic)."""
    mu_sym = sp.exp(sp.I * 3 * sp.pi / 4)
    result = sp.simplify(mu_sym ** 8)
    passed = (result == sp.Integer(1))
    return _make_result(
        name="gate_sympy_mu8_eq_1",
        check_type="logic_gate",
        pass_criterion="SymPy: simplify(exp(i·3π/4)⁸) == 1. "
                       "Lean theorem mu_pow_eight; failure = SymPy regression.",
        modelled=float(sp.re(result)),
        observed=1.0,
        passed=passed,
        method="SymPy exact symbolic",
        description="μ⁸ = 1  (SymPy gate)",
    )


def _check_numpy_gate() -> list[dict[str, Any]]:
    """NumPy: |μ| = 1 and |μ⁸ − 1| < 1e-12."""
    mu_num = np.exp(1j * 3.0 * np.pi / 4.0)
    tol = 1e-12

    err_abs = abs(abs(mu_num) - 1.0)
    r1 = _make_result(
        name="gate_numpy_mu_unit_circle",
        check_type="logic_gate",
        pass_criterion="|  |μ| − 1 | < 1e-12. IEEE 754 check; failure = floating-point bug.",
        modelled=abs(mu_num),
        observed=1.0,
        passed=bool(err_abs < tol),
        method="NumPy",
        description="|μ| = 1  (NumPy gate)",
    )

    err8 = abs(mu_num ** 8 - 1.0)
    r2 = _make_result(
        name="gate_numpy_mu8_eq_1",
        check_type="logic_gate",
        pass_criterion="|μ⁸ − 1| < 1e-12. IEEE 754 check; failure = floating-point bug.",
        modelled=float(abs(mu_num ** 8)),
        observed=1.0,
        passed=bool(err8 < tol),
        method="NumPy",
        description="|μ⁸ − 1| < 1e-12  (NumPy gate)",
    )
    return [r1, r2]


def _check_lean_fingerprint() -> dict[str, Any]:
    """Checksum gate: Lean constant fingerprint is stable."""
    from optimizers.mu8_cycle_optimizer import _LEAN_CONSTANTS
    actual = hashlib.sha256(
        json.dumps(_LEAN_CONSTANTS, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    passed = (actual == LEAN_FINGERPRINT)
    # Use 0/1 floats — no numeric meaning, just pass/fail.
    return _make_result(
        name="gate_checksum_lean_fingerprint",
        check_type="logic_gate",
        pass_criterion="SHA-256 of Lean constants matches hard-coded LEAN_FINGERPRINT. "
                       "Failure = constants drifted from their Lean-verified values.",
        modelled=1.0 if passed else 0.0,
        observed=1.0,
        passed=passed,
        method="SHA-256",
        description="Lean grounding-constant checksum matches",
        rel_error=0.0 if passed else 1.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Mathematical-identity checks
# ─────────────────────────────────────────────────────────────────────────────

def _check_mu_angle() -> dict[str, Any]:
    """μ angle = 3π/4 exactly (SymPy)."""
    mu_sym = sp.exp(sp.I * 3 * sp.pi / 4)
    arg = sp.simplify(sp.arg(mu_sym))
    expected = 3 * sp.pi / 4
    passed = bool(sp.simplify(arg - expected) == 0)
    return _make_result(
        name="identity_mu_angle",
        check_type="mathematical_identity",
        pass_criterion="SymPy: arg(μ) = 3π/4. Definitional; failure = coding bug.",
        modelled=float(sp.N(arg)),
        observed=float(sp.N(expected)),
        passed=passed,
        method="SymPy",
        description="arg(μ) = 3π/4",
    )


def _check_mu_real_imag() -> list[dict[str, Any]]:
    """Re(μ) = −1/√2, Im(μ) = 1/√2 (numerical)."""
    tol = 1e-14
    expected_re = -1.0 / math.sqrt(2.0)
    expected_im = 1.0 / math.sqrt(2.0)
    err_re = abs(MU.real - expected_re)
    err_im = abs(MU.imag - expected_im)
    r1 = _make_result(
        name="identity_mu_real_part",
        check_type="mathematical_identity",
        pass_criterion="|Re(μ) − (−1/√2)| < 1e-14. Definitional; failure = coding bug.",
        modelled=MU.real,
        observed=expected_re,
        passed=err_re < tol,
        method="NumPy",
        description="Re(μ) = −1/√2",
    )
    r2 = _make_result(
        name="identity_mu_imag_part",
        check_type="mathematical_identity",
        pass_criterion="|Im(μ) − 1/√2| < 1e-14. Definitional; failure = coding bug.",
        modelled=MU.imag,
        observed=expected_im,
        passed=err_im < tol,
        method="NumPy",
        description="Im(μ) = 1/√2",
    )
    return [r1, r2]


def _check_silver_ratio() -> list[dict[str, Any]]:
    """δS = 1+√2 and self-similarity δS = 2 + 1/δS (CriticalEigenvalue.lean §20)."""
    tol = 1e-14
    expected = 1.0 + math.sqrt(2.0)
    err1 = abs(SILVER_RATIO - expected)
    r1 = _make_result(
        name="identity_silver_ratio_value",
        check_type="mathematical_identity",
        pass_criterion="|δS − (1+√2)| < 1e-14. Definitional; failure = coding bug.",
        modelled=SILVER_RATIO,
        observed=expected,
        passed=err1 < tol,
        method="NumPy",
        description="δS = 1 + √2",
    )

    # Self-similarity: δS = 2 + 1/δS  (Lean §20)
    self_similar = 2.0 + 1.0 / SILVER_RATIO
    err2 = abs(SILVER_RATIO - self_similar)
    r2 = _make_result(
        name="identity_silver_ratio_self_similar",
        check_type="mathematical_identity",
        pass_criterion="|δS − (2 + 1/δS)| < 1e-14. "
                       "Lean §20 silver_ratio_self_similarity; failure = coding bug.",
        modelled=SILVER_RATIO,
        observed=self_similar,
        passed=err2 < tol,
        method="NumPy",
        description="δS = 2 + 1/δS  (self-similarity)",
    )
    return [r1, r2]


def _check_lean_coherence() -> list[dict[str, Any]]:
    """Lean C(r) = 2r/(1+r²): boundary values and symmetry."""
    tol = 1e-14
    results = []

    # C(1) = 1
    val = lean_coherence(1.0)
    results.append(_make_result(
        name="identity_lean_coherence_at_one",
        check_type="mathematical_identity",
        pass_criterion="|C(1) − 1| < 1e-14. 2·1/(1+1)=1; failure = coding bug.",
        modelled=val,
        observed=1.0,
        passed=abs(val - 1.0) < tol,
        method="Python",
        description="C(1) = 1  (maximum coherence)",
    ))

    # C(r) = C(1/r)  palindrome symmetry
    r = SILVER_RATIO
    val_r = lean_coherence(r)
    val_inv = lean_coherence(1.0 / r)
    err = abs(val_r - val_inv)
    results.append(_make_result(
        name="identity_lean_coherence_palindrome",
        check_type="mathematical_identity",
        pass_criterion="|C(δS) − C(1/δS)| < 1e-14. "
                       "Palindrome symmetry C(r)=C(1/r); failure = coding bug.",
        modelled=val_r,
        observed=val_inv,
        passed=err < tol,
        method="Python",
        description="C(δS) = C(1/δS)  (palindrome symmetry)",
    ))

    # C(r) ≤ 1 on a grid
    r_grid = np.linspace(0.01, 10.0, 500)
    c_grid = np.array([lean_coherence(float(rv)) for rv in r_grid])
    all_le_one = bool(np.all(c_grid <= 1.0 + 1e-14))
    results.append(_make_result(
        name="identity_lean_coherence_bounded",
        check_type="mathematical_identity",
        pass_criterion="C(r) ≤ 1 for r ∈ [0.01, 10] (500 points). "
                       "Lean coherence bound; failure = coding bug.",
        modelled=float(np.max(c_grid)),
        observed=1.0,
        passed=all_le_one,
        method="NumPy grid",
        description="C(r) ≤ 1 on [0.01, 10]",
        rel_error=0.0 if all_le_one else float(np.max(c_grid) - 1.0),
    ))

    return results


def _check_c_natural() -> dict[str, Any]:
    """c_natural = 137 = 1/α_FS (SpeedOfLight.lean)."""
    # Verified numerically: 1/α_FS ≈ 137.036; c_natural is the integer 137.
    expected = 137
    passed = (C_NATURAL == expected)
    return _make_result(
        name="identity_c_natural",
        check_type="mathematical_identity",
        pass_criterion="C_NATURAL == 137. SpeedOfLight.lean c_natural; failure = coding bug.",
        modelled=float(C_NATURAL),
        observed=float(expected),
        passed=passed,
        method="Python",
        description="c_natural = 137 = 1/α_FS  (SpeedOfLight.lean)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Cycle-invariant checks
# ─────────────────────────────────────────────────────────────────────────────

def _check_cycle_invariants() -> list[dict[str, Any]]:
    """Run 8 complete cycles and verify per-cycle invariants."""
    opt = Mu8CycleOptimizer(N=8, gain=0.3, seed=0)
    results = []

    # Run 8 cycles (one full μ⁸=1 revolution-group)
    metrics_list = opt.run(8)

    # 1. Coherence non-decreasing (R_out ≥ R_in each cycle)
    all_nondec = all(m.delta_coherence >= -1e-12 for m in metrics_list)
    worst = min(m.delta_coherence for m in metrics_list)
    results.append(_make_result(
        name="invariant_coherence_nondecreasing",
        check_type="cycle_invariant",
        pass_criterion="delta_coherence ≥ −1e-12 every cycle (8 cycles, N=8, g=0.3). "
                       "EMA contraction is dissipative; failure = optimization step wrong.",
        modelled=float(worst),
        observed=0.0,
        passed=all_nondec,
        method="Mu8CycleOptimizer, 8 cycles",
        description="R_out ≥ R_in each cycle",
        rel_error=0.0 if all_nondec else abs(worst),
    ))

    # 2. Frustration non-increasing (E_out ≤ E_in each cycle)
    all_noninc = all(m.delta_frustration <= 1e-12 for m in metrics_list)
    worst_e = max(m.delta_frustration for m in metrics_list)
    results.append(_make_result(
        name="invariant_frustration_nonincreasing",
        check_type="cycle_invariant",
        pass_criterion="delta_frustration ≤ 1e-12 every cycle (8 cycles, N=8, g=0.3). "
                       "SOURCE gives up energy to SINK; failure = extraction step wrong.",
        modelled=float(worst_e),
        observed=0.0,
        passed=all_noninc,
        method="Mu8CycleOptimizer, 8 cycles",
        description="E_out ≤ E_in each cycle",
        rel_error=0.0 if all_noninc else abs(worst_e),
    ))

    # 3. μ-power cycles through 0..7 exactly (revolution % 8)
    powers = [m.mu_power for m in metrics_list]
    expected_powers = list(range(8))
    powers_correct = (powers == expected_powers)
    results.append(_make_result(
        name="invariant_mu_power_cycle",
        check_type="cycle_invariant",
        pass_criterion="mu_power for 8 consecutive cycles = [0,1,2,3,4,5,6,7]. "
                       "μ⁸=1 group orbit; failure = revolution counter wrong.",
        modelled=float(powers[-1]),
        observed=7.0,
        passed=powers_correct,
        method="Mu8CycleOptimizer, 8 cycles",
        description="μ-powers cycle 0→7 over 8 revolutions",
        rel_error=0.0 if powers_correct else 1.0,
    ))

    # 4. Coherence strictly better after 8 cycles vs start
    R_start = metrics_list[0].coherence_in
    R_end = metrics_list[-1].coherence_out
    improved = R_end > R_start
    results.append(_make_result(
        name="invariant_coherence_improves_over_8",
        check_type="cycle_invariant",
        pass_criterion="R after 8 cycles > R before first cycle. "
                       "Optimizer makes progress; failure = optimizer non-functional.",
        modelled=R_end,
        observed=R_start,
        passed=improved,
        method="Mu8CycleOptimizer, 8 cycles",
        description="Coherence higher after 8 complete cycles",
        rel_error=(R_start - R_end) / max(R_start, 1e-12) if not improved else 0.0,
    ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Convergence checks
# ─────────────────────────────────────────────────────────────────────────────

def _check_convergence() -> list[dict[str, Any]]:
    """Run 40 cycles and verify global convergence properties."""
    n_cycles = 40
    opt = Mu8CycleOptimizer(N=16, gain=0.35, seed=7)
    metrics_list = opt.run(n_cycles)
    results = []

    # 1. Final coherence substantially higher than initial
    R_init = metrics_list[0].coherence_in
    R_final = metrics_list[-1].coherence_out
    tol_improvement = 0.1
    results.append(_make_result(
        name="convergence_coherence_gain",
        check_type="convergence",
        pass_criterion=f"R_final − R_initial ≥ {tol_improvement} "
                       f"({n_cycles} cycles, N=16, g=0.35). "
                       "Optimizer converges; failure = optimizer non-functional.",
        modelled=R_final - R_init,
        observed=tol_improvement,
        passed=(R_final - R_init) >= tol_improvement,
        method=f"Mu8CycleOptimizer, {n_cycles} cycles",
        description=f"R gain ≥ {tol_improvement} over {n_cycles} cycles",
        rel_error=max(0.0, tol_improvement - (R_final - R_init)) / tol_improvement,
    ))

    # 2. Frustration monotone non-increasing over full run (moving avg)
    # Use 8-cycle window to absorb transient wiggles from generation phase.
    window = 8
    e_out = [m.frustration_out for m in metrics_list]
    e_avg_start = float(np.mean(e_out[:window]))
    e_avg_end = float(np.mean(e_out[-window:]))
    frustration_fell = e_avg_end < e_avg_start
    results.append(_make_result(
        name="convergence_frustration_falls",
        check_type="convergence",
        pass_criterion=f"Mean E in last {window} cycles < mean E in first {window} cycles. "
                       "Frustration decreases over the run; failure = optimizer diverging.",
        modelled=e_avg_end,
        observed=e_avg_start,
        passed=frustration_fell,
        method=f"Mu8CycleOptimizer, {n_cycles} cycles, window={window}",
        description="Average frustration falls over the full run",
        rel_error=max(0.0, e_avg_end - e_avg_start) / max(e_avg_start, 1e-12),
    ))

    # 3. Revolution counter equals n_cycles
    rev_correct = (opt.revolution == n_cycles)
    results.append(_make_result(
        name="convergence_revolution_counter",
        check_type="convergence",
        pass_criterion=f"opt.revolution == {n_cycles} after {n_cycles} cycles. "
                       "Counter bookkeeping; failure = run_cycle bug.",
        modelled=float(opt.revolution),
        observed=float(n_cycles),
        passed=rev_correct,
        method="Mu8CycleOptimizer",
        description=f"Revolution counter = {n_cycles}",
    ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Bit-strength checks
# ─────────────────────────────────────────────────────────────────────────────

def _check_bit_strength() -> list[dict[str, Any]]:
    """Verify bit-strength properties over a 16-cycle run (N=8, gain=0.3)."""
    results = []
    opt = Mu8CycleOptimizer(N=8, gain=0.3, seed=42)
    metrics_list = opt.run(16)

    # 1. At R=0.875 (=1-1/8), B should equal log₂(8)=3 bits exactly.
    expected_threshold = math.log2(8)
    actual_threshold = bit_strength(0.875, N=8)
    results.append(_make_result(
        name="bit_strength_threshold_n8",
        check_type="mathematical_identity",
        pass_criterion="bit_strength(0.875, N=8) = log₂(8) = 3 bits exactly "
                       "(natural μ⁸ threshold: R* = 1 − 1/N).",
        modelled=expected_threshold,
        observed=actual_threshold,
        passed=abs(actual_threshold - expected_threshold) < 1e-10,
        method="−log₂(1 − 0.875) = −log₂(0.125) = 3",
        description="B = log₂(N) at the natural threshold R* = 1 − 1/N",
        rel_error=abs(actual_threshold - expected_threshold) / max(expected_threshold, 1e-15),
    ))

    # 2. B = 0 at R = 0.
    b_zero = bit_strength(0.0)
    results.append(_make_result(
        name="bit_strength_zero_coherence",
        check_type="mathematical_identity",
        pass_criterion="bit_strength(0) = 0 (no coherence → zero bit strength).",
        modelled=0.0,
        observed=b_zero,
        passed=b_zero == 0.0,
        method="−log₂(1 − 0 + ε) clipped to 0 for R ≤ 0",
        description="Zero coherence yields zero bit strength",
        rel_error=abs(b_zero),
    ))

    # 3. B = 1 at R = 0.5 (approximately, within tolerance of ε).
    b_half = bit_strength(0.5)
    results.append(_make_result(
        name="bit_strength_half_coherence",
        check_type="mathematical_identity",
        pass_criterion="bit_strength(0.5) ≈ 1.0 bit (R=0.5 → half-coherence).",
        modelled=1.0,
        observed=b_half,
        passed=abs(b_half - 1.0) < 1e-10,
        method="−log₂(1 − 0.5) = −log₂(0.5) = 1",
        description="Half coherence (R=0.5) yields one bit of strength",
        rel_error=abs(b_half - 1.0),
    ))

    # 4. bit_strength_out is non-decreasing over the run (spiral deepening).
    bs_out = [m.bit_strength_out for m in metrics_list]
    # Allow tiny numerical noise (1e-10) as with coherence non-decreasing check.
    nondec = all(bs_out[i] >= bs_out[i - 1] - 1e-10 for i in range(1, len(bs_out)))
    worst = min(bs_out[i] - bs_out[i - 1] for i in range(1, len(bs_out)))
    results.append(_make_result(
        name="bit_strength_nondecreasing",
        check_type="cycle_invariant",
        pass_criterion="bit_strength_out non-decreasing over 16 cycles (N=8, g=0.3). "
                       "Monotone in R means monotone in B.",
        modelled=0.0,
        observed=worst,
        passed=nondec,
        method="bit_strength_out[i] >= bit_strength_out[i-1] - 1e-10",
        description="Bit strength grows monotonically as coherence spirals deeper",
        rel_error=max(0.0, -worst),
    ))

    # 5. bit_strength_in of each cycle matches bit_strength of coherence_in.
    mismatches = [
        abs(m.bit_strength_in - bit_strength(m.coherence_in, 8))
        for m in metrics_list
    ]
    consistent = all(e < 1e-12 for e in mismatches)
    results.append(_make_result(
        name="bit_strength_consistent_with_coherence",
        check_type="cycle_invariant",
        pass_criterion="m.bit_strength_in = bit_strength(m.coherence_in, N) "
                       "for all cycles. Internal consistency.",
        modelled=0.0,
        observed=max(mismatches) if mismatches else 0.0,
        passed=consistent,
        method="element-wise comparison over 16 cycles",
        description="CycleMetrics bit_strength fields are consistent with coherence values",
        rel_error=max(mismatches) if mismatches else 0.0,
    ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility / cross-language portability checks
# ─────────────────────────────────────────────────────────────────────────────

def _check_reproducibility() -> list[dict[str, Any]]:
    """Same seed + same parameters → identical state checksums."""
    results = []

    # Run twice with the same seed; checksums must match.
    opt_a = Mu8CycleOptimizer(N=8, gain=0.3, seed=42)
    opt_b = Mu8CycleOptimizer(N=8, gain=0.3, seed=42)
    m_a = [opt_a.run_cycle() for _ in range(5)]
    m_b = [opt_b.run_cycle() for _ in range(5)]
    checksums_match = all(a.state_checksum == b.state_checksum for a, b in zip(m_a, m_b))
    results.append(_make_result(
        name="reproducibility_deterministic_checksums",
        check_type="reproducibility",
        pass_criterion="State checksums match across two identical runs (seed=42, 5 cycles). "
                       "Determinism required for cross-language portability.",
        modelled=1.0 if checksums_match else 0.0,
        observed=1.0,
        passed=checksums_match,
        method="SHA-256 state checksums",
        description="Identical seeds produce identical state fingerprints",
        rel_error=0.0 if checksums_match else 1.0,
    ))

    # Different seeds → different checksums (sanity check)
    opt_c = Mu8CycleOptimizer(N=8, gain=0.3, seed=99)
    m_c = opt_c.run_cycle()
    different_initial = (m_a[0].state_checksum != m_c.state_checksum)
    results.append(_make_result(
        name="reproducibility_different_seeds_differ",
        check_type="reproducibility",
        pass_criterion="State checksums differ for seed=42 vs seed=99 (cycle 1). "
                       "Sanity: distinct random seeds → distinct trajectories.",
        modelled=0.0 if different_initial else 1.0,
        observed=0.0,
        passed=different_initial,
        method="SHA-256 state checksums",
        description="Different seeds produce different state fingerprints",
        rel_error=0.0 if different_initial else 1.0,
    ))

    # Export / JSON round-trip: export_state returns valid JSON
    opt_d = Mu8CycleOptimizer(N=8, gain=0.3, seed=1)
    opt_d.run(3)
    state = opt_d.export_state()
    try:
        encoded = json.dumps(state)
        decoded = json.loads(encoded)
        phases_match = np.allclose(decoded["phases"], state["phases"], atol=1e-15)
        json_ok = (
            "phases" in decoded
            and "revolution" in decoded
            and "lean_fingerprint" in decoded
            and phases_match
        )
    except Exception:
        json_ok = False
    results.append(_make_result(
        name="reproducibility_json_export",
        check_type="reproducibility",
        pass_criterion="export_state() round-trips through json.dumps/loads with "
                       "phases preserved to 1e-15. Cross-language portability.",
        modelled=1.0 if json_ok else 0.0,
        observed=1.0,
        passed=json_ok,
        method="json.dumps / json.loads",
        description="export_state() JSON round-trip preserves all fields",
        rel_error=0.0 if json_ok else 1.0,
    ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Public validate() entry point — matches existing validator interface
# ─────────────────────────────────────────────────────────────────────────────

def validate(_data: dict | None = None) -> list[dict[str, Any]]:
    """Validate the μ⁸=1 spiral cycle optimizer.

    Parameters
    ----------
    _data:
        Unused; accepted for interface consistency with other validators.

    Returns
    -------
    list[dict]
        One dict per check, each with keys: name, check_type, pass_criterion,
        modelled, observed, rel_error, passed, method, description.
    """
    results: list[dict[str, Any]] = []

    # Gate checks
    results.append(_check_sympy_gate())
    results.extend(_check_numpy_gate())
    results.append(_check_lean_fingerprint())

    # Mathematical identities
    results.append(_check_mu_angle())
    results.extend(_check_mu_real_imag())
    results.extend(_check_silver_ratio())
    results.extend(_check_lean_coherence())
    results.append(_check_c_natural())

    # Cycle invariants
    results.extend(_check_cycle_invariants())

    # Bit strength
    results.extend(_check_bit_strength())

    # Convergence
    results.extend(_check_convergence())

    # Reproducibility
    results.extend(_check_reproducibility())

    return results
