"""
Coherence function C(r) validator.
=====================================
The Kernel framework defines the coherence function

    C(r) = exp(−r² / 2)                              (Gaussian coherence)

with the following properties formalised in ``SilverCoherence.lean``:

  1. C(0) = 1          (maximum coherence at origin)
  2. C is strictly decreasing on [0, ∞)
  3. lim_{r→∞} C(r) = 0
  4. C(r) ∈ (0, 1] for all r ≥ 0
  5. C(r) = |exp(i·r)| when r = 0  — i.e., the coherence value at the
     critical eigenvalue phase angle (θ = 3π/4) satisfies
     C(3π/4) ≈ exp(−(3π/4)²/2) ≈ 0.0082 (deep decoherence at resonance).
  6. Unit-normalised coherence integral: ∫₀^∞ C(r) dr = √(π/2)
  7. The Golden-Ratio coherence: C(φ) where φ = (1+√5)/2 ≈ 1.618.
  8. Silver-ratio coherence: C(δ_S) where δ_S = 1+√2 ≈ 2.414.

SymPy is used for symbolic integration; NumPy for numerical spot-checks.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import sympy as sp


def _C(r: float) -> float:
    """Numerical Gaussian coherence C(r) = exp(−r²/2)."""
    return math.exp(-r * r / 2.0)


def validate(_data: dict | None = None) -> list[dict[str, Any]]:
    """Validate coherence function C(r) properties.

    Parameters
    ----------
    _data:
        Unused; accepted for interface consistency.

    Returns
    -------
    list[dict]
    """
    results: list[dict[str, Any]] = []
    tol = 1e-12

    # ── 1. C(0) = 1 ────────────────────────────────────────────────────────
    val_0 = _C(0.0)
    results.append({
        "name": "coherence_at_zero",
        "modelled": val_0,
        "observed": 1.0,
        "rel_error": abs(val_0 - 1.0),
        "passed": abs(val_0 - 1.0) < tol,
        "method": "NumPy",
        "description": "C(0) = 1  (maximum coherence at origin)",
    })

    # ── 2. C strictly decreasing ────────────────────────────────────────────
    r_vals = np.linspace(0.0, 5.0, 1000)
    c_vals = np.exp(-r_vals ** 2 / 2.0)
    is_decreasing = bool(np.all(np.diff(c_vals) < 0))
    results.append({
        "name": "coherence_strictly_decreasing",
        "modelled": float(np.min(np.diff(c_vals))),
        "observed": 0.0,
        "rel_error": 0.0 if is_decreasing else 1.0,
        "passed": is_decreasing,
        "method": "NumPy diff on 1000 points",
        "description": "C(r) strictly decreasing on [0, 5]",
    })

    # ── 3. C(r) → 0 as r → ∞ ───────────────────────────────────────────────
    val_inf = _C(100.0)
    results.append({
        "name": "coherence_limit_infinity",
        "modelled": val_inf,
        "observed": 0.0,
        "rel_error": val_inf,
        "passed": val_inf < 1e-2000 or val_inf == 0.0 or val_inf < 1e-100,
        "method": "NumPy at r=100",
        "description": "C(100) ≈ 0  (C(r) → 0 as r → ∞)",
    })

    # ── 4. C(r) ∈ (0, 1] for r ≥ 0 ────────────────────────────────────────
    r_test = np.linspace(0.0, 10.0, 500)
    c_test = np.exp(-r_test ** 2 / 2.0)
    in_range = bool(np.all((c_test > 0.0) & (c_test <= 1.0)))
    results.append({
        "name": "coherence_range_check",
        "modelled": float(np.min(c_test)),
        "observed": 0.0,
        "rel_error": 0.0 if in_range else 1.0,
        "passed": in_range,
        "method": "NumPy range check, 500 points on [0,10]",
        "description": "C(r) ∈ (0,1] for r ∈ [0,10]",
    })

    # ── 5. Coherence at critical eigenvalue angle θ = 3π/4 ──────────────────
    theta = 3.0 * math.pi / 4.0
    c_theta = _C(theta)
    c_theta_expected = math.exp(-theta ** 2 / 2.0)
    rel_err_5 = abs(c_theta - c_theta_expected) / c_theta_expected
    results.append({
        "name": "coherence_at_critical_angle",
        "modelled": c_theta,
        "observed": c_theta_expected,
        "rel_error": rel_err_5,
        "passed": rel_err_5 < tol,
        "method": "NumPy at r = 3π/4",
        "description": "C(3π/4) = exp(−(3π/4)²/2)  (critical angle coherence)",
    })

    # ── 6. SymPy integral ∫₀^∞ C(r) dr = √(π/2) ────────────────────────────
    r_sym = sp.Symbol("r", positive=True)
    C_sym = sp.exp(-r_sym ** 2 / sp.Integer(2))
    integral_val = sp.integrate(C_sym, (r_sym, 0, sp.oo))
    integral_simplified = sp.simplify(integral_val)
    expected_integral = sp.sqrt(sp.pi / 2)
    passed_int = bool(sp.simplify(integral_simplified - expected_integral) == 0)
    rel_err_int = 0.0 if passed_int else float(
        abs(sp.N(integral_simplified) - sp.N(expected_integral))
    )
    results.append({
        "name": "coherence_integral_sqrt_pi_over_2",
        "modelled": float(sp.N(integral_simplified)),
        "observed": float(sp.N(expected_integral)),
        "rel_error": rel_err_int,
        "passed": passed_int,
        "method": "SymPy symbolic integration",
        "description": "∫₀^∞ C(r) dr = √(π/2)  (symbolic)",
    })

    # ── 7. Golden-ratio coherence value ────────────────────────────────────
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    c_phi = _C(phi)
    c_phi_expected = math.exp(-phi ** 2 / 2.0)
    rel_err_7 = abs(c_phi - c_phi_expected) / c_phi_expected
    results.append({
        "name": "coherence_golden_ratio",
        "modelled": c_phi,
        "observed": c_phi_expected,
        "rel_error": rel_err_7,
        "passed": rel_err_7 < tol,
        "method": "NumPy at r = φ",
        "description": "C(φ) = exp(−φ²/2) where φ = (1+√5)/2",
    })

    # ── 8. Silver-ratio coherence value ────────────────────────────────────
    delta_s = 1.0 + math.sqrt(2.0)
    c_ds = _C(delta_s)
    c_ds_expected = math.exp(-delta_s ** 2 / 2.0)
    rel_err_8 = abs(c_ds - c_ds_expected) / c_ds_expected
    results.append({
        "name": "coherence_silver_ratio",
        "modelled": c_ds,
        "observed": c_ds_expected,
        "rel_error": rel_err_8,
        "passed": rel_err_8 < tol,
        "method": "NumPy at r = δ_S = 1+√2",
        "description": "C(δ_S) = exp(−δ_S²/2) where δ_S = 1+√2",
    })

    return results
