"""
Fine-structure constant validator.
====================================
Validates the fine-structure constant α against CODATA 2018 reference data
and checks key relationships established in ``FineStructure.lean``:

  1. α ≈ 7.2973525693 × 10⁻³  (CODATA 2018 reference)
  2. 1/α ≈ 137.035999084       (reciprocal)
  3. α = e² / (4πε₀ℏc)        (definition, dimensionless — verified via
                                 CODATA values of e, ℏ, c; ε₀ = 1/(μ₀c²))
  4. α < 1/137                 (sub-unity check)
  5. α × 137 ≈ 1               (near-unity product with nominal 137)

Both exact SymPy and numerical NumPy checks are included.
"""

from __future__ import annotations

import math
from typing import Any

import sympy as sp


# CODATA 2018 exact values used in the definition check
_ALPHA_CODATA = 7.2973525693e-3
_E = 1.602176634e-19        # elementary charge (C)   — exact since 2019 SI
_HBAR = 1.054571817e-34     # ℏ (J·s)                 — exact since 2019 SI
_C = 299_792_458.0          # speed of light (m/s)    — exact
# ε₀ = 1/(μ₀c²); CODATA 2018 value (μ₀ no longer exact post-2019 SI redefinition)
_EPS0 = 8.8541878188e-12    # ε₀ (F/m)                — CODATA 2018 (scipy value)


def _compute_alpha_from_definition() -> float:
    """Reconstruct α from its definition α = e² / (4πε₀ℏc)."""
    return (_E ** 2) / (4 * math.pi * _EPS0 * _HBAR * _C)


def validate(data: dict | None = None) -> list[dict[str, Any]]:
    """Validate fine-structure constant properties.

    Parameters
    ----------
    data:
        Combined dataset dict.  If it contains ``"fine_structure_constant"``
        (from the CODATA ingestion module) that value is used as the
        observed reference; otherwise the built-in CODATA 2018 value is used.

    Returns
    -------
    list[dict]
        One result dict per individual check.
    """
    data = data or {}
    alpha_ref = (
        data.get("fine_structure_constant", {}).get("value", _ALPHA_CODATA)
    )
    results: list[dict[str, Any]] = []

    # ── 1. Reconstructed α vs CODATA reference ─────────────────────────────
    alpha_computed = _compute_alpha_from_definition()
    rel_err_1 = abs(alpha_computed - alpha_ref) / alpha_ref
    results.append({
        "name": "fine_structure_constant_definition",
        "modelled": alpha_computed,
        "observed": alpha_ref,
        "rel_error": rel_err_1,
        "passed": rel_err_1 < 1e-9,
        "method": "NumPy from definition e²/(4πε₀ℏc)",
        "description": "α reconstructed from definition matches CODATA 2018",
    })

    # ── 2. Reciprocal 1/α ≈ 137.035999084 ─────────────────────────────────
    inv_alpha = 1.0 / alpha_ref
    inv_alpha_nominal = 137.035999084
    rel_err_2 = abs(inv_alpha - inv_alpha_nominal) / inv_alpha_nominal
    results.append({
        "name": "fine_structure_constant_inverse",
        "modelled": inv_alpha,
        "observed": inv_alpha_nominal,
        "rel_error": rel_err_2,
        "passed": rel_err_2 < 1e-9,
        "method": "NumPy arithmetic",
        "description": "1/α ≈ 137.035999084",
    })

    # ── 3. α < 1 (weak coupling) ───────────────────────────────────────────
    results.append({
        "name": "fine_structure_constant_sub_unity",
        "modelled": alpha_ref,
        "observed": 1.0,
        "rel_error": alpha_ref,
        "passed": alpha_ref < 1.0,
        "method": "numeric comparison",
        "description": "α < 1  (weak electromagnetic coupling)",
    })

    # ── 4. α × 137 close to 1 ──────────────────────────────────────────────
    prod = alpha_ref * 137
    rel_err_4 = abs(prod - 1.0)
    results.append({
        "name": "fine_structure_constant_times_137",
        "modelled": prod,
        "observed": 1.0,
        "rel_error": rel_err_4,
        "passed": rel_err_4 < 3e-4,          # within 0.03 % of unity
        "method": "NumPy arithmetic",
        "description": "α × 137 ≈ 1  (within 0.03 %)",
    })

    # ── 5. SymPy symbolic: α reciprocal ────────────────────────────────────
    alpha_sym = sp.Rational(72973525693, 10000000000000)   # CODATA 2018 value
    inv_sym = sp.Integer(1) / alpha_sym
    inv_numeric = float(inv_sym.evalf())
    rel_err_5 = abs(inv_numeric - inv_alpha_nominal) / inv_alpha_nominal
    results.append({
        "name": "fine_structure_constant_inverse_sympy",
        "modelled": inv_numeric,
        "observed": inv_alpha_nominal,
        "rel_error": rel_err_5,
        "passed": rel_err_5 < 1e-9,
        "method": "SymPy rational arithmetic",
        "description": "1/α from SymPy rational matches CODATA 2018",
    })

    # ── 6. α < 1/137 is FALSE (α is slightly *larger* than 1/137.036) ──────
    # The actual CODATA value α = 7.297…×10⁻³ > 1/137 = 7.299…×10⁻³? No:
    # 1/137 = 0.007299... but α = 0.007297... so α < 1/137 is TRUE.
    alpha_lt_1_over_137 = alpha_ref < (1.0 / 137.0)
    results.append({
        "name": "fine_structure_constant_lt_1_over_137",
        "modelled": alpha_ref,
        "observed": 1.0 / 137.0,
        "rel_error": abs(alpha_ref - 1.0 / 137.0) / (1.0 / 137.0),
        "passed": alpha_lt_1_over_137,
        "method": "numeric comparison",
        "description": "α < 1/137  (fine-tuning check)",
    })

    return results
