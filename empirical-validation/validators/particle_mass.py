"""
Particle mass ratio validator.
================================
Validates the proton-to-electron mass ratio and related constructs that
appear in ``ParticleMass.lean``.

Check-type taxonomy used in this module
----------------------------------------
``empirical``
    Comparisons between a *predicted/derived* value and an *independent*
    external reference (CODATA 2018, PDG 2022).  These checks CAN fail
    and distinguish whether the framework is consistent with measurement.

``numerical_precision``
    Verifies that SymPy rational arithmetic agrees with the floating-point
    CODATA value to within rounding tolerance.

Validation checks
-----------------
  1. CODATA 2018 direct ratio vs reference value — EMPIRICAL
     Verifies that data ingestion returned the correct ratio.
     Reference: CODATA 2018, m_p/m_e = 1836.15267343.
     Tolerance: 1e-9 (CODATA precision).

  2. Ratio reconstructed from individual CODATA masses — EMPIRICAL
     Tests internal CODATA consistency: m_p/m_e derived from m_p and m_e
     separately must agree with the direct tabulation to 1 ppm.
     A discrepancy > 1e-6 would indicate a data ingestion error.

  3. Koide formula Q = 2/3 — EMPIRICAL
     Q = (m_e + m_μ + m_τ) / (√m_e + √m_μ + √m_τ)² uses PDG 2022 measured
     lepton masses; the predicted value is Q = 2/3 (Koide, 1982).
     Tolerance 1e-3 (0.1%) reflects PDG 2022 mass uncertainties.
     Failure would mean PDG masses contradict the Koide relation.

  4. Ratio within 0.1% of integer 1836 — EMPIRICAL
     The Kernel framework treats 1836 as a round-number approximation.
     Tolerance 1e-3 (0.1%) is set to flag gross data errors.

  5. Wyler approximation 6π⁵ ≈ m_p/m_e — EMPIRICAL
     A well-known numerical coincidence: 6π⁵ = 1836.118…
     Tolerance 5e-4 (0.05%); failure means CODATA has changed significantly
     relative to this approximation's known accuracy.

  6. SymPy rational representation — NUMERICAL_PRECISION
     Verifies SymPy can represent the CODATA value; not an empirical check.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import sympy as sp


# CODATA 2018 particle masses (PDG 2022 for lepton masses)
_M_E_KG   = 9.1093837015e-31   # electron mass (kg)
_M_P_KG   = 1.67262192369e-27  # proton mass (kg)
_M_E_MEV  = 0.51099895000      # electron mass (MeV/c²)
_M_MU_MEV = 105.6583755        # muon mass (MeV/c²)     PDG 2022
_M_TAU_MEV = 1776.86           # tau mass (MeV/c²)      PDG 2022


def _koide_q(me: float, mmu: float, mtau: float) -> float:
    """Compute Koide Q = (m_e + m_μ + m_τ) / (√m_e + √m_μ + √m_τ)²."""
    num = me + mmu + mtau
    den = (math.sqrt(me) + math.sqrt(mmu) + math.sqrt(mtau)) ** 2
    return num / den


def validate(data: dict | None = None) -> list[dict[str, Any]]:
    """Validate particle mass ratio properties.

    Parameters
    ----------
    data:
        Combined dataset dict.  CODATA values for masses are used when
        available; built-in CODATA 2018 values are the fallback.

    Returns
    -------
    list[dict]
        Each dict includes ``check_type`` and ``pass_criterion``.
    """
    data = data or {}
    results: list[dict[str, Any]] = []

    # Fetch from ingested data if available
    m_e = data.get("electron_mass", {}).get("value", _M_E_KG)
    m_p = data.get("proton_mass",   {}).get("value", _M_P_KG)
    ratio_ref = data.get("proton_electron_mass_ratio", {}).get("value", 1836.15267343)

    # ── 1. Direct CODATA ratio ──────────────────────────────────────────────
    results.append({
        "name": "proton_electron_mass_ratio_codata",
        "check_type": "empirical",
        "pass_criterion": (
            "Relative error |ratio_ingested − 1836.15267343| / 1836.15267343 < 1e-9. "
            "Reference: CODATA 2018 direct measurement. "
            "Failure means data ingestion returned a wrong value."
        ),
        "modelled": ratio_ref,
        "observed": 1836.15267343,
        "rel_error": abs(ratio_ref - 1836.15267343) / 1836.15267343,
        "passed": abs(ratio_ref - 1836.15267343) / 1836.15267343 < 1e-9,
        "method": "CODATA 2018 direct",
        "description": "m_p/m_e ≈ 1836.15267343 (CODATA 2018)",
    })

    # ── 2. Reconstructed ratio from individual masses ───────────────────────
    ratio_computed = m_p / m_e
    rel_err_2 = abs(ratio_computed - ratio_ref) / ratio_ref
    results.append({
        "name": "proton_electron_mass_ratio_reconstructed",
        "check_type": "empirical",
        "pass_criterion": (
            "Relative error |m_p/m_e − ratio_direct| / ratio_direct < 1e-6. "
            "Tests CODATA internal consistency between individual masses and the "
            "tabulated ratio. Failure means the ingested m_p and m_e are inconsistent."
        ),
        "modelled": ratio_computed,
        "observed": ratio_ref,
        "rel_error": rel_err_2,
        "passed": rel_err_2 < 1e-6,
        "method": "NumPy from individual CODATA masses",
        "description": "m_p/m_e reconstructed from individual masses",
    })

    # ── 3. Koide formula  Q = 2/3 ──────────────────────────────────────────
    Q = _koide_q(_M_E_MEV, _M_MU_MEV, _M_TAU_MEV)
    Q_expected = 2.0 / 3.0
    rel_err_3 = abs(Q - Q_expected) / Q_expected
    results.append({
        "name": "koide_formula",
        "check_type": "empirical",
        "pass_criterion": (
            "Relative error |Q − 2/3| / (2/3) < 1e-3 (0.1%). "
            "Q computed from PDG 2022 lepton masses; Koide (1982) predicts Q = 2/3. "
            "Tolerance 0.1% covers PDG 2022 τ-mass uncertainty (~0.006%). "
            "Failure means measured lepton masses deviate significantly from prediction."
        ),
        "modelled": Q,
        "observed": Q_expected,
        "rel_error": rel_err_3,
        "passed": rel_err_3 < 1e-3,
        "method": "NumPy from PDG 2022 lepton masses",
        "description": "Koide Q = (m_e+m_μ+m_τ)/(√m_e+√m_μ+√m_τ)² ≈ 2/3",
    })

    # ── 4. Small fractional deviation from integer 1836 ────────────────────
    frac_dev = abs(ratio_ref - 1836.0) / 1836.0
    results.append({
        "name": "proton_electron_mass_ratio_near_1836",
        "check_type": "empirical",
        "pass_criterion": (
            "|m_p/m_e − 1836| / 1836 < 1e-3 (0.1%). "
            "The Kernel framework uses 1836 as a first-order approximation. "
            "Tolerance 0.1% flags gross data errors while allowing for the "
            "known 0.008% deviation of the true ratio from 1836."
        ),
        "modelled": ratio_ref,
        "observed": 1836.0,
        "rel_error": frac_dev,
        "passed": frac_dev < 1e-3,
        "method": "numeric comparison",
        "description": "m_p/m_e within 0.1% of integer 1836",
    })

    # ── 5. Wyler-type approximation: m_p/m_e ≈ 6π⁵ ─────────────────────────
    wyler = 6.0 * math.pi ** 5
    rel_err_5 = abs(wyler - ratio_ref) / ratio_ref
    results.append({
        "name": "proton_electron_mass_ratio_wyler",
        "check_type": "empirical",
        "pass_criterion": (
            "Relative error |6π⁵ − m_p/m_e| / (m_p/m_e) < 5e-4 (0.05%). "
            "6π⁵ = 1836.118… is an empirical approximation (Wyler-type). "
            "Known accuracy ~0.02%; tolerance 0.05% gives headroom for CODATA updates. "
            "Failure means CODATA has changed enough to break this approximation."
        ),
        "modelled": wyler,
        "observed": ratio_ref,
        "rel_error": rel_err_5,
        "passed": rel_err_5 < 5e-4,
        "method": "NumPy: 6π⁵",
        "description": "m_p/m_e ≈ 6π⁵ (Wyler approximation, ~0.05%)",
    })

    # ── 6. SymPy exact: ratio_ref is close to 1836 + 0.15267343 ────────────
    # Use str() conversion to avoid precision loss from float→Rational.
    ratio_sym = sp.Rational(str(round(ratio_ref, 8)))
    modelled_val = float(ratio_sym)
    rel_err_6 = abs(modelled_val - ratio_ref) / ratio_ref
    results.append({
        "name": "proton_electron_mass_ratio_sympy_rational",
        "check_type": "numerical_precision",
        "pass_criterion": (
            "Relative error between SymPy rational representation and CODATA float < 1e-8. "
            "Verifies SymPy str→Rational round-trip precision; not an empirical check."
        ),
        "modelled": modelled_val,
        "observed": ratio_ref,
        "rel_error": rel_err_6,
        "passed": rel_err_6 < 1e-8,
        "method": "SymPy rational approximation",
        "description": "m_p/m_e represented as SymPy rational matches CODATA",
    })

    return results


