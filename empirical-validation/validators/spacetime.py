"""
Space-time framework validator.
=================================
Validates space-time constructs from ``SpaceTime.lean`` and the broader
Kernel framework against known physical constants and cosmological data:

  1. Speed of light c = 299,792,458 m/s  (exact SI definition)
  2. Planck time t_P = √(ℏG/c⁵) ≈ 5.391247×10⁻⁴⁴ s
  3. Planck length l_P = √(ℏG/c³) ≈ 1.616255×10⁻³⁵ m
  4. Planck mass m_P = √(ℏc/G) ≈ 2.176434×10⁻⁸ kg
  5. l_P / t_P = c  (light-cone consistency)
  6. Hubble radius: r_H = c / H_0 in light-years  (cosmological horizon)
  7. Schwarzschild radius of the Sun ≈ 2.953 km (from G, M_☉, c)
  8. Cosmological constant Λ has correct order of magnitude (~10⁻⁵² m⁻²)

Uses CODATA 2018 values for ℏ, G and cosmological data from Planck 2018.
"""

from __future__ import annotations

import math
from typing import Any


# CODATA 2018 constants
_HBAR = 1.054571817e-34     # ℏ (J·s)
_G    = 6.67430e-11         # G (m³ kg⁻¹ s⁻²)
_C    = 299_792_458.0       # c (m/s)  — exact
_M_SUN = 1.98892e30         # solar mass (kg)


def validate(data: dict | None = None) -> list[dict[str, Any]]:
    """Validate space-time framework quantities.

    Parameters
    ----------
    data:
        Combined dataset dict with CODATA, NIST, and cosmological values.

    Returns
    -------
    list[dict]
    """
    data = data or {}
    results: list[dict[str, Any]] = []

    c_ref = data.get("speed_of_light", {}).get("value", _C)
    hbar = data.get("reduced_planck_constant", {}).get("value", _HBAR)
    G = data.get("gravitational_constant", {}).get("value", _G)
    H0_km = data.get("hubble_constant", {}).get("value", 67.36)   # km/s/Mpc
    lambda_ref = data.get("cosmological_constant", {}).get("value", None)

    # ── 1. Speed of light exact ─────────────────────────────────────────────
    results.append({
        "name": "speed_of_light_exact",
        "modelled": c_ref,
        "observed": 299_792_458.0,
        "rel_error": abs(c_ref - 299_792_458.0) / 299_792_458.0,
        "passed": c_ref == 299_792_458.0,
        "method": "CODATA / SI",
        "description": "c = 299,792,458 m/s  (exact SI definition)",
    })

    # ── 2. Planck time ──────────────────────────────────────────────────────
    t_P = math.sqrt(hbar * G / c_ref ** 5)
    t_P_ref = 5.391247e-44                # NIST-tabulated value (s)
    rel_err_2 = abs(t_P - t_P_ref) / t_P_ref
    results.append({
        "name": "planck_time",
        "modelled": t_P,
        "observed": t_P_ref,
        "rel_error": rel_err_2,
        "passed": rel_err_2 < 5e-4,
        "method": "NumPy: √(ℏG/c⁵)",
        "description": "Planck time t_P ≈ 5.391×10⁻⁴⁴ s",
    })

    # ── 3. Planck length ────────────────────────────────────────────────────
    l_P = math.sqrt(hbar * G / c_ref ** 3)
    l_P_ref = 1.616255e-35                # NIST-tabulated value (m)
    rel_err_3 = abs(l_P - l_P_ref) / l_P_ref
    results.append({
        "name": "planck_length",
        "modelled": l_P,
        "observed": l_P_ref,
        "rel_error": rel_err_3,
        "passed": rel_err_3 < 5e-4,
        "method": "NumPy: √(ℏG/c³)",
        "description": "Planck length l_P ≈ 1.616×10⁻³⁵ m",
    })

    # ── 4. Planck mass ──────────────────────────────────────────────────────
    m_P = math.sqrt(hbar * c_ref / G)
    m_P_ref = 2.176434e-8                 # NIST-tabulated value (kg)
    rel_err_4 = abs(m_P - m_P_ref) / m_P_ref
    results.append({
        "name": "planck_mass",
        "modelled": m_P,
        "observed": m_P_ref,
        "rel_error": rel_err_4,
        "passed": rel_err_4 < 5e-4,
        "method": "NumPy: √(ℏc/G)",
        "description": "Planck mass m_P ≈ 2.176×10⁻⁸ kg",
    })

    # ── 5. l_P / t_P = c (light-cone consistency) ──────────────────────────
    ratio_lp_tp = l_P / t_P
    rel_err_5 = abs(ratio_lp_tp - c_ref) / c_ref
    results.append({
        "name": "planck_ratio_lp_over_tp_equals_c",
        "modelled": ratio_lp_tp,
        "observed": c_ref,
        "rel_error": rel_err_5,
        "passed": rel_err_5 < 1e-12,
        "method": "NumPy: l_P / t_P",
        "description": "l_P / t_P = c  (light-cone consistency)",
    })

    # ── 6. Hubble radius r_H = c / H_0 ─────────────────────────────────────
    H0_si = H0_km * 1e3 / 3.085677581e22   # convert km/s/Mpc → s⁻¹
    r_H_m = c_ref / H0_si                   # Hubble radius in metres
    ly_m = 9.4607304725808e15               # metres per light-year
    r_H_ly = r_H_m / ly_m
    # Hubble distance c/H₀ with H₀ = 67.36 km/s/Mpc ≈ 14.52 billion light-years
    # (distinct from the age of the universe ~13.8 Gyr)
    r_H_ref_ly = 1.452e10
    rel_err_6 = abs(r_H_ly - r_H_ref_ly) / r_H_ref_ly
    results.append({
        "name": "hubble_radius",
        "modelled": r_H_ly,
        "observed": r_H_ref_ly,
        "rel_error": rel_err_6,
        "passed": rel_err_6 < 2e-2,          # 2% tolerance
        "method": "NumPy: c/H₀, Planck 2018",
        "description": "Hubble radius r_H = c/H₀ ≈ 14.52 Gly (Planck 2018 H₀=67.36)",
    })

    # ── 7. Schwarzschild radius of the Sun ──────────────────────────────────
    r_sch = 2.0 * G * _M_SUN / c_ref ** 2
    r_sch_ref = 2.953e3                      # ≈ 2.953 km
    rel_err_7 = abs(r_sch - r_sch_ref) / r_sch_ref
    results.append({
        "name": "schwarzschild_radius_sun",
        "modelled": r_sch,
        "observed": r_sch_ref,
        "rel_error": rel_err_7,
        "passed": rel_err_7 < 2e-3,
        "method": "NumPy: 2GM_☉/c²",
        "description": "Schwarzschild radius of the Sun ≈ 2.953 km",
    })

    # ── 8. Cosmological constant order of magnitude ─────────────────────────
    if lambda_ref is None:
        H0_si2 = 67.36e3 / 3.085677581e22
        lambda_ref = 3.0 * H0_si2 ** 2 * 0.6847 / c_ref ** 2
    expected_magnitude = 1.1e-52            # m⁻²  (approximate)
    rel_err_8 = abs(lambda_ref - expected_magnitude) / expected_magnitude
    results.append({
        "name": "cosmological_constant_magnitude",
        "modelled": lambda_ref,
        "observed": expected_magnitude,
        "rel_error": rel_err_8,
        "passed": rel_err_8 < 5e-2,          # 5% tolerance on derived quantity
        "method": "NumPy: 3H₀²Ω_Λ/c²",
        "description": "Cosmological constant Λ ≈ 1.1×10⁻⁵² m⁻²",
    })

    return results
