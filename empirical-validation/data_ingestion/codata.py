"""
CODATA 2018 physical constants ingestion module.
=================================================
Sources
-------
* Primary   : ``scipy.constants`` — CODATA 2018 values shipped with SciPy.
* Fallback  : Hard-coded CODATA 2018 recommended values from the NIST/BIPM
              publication (https://physics.nist.gov/cuu/Constants/).

``load()`` returns a dict whose keys are human-readable constant names and
whose values are ``{"value": float, "unit": str, "source": str}`` records.
The ``"source"`` field distinguishes live SciPy values from the built-in
fallback so that callers can log the data provenance.

Only constants directly relevant to the mathematical constructs validated
in this repository are included:

  - Fine-structure constant  (α)
  - Elementary charge        (e)
  - Speed of light           (c)
  - Planck constant          (h, ℏ)
  - Boltzmann constant       (k_B)
  - Electron rest mass       (m_e)
  - Proton rest mass         (m_p)
  - Proton/electron mass ratio
  - Avogadro constant        (N_A)
  - Gravitational constant   (G)
  - Electron magnetic moment anomaly (a_e)
"""

from __future__ import annotations

from typing import Any

# ──────────────────────────────────────────────────────────────────────────────
# Fallback: CODATA 2018 recommended values
# Source: https://physics.nist.gov/cuu/Constants/
# ──────────────────────────────────────────────────────────────────────────────
_CODATA_2018_FALLBACK: dict[str, dict[str, Any]] = {
    "fine_structure_constant": {
        "value": 7.2973525693e-3,
        "unit": "dimensionless",
        "source": "CODATA 2018 fallback",
    },
    "elementary_charge": {
        "value": 1.602176634e-19,
        "unit": "C",
        "source": "CODATA 2018 fallback",
    },
    "speed_of_light": {
        "value": 299_792_458.0,
        "unit": "m/s",
        "source": "CODATA 2018 fallback",
    },
    "planck_constant": {
        "value": 6.62607015e-34,
        "unit": "J·s",
        "source": "CODATA 2018 fallback",
    },
    "reduced_planck_constant": {
        "value": 1.054571817e-34,
        "unit": "J·s",
        "source": "CODATA 2018 fallback",
    },
    "boltzmann_constant": {
        "value": 1.380649e-23,
        "unit": "J/K",
        "source": "CODATA 2018 fallback",
    },
    "electron_mass": {
        "value": 9.1093837015e-31,
        "unit": "kg",
        "source": "CODATA 2018 fallback",
    },
    "proton_mass": {
        "value": 1.67262192369e-27,
        "unit": "kg",
        "source": "CODATA 2018 fallback",
    },
    "proton_electron_mass_ratio": {
        "value": 1836.15267343,
        "unit": "dimensionless",
        "source": "CODATA 2018 fallback",
    },
    "avogadro_constant": {
        "value": 6.02214076e23,
        "unit": "mol⁻¹",
        "source": "CODATA 2018 fallback",
    },
    "gravitational_constant": {
        "value": 6.67430e-11,
        "unit": "m³/(kg·s²)",
        "source": "CODATA 2018 fallback",
    },
    "electron_mag_moment_anomaly": {
        "value": 1.15965218128e-3,
        "unit": "dimensionless",
        "source": "CODATA 2018 fallback",
    },
}


def _load_from_scipy() -> dict[str, dict[str, Any]]:
    """Attempt to read constants from ``scipy.constants`` (CODATA 2018)."""
    import scipy.constants as sc  # type: ignore[import]

    # Map our key names to scipy attribute names and units.
    _SCIPY_MAP = {
        "fine_structure_constant": ("fine_structure",  "dimensionless"),
        "elementary_charge":       ("e",               "C"),
        "speed_of_light":          ("c",               "m/s"),
        "planck_constant":         ("h",               "J·s"),
        "reduced_planck_constant": ("hbar",            "J·s"),
        "boltzmann_constant":      ("k",               "J/K"),
        "electron_mass":           ("electron_mass",   "kg"),
        "proton_mass":             ("proton_mass",     "kg"),
        "avogadro_constant":       ("Avogadro",        "mol⁻¹"),
        "gravitational_constant":  ("G",               "m³/(kg·s²)"),
    }

    out: dict[str, dict[str, Any]] = {}
    for key, (attr, unit) in _SCIPY_MAP.items():
        val = getattr(sc, attr, None)
        if val is not None:
            out[key] = {"value": float(val), "unit": unit, "source": "scipy CODATA 2018"}

    # proton/electron mass ratio — derive from individual masses when not a
    # direct scipy attribute (scipy 1.10+ does not expose this as a constant).
    if "electron_mass" in out and "proton_mass" in out:
        ratio = out["proton_mass"]["value"] / out["electron_mass"]["value"]
        out["proton_electron_mass_ratio"] = {
            "value": ratio,
            "unit": "dimensionless",
            "source": "scipy CODATA 2018 derived: m_p / m_e",
        }

    # Electron magnetic moment anomaly is not directly in scipy.constants.
    # Compute from the value that is: a_e = (g_e − 2)/2 ≈ 1.15965218128e-3
    out["electron_mag_moment_anomaly"] = {
        "value": 1.15965218128e-3,
        "unit": "dimensionless",
        "source": "CODATA 2018 fallback (not in scipy)",
    }
    return out


def load() -> dict[str, dict[str, Any]]:
    """Return a dict of CODATA physical constants.

    Tries ``scipy.constants`` first; falls back to hard-coded CODATA 2018
    values if SciPy is unavailable.

    Returns
    -------
    dict
        ``{name: {"value": float, "unit": str, "source": str}}``
    """
    try:
        data = _load_from_scipy()
        if data:
            return data
    except Exception:  # noqa: BLE001
        pass
    return dict(_CODATA_2018_FALLBACK)
