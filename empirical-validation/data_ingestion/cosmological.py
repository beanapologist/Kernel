"""
Cosmological and astrophysical data ingestion module.
======================================================
Sources
-------
* Planck 2018 Results (Planck Collaboration, A&A 641, A6, 2020)
  https://doi.org/10.1051/0004-6361/201833910
* PDG 2022 Review of Particle Physics
  https://pdg.lbl.gov/
* IAU 2012 System of Astronomical Constants
  https://www.iau.org/
* NASA/JPL Solar System Dynamics

``load()`` returns a dict of cosmological and astrophysical constants
following the same ``{"value", "unit", "source"}`` schema used by the
other data-ingestion modules.

Constants included
------------------
Cosmological parameters (Planck 2018, TT,TE,EE+lowE+lensing)
  * Hubble constant H_0
  * Baryon density Ω_b·h²
  * Cold dark matter density Ω_c·h²
  * Dark energy density Ω_Λ
  * Optical depth to reionisation τ
  * Scalar spectral index n_s
  * Amplitude of primordial perturbations A_s
  * CMB temperature T_CMB

Astrophysical / particle-physics constants
  * Solar mass M_☉
  * Solar luminosity L_☉
  * Solar radius R_☉
  * Parsec (pc) in metres
  * Light year (ly) in metres
  * Schwarzschild radius of the Sun
  * Proton-neutron mass difference
  * Neutron lifetime
  * Cosmological constant Λ (derived from H_0, Ω_Λ)
"""

from __future__ import annotations

from typing import Any

# ──────────────────────────────────────────────────────────────────────────────
# Planck 2018 best-fit cosmological parameters
# (Table 2, column "TT,TE,EE+lowE+lensing", Planck 2018 Paper VI)
# ──────────────────────────────────────────────────────────────────────────────
_PLANCK_2018: dict[str, dict[str, Any]] = {
    "hubble_constant": {
        "value": 67.36,
        "unit": "km/s/Mpc",
        "source": "Planck 2018 VI, Table 2",
    },
    "baryon_density_omega_b_h2": {
        "value": 0.02237,
        "unit": "dimensionless",
        "source": "Planck 2018 VI, Table 2",
    },
    "cdm_density_omega_c_h2": {
        "value": 0.1200,
        "unit": "dimensionless",
        "source": "Planck 2018 VI, Table 2",
    },
    "dark_energy_density_omega_lambda": {
        "value": 0.6847,
        "unit": "dimensionless",
        "source": "Planck 2018 VI, Table 2",
    },
    "optical_depth_reionisation": {
        "value": 0.0544,
        "unit": "dimensionless",
        "source": "Planck 2018 VI, Table 2",
    },
    "scalar_spectral_index": {
        "value": 0.9649,
        "unit": "dimensionless",
        "source": "Planck 2018 VI, Table 2",
    },
    "primordial_amplitude_ln_As_10_10": {
        "value": 3.044,
        "unit": "dimensionless",
        "source": "Planck 2018 VI, Table 2 (ln(10^10 A_s))",
    },
    "cmb_temperature": {
        "value": 2.7255,
        "unit": "K",
        "source": "Fixsen 2009 / Planck 2018",
    },
}

def _derive_lambda() -> float:
    """Compute cosmological constant Λ = 3 H₀² Ω_Λ / c²."""
    H0_si = 67.36e3 / 3.085677581e22    # s⁻¹
    Omega_Lambda = 0.6847
    c = 299_792_458.0                    # m/s
    return 3.0 * H0_si**2 * Omega_Lambda / c**2


# ──────────────────────────────────────────────────────────────────────────────
# Astrophysical constants (IAU 2012 / PDG 2022 / SI)
# ──────────────────────────────────────────────────────────────────────────────
_ASTROPHYSICAL: dict[str, dict[str, Any]] = {
    "solar_mass": {
        "value": 1.98892e30,
        "unit": "kg",
        "source": "IAU 2012",
    },
    "solar_luminosity": {
        "value": 3.828e26,
        "unit": "W",
        "source": "IAU 2015 nominal solar luminosity",
    },
    "solar_radius": {
        "value": 6.957e8,
        "unit": "m",
        "source": "IAU 2015 nominal solar radius",
    },
    "parsec_in_metres": {
        "value": 3.085677581e16,
        "unit": "m",
        "source": "IAU 2012 (1 pc = 648000/π AU)",
    },
    "light_year_in_metres": {
        "value": 9.4607304725808e15,
        "unit": "m",
        "source": "IAU 2012",
    },
    "schwarzschild_radius_sun": {
        "value": 2.0 * 6.67430e-11 * 1.98892e30 / 299_792_458.0 ** 2,
        "unit": "m",
        "source": "Derived: 2·G·M_☉/c²  (CODATA 2018 G, IAU 2012 M_☉, exact c)",
    },
    "proton_neutron_mass_difference": {
        "value": 1.293_300e-3,
        "unit": "u",
        "source": "PDG 2022",
    },
    "neutron_lifetime": {
        "value": 878.4,
        "unit": "s",
        "source": "PDG 2022 (free neutron mean lifetime)",
    },
    # Cosmological constant Λ derived from Planck 2018 H_0, Ω_Λ
    # Λ = 3 H_0² Ω_Λ / c²
    # H_0 in SI: 67.36 km/s/Mpc = 67.36e3 / 3.085677581e22 s⁻¹
    "cosmological_constant": {
        "value": _derive_lambda(),
        "unit": "m⁻²",
        "source": "Derived from Planck 2018 H_0 and Ω_Λ",
    },
}


def load() -> dict[str, dict[str, Any]]:
    """Return a dict of cosmological and astrophysical constants.

    Returns
    -------
    dict
        ``{name: {"value": float, "unit": str, "source": str}}``
    """
    data: dict[str, dict[str, Any]] = {}
    data.update(_PLANCK_2018)
    data.update(_ASTROPHYSICAL)
    return data
