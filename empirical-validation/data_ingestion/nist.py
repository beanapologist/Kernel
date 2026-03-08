"""
NIST standards ingestion module.
=================================
Sources
-------
* NIST Special Publication 330 (SI base units)
* NIST CODATA 2018 recommended values
  https://physics.nist.gov/cuu/Constants/
* Mathematical constants with high-precision NIST-tabulated values
  https://oeis.org / Wolfram Research / NIST DLMF

``load()`` returns a dict of NIST-sourced constants and mathematical
standards relevant to the Kernel framework.  Returned values follow the
same ``{"value", "unit", "source"}`` schema as the CODATA module.

Constants included
------------------
* Golden ratio φ = (1 + √5) / 2
* Silver ratio δ_S = 1 + √2  (silver mean / Pell constant)
* Euler's number e
* π (pi)
* √2 (Pythagoras constant)
* √5
* Von Klitzing constant R_K = h / e²  (quantum of resistance)
* Josephson constant K_J = 2e / h
* Conductance quantum G_0 = 2e² / h
* Magnetic flux quantum Φ_0 = h / (2e)
"""

from __future__ import annotations

import math
from typing import Any


def load() -> dict[str, dict[str, Any]]:
    """Return a dict of NIST-sourced mathematical and SI constants.

    Returns
    -------
    dict
        ``{name: {"value": float, "unit": str, "source": str}}``
    """
    # Mathematical constants — high-precision (50+ significant digits in
    # IEEE 754 double the precision is limited to ~15–17 digits).
    phi = (1.0 + math.sqrt(5.0)) / 2.0          # golden ratio
    delta_s = 1.0 + math.sqrt(2.0)              # silver ratio / silver mean
    sqrt2 = math.sqrt(2.0)
    sqrt5 = math.sqrt(5.0)

    # Derived SI constants (CODATA 2018, NIST SP 330)
    h = 6.62607015e-34          # Planck constant
    e = 1.602176634e-19         # elementary charge

    R_K = h / (e ** 2)          # von Klitzing constant (quantum Hall resistance)
    K_J = 2.0 * e / h           # Josephson constant
    G_0 = 2.0 * e**2 / h        # conductance quantum
    Phi_0 = h / (2.0 * e)       # magnetic flux quantum

    data: dict[str, dict[str, Any]] = {
        "golden_ratio": {
            "value": phi,
            "unit": "dimensionless",
            "source": "NIST DLMF / computed (1+√5)/2",
        },
        "silver_ratio": {
            "value": delta_s,
            "unit": "dimensionless",
            "source": "NIST DLMF / computed 1+√2",
        },
        "euler_number": {
            "value": math.e,
            "unit": "dimensionless",
            "source": "NIST DLMF / math.e",
        },
        "pi": {
            "value": math.pi,
            "unit": "dimensionless",
            "source": "NIST DLMF / math.pi",
        },
        "sqrt2": {
            "value": sqrt2,
            "unit": "dimensionless",
            "source": "NIST DLMF / √2",
        },
        "sqrt5": {
            "value": sqrt5,
            "unit": "dimensionless",
            "source": "NIST DLMF / √5",
        },
        "von_klitzing_constant": {
            "value": R_K,
            "unit": "Ω",
            "source": "NIST CODATA 2018 derived: h/e²",
        },
        "josephson_constant": {
            "value": K_J,
            "unit": "Hz/V",
            "source": "NIST CODATA 2018 derived: 2e/h",
        },
        "conductance_quantum": {
            "value": G_0,
            "unit": "S",
            "source": "NIST CODATA 2018 derived: 2e²/h",
        },
        "magnetic_flux_quantum": {
            "value": Phi_0,
            "unit": "Wb",
            "source": "NIST CODATA 2018 derived: h/(2e)",
        },
    }
    return data
