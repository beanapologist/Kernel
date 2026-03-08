"""
Eigenvalue validator: |μ|² = 1 for μ = exp(i·3π/4).
=====================================================
The Kernel framework postulates a critical eigenvalue μ = e^{i3π/4} that
lies on the unit circle, satisfying |μ|² = 1 and μ⁸ = 1 (8-cycle orbit).

Check-type taxonomy used in this module
----------------------------------------
``mathematical_identity``
    Pure algebra or complex-analysis facts that follow directly from the
    definition of the complex exponential.  These checks verify that the
    *code* correctly evaluates the postulated μ; they can NEVER distinguish
    a correct framework from an incorrect one — a failing result indicates
    a coding bug, not a physical discrepancy.

``numerical_precision``
    The same identities evaluated with IEEE 754 floating-point arithmetic.
    Pass criterion: absolute error < machine-epsilon × small factor (1e-14).
    A failure signals a catastrophic floating-point regression, not physics.

Neither category constitutes empirical validation.  The framework's
eigenvalue postulate is *postulated*, not derived from measurement; the
checks here confirm mathematical self-consistency only.

Validation checks
-----------------
  1. |μ|² = 1  (mathematical_identity / numerical_precision)
  2. μ⁸ = 1   (mathematical_identity / numerical_precision)
  3. Re(μ) = −1/√2, Im(μ) = 1/√2  (mathematical_identity)
  4. μ + μ* = −√2  (mathematical_identity)
  5. Rotation matrix R(3π/4) eigenvalues on unit circle (numerical_precision)
  6. R(3π/4)⁸ = I  (numerical_precision)
"""

from __future__ import annotations

import cmath
import math
from typing import Any

import numpy as np
import sympy as sp


# ─────────────────────────────────────────────────────────────────────────────
# SymPy symbolic exact verification
# ─────────────────────────────────────────────────────────────────────────────

def _sympy_checks() -> list[dict[str, Any]]:
    """Run exact symbolic checks via SymPy."""
    results: list[dict[str, Any]] = []

    # Define μ symbolically
    mu = sp.exp(sp.I * 3 * sp.pi / 4)

    # 1. |μ|² = 1
    norm_sq = sp.Abs(mu) ** 2
    norm_sq_simplified = sp.simplify(norm_sq)
    passed = bool(sp.Eq(norm_sq_simplified, sp.Integer(1)))
    results.append({
        "name": "eigenvalue_norm_sq_exact",
        "check_type": "mathematical_identity",
        "pass_criterion": (
            "SymPy symbolic equality: sp.Eq(|μ|², 1) must be True. "
            "Failure = coding error in SymPy expression, not a physical discrepancy."
        ),
        "modelled": float(sp.re(norm_sq_simplified)),
        "observed": 1.0,
        "rel_error": 0.0 if passed else float(abs(norm_sq_simplified - 1)),
        "passed": passed,
        "method": "SymPy symbolic",
        "description": "|μ|² = 1  (unit modulus, symbolic)",
    })

    # 2. μ⁸ = 1
    mu8 = sp.simplify(mu ** 8)
    passed_mu8 = bool(sp.Eq(sp.expand_complex(mu8), sp.Integer(1)))
    results.append({
        "name": "eigenvalue_8th_power_exact",
        "check_type": "mathematical_identity",
        "pass_criterion": (
            "SymPy symbolic equality: expand_complex(μ⁸) == 1 must be True. "
            "Consequence of e^{i·6π} = 1; failure = coding bug."
        ),
        "modelled": float(sp.re(sp.N(mu8))),
        "observed": 1.0,
        "rel_error": 0.0 if passed_mu8 else float(abs(sp.N(mu8) - 1)),
        "passed": passed_mu8,
        "method": "SymPy symbolic",
        "description": "μ⁸ = 1  (8-cycle periodicity, symbolic)",
    })

    # 3. Re(μ) = −1/√2
    re_mu = sp.re(sp.expand_complex(mu))
    re_expected = -1 / sp.sqrt(2)
    passed_re = bool(sp.simplify(re_mu - re_expected) == 0)
    results.append({
        "name": "eigenvalue_real_part_exact",
        "check_type": "mathematical_identity",
        "pass_criterion": (
            "SymPy: simplify(Re(μ) − (−1/√2)) == 0. "
            "Re(e^{i3π/4}) = cos(3π/4) = −1/√2 by Euler's formula; failure = coding bug."
        ),
        "modelled": float(sp.N(re_mu)),
        "observed": float(sp.N(re_expected)),
        "rel_error": 0.0 if passed_re else float(abs(sp.N(re_mu - re_expected))),
        "passed": passed_re,
        "method": "SymPy symbolic",
        "description": "Re(μ) = −1/√2  (symbolic)",
    })

    # 4. Im(μ) = 1/√2
    im_mu = sp.im(sp.expand_complex(mu))
    im_expected = 1 / sp.sqrt(2)
    passed_im = bool(sp.simplify(im_mu - im_expected) == 0)
    results.append({
        "name": "eigenvalue_imag_part_exact",
        "check_type": "mathematical_identity",
        "pass_criterion": (
            "SymPy: simplify(Im(μ) − 1/√2) == 0. "
            "Im(e^{i3π/4}) = sin(3π/4) = 1/√2 by Euler's formula; failure = coding bug."
        ),
        "modelled": float(sp.N(im_mu)),
        "observed": float(sp.N(im_expected)),
        "rel_error": 0.0 if passed_im else float(abs(sp.N(im_mu - im_expected))),
        "passed": passed_im,
        "method": "SymPy symbolic",
        "description": "Im(μ) = 1/√2  (symbolic)",
    })

    # 5. μ + μ* = −√2
    mu_conj = sp.conjugate(mu)
    trace = sp.simplify(mu + mu_conj)
    trace_expected = -sp.sqrt(2)
    # Use numerical comparison: symbolic simplification may leave the expression
    # in a form like (-1)^(1/4)·(−1+i) which is numerically equal to −√2.
    trace_numeric = complex(sp.N(trace))
    expected_numeric = float(sp.N(trace_expected))
    _tol_trace = 1e-14   # tolerance for the trace numerical check
    passed_trace = (
        abs(trace_numeric.real - expected_numeric) < _tol_trace
        and abs(trace_numeric.imag) < _tol_trace
    )
    results.append({
        "name": "eigenvalue_trace_exact",
        "check_type": "mathematical_identity",
        "pass_criterion": (
            "Numerical: |N(μ+μ*).real − (−√2)| < 1e-14 and |N(μ+μ*).imag| < 1e-14. "
            "e^{iθ} + e^{−iθ} = 2cos(θ) = 2cos(3π/4) = −√2; failure = coding bug."
        ),
        "modelled": trace_numeric.real,
        "observed": expected_numeric,
        "rel_error": abs(trace_numeric.real - expected_numeric) / abs(expected_numeric) if expected_numeric != 0 else 0.0,
        "passed": passed_trace,
        "method": "SymPy symbolic + numerical",
        "description": "μ + μ* = −√2  (Ohm–Coherence trace, symbolic)",
    })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# NumPy numerical verification
# ─────────────────────────────────────────────────────────────────────────────

def _numpy_checks() -> list[dict[str, Any]]:
    """Run floating-point numerical checks via NumPy/cmath."""
    results: list[dict[str, Any]] = []
    # IEEE 754 double has machine epsilon ~2.2e-16; allowing 64× headroom.
    tol = 1e-14

    mu = cmath.exp(1j * 3 * math.pi / 4)

    # 1. |μ|² = 1
    norm_sq = abs(mu) ** 2
    err = abs(norm_sq - 1.0)
    results.append({
        "name": "eigenvalue_norm_sq_numerical",
        "check_type": "numerical_precision",
        "pass_criterion": (
            "Absolute error |norm_sq − 1| < 1e-14 (64× machine epsilon). "
            "Verifies IEEE 754 precision of cmath.exp; cannot validate physics."
        ),
        "modelled": norm_sq,
        "observed": 1.0,
        "rel_error": err,
        "passed": err < tol,
        "method": "NumPy/cmath",
        "description": "|μ|² = 1  (unit modulus, numerical)",
    })

    # 2. μ⁸ = 1
    mu8 = mu ** 8
    err_mu8 = abs(mu8 - 1.0)
    results.append({
        "name": "eigenvalue_8th_power_numerical",
        "check_type": "numerical_precision",
        "pass_criterion": (
            "Absolute error |μ⁸ − 1| < 1e-14 (64× machine epsilon). "
            "Accumulated floating-point error from 8 complex multiplications."
        ),
        "modelled": mu8.real,
        "observed": 1.0,
        "rel_error": err_mu8,
        "passed": err_mu8 < tol,
        "method": "NumPy/cmath",
        "description": "μ⁸ = 1  (8-cycle periodicity, numerical)",
    })

    # 3. Re(μ) = −1/√2
    re_expected = -1.0 / math.sqrt(2.0)
    err_re = abs(mu.real - re_expected)
    rel_re = err_re / abs(re_expected)
    results.append({
        "name": "eigenvalue_real_part_numerical",
        "check_type": "numerical_precision",
        "pass_criterion": (
            "Absolute error |Re(μ) − (−1/√2)| < 1e-14. "
            "Identity Re(e^{i3π/4}) = cos(3π/4); failure = floating-point bug."
        ),
        "modelled": mu.real,
        "observed": re_expected,
        "rel_error": rel_re,
        "passed": err_re < tol,
        "method": "NumPy/cmath",
        "description": "Re(μ) = −1/√2  (numerical)",
    })

    # 4. Im(μ) = 1/√2
    im_expected = 1.0 / math.sqrt(2.0)
    err_im = abs(mu.imag - im_expected)
    rel_im = err_im / abs(im_expected)
    results.append({
        "name": "eigenvalue_imag_part_numerical",
        "check_type": "numerical_precision",
        "pass_criterion": (
            "Absolute error |Im(μ) − 1/√2| < 1e-14. "
            "Identity Im(e^{i3π/4}) = sin(3π/4); failure = floating-point bug."
        ),
        "modelled": mu.imag,
        "observed": im_expected,
        "rel_error": rel_im,
        "passed": err_im < tol,
        "method": "NumPy/cmath",
        "description": "Im(μ) = 1/√2  (numerical)",
    })

    # 5. Rotation matrix R(3π/4): eigenvalues lie on unit circle
    angle = 3 * math.pi / 4
    R = np.array([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle),  math.cos(angle)],
    ])
    evals = np.linalg.eigvals(R)
    norms = np.abs(evals)
    err_eig = float(np.max(np.abs(norms - 1.0)))
    results.append({
        "name": "rotation_matrix_eigenvalue_norm",
        "check_type": "numerical_precision",
        "pass_criterion": (
            "max|‖eigenvalue‖ − 1| < 1e-14 for both eigenvalues of R(3π/4). "
            "Rotation matrices are unitary; failure = numerical linalg issue."
        ),
        "modelled": float(np.mean(norms)),
        "observed": 1.0,
        "rel_error": err_eig,
        "passed": err_eig < tol,
        "method": "NumPy linalg",
        "description": "Rotation matrix R(3π/4) eigenvalues on unit circle",
    })

    # 6. 8-cycle orbit closes: R^8 = I (identity)
    R8 = np.linalg.matrix_power(R, 8)
    err_I = np.max(np.abs(R8 - np.eye(2)))
    results.append({
        "name": "rotation_matrix_8th_power_identity",
        "check_type": "numerical_precision",
        "pass_criterion": (
            "max|R(3π/4)⁸ − I| < 1e-14. "
            "8×(3π/4) = 6π, so R^8 = I exactly; failure = matrix-power precision."
        ),
        "modelled": float(R8[0, 0]),
        "observed": 1.0,
        "rel_error": float(err_I),
        "passed": float(err_I) < tol,
        "method": "NumPy linalg",
        "description": "R(3π/4)⁸ = I  (8-orbit closure)",
    })

    return results


def validate(_data: dict | None = None) -> list[dict[str, Any]]:
    """Validate eigenvalue properties.

    Parameters
    ----------
    _data:
        Unused (eigenvalue checks are purely mathematical); accepted for
        interface consistency with other validators.

    Returns
    -------
    list[dict]
        One dict per individual check.  Each dict includes ``check_type``
        (``"mathematical_identity"`` or ``"numerical_precision"``) and
        ``pass_criterion`` explaining what constitutes a pass.

    Note
    ----
    All checks here are mathematical in nature.  None of them can confirm
    or refute the physical validity of the framework's eigenvalue postulate —
    they only verify that the code correctly evaluates the postulated value.
    """
    return _sympy_checks() + _numpy_checks()

