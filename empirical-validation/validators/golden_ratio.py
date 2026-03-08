"""
Golden ratio and silver ratio validator.
==========================================
The Kernel framework frequently uses the golden ratio φ and silver ratio δ_S.

Check-type taxonomy used in this module
----------------------------------------
``mathematical_identity``
    Algebraic facts that follow from the definitions φ = (1+√5)/2 and
    δ_S = 1+√2.  These are true by construction and CANNOT validate the
    physical claims of the framework.  A failure indicates a coding bug.

``empirical``
    Checks that compare the computed irrational-number value against an
    independent NIST tabulation, or verify that Fibonacci integer ratios
    converge to φ as claimed.  These CAN distinguish a correct computation
    from a buggy one AND anchor the value to an external reference.

``numerical_precision``
    IEEE 754 floating-point precision of an algebraic identity.

Validation checks
-----------------
  Symbolic (SymPy — mathematical_identity):
  1. φ² = φ + 1          (definition consequence)
  2. φ − 1 = 1/φ         (definition consequence)
  3. φ² − φ − 1 = 0      (minimal polynomial)
  4. δ_S² − 2δ_S − 1 = 0 (silver ratio minimal polynomial)
  5. δ_S·(√2−1) = 1      (silver conservation identity)
  6. φ + 1/φ = √5         (sum identity)

  Numerical (NumPy):
  7. φ² − φ − 1 ≈ 0       (numerical_precision: IEEE 754 residual)
  8. δ_S·(√2−1) ≈ 1       (numerical_precision: IEEE 754 residual)
  9. F(72)/F(71) → φ       (empirical: Fibonacci integer sequence vs NIST φ)
 10. Computed φ vs NIST    (empirical: cross-check against external tabulation)
 11. Computed δ_S vs NIST  (empirical: cross-check against external tabulation)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import sympy as sp


def validate(data: dict | None = None) -> list[dict[str, Any]]:
    """Validate golden ratio and silver ratio properties.

    Parameters
    ----------
    data:
        Combined dataset dict.  NIST ``"golden_ratio"`` and
        ``"silver_ratio"`` values are used when present.

    Returns
    -------
    list[dict]
        Each dict includes ``check_type`` and ``pass_criterion``.
    """
    data = data or {}
    results: list[dict[str, Any]] = []
    tol_num = 1e-14   # IEEE 754 double precision, allowing 64× headroom

    # Reference values (from data ingestion or computed)
    phi_ref = data.get("golden_ratio", {}).get("value", (1.0 + math.sqrt(5.0)) / 2.0)
    delta_s_ref = data.get("silver_ratio", {}).get("value", 1.0 + math.sqrt(2.0))

    # ── Symbolic (SymPy exact) ──────────────────────────────────────────────
    sqrt5_s = sp.sqrt(5)
    phi_s = (1 + sqrt5_s) / 2
    sqrt2_s = sp.sqrt(2)
    delta_s_s = 1 + sqrt2_s

    # 1. φ² = φ + 1 (exact)
    lhs = sp.expand(phi_s ** 2)
    rhs = phi_s + 1
    passed_1 = bool(sp.simplify(lhs - rhs) == 0)
    results.append({
        "name": "golden_ratio_quadratic_identity_sympy",
        "check_type": "mathematical_identity",
        "pass_criterion": (
            "SymPy: simplify(φ² − (φ+1)) == 0. "
            "Algebraic consequence of φ = (1+√5)/2; failure = coding bug."
        ),
        "modelled": float(sp.N(lhs)),
        "observed": float(sp.N(rhs)),
        "rel_error": 0.0,
        "passed": passed_1,
        "method": "SymPy symbolic",
        "description": "φ² = φ + 1  (minimal polynomial identity, exact)",
    })

    # 2. φ − 1 = 1/φ (exact)
    lhs2 = phi_s - 1
    rhs2 = sp.Integer(1) / phi_s
    passed_2 = bool(sp.simplify(lhs2 - rhs2) == 0)
    results.append({
        "name": "golden_ratio_reciprocal_identity_sympy",
        "check_type": "mathematical_identity",
        "pass_criterion": (
            "SymPy: simplify((φ−1) − 1/φ) == 0. "
            "Follows from φ²=φ+1 by dividing by φ; failure = coding bug."
        ),
        "modelled": float(sp.N(lhs2)),
        "observed": float(sp.N(rhs2)),
        "rel_error": 0.0,
        "passed": passed_2,
        "method": "SymPy symbolic",
        "description": "φ − 1 = 1/φ  (reciprocal identity, exact)",
    })

    # 3. φ minimal polynomial root: φ² − φ − 1 = 0
    poly_val = sp.expand(phi_s ** 2 - phi_s - 1)
    passed_3 = bool(sp.simplify(poly_val) == 0)
    results.append({
        "name": "golden_ratio_minimal_polynomial_sympy",
        "check_type": "mathematical_identity",
        "pass_criterion": (
            "SymPy: simplify(φ²−φ−1) == 0. "
            "φ is defined as root of x²−x−1; failure = coding bug."
        ),
        "modelled": float(sp.N(poly_val)),
        "observed": 0.0,
        "rel_error": float(abs(sp.N(poly_val))),
        "passed": passed_3,
        "method": "SymPy symbolic",
        "description": "φ² − φ − 1 = 0  (minimal polynomial, exact)",
    })

    # 4. δ_S minimal polynomial: δ_S² − 2·δ_S − 1 = 0
    poly_ds = sp.expand(delta_s_s ** 2 - 2 * delta_s_s - 1)
    passed_4 = bool(sp.simplify(poly_ds) == 0)
    results.append({
        "name": "silver_ratio_minimal_polynomial_sympy",
        "check_type": "mathematical_identity",
        "pass_criterion": (
            "SymPy: simplify(δ_S²−2δ_S−1) == 0. "
            "δ_S=1+√2 is a root of x²−2x−1; failure = coding bug."
        ),
        "modelled": float(sp.N(poly_ds)),
        "observed": 0.0,
        "rel_error": float(abs(sp.N(poly_ds))),
        "passed": passed_4,
        "method": "SymPy symbolic",
        "description": "δ_S² − 2δ_S − 1 = 0  (minimal polynomial, exact)",
    })

    # 5. δ_S · (√2 − 1) = 1 (Silver conservation)
    silver_prod = sp.expand(delta_s_s * (sqrt2_s - 1))
    passed_5 = bool(sp.simplify(silver_prod - 1) == 0)
    results.append({
        "name": "silver_conservation_sympy",
        "check_type": "mathematical_identity",
        "pass_criterion": (
            "SymPy: simplify((1+√2)(√2−1) − 1) == 0. "
            "(1+√2)(√2−1) = 2−1 = 1 by difference of squares; failure = coding bug."
        ),
        "modelled": float(sp.N(silver_prod)),
        "observed": 1.0,
        "rel_error": 0.0 if passed_5 else float(abs(sp.N(silver_prod) - 1)),
        "passed": passed_5,
        "method": "SymPy symbolic",
        "description": "δ_S·(√2−1) = 1  (silver conservation, exact)",
    })

    # 6. φ + 1/φ = √5 (exact)
    sum_sym = sp.simplify(phi_s + 1 / phi_s - sqrt5_s)
    passed_6 = bool(sum_sym == 0)
    results.append({
        "name": "golden_ratio_sum_identity_sympy",
        "check_type": "mathematical_identity",
        "pass_criterion": (
            "SymPy: simplify(φ + 1/φ − √5) == 0. "
            "φ+1/φ = (1+√5)/2 + 2/(1+√5) = √5 by algebra; failure = coding bug."
        ),
        "modelled": float(sp.N(phi_s + 1 / phi_s)),
        "observed": float(sp.N(sqrt5_s)),
        "rel_error": 0.0 if passed_6 else float(abs(sp.N(sum_sym))),
        "passed": passed_6,
        "method": "SymPy symbolic",
        "description": "φ + 1/φ = √5  (sum identity, exact)",
    })

    # ── Numerical (NumPy) ───────────────────────────────────────────────────

    # 7. φ² − φ − 1 ≈ 0 (numerical)
    poly_num = phi_ref ** 2 - phi_ref - 1.0
    results.append({
        "name": "golden_ratio_minimal_polynomial_numerical",
        "check_type": "numerical_precision",
        "pass_criterion": (
            "|φ_float² − φ_float − 1| < 1e-14 (64× machine epsilon). "
            "IEEE 754 residual of the minimal polynomial; failure = FP precision loss."
        ),
        "modelled": poly_num,
        "observed": 0.0,
        "rel_error": abs(poly_num),
        "passed": abs(poly_num) < tol_num,
        "method": "NumPy numerical",
        "description": "φ² − φ − 1 ≈ 0  (minimal polynomial, numerical)",
    })

    # 8. δ_S · (√2 − 1) ≈ 1 (numerical)
    silver_num = delta_s_ref * (math.sqrt(2.0) - 1.0)
    rel_err_8 = abs(silver_num - 1.0)
    results.append({
        "name": "silver_conservation_numerical",
        "check_type": "numerical_precision",
        "pass_criterion": (
            "|(1+√2_float)(√2_float−1) − 1| < 1e-14. "
            "IEEE 754 residual of the silver conservation identity; failure = FP issue."
        ),
        "modelled": silver_num,
        "observed": 1.0,
        "rel_error": rel_err_8,
        "passed": rel_err_8 < tol_num,
        "method": "NumPy numerical",
        "description": "δ_S·(√2−1) ≈ 1  (silver conservation, numerical)",
    })

    # 9. Fibonacci convergence: F(n+1)/F(n) → φ
    a, b = 1, 1
    for _ in range(70):
        a, b = b, a + b
    fib_ratio = b / a
    rel_err_9 = abs(fib_ratio - phi_ref) / phi_ref
    results.append({
        "name": "fibonacci_convergence_golden_ratio",
        "check_type": "empirical",
        "pass_criterion": (
            "Relative error |F(72)/F(71) − φ_NIST| / φ_NIST < 1e-12. "
            "The 72nd Fibonacci ratio converges to φ exponentially; "
            "independent integer sequence compared to NIST-tabulated φ. "
            "Failure would mean the NIST value is wrong or the code has a bug."
        ),
        "modelled": fib_ratio,
        "observed": phi_ref,
        "rel_error": rel_err_9,
        "passed": rel_err_9 < 1e-12,
        "method": "NumPy: 72nd Fibonacci ratio",
        "description": "F(72)/F(71) converges to φ within 1e-12",
    })

    # 10. Numerical value of φ matches NIST
    phi_computed = (1.0 + math.sqrt(5.0)) / 2.0
    rel_err_10 = abs(phi_computed - phi_ref) / phi_ref
    results.append({
        "name": "golden_ratio_value_nist",
        "check_type": "empirical",
        "pass_criterion": (
            "Relative error |(1+√5)/2_float − φ_NIST| / φ_NIST < 1e-15 (~machine eps). "
            "Cross-checks our computation against the NIST-tabulated value loaded "
            "by the data_ingestion module; failure = data ingestion error."
        ),
        "modelled": phi_computed,
        "observed": phi_ref,
        "rel_error": rel_err_10,
        "passed": rel_err_10 < 1e-15,
        "method": "NumPy vs NIST",
        "description": "Computed φ = (1+√5)/2 matches NIST tabulated value",
    })

    # 11. Silver ratio value matches NIST
    delta_s_computed = 1.0 + math.sqrt(2.0)
    rel_err_11 = abs(delta_s_computed - delta_s_ref) / delta_s_ref
    results.append({
        "name": "silver_ratio_value_nist",
        "check_type": "empirical",
        "pass_criterion": (
            "Relative error |(1+√2)_float − δ_S_NIST| / δ_S_NIST < 1e-15 (~machine eps). "
            "Cross-checks our computation against the NIST-tabulated value; "
            "failure = data ingestion error."
        ),
        "modelled": delta_s_computed,
        "observed": delta_s_ref,
        "rel_error": rel_err_11,
        "passed": rel_err_11 < 1e-15,
        "method": "NumPy vs NIST",
        "description": "Computed δ_S = 1+√2 matches NIST tabulated value",
    })

    return results


