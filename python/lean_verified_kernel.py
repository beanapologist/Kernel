#!/usr/bin/env python3
"""
Kernel — Lean-Verified Mathematics: Numerical Demonstration
============================================================
Numerically validates every core theorem proved in the Lean 4 formalization
(formal-lean/) for the Kernel project.  Each check is annotated with the
corresponding Lean theorem name so it can be traced back to a machine-checked
proof.

Mathematical objects (all machine-checked in Lean 4):
  μ       = exp(I·3π/4) = (−1 + i)/√2   critical eigenvalue
  η       = 1/√2                          canonical amplitude
  C(r)    = 2r / (1 + r²)                 coherence function
  δS      = 1 + √2                         silver ratio
  R(θ)    — 2×2 rotation matrix           det = 1, orthogonal, R^8 = I
  Res(r)  = (r − 1/r) / δS               palindrome residual
  U(H,t)  = exp(−I·H·t)                   time evolution operator
  reality(s,t) = t + I·s                  SpaceTime coordinate

Lean source:
  formal-lean/CriticalEigenvalue.lean  (71 theorems)
  formal-lean/TimeCrystal.lean         (33 theorems)
  formal-lean/SpaceTime.lean

Usage:
    python python/lean_verified_kernel.py
Each check prints ✓ or raises AssertionError on failure.
"""

import cmath
import math

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = "✓"
FAIL = "✗"
_results: list[tuple[str, bool]] = []


def _check(name: str, condition: bool) -> None:
    status = PASS if condition else FAIL
    print(f"  {status} {name}")
    _results.append((name, condition))
    if not condition:
        raise AssertionError(f"FAILED: {name}")


def _section(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


# ─────────────────────────────────────────────────────────────────────────────
# Core mathematical objects
# ─────────────────────────────────────────────────────────────────────────────

# Critical eigenvalue: μ = exp(I · 3π/4)
# Lean: CriticalEigenvalue.lean §1 — mu_eq_cart, mu_abs_one, mu_pow_eight
MU: complex = cmath.exp(1j * 3 * math.pi / 4)

# Canonical amplitude: η = 1/√2
# Lean: CriticalEigenvalue.lean §6 — canonical_norm
ETA: float = 1.0 / math.sqrt(2)

# Silver ratio: δS = 1 + √2
# Lean: CriticalEigenvalue.lean §7 — silverRatio_mul_conj, silverRatio_sq
DELTA_S: float = 1.0 + math.sqrt(2)


def coherence(r: float) -> float:
    """Coherence function C(r) = 2r / (1 + r²).

    Lean: CriticalEigenvalue.lean §5 — coherence_le_one, coherence_eq_one_iff,
          coherence_pos, coherence_strictMono, coherence_strictAnti,
          coherence_is_sech_of_log.
    """
    return 2.0 * r / (1.0 + r * r)


def palindrome_residual(r: float) -> float:
    """Palindrome residual Res(r) = (r − 1/r) / δS.

    Lean: CriticalEigenvalue.lean §9 — palindrome_residual_def,
          palindrome_sum_zero.
    """
    return (r - 1.0 / r) / DELTA_S


def rotation_matrix(theta: float) -> np.ndarray:
    """2×2 rotation matrix R(θ) = [[cos θ, −sin θ], [sin θ, cos θ]].

    Lean: CriticalEigenvalue.lean §4 — rotMat_det, rotMat_orthog,
          rotMat_pow_eight.
    """
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def time_evolution(H: float, t: float) -> complex:
    """Time evolution operator U(H, t) = exp(−I·H·t).

    Lean: TimeCrystal.lean §1 — timeEvolution_zero, timeEvolution_abs_one,
          timeEvolution_group_law.
    """
    return cmath.exp(-1j * H * t)


def reality(s: float, t: float) -> complex:
    """SpaceTime reality coordinate: reality(s, t) = t + I·s.

    Lean: SpaceTime.lean §2 — reality_time_component, reality_space_component,
          reality_real_axis, reality_imag_axis.
    """
    return complex(t, s)


# ─────────────────────────────────────────────────────────────────────────────
# §1  Critical Eigenvalue μ = exp(I · 3π/4)
# ─────────────────────────────────────────────────────────────────────────────

def verify_critical_eigenvalue() -> None:
    """§1–3 — Critical eigenvalue μ = exp(I · 3π/4).

    Lean theorems: mu_eq_cart, mu_abs_one, mu_pow_eight, mu_powers_distinct,
                   canonical_norm.
    """
    _section("§1  Critical Eigenvalue  μ = exp(I · 3π/4)")

    # mu_eq_cart: μ = (−1 + I)/√2
    expected = (-1 + 1j) / math.sqrt(2)
    _check("μ = (−1 + i)/√2  [mu_eq_cart]",
           abs(MU - expected) < 1e-12)

    # mu_abs_one: |μ| = 1
    _check("|μ| = 1  [mu_abs_one]",
           abs(abs(MU) - 1.0) < 1e-12)

    # mu_pow_eight: μ^8 = 1
    _check("μ^8 = 1  [mu_pow_eight]",
           abs(MU ** 8 - 1.0) < 1e-12)

    # mu_powers_distinct: {μ^0, …, μ^7} are pairwise distinct
    orbit = [MU ** k for k in range(8)]
    distinct_points = {round(p.real, 9) + round(p.imag, 9) * 1j for p in orbit}
    _check("{μ^0,…,μ^7} are 8 distinct points  [mu_powers_distinct]",
           len(distinct_points) == 8)

    # canonical_norm: η² + |μ·η|² = 1
    _check("η² + |μ·η|² = 1  [canonical_norm]",
           abs(ETA ** 2 + abs(MU * ETA) ** 2 - 1.0) < 1e-12)

    # mu_inv_eq_pow7: μ⁷ = μ⁻¹
    _check("μ^7 = μ⁻¹  [mu_inv_eq_pow7]",
           abs(MU ** 7 - 1.0 / MU) < 1e-12)

    print(f"\n  μ      = {MU:.6f}")
    print(f"  η      = {ETA:.6f}")
    print(f"  arg(μ) = 3π/4 = {3 * math.pi / 4:.6f} rad")


# ─────────────────────────────────────────────────────────────────────────────
# §2  Coherence Function C(r) = 2r / (1 + r²)
# ─────────────────────────────────────────────────────────────────────────────

def verify_coherence_function() -> None:
    """§4–8 — Coherence function C(r) = 2r / (1 + r²).

    Lean theorems: coherence_le_one, coherence_eq_one_iff, coherence_pos,
                   coherence_symm, coherence_strictMono, coherence_strictAnti,
                   coherence_is_sech_of_log, coherence_at_silver_is_eta.
    """
    _section("§2  Coherence Function  C(r) = 2r / (1 + r²)")

    test_rs = [0.1, 0.5, 1.0, 2.0, 10.0, 100.0]

    # coherence_le_one: C(r) ≤ 1 for all r ≥ 0
    for r in test_rs:
        _check(f"C({r}) ≤ 1  [coherence_le_one]",
               coherence(r) <= 1.0 + 1e-12)

    # coherence_eq_one_iff: C(r) = 1 ↔ r = 1
    _check("C(1) = 1  [coherence_eq_one_iff]",
           abs(coherence(1.0) - 1.0) < 1e-12)
    _check("C(2) < 1  [coherence_eq_one_iff]",
           coherence(2.0) < 1.0)

    # coherence_pos: C(r) > 0 for all r > 0
    for r in test_rs:
        _check(f"C({r}) > 0  [coherence_pos]",
               coherence(r) > 0.0)

    # coherence_symm: C(r) = C(1/r)
    for r in [0.5, 2.0, 3.0]:
        _check(f"C({r}) = C(1/{r})  [coherence_symm]",
               abs(coherence(r) - coherence(1.0 / r)) < 1e-12)

    # coherence_is_sech_of_log: C(exp(λ)) = sech(λ)  (Theorem 14 / §22)
    for lam in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        r = math.exp(lam)
        c_sech = 1.0 / math.cosh(lam)
        _check(f"C(exp({lam:+.0f})) = sech({lam:+.0f})  [coherence_is_sech_of_log]",
               abs(coherence(r) - c_sech) < 1e-12)

    # coherence_strictMono on (0, 1]: C(r) < C(s) when 0 < r < s ≤ 1
    _check("C(0.5) < C(0.8) < C(1.0)  [coherence_strictMono]",
           coherence(0.5) < coherence(0.8) < coherence(1.0))

    # coherence_strictAnti on [1, ∞): C(r) > C(s) when 1 ≤ r < s
    _check("C(1.0) > C(1.5) > C(3.0)  [coherence_strictAnti]",
           coherence(1.0) > coherence(1.5) > coherence(3.0))

    # coherence_at_silver_is_eta: C(δS) = η  (§22 machine-discovered cross-section)
    c_silver = coherence(DELTA_S)
    _check("C(δS) = η  [coherence_at_silver_is_eta]",
           abs(c_silver - ETA) < 1e-12)

    print(f"\n  C(1)  = {coherence(1.0):.6f}  (maximum, C(r)=1 iff r=1)")
    print(f"  C(δS) = {c_silver:.6f}  = η = {ETA:.6f}  (cross-section identity)")
    print(f"  C(2)  = {coherence(2.0):.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# §3  Silver Ratio δS = 1 + √2
# ─────────────────────────────────────────────────────────────────────────────

def verify_silver_ratio() -> None:
    """§7 — Silver ratio δS = 1 + √2.

    Lean theorems: silverRatio_mul_conj, silverRatio_sq, silverRatio_inv,
                   silverRatio_self_similar.
    """
    _section("§3  Silver Ratio  δS = 1 + √2")

    # silverRatio_mul_conj: δS · (√2 − 1) = 1
    _check("δS · (√2 − 1) = 1  [silverRatio_mul_conj]",
           abs(DELTA_S * (math.sqrt(2) - 1) - 1.0) < 1e-12)

    # silverRatio_sq: δS² = 2·δS + 1
    _check("δS² = 2·δS + 1  [silverRatio_sq]",
           abs(DELTA_S ** 2 - (2 * DELTA_S + 1)) < 1e-12)

    # silverRatio_inv: 1/δS = √2 − 1
    _check("1/δS = √2 − 1  [silverRatio_inv]",
           abs(1.0 / DELTA_S - (math.sqrt(2) - 1)) < 1e-12)

    # silverRatio_self_similar: δS = 2 + 1/δS
    _check("δS = 2 + 1/δS  [silverRatio_self_similar]",
           abs(DELTA_S - (2 + 1.0 / DELTA_S)) < 1e-12)

    print(f"\n  δS   = {DELTA_S:.6f}")
    print(f"  1/δS = {1.0/DELTA_S:.6f}  = √2−1 = {math.sqrt(2)-1:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# §4  Rotation Matrix R(3π/4)
# ─────────────────────────────────────────────────────────────────────────────

def verify_rotation_matrix() -> None:
    """§4 — Rotation matrix R(3π/4).

    Lean theorems: rotMat_det, rotMat_orthog, rotMat_pow_eight.
    """
    _section("§4  Rotation Matrix  R(3π/4)")

    R = rotation_matrix(3 * math.pi / 4)

    # rotMat_det: det R(3π/4) = 1
    det = np.linalg.det(R)
    _check("det R(3π/4) = 1  [rotMat_det]",
           abs(det - 1.0) < 1e-12)

    # rotMat_orthog: R · Rᵀ = I
    _check("R · Rᵀ = I  [rotMat_orthog]",
           np.allclose(R @ R.T, np.eye(2), atol=1e-12))

    # R^4 = −I (intermediate result used in the Lean proof)
    R4 = np.linalg.matrix_power(R, 4)
    _check("R^4 = −I",
           np.allclose(R4, -np.eye(2), atol=1e-10))

    # rotMat_pow_eight: R^8 = I
    R8 = np.linalg.matrix_power(R, 8)
    _check("R^8 = I  [rotMat_pow_eight]",
           np.allclose(R8, np.eye(2), atol=1e-10))

    print(f"\n  R(3π/4) =\n{R}")
    print(f"  det R = {det:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# §5  Palindrome Residual and Pythagorean Coherence Identity
# ─────────────────────────────────────────────────────────────────────────────

def verify_palindrome_and_pythagorean() -> None:
    """§9, §18 — Palindrome residual and Pythagorean coherence identity.

    Lean theorems: palindrome_sum_zero, coherence_pythagorean.
    """
    _section("§5  Palindrome Residual & Pythagorean Coherence Identity")

    # palindrome_sum_zero: Res(r) + Res(1/r) = 0
    for r in [0.5, 1.5, 2.0, DELTA_S]:
        res_r = palindrome_residual(r)
        res_inv = palindrome_residual(1.0 / r)
        _check(f"Res({r:.4f}) + Res(1/{r:.4f}) = 0  [palindrome_sum_zero]",
               abs(res_r + res_inv) < 1e-12)

    # coherence_pythagorean: C(r)² + ((r² − 1)/(1 + r²))² = 1
    for r in [0.5, 1.0, 2.0, DELTA_S]:
        c = coherence(r)
        s = (r ** 2 - 1) / (1 + r ** 2)
        _check(f"C({r:.4f})² + sin_term² = 1  [coherence_pythagorean]",
               abs(c ** 2 + s ** 2 - 1.0) < 1e-12)

    r_ex = 2.0
    c_ex = coherence(r_ex)
    s_ex = (r_ex ** 2 - 1) / (1 + r_ex ** 2)
    print(f"\n  Res(δS)    = {palindrome_residual(DELTA_S):.6f}")
    print(f"  C({r_ex})² + sin² = {c_ex**2 + s_ex**2:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# §6  Time Crystal — Floquet Theory
# ─────────────────────────────────────────────────────────────────────────────

def verify_time_crystal() -> None:
    """§6 — Discrete time crystal via Floquet theory.

    Lean theorems (TimeCrystal.lean):
      timeEvolution_abs_one, timeEvolution_zero, timeEvolution_group_law,
      floquet_state_period, time_crystal_period_double,
      quasi_energy_half_drive, kernel_time_crystal_8period.
    """
    _section("§6  Time Crystal — Floquet Theory")

    # timeEvolution_zero: U(H, 0) = 1
    _check("U(H, 0) = 1  [timeEvolution_zero]",
           abs(time_evolution(1.0, 0.0) - 1.0) < 1e-12)

    # timeEvolution_abs_one: |U(H, t)| = 1
    for H, t in [(1.0, 0.5), (2.0, 1.0), (0.5, 3.0)]:
        U = time_evolution(H, t)
        _check(f"|U({H}, {t})| = 1  [timeEvolution_abs_one]",
               abs(abs(U) - 1.0) < 1e-12)

    # timeEvolution_group_law: U(H, t+s) = U(H, t) · U(H, s)
    H_test, t1, t2 = 1.5, 0.3, 0.7
    _check("U(t+s) = U(t)·U(s)  [timeEvolution_group_law]",
           abs(time_evolution(H_test, t1 + t2)
               - time_evolution(H_test, t1) * time_evolution(H_test, t2)) < 1e-12)

    # time_crystal_period_double: Floquet phase φ = π gives period doubling
    # ψ(t + T) = exp(−iπ) · ψ(t) = −ψ(t)  →  ψ(t + 2T) = ψ(t)
    phi = math.pi
    floquet_factor = cmath.exp(-1j * phi)
    _check("exp(−iπ) = −1  [time_crystal_period_double]",
           abs(floquet_factor - (-1.0)) < 1e-12)
    _check("(exp(−iπ))² = 1  (period doubling)  [time_crystal_period_double]",
           abs(floquet_factor ** 2 - 1.0) < 1e-12)

    # quasi_energy_half_drive: ε_F · T = π
    T = 1.0
    eps_F = math.pi / T
    _check("ε_F · T = π  [quasi_energy_half_drive]",
           abs(eps_F * T - math.pi) < 1e-12)

    # kernel_time_crystal_8period: orbit of μ gives 8-period crystal
    orbit_8 = [MU ** k for k in range(9)]
    _check("μ^8 = μ^0  (8-period orbit closes)  [kernel_time_crystal_8period]",
           abs(orbit_8[8] - orbit_8[0]) < 1e-12)

    print(f"\n  Floquet phase φ = π  →  period doubling")
    print(f"  Quasi-energy ε_F = π/T = {eps_F:.6f}")
    print(f"  8-period crystal: μ^8 = {MU**8:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# §7  SpaceTime — reality(s, t) = t + I·s
# ─────────────────────────────────────────────────────────────────────────────

def verify_spacetime() -> None:
    """§7 — SpaceTime reality function reality(s, t) = t + I·s.

    Lean theorems (SpaceTime.lean):
      reality_time_component, reality_space_component, reality_real_axis,
      reality_imag_axis, reality_in_complex_plane.
    """
    _section("§7  SpaceTime  —  reality(s, t) = t + I·s")

    s, t = 3.0, -2.0
    r = reality(s, t)

    # reality_time_component: Re(reality(s, t)) = t
    _check("Re(reality(s, t)) = t  [reality_time_component]",
           abs(r.real - t) < 1e-12)

    # reality_space_component: Im(reality(s, t)) = s
    _check("Im(reality(s, t)) = s  [reality_space_component]",
           abs(r.imag - s) < 1e-12)

    # reality_real_axis: reality(0, t) = t  (pure time → real axis)
    r_time_only = reality(0.0, -1.0)
    _check("reality(0, t) = t  [reality_real_axis]",
           abs(r_time_only - (-1.0)) < 1e-12)

    # reality_imag_axis: reality(s, 0) = I·s  (pure space → imaginary axis)
    r_space_only = reality(1.0, 0.0)
    _check("reality(s, 0) = I·s  [reality_imag_axis]",
           abs(r_space_only - 1j) < 1e-12)

    print(f"\n  reality({s}, {t}) = {r}")
    print(f"  Time axis  (s=0): reality(0, −1) = {reality(0.0, -1.0)}")
    print(f"  Space axis (t=0): reality(1,  0) = {reality(1.0, 0.0)}")


# ─────────────────────────────────────────────────────────────────────────────
# §8  Orbit Lyapunov Connection
# ─────────────────────────────────────────────────────────────────────────────

def verify_orbit_lyapunov() -> None:
    """§19, §22 — Orbit Lyapunov connection.

    Lean theorems: orbit_magnitude_pow, orbit_lyapunov_connection,
                   coherence_orbit_sech, orbit_decoherence_rate.
    """
    _section("§8  Orbit Lyapunov Connection")

    r_orb = 1.2
    theta_orb = 3 * math.pi / 4
    xi = r_orb * cmath.exp(1j * theta_orb)

    # orbit_magnitude_pow: |ξ^n| = r^n
    for n in [1, 2, 3, 5, 8]:
        _check(f"|ξ^{n}| = r^{n}  [orbit_magnitude_pow]",
               abs(abs(xi ** n) - r_orb ** n) < 1e-10)

    # orbit_lyapunov_connection: |ξ^n| = exp(n · log r)
    for n in [1, 3, 8]:
        _check(f"|ξ^{n}| = exp({n}·log r)  [orbit_lyapunov_connection]",
               abs(abs(xi ** n) - math.exp(n * math.log(r_orb))) < 1e-10)

    # coherence_orbit_sech: C(r^n) = sech(n · log r)
    for n in [1, 2, 3]:
        c_rn = coherence(r_orb ** n)
        sech_n = 1.0 / math.cosh(n * math.log(r_orb))
        _check(f"C(r^{n}) = sech({n}·log r)  [coherence_orbit_sech]",
               abs(c_rn - sech_n) < 1e-10)

    # orbit_decoherence_rate: C(r^n) ≤ 2/r^n for r > 1
    for n in [1, 2, 3, 5]:
        c_rn = coherence(r_orb ** n)
        bound = 2.0 / r_orb ** n
        _check(f"C(r^{n}) ≤ 2/r^{n}  [orbit_decoherence_rate]",
               c_rn <= bound + 1e-12)

    print(f"\n  r = {r_orb},  C(r) = {coherence(r_orb):.6f}")
    print(f"  C(r^3) = {coherence(r_orb**3):.6f}  (coherence decays over orbit)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_all() -> None:
    """Run all numerical verifications of the Lean-proved theorems."""
    print("=" * 70)
    print("  Kernel — Lean-Verified Mathematics: Numerical Demonstration")
    print("=" * 70)
    print()
    print("  All checks correspond to machine-checked theorems in formal-lean/.")
    print("  Sources: CriticalEigenvalue.lean (71 thms), TimeCrystal.lean (33 thms),")
    print("           SpaceTime.lean")
    print()

    verify_critical_eigenvalue()
    verify_coherence_function()
    verify_silver_ratio()
    verify_rotation_matrix()
    verify_palindrome_and_pythagorean()
    verify_time_crystal()
    verify_spacetime()
    verify_orbit_lyapunov()

    # Summary
    passed = sum(1 for _, ok in _results if ok)
    total = len(_results)
    print(f"\n{'=' * 70}")
    print(f"  Summary: {passed}/{total} checks passed")
    if passed == total:
        print("  All numerical checks agree with Lean-verified proofs. ✓")
    else:
        failed_names = [n for n, ok in _results if not ok]
        print(f"  FAILURES: {failed_names}")
    print("=" * 70)


if __name__ == "__main__":
    run_all()
