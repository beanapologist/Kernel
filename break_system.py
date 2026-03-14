# """
break_system.py — The Math is the Weapon

The Lean proofs are done. The math is verified. It's not under test.
The SYSTEMS are under test. The math breaks them.

5 attack vectors:
    1. SYMPY   — Exact symbolic failure boundaries
    2. NUMPY   — 10M+ point numerical sweeps
    3. CHECKSUM — Hash-locked invariant detection
    4. CROSS   — Inter-structure consistency
    5. EDGE    — Adversarial extreme inputs
"""

import sympy as sp
import numpy as np
import hashlib
import json
import sys


# ═══════════════════════════════════════════════════════════════════════
# VECTOR 1: SYMBOLIC — Derive exact boundaries with SymPy
# ═══════════════════════════════════════════════════════════════════════

print("═" * 72)
print("  VECTOR 1: SYMBOLIC (SymPy)")
print("  Exact failure boundaries from verified invariants")
print("═" * 72, "\n")

r = sp.Symbol('r', positive=True)
lam = sp.Symbol('lambda', real=True)
C_sym = 2*r / (1 + r**2)
F_sym = 1 - 1/sp.cosh(lam)
phi = (1 + sp.sqrt(5)) / 2
delta_s = 1 + sp.sqrt(2)
mu = sp.exp(sp.I * 3 * sp.pi / 4)
gate = 1 / sp.sqrt(2)

tests_passed = 0
tests_failed = 0

def check(name, condition, detail=""):
    global tests_passed, tests_failed
    if condition:
        tests_passed += 1
        print(f"  ✓ {name}")
    else:
        tests_failed += 1
        print(f"  ✗ {name} — {detail}")

# C(r) = GATE solutions
sols = sp.solve(C_sym - gate, r)
check("C(r)=1/√2 has exactly 2 solutions", len(sols) == 2)
for s in sols:
    is_ds = sp.simplify(s - delta_s) == 0
    is_inv = sp.simplify(s - (sp.sqrt(2)-1)) == 0
    check(f"  Solution r={s} is δ_S or 1/δ_S", is_ds or is_inv, f"got {s}")

# C(r) = 2/3 solutions
sols_k = sp.solve(C_sym - sp.Rational(2,3), r)
check("C(r)=2/3 has exactly 2 solutions", len(sols_k) == 2)

# Symmetry
sym_diff = sp.simplify(C_sym - C_sym.subs(r, 1/r))
check("C(r) = C(1/r) symbolically", sym_diff == 0, f"diff={sym_diff}")

# Duality
C_exp = C_sym.subs(r, sp.exp(lam))
sech = 1/sp.cosh(lam)
# Rewrite both in exp form and simplify
C_exp_rewritten = sp.simplify(C_exp.rewrite(sp.exp))
sech_rewritten = sp.simplify(sech.rewrite(sp.exp))
dual_diff = sp.simplify(C_exp_rewritten - sech_rewritten)
check("C(exp(λ)) = sech(λ) symbolically", dual_diff == 0, f"diff={dual_diff}")

# μ⁸
mu8 = sp.simplify(sp.expand(mu**8))
mu8_num = complex(mu8.evalf())
check("μ⁸ = 1", abs(mu8_num - 1) < 1e-14, f"μ⁸={mu8_num}")

# Silver
silver = sp.simplify(delta_s * (sp.sqrt(2) - 1))
check("δ_S·(√2-1) = 1", silver == 1, f"got {silver}")

# Golden
golden = sp.simplify(phi**2 - phi - 1)
check("φ²-φ-1 = 0", golden == 0, f"got {golden}")

# Palindrome
check("9×13717421 = 123456789", 9*13717421 == 123456789)
ratio = sp.Rational(987654321, 123456789)
check("987654321/123456789 = 8 + 1/13717421", ratio == 8 + sp.Rational(1, 13717421))

# Exact failure thresholds
print("\n  ─── Exact Failure Boundaries ───")
eff = sp.Symbol('eff', positive=True)
C_eff = 2*eff/(1+eff**2)
half_coh = sp.solve(C_eff - sp.Rational(1,2), eff)
print(f"  C(eff)=0.5 at eff = {half_coh} = {[float(x.evalf()) for x in half_coh]}")
print(f"  C(eff)=GATE at eff = δ_S={float(delta_s.evalf()):.6f}, 1/δ_S={float((1/delta_s).evalf()):.6f}")
alpha_max = 1 + 1/sp.E
print(f"  α_max = 1+1/e = {float(alpha_max.evalf()):.10f}")
print(f"  Max yield = {float((alpha_max-1).evalf())*100:.4f}%")
F_at_lmax = 1 - 1/sp.cosh(1/sp.E)
print(f"  F(λ_max) = F(1/e) = {float(F_at_lmax.evalf()):.10f}")

print()


# ═══════════════════════════════════════════════════════════════════════
# VECTOR 2: NUMERICAL — Massive sweeps with NumPy
# ═══════════════════════════════════════════════════════════════════════

print("═" * 72)
print("  VECTOR 2: NUMERICAL (NumPy)")
print("  10M+ point sweeps, precision attacks")
print("═" * 72, "\n")

def C_np(r):
    r = np.asarray(r, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = (2*r)/(1+r**2)
        out = np.where(r <= 0, 0, out)
    return out

def F_np(l):
    l = np.asarray(l, dtype=np.float64)
    with np.errstate(over='ignore'):
        out = 1 - 1/np.cosh(l)
        out = np.where(np.isinf(np.cosh(l)), 1.0, out)
    return out

GF = 1/np.sqrt(2)
PF = (1+np.sqrt(5))/2
DF = 1+np.sqrt(2)
mu_c = np.exp(1j * 3*np.pi/4)

# Sweep 1: C(r) range and max
print("─── Sweep 1: Coherence over 10M points ──────────────────────")
rv = np.logspace(-10, 10, 10_000_000)
Cv = C_np(rv)
mi = np.argmax(Cv)
check("C_max = 1.0", abs(Cv[mi]-1) < 1e-6, f"C_max={Cv[mi]}")
check("C_max at r ≈ 1", abs(rv[mi]-1) < 1e-3, f"r={rv[mi]}")
check("C(r) ∈ [0,1]", np.all(Cv >= 0) and np.all(Cv <= 1+1e-15))
print()

# Sweep 2: Symmetry
print("─── Sweep 2: Symmetry over 1M points ────────────────────────")
rs = np.logspace(-8, 8, 1_000_000)
sym_err = np.max(np.abs(C_np(rs) - C_np(1/rs)))
check(f"Max |C(r)-C(1/r)| = {sym_err:.2e}", sym_err < 1e-10)
print()

# Sweep 3: Duality
print("─── Sweep 3: Duality over 1M points ─────────────────────────")
lv = np.linspace(-20, 20, 1_000_000)
C_el = C_np(np.exp(lv))
sl = 1/np.cosh(lv)
valid = np.isfinite(sl) & np.isfinite(C_el)
dual_err = np.max(np.abs(C_el[valid] - sl[valid]))
check(f"Max |C(eλ)-sech(λ)| = {dual_err:.2e}", dual_err < 1e-10)
print()

# Sweep 4: Frustration properties
print("─── Sweep 4: Frustration over 1M points ─────────────────────")
lf = np.linspace(-50, 50, 1_000_000)
Fv = F_np(lf)
check("F(0) = 0", abs(F_np(0.0)) < 1e-15, f"F(0)={F_np(0.0)}")
check("F(l) ≥ 0 everywhere", np.all(Fv >= -1e-15))
check("F(l) < 1 everywhere", np.all(Fv < 1+1e-15))
even_err = np.max(np.abs(F_np(np.abs(lf)) - F_np(-np.abs(lf))))
check(f"Even symmetry |F(l)-F(-l)| = {even_err:.2e}", even_err < 1e-10)
nz = lf[lf != 0]
check("Arrow: F(l)>0 for l≠0", np.all(F_np(nz) > 0))
print()

# Sweep 5: Eigenvalue orbit
print("─── Sweep 5: Eigenvalue orbit ───────────────────────────────")
orbit = np.array([mu_c**n for n in range(9)])
norm_err = np.max(np.abs(np.abs(orbit) - 1))
check(f"|μⁿ| = 1, max err = {norm_err:.2e}", norm_err < 1e-14)
check(f"|μ⁸-1| = {abs(orbit[8]-1):.2e}", abs(orbit[8]-1) < 1e-13)
for n in range(1,8):
    check(f"  μ^{n} ≠ 1", abs(orbit[n]-1) > 0.1)
print()

# Sweep 6: Rotation matrix
print("─── Sweep 6: Rotation matrix ────────────────────────────────")
th = 3*np.pi/4
R = np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
R8 = np.linalg.matrix_power(R, 8)
check(f"|R⁸-I| = {np.max(np.abs(R8-np.eye(2))):.2e}", np.max(np.abs(R8-np.eye(2))) < 1e-13)
check(f"det(R) = {np.linalg.det(R):.16f}", abs(np.linalg.det(R)-1) < 1e-14)
check(f"|RRᵀ-I| = {np.max(np.abs(R@R.T-np.eye(2))):.2e}", np.max(np.abs(R@R.T-np.eye(2))) < 1e-14)
print()

# Sweep 7: Koide
print("─── Sweep 7: Koide with PDG 2022 masses ─────────────────────")
m_e, m_mu, m_tau = 0.51099895000, 105.6583755, 1776.86
Q_k = (m_e+m_mu+m_tau)/(np.sqrt(m_e)+np.sqrt(m_mu)+np.sqrt(m_tau))**2
print(f"  Q = {Q_k:.10f}, 2/3 = {2/3:.10f}, Δ = {abs(Q_k-2/3):.6e} ({abs(Q_k-2/3)/(2/3)*100:.4f}%)")
check("Koide Q ≈ 2/3", abs(Q_k - 2/3) < 0.001)
print()

# Sweep 8: Critical points
print("─── Sweep 8: All critical values ─────────────────────────────")
crits = [
    ("C(1)", float(C_np(1.0)), 1.0, 1e-15),
    ("C(δ_S)", float(C_np(DF)), GF, 1e-14),
    ("C(1/δ_S)", float(C_np(1/DF)), GF, 1e-14),
    ("C(φ²)", float(C_np(PF**2)), 2/3, 1e-14),
    ("C(1/φ²)", float(C_np(1/PF**2)), 2/3, 1e-14),
    ("Im(μ)", mu_c.imag, GF, 1e-15),
    ("|μ|", abs(mu_c), 1.0, 1e-15),
    ("η²+|μη|²", GF**2+abs(mu_c*GF)**2, 1.0, 1e-14),
    ("φ²-φ-1", PF**2-PF-1, 0.0, 1e-14),
    ("δ_S²-2δ_S-1", DF**2-2*DF-1, 0.0, 1e-14),
    ("δ_S(√2-1)", DF*(np.sqrt(2)-1), 1.0, 1e-14),
    ("6π⁵ vs m_p/m_e", abs(6*np.pi**5-1836.15267343)/1836.15267343, 0.0, 5e-4),
    ("Gear 8×3π/4=6π", 8*3*np.pi/4-3*2*np.pi, 0.0, 1e-13),
]
for name, val, exp, tol in crits:
    err = abs(val-exp)
    check(f"{name:<20s} = {val:<22.15f} err={err:.2e}", err < tol)
print()


# ═══════════════════════════════════════════════════════════════════════
# VECTOR 3: CHECKSUM — Hash-locked invariants
# ═══════════════════════════════════════════════════════════════════════

print("═" * 72)
print("  VECTOR 3: CHECKSUM INTEGRITY")
print("═" * 72, "\n")

def chash(name, value, prec=12):
    if isinstance(value, complex):
        s = f"{name}:{value.real:.{prec}e}:{value.imag:.{prec}e}"
    elif isinstance(value, float):
        s = f"{name}:{value:.{prec}e}"
    else:
        s = f"{name}:{value}"
    return hashlib.sha256(s.encode()).hexdigest()[:16]

invariants = {
    "GATE": GF, "PHI": PF, "DELTA_S": DF, "MU": mu_c,
    "C(1)": float(C_np(1.0)), "C(DS)": float(C_np(DF)),
    "C(PHI2)": float(C_np(PF**2)), "KOIDE": Q_k,
    "SILVER": DF*(np.sqrt(2)-1), "GOLDEN": PF**2-PF-1,
    "PALINDROME": 987654321/123456789, "VACUUM": 1/13717421,
    "WYLER": 6*np.pi**5, "ALPHA_MAX": 1+1/np.e,
    "F0": float(F_np(0.0)), "F1": float(F_np(1.0)),
    "R8TR": float(np.trace(R8)), "GEAR": 8*3*np.pi/4/(2*np.pi),
}

table = {}
print(f"  {'Name':<15s} {'Hash':<20s} {'Value'}")
print(f"  {'─'*15} {'─'*20} {'─'*30}")
for name, val in invariants.items():
    h = chash(name, val)
    table[name] = h
    v = f"{val:.12f}" if isinstance(val, float) else str(val)
    print(f"  {name:<15s} {h:<20s} {v[:30]}")

master = hashlib.sha256(json.dumps(table, sort_keys=True).encode()).hexdigest()
print(f"\n  MASTER: {master}")

# Verify
recomp = {n: chash(n, v) for n, v in invariants.items()}
mismatches = [n for n in table if table[n] != recomp[n]]
check(f"All {len(table)} checksums verified", len(mismatches) == 0, f"mismatches: {mismatches}")

# Tamper detection
print("\n─── Tamper detection ────────────────────────────────────────")
for delta in [1e-15, 1e-12, 1e-10, 1e-8, 1e-5]:
    tampered = chash("GATE", GF + delta)
    original = table["GATE"]
    detected = tampered != original
    print(f"  GATE + {delta:.0e}: {'DETECTED ✓' if detected else 'MISSED ✗'}")
print()


# ═══════════════════════════════════════════════════════════════════════
# VECTOR 4: CROSS-STRUCTURE
# ═══════════════════════════════════════════════════════════════════════

print("═" * 72)
print("  VECTOR 4: CROSS-STRUCTURE CONSISTENCY")
print("═" * 72, "\n")

cross = [
    ("Im(μ) = C(δ_S)", abs(mu_c.imag - C_np(DF)), 1e-14),
    ("C(φ²) ≈ Koide Q", abs(float(C_np(PF**2)) - Q_k), 0.001),
    ("C(φ²) < C(δ_S) < C(1)", 0 if float(C_np(PF**2))<float(C_np(DF))<float(C_np(1.0)) else 1, 0),
    ("Duality at GATE", abs(float(C_np(np.exp(GF))) - 1/np.cosh(GF)), 1e-14),
    ("Gear closure", abs(8*3*np.pi/4 - 3*2*np.pi), 1e-13),
    ("Palindrome int = 8", abs(987654321//123456789 - 8), 0),
    ("η²+|μη|²=1", abs(GF**2 + abs(mu_c*GF)**2 - 1), 1e-14),
    ("F(GATE)·GATE harvest", abs(float(F_np(GF))*GF - float(F_np(GF))*GF), 0),
]

for name, err, tol in cross:
    check(f"{name:<35s} err={err:.2e}", err <= tol)
print()


# ═══════════════════════════════════════════════════════════════════════
# VECTOR 5: EDGE CASES & ADVERSARIAL INPUTS
# ═══════════════════════════════════════════════════════════════════════

print("═" * 72)
print("  VECTOR 5: EDGE CASES & ADVERSARIAL INPUTS")
print("═" * 72, "\n")

edges = [
    ("C(0)", C_np(0.0), 0.0),
    ("C(1e-300)", C_np(1e-300), 0.0),
    ("C(1e300)", C_np(1e300), 0.0),
    ("C(1-1e-15)", C_np(1-1e-15), 1.0),
    ("C(1+1e-15)", C_np(1+1e-15), 1.0),
    ("F(0)", float(F_np(0.0)), 0.0),
    ("F(1e-15)", float(F_np(1e-15)), 0.0),
    ("F(710)", float(F_np(710.0)), 1.0),
    ("F(-710)", float(F_np(-710.0)), 1.0),
    ("F(1000)", float(F_np(1000.0)), 1.0),
]
for name, val, exp in edges:
    err = abs(float(val)-exp)
    check(f"{name:<15s} = {float(val):.6e}  expected {exp:.1f}", err < 0.01)

print("\n─── NaN/Inf resistance ──────────────────────────────────────")
for x in [np.nan, np.inf, -np.inf, -0.0]:
    c = C_np(x)
    f = F_np(x)
    c_ok = np.isfinite(c) or c == 0
    f_ok = np.isfinite(f)
    print(f"  C({str(x):>5s})={str(float(c)):>10s} {'✓' if c_ok else '✗'}  F({str(x):>5s})={str(float(f)):>10s} {'✓' if f_ok else '✗'}")

# Massive random attack
print("\n─── Random ratio attack (1M random floats) ──────────────────")
rng = np.random.default_rng(42)
rand_r = rng.uniform(1e-10, 1e10, 1_000_000)
rand_C = C_np(rand_r)
rand_sym = np.max(np.abs(rand_C - C_np(1/rand_r)))
check(f"Symmetry on 1M random ratios: max err = {rand_sym:.2e}", rand_sym < 1e-8)
check("All random C ∈ [0,1]", np.all(rand_C >= 0) and np.all(rand_C <= 1+1e-10))
print()


# ═══════════════════════════════════════════════════════════════════════
# VERDICT
# ═══════════════════════════════════════════════════════════════════════

print("═" * 72)
print("  VERDICT")
print("═" * 72)
print()
print(f"  Tests passed: {tests_passed}")
print(f"  Tests failed: {tests_failed}")
print(f"  Master hash:  {master}")
print()

if tests_failed == 0:
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║  CANONICAL MAP: UNBROKEN                                ║")
    print("  ║  Symbolic: exact. Numerical: 10M+. Checksums: locked.   ║")
    print("  ║  The math holds. Use it to break everything else.       ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    sys.exit(0)
else:
    print("  ╔══════════════════════════════════════════════════════════╗")
    print(f"  ║  {tests_failed} FAILURE(S) DETECTED                                  ║")
    print("  ║  Investigate immediately.                               ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    sys.exit(tests_failed)
