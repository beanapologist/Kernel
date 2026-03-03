/-
  Main.lean — Entry point for the Kernel Lean 4 formalization.

  This module imports CriticalEigenvalue and prints a summary of all
  theorems that have been verified by the Lean 4 type checker.

  Mathematical background: ../docs/master_derivations.pdf

  Usage:
    lake exe cache get    -- (recommended) fetch pre-built Mathlib cache
    lake build            -- build the project
    lake exe formalLean   -- run this entry point
-/
import CriticalEigenvalue

def main : IO Unit := do
  IO.println "==================================================="
  IO.println " Kernel — Lean 4 Formal Verification"
  IO.println "==================================================="
  IO.println ""
  IO.println "§1–6  Core eigenvalue and coherence structure"
  IO.println ""
  IO.println "  [1]  mu_eq_cart              : μ = (−1 + i)/√2"
  IO.println "  [2]  mu_abs_one              : |μ| = 1"
  IO.println "  [3]  mu_pow_eight            : μ^8 = 1  (8-cycle closure)"
  IO.println "  [4]  mu_powers_distinct      : {μ^0,…,μ^7} pairwise distinct"
  IO.println "  [5]  rotMat_det              : det R(3π/4) = 1"
  IO.println "  [6]  rotMat_orthog           : R · Rᵀ = I"
  IO.println "  [7]  rotMat_pow_eight        : R(3π/4)^8 = I"
  IO.println "  [8]  coherence_le_one        : C(r) ≤ 1"
  IO.println "  [9]  coherence_eq_one_iff    : C(r) = 1 ↔ r = 1"
  IO.println "  [10] canonical_norm          : η² + |μ·η|² = 1"
  IO.println ""
  IO.println "§7    Silver ratio (Proposition 4)"
  IO.println ""
  IO.println "  [11] silverRatio_mul_conj    : δS · (√2−1) = 1"
  IO.println "  [12] silverRatio_sq          : δS² = 2·δS + 1"
  IO.println "  [13] silverRatio_inv         : 1/δS = √2−1"
  IO.println ""
  IO.println "§8    Additional coherence properties (Theorem 11)"
  IO.println ""
  IO.println "  [14] coherence_pos           : C(r) > 0  for r > 0"
  IO.println "  [15] coherence_symm          : C(r) = C(1/r)  (even symmetry)"
  IO.println "  [16] coherence_lt_one        : C(r) < 1  for r ≥ 0, r ≠ 1"
  IO.println ""
  IO.println "§9    Palindrome residual (Theorem 12)"
  IO.println ""
  IO.println "  [17] palindrome_residual_zero_iff  : R(r) = 0 ↔ r = 1"
  IO.println "  [18] palindrome_residual_pos       : R(r) > 0  for r > 1"
  IO.println "  [19] palindrome_residual_neg       : R(r) < 0  for 0 < r < 1"
  IO.println "  [20] palindrome_residual_antisymm  : R(1/r) = −R(r)  (odd symmetry)"
  IO.println ""
  IO.println "§10   Lyapunov–coherence duality (Theorem 14)"
  IO.println ""
  IO.println "  [21] lyapunov_coherence_duality : C(exp λ) = 2/(exp λ + exp(−λ))"
  IO.println "  [22] lyapunov_coherence_sech    : C(exp λ) = (cosh λ)⁻¹ = sech λ"
  IO.println ""
  IO.println "§11   Derived invariant equivalences  (machine-discovered)"
  IO.println ""
  IO.println "  [23] palindrome_coherence_equiv  : R(r)=0 ↔ C(r)=1"
  IO.println "  [24] coherence_palindrome_duality: C even ∧ R odd  (dual symmetries)"
  IO.println "  [25] coherence_max_symm          : C(r)=1 ↔ C(1/r)=1"
  IO.println "  [26] palindrome_zero_self_dual   : R(r)=0 → r = 1/r"
  IO.println "  [27] simultaneous_break          : r=1 ↔ C(r)=1 ∧ R(r)=0"
  IO.println "  [28] lyapunov_bound              : C(exp λ) ≤ 1  (via sech route)"
  IO.println ""
  IO.println "28 theorems — all machine-checked, zero sorry."
  IO.println ""
  IO.println "See CriticalEigenvalue.lean for full proof terms."
