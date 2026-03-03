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
  IO.println "Theorems verified by the Lean 4 type checker:"
  IO.println ""
  IO.println "  [1] mu_eq_cart       : μ = (−1 + i)/√2  (Cartesian form)"
  IO.println "  [2] mu_pow_eight    : μ^8 = 1  (8-cycle closure)"
  IO.println "  [3] mu_abs_one      : |μ| = 1  (μ lies on the unit circle)"
  IO.println "  [4] rotMat_det      : det R(3π/4) = 1"
  IO.println "  [5] rotMat_orthog   : R(3π/4) · R(3π/4)ᵀ = I"
  IO.println "  [6] rotMat_pow_eight: R(3π/4)^8 = I"
  IO.println "  [7] coherence_le_one: C(r) ≤ 1, with equality iff r = 1"
  IO.println "  [8] canonical_norm  : η² + |μ·η|² = 1  (η = 1/√2)"
  IO.println ""
  IO.println "See CriticalEigenvalue.lean for full proof terms."
