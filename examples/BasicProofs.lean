/-
  examples/BasicProofs.lean — Introductory examples from the mathematical universe.

  This file demonstrates how to use the verified theorems from the Lean
  mathematical universe in downstream proofs.  The examples are designed
  to be run with:

      cd formal-lean/
      lake build
      lake exe formalLean

  ── Example 1: 8-cycle closure ──────────────────────────────────────────────
  The critical eigenvalue μ = exp(I · 3π/4) satisfies μ^8 = 1.
  Proof: CriticalEigenvalue.mu_pow_eight

  ── Example 2: Coherence upper bound ────────────────────────────────────────
  For all r > 0, C(r) = 2r/(1+r²) ≤ 1 with equality iff r = 1.
  Proof: CriticalEigenvalue.coherence_le_one

  ── Example 3: Speed of light identity ──────────────────────────────────────
  c² · μ₀ · ε₀ = 1, i.e. c = 1/√(μ₀ε₀).
  Proof: SpeedOfLight.c_sq_mu0_eps0

  ── Example 4: Floquet period-doubling ──────────────────────────────────────
  A Floquet state with phase φ = π satisfies ψ(t + 2T) = ψ(t).
  Proof: TimeCrystal.time_crystal_period_doubling

  ── Example 5: Silver ratio uniqueness ──────────────────────────────────────
  δS = 1 + √2 is the unique positive root of x² - 2x - 1 = 0.
  Proof: SilverCoherence.silver_ratio_unique

  See formal-lean/*.lean for full proof terms.
-/

-- Import the top-level entry point.
-- (Uncomment after running `lake build` from formal-lean/)
-- import FormalLean.CriticalEigenvalue
-- import FormalLean.TimeCrystal
-- import FormalLean.SpeedOfLight
-- import FormalLean.SilverCoherence

/-
  Example usage in a downstream theorem:

  theorem my_result : ∀ r : ℝ, 0 < r → coherence r ≤ 1 :=
    CriticalEigenvalue.coherence_le_one

  theorem energy_cycle (n : ℕ) : μ ^ (n + 8) = μ ^ n :=
    CriticalEigenvalue.mu_pow_period n
-/

#check @id  -- placeholder to keep the file valid before build
