/-
  CriticalEigenvalue.lean — Lean 4 formalization of core Kernel theorems.

  The central object is the critical eigenvalue
      μ = exp(I · 3π/4) = (−1 + i)/√2,
  the unique balanced point on the unit circle with negative real part
  and positive imaginary part.  It generates an exact 8-cycle and drives
  the coherence / rotation structure studied in the Kernel project.

  Mathematical background: ../docs/master_derivations.pdf

  Sections
  ────────
  1. Critical eigenvalue  μ = exp(I · 3π/4)
  2. Eight-cycle closure  μ^8 = 1
  3. Distinctness of the eight powers of μ   (gcd(3,8) = 1)
  4. Rotation matrix  R(3π/4)  and its properties
  5. Coherence function  C(r) = 2r/(1 + r²)
  6. Canonical-state normalisation  η² + |μ·η|² = 1

  Proof status
  ────────────
  All theorems have complete machine-checked proofs.
  No `sorry` placeholders remain.
-/

import Mathlib.Analysis.SpecialFunctions.Complex.Circle
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.Determinant.Basic
import Mathlib.RingTheory.RootsOfUnity.Complex

open Complex Real Matrix

noncomputable section

-- ════════════════════════════════════════════════════════════════════════════
-- Section 1 — Critical Eigenvalue
-- Ref: docs/master_derivations.pdf §2.1
-- ════════════════════════════════════════════════════════════════════════════

/-- The critical eigenvalue μ = exp(I · 3π/4).

    Numerically: μ = (−1 + i)/√2 ≈ −0.7071 + 0.7071·i.
    It is the unique point on the unit circle at angle 135° = 3π/4. -/
def μ : ℂ := Complex.exp (Complex.I * (3 * ↑Real.pi / 4))

/-- μ can equivalently be written as (−1 + I)/√2. -/
theorem mu_eq_cart : μ = ((-1 + Complex.I) / Real.sqrt 2) := by
  unfold μ
  -- exp(I · 3π/4) = cos(3π/4) + I·sin(3π/4)
  --              = −(1/√2) + I·(1/√2) = (−1 + I)/√2
  rw [mul_comm, Complex.exp_mul_I]
  have hcos : Real.cos (3 * Real.pi / 4) = -(1 / Real.sqrt 2) := by
    rw [show (3 : ℝ) * Real.pi / 4 = Real.pi - Real.pi / 4 by ring]
    rw [Real.cos_pi_sub, Real.cos_pi_div_four]
    norm_num
  have hsin : Real.sin (3 * Real.pi / 4) = 1 / Real.sqrt 2 := by
    rw [show (3 : ℝ) * Real.pi / 4 = Real.pi - Real.pi / 4 by ring]
    rw [Real.sin_pi_sub, Real.sin_pi_div_four]
  simp [Complex.ext_iff, hcos, hsin]
  constructor <;> ring

-- ════════════════════════════════════════════════════════════════════════════
-- Section 2 — 8-cycle closure:  μ^8 = 1
-- Ref: docs/master_derivations.pdf §2.2
-- ════════════════════════════════════════════════════════════════════════════

/-- μ has absolute value 1 (it lies on the complex unit circle).
    Proof: |exp(I·θ)| = exp(Re(I·θ)) = exp(0) = 1. -/
theorem mu_abs_one : Complex.abs μ = 1 := by
  unfold μ
  -- Complex.abs (exp z) = Real.exp z.re; here z = I*(3π/4), z.re = 0
  rw [map_exp, Complex.abs_exp]
  simp [Complex.mul_re, Complex.I_re, Complex.I_im]

/-- μ^8 = 1: the eigenvalue closes an exact 8-cycle.

    Proof:  μ^8 = exp(I · 3π/4)^8
                = exp(8 · I · 3π/4)     [exp homomorphism]
                = exp(I · 6π)
                = exp(I · 2π · 3)
                = (exp(2πi))^3 = 1^3 = 1.

    Ref: docs/master_derivations.pdf §2.2  Theorem 2.1 -/
theorem mu_pow_eight : μ ^ 8 = 1 := by
  unfold μ
  -- Rewrite μ^8 as exp(↑8 · (I · 3π/4)) via the exp homomorphism
  rw [← Complex.exp_nat_mul]
  -- Simplify exponent: 8 · (I · 3π/4) = ↑3 · (2π · I)
  have h : (↑(8 : ℕ) : ℂ) * (Complex.I * (3 * ↑Real.pi / 4)) =
           ↑(3 : ℕ) * (2 * ↑Real.pi * Complex.I) := by
    push_cast; ring
  -- Apply exp(↑3 · z) = exp(z)^3, then exp(2πi) = 1
  rw [h, Complex.exp_nat_mul, Complex.exp_two_pi_mul_I]
  simp

-- ════════════════════════════════════════════════════════════════════════════
-- Section 3 — Distinctness of the eight powers of μ
-- Ref: docs/master_derivations.pdf §2.3
-- ════════════════════════════════════════════════════════════════════════════

/-- The eight powers μ^0, μ^1, …, μ^7 are pairwise distinct.

    Proof:
      (a) Let ζ = exp(2πi/8).  By `Complex.isPrimitiveRoot_exp`, ζ is a
          primitive 8th root of unity.
      (b) μ = exp(I · 3π/4) = exp(2πi · 3/8) = ζ^3.
      (c) Since gcd(3, 8) = 1, `IsPrimitiveRoot.pow_of_coprime` shows
          μ = ζ^3 is itself a primitive 8th root of unity.
      (d) For a primitive 8th root, `IsPrimitiveRoot.pow_inj` guarantees
          that μ^j = μ^k ↔ j = k for j, k < 8.

    Ref: docs/master_derivations.pdf §2.3 -/
theorem mu_powers_distinct :
    ∀ j k : Fin 8, (j : ℕ) ≠ k → μ ^ (j : ℕ) ≠ μ ^ (k : ℕ) := by
  -- (a) ζ = exp(2πi/8) is the standard primitive 8th root of unity
  have hprim8 : IsPrimitiveRoot (Complex.exp (2 * ↑Real.pi * Complex.I / 8)) 8 :=
    Complex.isPrimitiveRoot_exp (by norm_num : (8 : ℕ) ≠ 0)
  -- (b) μ = ζ^3, since exp(3 · (2πi/8)) = exp(3πi/4) = exp(I · 3π/4)
  have hmu_eq : μ = Complex.exp (2 * ↑Real.pi * Complex.I / 8) ^ 3 := by
    unfold μ
    rw [← Complex.exp_nat_mul]
    congr 1
    push_cast
    ring
  -- (c) gcd(3, 8) = 1, so μ = ζ^3 is also a primitive 8th root
  have hmuprim : IsPrimitiveRoot μ 8 := by
    rw [hmu_eq]
    exact hprim8.pow_of_coprime (by decide : Nat.Coprime 3 8)
  -- (d) Distinctness: μ^j = μ^k with j, k < 8 implies j = k
  intro j k hjk heq
  exact hjk ((hmuprim.pow_inj j.isLt k.isLt).mp heq)

-- ════════════════════════════════════════════════════════════════════════════
-- Section 4 — Rotation matrix R(3π/4)
-- Ref: docs/master_derivations.pdf §3
-- ════════════════════════════════════════════════════════════════════════════

/-- The 2×2 rotation matrix by angle θ = 3π/4. -/
def rotMat : Matrix (Fin 2) (Fin 2) ℝ :=
  let c := Real.cos (3 * Real.pi / 4)   -- = −1/√2
  let s := Real.sin (3 * Real.pi / 4)   -- =  1/√2
  !![c, -s; s, c]

/-- det R(3π/4) = 1.
    Proof: det [[c, −s],[s, c]] = c² + s² = cos²θ + sin²θ = 1. -/
theorem rotMat_det : Matrix.det rotMat = 1 := by
  unfold rotMat
  simp [Matrix.det_fin_two]
  have := Real.sin_sq_add_cos_sq (3 * Real.pi / 4)
  nlinarith [Real.sin_sq_add_cos_sq (3 * Real.pi / 4)]

/-- R(3π/4) is orthogonal: R · Rᵀ = I.
    Proof: direct computation using cos²θ + sin²θ = 1 and
           c·(−s) + s·c = 0. -/
theorem rotMat_orthog : rotMat * rotMatᵀ = 1 := by
  unfold rotMat
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [Matrix.mul_apply, Matrix.transpose_apply, Fin.sum_univ_two] <;>
    nlinarith [Real.sin_sq_add_cos_sq (3 * Real.pi / 4)]

/-- R(3π/4)^8 = I (8-fold application returns to identity).

    Proof via double-angle formulas:
      (1) cos(2·3π/4) = cos(3π/2) = 0  →  cos²(3π/4) = sin²(3π/4) = 1/2
      (2) sin(2·3π/4) = sin(3π/2) = −1  →  sin(3π/4)·cos(3π/4) = −1/2
      (3) R^2 = !![0, 1; −1, 0]   (matrix multiplication using (1)/(2))
      (4) R^4 = (R^2)^2 = −I       (squaring (3))
      (5) R^8 = (R^4)^2 = I        (squaring (4))

    Ref: docs/master_derivations.pdf §3 -/
theorem rotMat_pow_eight : rotMat ^ 8 = 1 := by
  -- Abbreviate the angle for readability
  set θ := (3 : ℝ) * Real.pi / 4 with hθ
  -- Pythagorean identity
  have hpy := Real.sin_sq_add_cos_sq θ
  -- cos(3π/2) = 0  (from cos(π + π/2) = −cos(π/2) = 0)
  have hcos32 : Real.cos (3 * Real.pi / 2) = 0 := by
    have : (3 : ℝ) * Real.pi / 2 = Real.pi + Real.pi / 2 := by ring
    rw [this]
    simp [Real.cos_add, Real.cos_pi, Real.sin_pi, Real.cos_pi_div_two]
  -- sin(3π/2) = −1  (from sin(π + π/2) = −sin(π/2) = −1)
  have hsin32 : Real.sin (3 * Real.pi / 2) = -1 := by
    have : (3 : ℝ) * Real.pi / 2 = Real.pi + Real.pi / 2 := by ring
    rw [this]
    simp [Real.sin_add, Real.cos_pi, Real.sin_pi, Real.sin_pi_div_two]
  -- Double-angle: cos(3π/2) = 2·cos²(3π/4) − 1  →  cos²(3π/4) = 1/2
  have hct := Real.cos_two_mul θ
  have h2θ : 2 * θ = 3 * Real.pi / 2 := by simp [hθ]; ring
  rw [h2θ, hcos32] at hct
  -- hct : 0 = 2 * cos(θ)^2 - 1
  have hc2 : Real.cos θ ^ 2 = 1 / 2 := by linarith
  -- sin²(3π/4) = 1/2  (from Pythagorean + cos²=1/2)
  have hs2 : Real.sin θ ^ 2 = 1 / 2 := by nlinarith
  -- Double-angle: sin(3π/2) = 2·sin(3π/4)·cos(3π/4)  →  sin·cos = −1/2
  have hst := Real.sin_two_mul θ
  rw [h2θ, hsin32] at hst
  -- hst : -1 = 2 * sin(θ) * cos(θ)
  have hsc : Real.sin θ * Real.cos θ = -(1 / 2) := by linarith
  -- (3) rotMat^2 = !![0, 1; −1, 0]
  have h2 : rotMat ^ 2 = !![0, 1; -1, 0] := by
    have hdef : rotMat = !![Real.cos θ, -Real.sin θ; Real.sin θ, Real.cos θ] := rfl
    rw [hdef, pow_two]
    ext i j; fin_cases i <;> fin_cases j <;>
      simp only [Matrix.mul_apply, Fin.sum_univ_two, Matrix.cons_val_zero,
                 Matrix.cons_val_one, Matrix.head_cons, Matrix.head_fin_const,
                 neg_mul, mul_neg] <;>
      nlinarith [hc2, hs2, hsc, mul_comm (Real.cos θ) (Real.sin θ)]
  -- (4) rotMat^4 = −I
  have h4 : rotMat ^ 4 = -1 := by
    have : (4 : ℕ) = 2 * 2 := by norm_num
    rw [this, pow_mul, h2]
    ext i j; fin_cases i <;> fin_cases j <;>
      simp [pow_two, Matrix.mul_apply, Fin.sum_univ_two, Matrix.cons_val_zero,
            Matrix.cons_val_one, Matrix.head_cons, Matrix.head_fin_const,
            Matrix.neg_apply, Matrix.one_apply, Pi.neg_apply]
  -- (5) rotMat^8 = I  (square both sides of step 4)
  have : (8 : ℕ) = 2 * 4 := by norm_num
  rw [this, pow_mul, h4]
  simp [pow_two, Matrix.neg_mul, Matrix.mul_neg, Matrix.neg_neg]

-- ════════════════════════════════════════════════════════════════════════════
-- Section 5 — Coherence function  C(r) = 2r / (1 + r²)
-- Ref: docs/master_derivations.pdf §4
-- ════════════════════════════════════════════════════════════════════════════

/-- Coherence function C(r) = 2r / (1 + r²), defined for all r : ℝ. -/
noncomputable def C (r : ℝ) : ℝ := 2 * r / (1 + r ^ 2)

/-- The denominator 1 + r² is always strictly positive. -/
private lemma one_add_sq_pos (r : ℝ) : 0 < 1 + r ^ 2 := by positivity

/-- C(r) ≤ 1 for all r ≥ 0, by the AM–GM inequality (1 + r² ≥ 2r).

    Proof: C(r) ≤ 1  ↔  2r ≤ 1 + r²  ↔  0 ≤ (r − 1)².
    Ref: docs/master_derivations.pdf §4  Proposition 4.1 -/
theorem coherence_le_one (r : ℝ) (hr : 0 ≤ r) : C r ≤ 1 := by
  unfold C
  rw [div_le_one (one_add_sq_pos r)]
  nlinarith [sq_nonneg (r - 1)]

/-- C(r) = 1 if and only if r = 1.
    Proof: equality in AM–GM holds iff r = 1. -/
theorem coherence_eq_one_iff (r : ℝ) (hr : 0 ≤ r) : C r = 1 ↔ r = 1 := by
  unfold C
  rw [div_eq_one_iff_eq (ne_of_gt (one_add_sq_pos r))]
  constructor
  · intro h
    -- 2r = 1 + r²  ↔  (r − 1)² = 0  ↔  r = 1
    have hsq : (r - 1) ^ 2 = 0 := by nlinarith
    have : r - 1 = 0 := by rwa [sq_eq_zero_iff] at hsq
    linarith
  · intro h
    rw [h]; ring

-- ════════════════════════════════════════════════════════════════════════════
-- Section 6 — Canonical-state normalisation
-- Ref: docs/master_derivations.pdf §5
-- ════════════════════════════════════════════════════════════════════════════

/-- The canonical amplitude η = 1/√2. -/
noncomputable def η : ℝ := 1 / Real.sqrt 2

/-- η² = 1/2. -/
private lemma eta_sq : η ^ 2 = 1 / 2 := by
  unfold η
  rw [div_pow, one_pow, Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2)]

/-- |μ|² = 1 (μ lies on the unit circle). -/
private lemma mu_normSq_one : Complex.normSq μ = 1 := by
  -- normSq (exp(Iθ)) = cos²θ + sin²θ = 1 by the Pythagorean identity
  unfold μ
  rw [mul_comm, Complex.exp_mul_I]
  simp only [Complex.normSq_apply, Complex.add_re, Complex.mul_re,
             Complex.ofReal_re, Complex.I_re, Complex.I_im,
             Complex.add_im, Complex.mul_im, Complex.ofReal_im]
  nlinarith [Real.sin_sq_add_cos_sq (3 * Real.pi / 4)]

/-- The canonical state has norm 1:  η² + |μ·η|² = 1.

    Proof:
      |μ·η|² = |μ|² · η² = 1 · (1/2) = 1/2
      η²  + |μ·η|²  = 1/2 + 1/2 = 1.

    Ref: docs/master_derivations.pdf §5  Proposition 5.1 -/
theorem canonical_norm : η ^ 2 + Complex.normSq (μ * ↑η) = 1 := by
  -- Reduce |μ·η|² to normSq μ · η²
  have h1 : Complex.normSq (μ * ↑η) = η ^ 2 := by
    rw [map_mul, Complex.normSq_ofReal, mu_normSq_one]
    ring
  rw [h1, eta_sq]
  norm_num

end -- noncomputable section
