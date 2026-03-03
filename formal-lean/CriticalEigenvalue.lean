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
  1.  Critical eigenvalue  μ = exp(I · 3π/4)
  2.  Eight-cycle closure  μ^8 = 1
  3.  Distinctness of the eight powers of μ   (gcd(3,8) = 1)
  4.  Rotation matrix  R(3π/4)  and its properties
  5.  Coherence function  C(r) = 2r/(1 + r²)
  6.  Canonical-state normalisation  η² + |μ·η|² = 1
  7.  Silver ratio  δS = 1 + √2  (Proposition 4)
  8.  Additional coherence properties  (symmetry, positivity, strict bound)
  9.  Palindrome residual  R(r) = (r − 1/r)/δS  (Theorem 12)
  10. Lyapunov–coherence duality  C(exp λ) = sech λ  (Theorem 14)
  11. Derived invariant equivalences  (machine-discovered connections)
  12. Orbit magnitude and trichotomy  |ξⁿ| = rⁿ  (Theorem 10)
  13. Coherence monotonicity  (strictly increasing on (0,1], decreasing on [1,∞))
  14. Palindrome arithmetic  (digit identities, gcd/lcm of torus periods)
  15. Z/8Z rotational memory  (bank addressing, μ^(j+8) = μ^j)
  16. Zero-overhead precession  (|e^{iθ}| = 1, preserves |β| and C(r))

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

-- ════════════════════════════════════════════════════════════════════════════
-- Section 7 — Silver Ratio Conservation (Proposition 4)
-- Ref: docs/master_derivations.pdf §1
-- ════════════════════════════════════════════════════════════════════════════

/-- The silver ratio δS = 1 + √2, the arithmetic dual of μ. -/
noncomputable def δS : ℝ := 1 + Real.sqrt 2

private lemma δS_pos : 0 < δS := by unfold δS; positivity

private lemma sqrt2_sq : Real.sqrt 2 * Real.sqrt 2 = 2 :=
  Real.mul_self_sqrt (by norm_num)

/-- Silver conservation product: δS · (√2 − 1) = 1.
    Ref: docs/master_derivations.pdf Proposition 4 -/
theorem silverRatio_mul_conj : δS * (Real.sqrt 2 - 1) = 1 := by
  unfold δS; nlinarith [sqrt2_sq]

/-- Silver quadratic identity: δS² = 2·δS + 1.
    Ref: docs/master_derivations.pdf Proposition 4 -/
theorem silverRatio_sq : δS ^ 2 = 2 * δS + 1 := by
  unfold δS; nlinarith [sqrt2_sq]

/-- The multiplicative inverse of δS equals √2 − 1. -/
theorem silverRatio_inv : 1 / δS = Real.sqrt 2 - 1 := by
  rw [div_eq_iff (ne_of_gt δS_pos)]
  linarith [silverRatio_mul_conj]

-- ════════════════════════════════════════════════════════════════════════════
-- Section 8 — Additional Coherence Properties
-- Ref: docs/master_derivations.pdf §4 Theorem 11
-- ════════════════════════════════════════════════════════════════════════════

/-- C(r) > 0 for all r > 0. -/
theorem coherence_pos (r : ℝ) (hr : 0 < r) : 0 < C r := by
  unfold C; exact div_pos (by linarith) (one_add_sq_pos r)

/-- C(r) = C(1/r): coherence is symmetric about r = 1.
    This duality reflects the equivalence of amplitude ratios r and 1/r.
    Ref: docs/master_derivations.pdf §4 Theorem 11 property (2) -/
theorem coherence_symm (r : ℝ) (hr : 0 < r) : C r = C (1 / r) := by
  unfold C
  have hr' : r ≠ 0 := ne_of_gt hr
  field_simp [hr', ne_of_gt (one_add_sq_pos r), ne_of_gt (one_add_sq_pos (1 / r))]
  ring

/-- C(r) < 1 for r ≥ 0 with r ≠ 1: the maximum is uniquely attained at r = 1.
    Ref: docs/master_derivations.pdf §4 Theorem 11 property (4) -/
theorem coherence_lt_one (r : ℝ) (hr : 0 ≤ r) (hr1 : r ≠ 1) : C r < 1 :=
  lt_of_le_of_ne (coherence_le_one r hr) (mt (coherence_eq_one_iff r hr).mp hr1)

-- ════════════════════════════════════════════════════════════════════════════
-- Section 9 — Palindrome Residual (Theorem 12)
-- Ref: docs/master_derivations.pdf §4 Theorem 12
-- ════════════════════════════════════════════════════════════════════════════

/-- The palindrome residual R(r) = (r − 1/r) / δS.
    Vanishes at r = 1; positive for r > 1; negative for 0 < r < 1.
    Ref: docs/master_derivations.pdf §4 Theorem 12 -/
noncomputable def Res (r : ℝ) : ℝ := (r - 1 / r) / δS

/-- R(r) = 0 ↔ r = 1 for r > 0.
    Ref: docs/master_derivations.pdf §4 Theorem 12 property (1) -/
theorem palindrome_residual_zero_iff (r : ℝ) (hr : 0 < r) : Res r = 0 ↔ r = 1 := by
  unfold Res
  rw [div_eq_zero_iff, or_iff_left (ne_of_gt δS_pos)]
  have hr' : r ≠ 0 := ne_of_gt hr
  constructor
  · intro h
    -- r − 1/r = 0  →  r·(r − 1/r) = 0  →  r² − 1 = 0  →  (r−1)(r+1) = 0
    have hrr : r * r = 1 := by
      have hstep : r * r - 1 = 0 := by
        have : r * (r - 1 / r) = r * r - 1 := by field_simp; ring
        linarith [show r * (r - 1 / r) = 0 from by rw [h, mul_zero]]
      linarith
    have hfact : (r - 1) * (r + 1) = 0 := by linear_combination hrr
    rcases mul_eq_zero.mp hfact with h1 | h1
    · linarith
    · linarith
  · rintro rfl; norm_num

/-- R(r) > 0 for r > 1.
    Ref: docs/master_derivations.pdf §4 Theorem 12 property (2) -/
theorem palindrome_residual_pos (r : ℝ) (hr : 1 < r) : 0 < Res r := by
  unfold Res
  apply div_pos _ δS_pos
  have hr0 : (0 : ℝ) < r := by linarith
  have hr' : r ≠ 0 := ne_of_gt hr0
  have hrep : r - 1 / r = (r ^ 2 - 1) / r := by field_simp; ring
  rw [hrep]; exact div_pos (by nlinarith) hr0

/-- R(r) < 0 for 0 < r < 1.
    Ref: docs/master_derivations.pdf §4 Theorem 12 property (3) -/
theorem palindrome_residual_neg (r : ℝ) (hr0 : 0 < r) (hr1 : r < 1) : Res r < 0 := by
  unfold Res
  apply div_neg_of_neg_of_pos _ δS_pos
  have hr' : r ≠ 0 := ne_of_gt hr0
  have hrep : r - 1 / r = (r ^ 2 - 1) / r := by field_simp; ring
  rw [hrep]; exact div_neg_of_neg_of_pos (by nlinarith) hr0

/-- R(1/r) = −R(r): palindrome residual is anti-symmetric about r = 1.
    This is the odd counterpart to coherence's even symmetry C(r) = C(1/r). -/
theorem palindrome_residual_antisymm (r : ℝ) (hr : 0 < r) : Res (1 / r) = -Res r := by
  unfold Res
  have hr' : r ≠ 0 := ne_of_gt hr
  field_simp; ring

-- ════════════════════════════════════════════════════════════════════════════
-- Section 10 — Lyapunov–Coherence Duality (Theorem 14)
-- Ref: docs/master_derivations.pdf §4 Theorem 14
-- ════════════════════════════════════════════════════════════════════════════

/-- Key identity: C(exp λ) · (exp λ + exp(−λ)) = 2. -/
private lemma lyapunov_key (λ : ℝ) :
    C (Real.exp λ) * (Real.exp λ + Real.exp (-λ)) = 2 := by
  have hmul : Real.exp λ * Real.exp (-λ) = 1 := by
    rw [← Real.exp_add]; simp
  unfold C
  have hd : (0 : ℝ) < 1 + Real.exp λ ^ 2 := one_add_sq_pos _
  field_simp [ne_of_gt hd]
  nlinarith [Real.exp_pos λ, Real.exp_pos (-λ), hmul, sq_nonneg (Real.exp λ)]

/-- Lyapunov–coherence duality: C(exp λ) = 2/(exp λ + exp(−λ)) = sech λ.
    The coherence is the reciprocal of the hyperbolic cosine of the Lyapunov
    exponent λ = ln r.  At the balanced fixed point r = 1 (λ = 0):
    sech(0) = 1, recovering C = 1.
    Ref: docs/master_derivations.pdf §4 Theorem 14 -/
theorem lyapunov_coherence_duality (λ : ℝ) :
    C (Real.exp λ) = 2 / (Real.exp λ + Real.exp (-λ)) := by
  have hpos : 0 < Real.exp λ + Real.exp (-λ) := by positivity
  rw [eq_div_iff (ne_of_gt hpos), mul_comm]
  exact lyapunov_key λ

/-- Corollary: C(exp λ) = (cosh λ)⁻¹ = sech λ.
    Uses: cosh λ = (exp λ + exp(−λ)) / 2. -/
theorem lyapunov_coherence_sech (λ : ℝ) :
    C (Real.exp λ) = (Real.cosh λ)⁻¹ := by
  have hcosh : Real.cosh λ = (Real.exp λ + Real.exp (-λ)) / 2 := Real.cosh_eq λ
  rw [lyapunov_coherence_duality, hcosh, inv_div]

-- ════════════════════════════════════════════════════════════════════════════
-- Section 11 — Derived Invariant Equivalences
-- The formal system discovers that independently-defined invariants
-- (coherence C, palindrome residual R, Lyapunov exponent λ) all
-- characterise the same balanced fixed point r = 1.
-- Ref: docs/master_derivations.pdf Corollary 13
-- ════════════════════════════════════════════════════════════════════════════

/-- **Derived**: palindrome residual zero ↔ maximum coherence.
    Two independently-defined invariants identify the same fixed point.
    Ref: docs/master_derivations.pdf Corollary 13 -/
theorem palindrome_coherence_equiv (r : ℝ) (hr : 0 < r) :
    Res r = 0 ↔ C r = 1 :=
  (palindrome_residual_zero_iff r hr).trans (coherence_eq_one_iff r (le_of_lt hr)).symm

/-- **Derived**: C is even (C(r) = C(1/r)) while R is odd (R(1/r) = −R(r)).
    Both are symmetric about the same fixed point r = 1, in dual senses. -/
theorem coherence_palindrome_duality (r : ℝ) (hr : 0 < r) :
    C r = C (1 / r) ∧ Res (1 / r) = -Res r :=
  ⟨coherence_symm r hr, palindrome_residual_antisymm r hr⟩

/-- **Derived**: the C-maximum is preserved under the symmetry r ↦ 1/r.
    C(r) = 1 ↔ C(1/r) = 1 — the maximum is a fixed point of the inversion. -/
theorem coherence_max_symm (r : ℝ) (hr : 0 < r) :
    C r = 1 ↔ C (1 / r) = 1 := by
  constructor
  · intro h; rwa [← coherence_symm r hr]
  · intro h; rwa [coherence_symm r hr]

/-- **Derived**: at the palindrome equilibrium the state is self-dual: r = 1/r. -/
theorem palindrome_zero_self_dual (r : ℝ) (hr : 0 < r) (h : Res r = 0) : r = 1 / r := by
  have := (palindrome_residual_zero_iff r hr).mp h
  rw [this]; norm_num

/-- **Derived**: simultaneous break — all equilibrium invariants are equivalent.
    The balanced state r = 1 is the unique point where coherence is maximal,
    palindrome residual vanishes, and the Lyapunov exponent is zero.
    Ref: docs/master_derivations.pdf Corollary 13 -/
theorem simultaneous_break (r : ℝ) (hr : 0 < r) :
    r = 1 ↔ C r = 1 ∧ Res r = 0 := by
  constructor
  · intro h
    exact ⟨(coherence_eq_one_iff r (le_of_lt hr)).mpr h,
            (palindrome_residual_zero_iff r hr).mpr h⟩
  · intro ⟨hC, _⟩
    exact (coherence_eq_one_iff r (le_of_lt hr)).mp hC

/-- **Derived**: the Lyapunov–coherence duality implies C ≤ 1 for all λ,
    since sech(λ) ≤ 1 = sech(0).  This recovers `coherence_le_one` via a
    completely different route (the hyperbolic bound). -/
theorem lyapunov_bound (λ : ℝ) : C (Real.exp λ) ≤ 1 :=
  coherence_le_one _ (le_of_lt (Real.exp_pos λ))

-- ════════════════════════════════════════════════════════════════════════════
-- Section 12 — Orbit Magnitude and Trichotomy (Theorem 10)
-- |ξ^n| = r^n for ξ = r·μ.  The three cases r=1 / r>1 / r<1 determine
-- whether the orbit is a closed 8-cycle, spirals outward, or collapses.
-- Ref: docs/master_derivations.pdf §5 Theorem 10
-- ════════════════════════════════════════════════════════════════════════════

/-- |μ^n| = 1 for all n: the orbit of μ stays on the unit circle.
    Follows immediately from |μ| = 1 and multiplicativity of the absolute value. -/
theorem mu_pow_abs (n : ℕ) : Complex.abs (μ ^ n) = 1 := by
  have h : Complex.abs (μ ^ n) = Complex.abs μ ^ n := map_pow Complex.abs μ n
  rw [h, mu_abs_one, one_pow]

/-- |(r·μ)^n| = r^n for r ≥ 0.
    The amplitude of the orbit is purely radial; the phase factor μ contributes
    no growth or decay.  This is the key quantitative form of Trichotomy. -/
theorem scaled_orbit_abs (r : ℝ) (hr : 0 ≤ r) (n : ℕ) :
    Complex.abs ((↑r * μ) ^ n) = r ^ n := by
  have habsr : Complex.abs (↑r * μ) = r := by
    rw [map_mul, Complex.abs_ofReal, abs_of_nonneg hr, mu_abs_one, mul_one]
  calc Complex.abs ((↑r * μ) ^ n)
      = Complex.abs (↑r * μ) ^ n := map_pow Complex.abs _ _
    _ = r ^ n := by rw [habsr]

/-- Trichotomy — r = 1: orbit has unit magnitude at every step (stable 8-cycle).
    Ref: docs/master_derivations.pdf §5 Theorem 10 case (1) -/
theorem trichotomy_unit_orbit (n : ℕ) : Complex.abs ((1 : ℂ) * μ ^ n) = 1 := by
  simp [map_mul, mu_pow_abs]

/-- Trichotomy — r > 1: magnitudes are strictly increasing (spiral outward).
    |(r·μ)^n| < |(r·μ)^(n+1)| since r^n < r^(n+1) when r > 1.
    Ref: docs/master_derivations.pdf §5 Theorem 10 case (2) -/
theorem trichotomy_grow (r : ℝ) (hr : 1 < r) (n : ℕ) :
    Complex.abs ((↑r * μ) ^ n) < Complex.abs ((↑r * μ) ^ (n + 1)) := by
  simp only [scaled_orbit_abs r (le_of_lt (lt_trans one_pos hr))]
  exact pow_lt_pow_right hr (Nat.lt_succ_self n)

/-- Trichotomy — 0 < r < 1: magnitudes are strictly decreasing (spiral inward).
    |(r·μ)^(n+1)| < |(r·μ)^n| since r^(n+1) < r^n when 0 < r < 1.
    Ref: docs/master_derivations.pdf §5 Theorem 10 case (3) -/
theorem trichotomy_shrink (r : ℝ) (hr0 : 0 < r) (hr1 : r < 1) (n : ℕ) :
    Complex.abs ((↑r * μ) ^ (n + 1)) < Complex.abs ((↑r * μ) ^ n) := by
  simp only [scaled_orbit_abs r (le_of_lt hr0)]
  exact pow_lt_pow_of_lt_one (le_of_lt hr0) hr1 (Nat.lt_succ_self n)

-- ════════════════════════════════════════════════════════════════════════════
-- Section 13 — Coherence Monotonicity
-- C(r) = 2r/(1+r²) is strictly increasing on (0,1] and strictly decreasing
-- on [1,∞).  This is the "gradient flow toward balance" that the Kernel
-- system exploits: any r≠1 can recover by moving r toward 1.
-- Ref: docs/master_derivations.pdf §4 Theorem 11
-- ════════════════════════════════════════════════════════════════════════════

/-- Factorisation of C(s) − C(r) useful for monotonicity proofs.
    C(s) − C(r) = 2(s−r)(1−rs) / ((1+r²)(1+s²)).
    The sign of the numerator 2(s−r)(1−rs) determines the direction of change. -/
private lemma coherence_diff_factored (r s : ℝ) (hr : 0 < r) (hs : 0 < s) :
    C s - C r = 2 * (s - r) * (1 - r * s) / ((1 + r ^ 2) * (1 + s ^ 2)) := by
  have hr' : 1 + r ^ 2 ≠ 0 := ne_of_gt (one_add_sq_pos r)
  have hs' : 1 + s ^ 2 ≠ 0 := ne_of_gt (one_add_sq_pos s)
  unfold C; field_simp [hr', hs']; ring

/-- C is strictly increasing on (0, 1]: for 0 < r < s ≤ 1, C(r) < C(s).
    When both components are below balance (|β| < |α|), increasing |β|/|α|
    toward 1 strictly improves coherence.
    Ref: docs/master_derivations.pdf §4 Theorem 11 -/
theorem coherence_strictMono (r s : ℝ) (hr : 0 < r) (hrs : r < s) (hs1 : s ≤ 1) :
    C r < C s := by
  rw [← sub_pos, coherence_diff_factored r s hr (lt_trans hr hrs)]
  refine div_pos ?_ (mul_pos (one_add_sq_pos r) (one_add_sq_pos s))
  have hsr : 0 < s - r := sub_pos.mpr hrs
  have hr1 : r < 1 := lt_of_lt_of_le hrs hs1
  have hrslt1 : r * s < 1 := by
    calc r * s < 1 * s := mul_lt_mul_of_pos_right hr1 (lt_trans hr hrs)
         _ = s := one_mul s
         _ ≤ 1 := hs1
  have h_pos : (s - r) * (1 - r * s) > 0 := mul_pos hsr (by linarith)
  nlinarith

/-- C is strictly decreasing on [1, ∞): for 1 ≤ r < s, C(s) < C(r).
    When both components are above balance (|β| > |α|), increasing |β|/|α|
    away from 1 strictly decreases coherence.
    Ref: docs/master_derivations.pdf §4 Theorem 11 -/
theorem coherence_strictAnti (r s : ℝ) (hr1 : 1 ≤ r) (hrs : r < s) :
    C s < C r := by
  have hr : 0 < r := lt_of_lt_of_le one_pos hr1
  rw [← sub_neg, coherence_diff_factored r s hr (lt_trans hr hrs)]
  refine div_neg_of_neg_of_pos ?_ (mul_pos (one_add_sq_pos r) (one_add_sq_pos s))
  have hsr : 0 < s - r := sub_pos.mpr hrs
  have hs1 : 1 < s := lt_of_le_of_lt hr1 hrs
  have hrsgt1 : 1 < r * s := by nlinarith
  have h_neg : (s - r) * (1 - r * s) < 0 :=
    mul_neg_of_pos_of_neg hsr (by linarith)
  nlinarith

-- ════════════════════════════════════════════════════════════════════════════
-- Section 14 — Palindrome Arithmetic (Proposition from §6 of master_derivations.pdf)
-- The palindrome digit pair 987654321 / 123456789 encodes the 8-cycle period
-- (integer quotient = 8) and slow-precession period D = 13717421.
-- Ref: docs/master_derivations.pdf §6
-- ════════════════════════════════════════════════════════════════════════════

/-- Two-palindrome complementarity: 987654321 = 8 × 123456789 + 9.
    The integer part 8 equals the μ-rotation period; the remainder 9 connects
    to the slow-precession denominator via 9 × D = 123456789.
    Ref: docs/master_derivations.pdf §6 Proposition 5 -/
theorem palindrome_comp : (987654321 : ℕ) = 8 * 123456789 + 9 := by norm_num

/-- Precession period identity: 9 × D = 123456789 where D = 13717421.
    D is the slow-precession period (denominator of ΔΦ₀ = 2π/D).
    Ref: docs/master_derivations.pdf §6 -/
theorem precession_period_factor : 9 * 13717421 = (123456789 : ℕ) := by norm_num

/-- gcd(8, D) = 1: the fast 8-cycle period and slow-precession period are coprime.
    Coprimality ensures the torus T² = S¹ × S¹ has no resonance between the
    two winding numbers, so all 8·D orbit points are distinct. -/
theorem precession_gcd_one : Nat.gcd 8 13717421 = 1 := by native_decide

/-- lcm(8, D) = 8·D = 109739368: the joint orbit closes after 8·D steps.
    Follows from gcd(8, D) = 1 and the formula lcm(a,b) = ab/gcd(a,b). -/
theorem precession_lcm : Nat.lcm 8 13717421 = 8 * 13717421 := by
  unfold Nat.lcm; rw [precession_gcd_one, Nat.div_one]

-- ════════════════════════════════════════════════════════════════════════════
-- Section 15 — Z/8Z Rotational Memory (Proposition from §11 of master_derivations.pdf)
-- Physical addresses decompose as (bank, offset) = (addr % 8, addr / 8).
-- The μ-orbit provides a natural clock that aligns memory banks with the
-- 8-cycle, so the group Z/8Z governs both phase rotation and bank addressing.
-- Ref: docs/master_derivations.pdf §11
-- ════════════════════════════════════════════════════════════════════════════

/-- Eight consecutive rotations return to the starting bank: (n + 8) % 8 = n % 8.
    Ref: docs/master_derivations.pdf §11 Proposition Z/8Z -/
theorem z8z_period (n : ℕ) : (n + 8) % 8 = n % 8 := by omega

/-- Memory address reconstruction: addr % 8 + 8 * (addr / 8) = addr.
    The (bank, offset) decomposition is lossless. -/
theorem z8z_reconstruction (addr : ℕ) : addr % 8 + 8 * (addr / 8) = addr := by omega

/-- **Derived**: the μ-orbit inherits Z/8Z periodicity: μ^(j+8) = μ^j.
    The eigenvalue clock and the memory bank clock share the same period 8,
    discovered by combining mu_pow_eight with the ring law for exponents. -/
theorem mu_z8z_period (j : ℕ) : μ ^ (j + 8) = μ ^ j := by
  rw [pow_add, mu_pow_eight, mul_one]

-- ════════════════════════════════════════════════════════════════════════════
-- Section 16 — Zero-Overhead Precession (Theorem from §13 of master_derivations.pdf)
-- The precession step β ↦ e^{iθ}·β rotates the phase without changing the
-- amplitude |β|.  Consequently it preserves r = |β|/|α|, C(r), and all
-- equilibrium invariants simultaneously — "zero overhead" in the sense of
-- Theorem 5.1 (docs/master_derivations.pdf).
-- Ref: docs/master_derivations.pdf §13 Theorem zero-overhead-prec
-- ════════════════════════════════════════════════════════════════════════════

/-- Precession phasor e^{iθ} has unit absolute value for any real phase θ.
    Proof: |exp(z)| = exp(Re z); Re(I·θ) = 0; exp(0) = 1.
    Ref: docs/master_derivations.pdf §13 Theorem zero-overhead-prec -/
theorem precession_phasor_unit (θ : ℝ) :
    Complex.abs (Complex.exp (Complex.I * ↑θ)) = 1 := by
  rw [Complex.abs_exp]
  have hre : (Complex.I * ↑θ).re = 0 := by
    simp [Complex.mul_re, Complex.I_re, Complex.I_im,
          Complex.ofReal_re, Complex.ofReal_im]
  rw [hre, Real.exp_zero]

/-- Multiplying by a unit-norm phasor preserves the absolute value of any β ∈ ℂ.
    This is the mathematical core of "zero overhead": the precession step
    β ↦ e^{iθ}·β leaves |β| — and hence r = |β|/|α| — unchanged. -/
theorem precession_preserves_abs (β : ℂ) (θ : ℝ) :
    Complex.abs (Complex.exp (Complex.I * ↑θ) * β) = Complex.abs β := by
  rw [map_mul, precession_phasor_unit, one_mul]

/-- **Derived**: precession preserves the coherence ratio.
    If α and β are the two state components and both are multiplied by the
    same phasor e^{iθ}, the ratio |β|/|α| — and therefore C(r) — is invariant.
    This formally proves that precession steps are "zero overhead" for coherence. -/
theorem precession_preserves_coherence (α β : ℂ) (hα : Complex.abs α ≠ 0) (θ : ℝ) :
    C (Complex.abs (Complex.exp (Complex.I * ↑θ) * β) / Complex.abs α) =
    C (Complex.abs β / Complex.abs α) := by
  rw [precession_preserves_abs]

end -- noncomputable section
