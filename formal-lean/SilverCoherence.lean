/-
  SilverCoherence.lean — A machine-checked answer to a question about the
  Kernel coherence function and the critical eigenvalue μ.

  ╔══════════════════════════════════════════════════════════════════════════╗
  ║ The question I was curious about:                                       ║
  ║                                                                         ║
  ║   The critical eigenvalue μ = (−1+i)/√2 has components of magnitude   ║
  ║   |Re(μ)| = |Im(μ)| = 1/√2 ≈ 0.7071.  Is there a scale r that        ║
  ║   already appears naturally in the Kernel framework at which the        ║
  ║   coherence function achieves exactly this value, C(r) = 1/√2?        ║
  ╚══════════════════════════════════════════════════════════════════════════╝

  The machine-checked answer: YES — uniquely at r = δS = 1 + √2,
  the very same silver ratio that appears as the denominator of the
  palindrome residual Res(r) = (r − 1/r)/δS in CriticalEigenvalue.lean §9.

  Proof sketch:
    C(δS) = 2δS / (1 + δS²)
          = 2δS / (1 + 2δS + 1)    [δS² = 2δS + 1, silverRatio_sq]
          = 2δS / (2 + 2δS)
          = δS / (1 + δS)
          = (1+√2) / (2+√2)
          = (1+√2)(2−√2) / ((2+√2)(2−√2))   [rationalize]
          = √2 / 2                            [since (1+√2)(2−√2) = √2, (2+√2)(2−√2) = 2]

  This is also Im(μ): since μ = exp(I·3π/4), Im(μ) = sin(3π/4) = sin(π/4) = √2/2.
  So C(δS) = Im(μ) — the silver-ratio scale is where the Kernel coherence
  equals the imaginary (and real) component magnitude of the critical eigenvalue.

  A further geometric fact: at r = δS, the coherence C(r) equals the
  "imbalance" (r²−1)/(1+r²), making the Pythagorean identity
      C(δS)² + C(δS)² = 1
  hold (the two terms are equal, both equal to 1/2).  The silver ratio is
  the unique positive scale where coherence and imbalance are identical —
  the "45-degree point" of the coherence-imbalance unit circle.

  In the coherence ordering, the silver scale sits strictly between the
  Koide wing (C = 2/3 ≈ 0.667) and the kernel maximum (C = 1):
      C(φ²) = 2/3  <  C(δS) = √2/2  <  C(1) = 1

  Sections
  ────────
  1.  Silver-ratio coherence  C(δS) = √2/2  (the main result)
  2.  Algebraic consequences  (C(δS)² = 1/2; isotropic diagonal property; 2C²=1)
  3.  Connection to μ  (Im(μ) = √2/2 = C(δS))
  4.  Ohm–Coherence at the silver scale  (G = √2/2, R = √2, G·R = 1)
  5.  Position in the coherence ordering  (Koide < silver < kernel)
  6.  Scale placement and symmetry
  7.  Uniqueness  (only r = δS or r = 1/δS satisfies C(r) = √2/2)
  8.  Physics at 45°  (scattering amplitude, Schwinger bound, EM coherence)

  Proof status
  ────────────
  All 27 theorems have complete machine-checked proofs.
  No `sorry` placeholders remain.
-/

import OhmTriality
import Mathlib.Analysis.Real.Pi.Bounds

open Complex Real

noncomputable section

-- ════════════════════════════════════════════════════════════════════════════
-- Section 1 — Silver-Ratio Coherence
-- The main result: C(δS) = √2/2, where δS = 1 + √2 is the silver ratio.
-- Proof: δS² = 2δS + 1 (silverRatio_sq) implies 1 + δS² = 2(1 + δS), so
--     C(δS) = 2δS / (2(1+δS)) = δS/(1+δS) = (1+√2)/(2+√2) = √2/2.
-- ════════════════════════════════════════════════════════════════════════════

/-- The silver ratio satisfies δS > 0.  (δS_pos is private in CriticalEigenvalue.) -/
private lemma silver_pos' : 0 < δS := by unfold δS; positivity

/-- `√2 · √2 = 2`.  (sqrt2_sq is private in CriticalEigenvalue.) -/
private lemma sqrt2_mul_self : Real.sqrt 2 * Real.sqrt 2 = 2 :=
  Real.mul_self_sqrt (by norm_num)

/-- **Main result**: C(δS) = √2/2.

    The Kernel coherence function, evaluated at the silver ratio δS = 1+√2,
    equals 1/√2 = √2/2 — the magnitude of the components of μ = (−1+i)/√2.

    Proof uses δS² = 2δS + 1 to simplify 1 + δS² = 2(1 + δS), then
    rationalises (1+√2)/(2+√2) = √2/2. -/
theorem silver_coherence : C δS = Real.sqrt 2 / 2 := by
  unfold C δS
  have hsq : Real.sqrt 2 * Real.sqrt 2 = 2 := sqrt2_mul_self
  have hpos : (0 : ℝ) < 1 + (1 + Real.sqrt 2) ^ 2 := by positivity
  field_simp [ne_of_gt hpos]
  nlinarith [Real.sqrt_nonneg 2, hsq]

-- ════════════════════════════════════════════════════════════════════════════
-- Section 2 — Algebraic Consequences
-- ════════════════════════════════════════════════════════════════════════════

/-- C(δS)² = 1/2.  The silver ratio coherence, squared, is exactly one-half. -/
theorem silver_coherence_sq : C δS ^ 2 = 1 / 2 := by
  rw [silver_coherence]
  have h : Real.sqrt 2 ^ 2 = 2 := by rw [sq]; exact sqrt2_mul_self
  nlinarith [h]

/-- **Isotropic/diagonal property**: at r = δS, the coherence equals the
    Pythagorean imbalance term: C(δS) = (δS² − 1) / (1 + δS²).

    For general r, the coherence 2r/(1+r²) and the imbalance (r²−1)/(1+r²)
    are different.  The silver ratio is the unique positive scale where they
    coincide.  This follows from δS² = 2δS + 1, which gives:
        (δS² − 1)/(1 + δS²) = 2δS/(2 + 2δS) = δS/(1+δS) = C(δS). -/
theorem silver_coherence_eq_imbalance :
    C δS = (δS ^ 2 - 1) / (1 + δS ^ 2) := by
  have hδsq : δS ^ 2 = 2 * δS + 1 := silverRatio_sq
  have hpos : (0 : ℝ) < 1 + δS ^ 2 := by positivity
  unfold C
  field_simp [ne_of_gt hpos, ne_of_gt silver_pos']
  nlinarith [silver_pos', hδsq]

/-- **Symmetric Pythagorean**: 2 · C(δS)² = 1.

    The standard Pythagorean coherence identity is C(r)² + ((r²−1)/(1+r²))² = 1.
    At r = δS both terms are equal (silver_coherence_eq_imbalance), so:
        C(δS)² + C(δS)² = 1  ⟹  2 · C(δS)² = 1.
    The silver ratio is the "45-degree point" of the coherence-imbalance circle. -/
theorem silver_pythagorean : 2 * C δS ^ 2 = 1 := by
  have h := coherence_pythagorean δS silver_pos'
  rw [← silver_coherence_eq_imbalance] at h
  linarith

-- ════════════════════════════════════════════════════════════════════════════
-- Section 3 — Connection to μ
-- Im(μ) = sin(3π/4) = √2/2 = C(δS).
-- The silver-ratio scale is where Kernel coherence meets the critical eigenvalue.
-- ════════════════════════════════════════════════════════════════════════════

/-- The imaginary part of μ is √2/2.

    Since μ = exp(I · 3π/4) and Im(exp(I·θ)) = sin θ, we have
    Im(μ) = sin(3π/4) = sin(π − π/4) = sin(π/4) = √2/2. -/
theorem mu_imaginary_part : Complex.im μ = Real.sqrt 2 / 2 := by
  have hsin : Real.sin (3 * Real.pi / 4) = Real.sqrt 2 / 2 := by
    rw [show (3 : ℝ) * Real.pi / 4 = Real.pi - Real.pi / 4 by ring,
        Real.sin_pi_sub, Real.sin_pi_div_four]
  unfold μ
  rw [show Complex.I * (3 * ↑Real.pi / 4) = ↑(3 * Real.pi / 4 : ℝ) * Complex.I by push_cast; ring,
      Complex.exp_mul_I]
  simp only [Complex.add_im, Complex.mul_im, Complex.ofReal_im, Complex.I_im,
             Complex.ofReal_re, Complex.I_re, Complex.cos_ofReal_im, Complex.sin_ofReal_re,
             mul_one, mul_zero, zero_add, add_zero]
  exact hsin

/-- **Central connection**: Im(μ) = C(δS).

    The coherence function, evaluated at the silver ratio δS = 1+√2,
    equals the imaginary component of the critical eigenvalue μ = (−1+i)/√2.
    Both equal √2/2 ≈ 0.7071.

    This is a machine-discovered equality between two independently-defined
    quantities: the palindrome-residual denominator δS and the eigenvalue μ. -/
theorem mu_im_eq_silver_coherence : Complex.im μ = C δS :=
  mu_imaginary_part.trans silver_coherence.symm

-- ════════════════════════════════════════════════════════════════════════════
-- Section 4 — Ohm–Coherence at the Silver Scale
-- By the Ohm–Coherence duality (CriticalEigenvalue §17, OhmTriality),
-- G_eff(δS) = C(δS) = √2/2 and R_eff(δS) = (C(δS))⁻¹ = √2.
-- ════════════════════════════════════════════════════════════════════════════

/-- Ohm conductance at the silver scale: G_eff(δS) = C(δS) = √2/2. -/
theorem silver_ohm_conductance : C δS = Real.sqrt 2 / 2 := silver_coherence

/-- Ohm resistance at the silver scale: R_eff(δS) = (C(δS))⁻¹ = √2.

    The inverse of √2/2 is 2/√2 = √2 (by rationalization 2/√2 = 2·√2/2 = √2). -/
theorem silver_ohm_resistance : (C δS)⁻¹ = Real.sqrt 2 := by
  rw [silver_coherence]
  have h2 : Real.sqrt 2 ≠ 0 := Real.sqrt_ne_zero'.mpr (by norm_num)
  field_simp [h2]

/-- Ohm's law at the silver scale: G_eff(δS) · R_eff(δS) = 1. -/
theorem silver_ohm_law : C δS * (C δS)⁻¹ = 1 := by
  rw [silver_coherence]
  have h2 : Real.sqrt 2 ≠ 0 := Real.sqrt_ne_zero'.mpr (by norm_num)
  field_simp [h2]

-- ════════════════════════════════════════════════════════════════════════════
-- Section 5 — Position in the Coherence Ordering
-- C(φ²) = 2/3 < C(δS) = √2/2 < C(1) = 1.
-- The silver scale sits strictly between the Koide wing and the kernel peak.
-- ════════════════════════════════════════════════════════════════════════════

/-- The Koide coherence is strictly less than the silver-ratio coherence:
    C(φ²) = 2/3 < √2/2 = C(δS).

    Numerically: 2/3 ≈ 0.667 < 0.707 ≈ √2/2.
    Algebraically: 2/3 < √2/2 ↔ 4/3 < √2 ↔ 16/9 < 2. ✓ -/
theorem koide_below_silver : C (φ ^ 2) < C δS := by
  rw [koide_coherence_bridge, silver_coherence]
  nlinarith [sqrt2_mul_self, Real.sqrt_nonneg 2]

/-- The silver-ratio coherence is strictly less than the kernel maximum:
    C(δS) = √2/2 < 1 = C(1).

    This follows from coherence_strictAnti since 1 ≤ 1 < δS. -/
theorem silver_below_kernel : C δS < C 1 :=
  coherence_strictAnti 1 δS le_rfl (by unfold δS; linarith [Real.sqrt_pos.mpr (by norm_num : (0:ℝ) < 2)])

/-- **Coherence ordering**: C(φ²) = 2/3 < C(δS) = √2/2 < C(1) = 1.

    The three natural constants in the framework satisfy a strict ordering:
    the Koide lepton scale, the silver palindrome scale, and the kernel peak. -/
theorem koide_silver_kernel_ordering :
    C (φ ^ 2) < C δS ∧ C δS < C 1 :=
  ⟨koide_below_silver, silver_below_kernel⟩

/-- The μ-orbit coherence C(|μⁿ|) = 1 strictly exceeds the silver coherence.
    The μ-orbit always runs at the perfectly-coherent kernel point, above δS. -/
theorem mu_orbit_exceeds_silver (n : ℕ) :
    C δS < C (Complex.abs (μ ^ n)) := by
  rw [mu_pow_abs]
  exact silver_below_kernel

-- ════════════════════════════════════════════════════════════════════════════
-- Section 6 — Scale Placement and Symmetry
-- δS lies in the meso turbulence domain [1, 100] and below the golden scale.
-- Its mirror 1/δS = √2 − 1 shares the same coherence by C(r) = C(1/r).
-- ════════════════════════════════════════════════════════════════════════════

/-- The silver ratio is strictly greater than 1: 1 < δS = 1 + √2. -/
theorem silver_gt_one : 1 < δS := by
  unfold δS; linarith [Real.sqrt_pos.mpr (by norm_num : (0:ℝ) < 2)]

/-- δS ≤ 100 (δS = 1+√2 ≈ 2.414). -/
theorem silver_le_hundred : δS ≤ 100 := by
  unfold δS
  have : Real.sqrt 2 ≤ 2 := by
    calc Real.sqrt 2 ≤ Real.sqrt 4 := Real.sqrt_le_sqrt (by norm_num)
      _ = 2 := by norm_num [Real.sqrt_eq_iff_eq_sq]
  linarith

/-- The silver ratio lies in the meso turbulence domain [1, 100]. -/
theorem silver_in_meso : δS ∈ mesoScaleDomain :=
  ⟨le_of_lt silver_gt_one, silver_le_hundred⟩

/-- The mirror of the silver scale, 1/δS = √2 − 1, has the same coherence.
    This follows from the general symmetry C(r) = C(1/r). -/
theorem silver_mirror_coherence : C (1 / δS) = C δS :=
  (coherence_symm δS silver_pos').symm

/-- The silver ratio δS ≈ 2.414 is strictly less than the golden scale φ² ≈ 2.618.

    Proof: δS < φ² ↔ 1+√2 < φ+1 ↔ √2 < φ ↔ √2 < (1+√5)/2.
    Holds since 2√2 < 1+√5: (2√2)² = 8 < 46 ≤ (1+√5)² (using √5 > 2). -/
theorem silver_lt_golden_sq : δS < φ ^ 2 := by
  have h2 : Real.sqrt 2 * Real.sqrt 2 = 2 := sqrt2_mul_self
  have h5 : Real.sqrt 5 * Real.sqrt 5 = 5 := Real.mul_self_sqrt (by norm_num)
  have h2pos : 0 ≤ Real.sqrt 2 := Real.sqrt_nonneg 2
  have h5pos : 0 ≤ Real.sqrt 5 := Real.sqrt_nonneg 5
  -- bound: √2 < 3/2 and √5 > 11/5 gives δS < 5/2 < φ^2
  have hlt2 : Real.sqrt 2 < 3 / 2 := by
    calc Real.sqrt 2 < Real.sqrt (9 / 4) :=
          Real.sqrt_lt_sqrt (by norm_num) (by norm_num)
      _ = 3 / 2 := by
          rw [show (9 / 4 : ℝ) = (3 / 2) ^ 2 by norm_num]
          exact Real.sqrt_sq (by norm_num)
  have hgt5 : 11 / 5 < Real.sqrt 5 := by
    rw [show (11 / 5 : ℝ) = Real.sqrt (121 / 25) from by
        rw [show (121 / 25 : ℝ) = (11 / 5) ^ 2 by norm_num]
        exact (Real.sqrt_sq (by norm_num)).symm]
    exact Real.sqrt_lt_sqrt (by norm_num) (by norm_num)
  rw [goldenRatio_sq]
  unfold δS φ
  linarith

/-- The real part of μ has magnitude |Re(μ)| = √2/2.

    μ = (−1+i)/√2 has Re(μ) = −1/√2 = −√2/2, so |Re(μ)| = √2/2.
    This equals C(δS) (mu_re_abs_eq_silver_coherence below). -/
theorem mu_real_part : Complex.re μ = -(Real.sqrt 2 / 2) := by
  have hcos : Real.cos (3 * Real.pi / 4) = -(Real.sqrt 2 / 2) := by
    rw [show (3 : ℝ) * Real.pi / 4 = Real.pi - Real.pi / 4 by ring,
        Real.cos_pi_sub, Real.cos_pi_div_four]
  unfold μ
  rw [show Complex.I * (3 * ↑Real.pi / 4) = ↑(3 * Real.pi / 4 : ℝ) * Complex.I by push_cast; ring,
      Complex.exp_mul_I]
  simp only [Complex.add_re, Complex.mul_re, Complex.ofReal_re, Complex.I_re,
             Complex.ofReal_im, Complex.I_im, Complex.cos_ofReal_re, Complex.sin_ofReal_im,
             mul_zero, zero_mul, sub_zero, add_zero]
  exact hcos

/-- **Symmetry**: both component magnitudes of μ equal C(δS) = √2/2.

    The critical eigenvalue μ = (−1+i)/√2 has |Re(μ)| = |Im(μ)| = √2/2 = C(δS).
    The silver ratio δS is the unique scale capturing both component magnitudes. -/
theorem mu_re_abs_eq_silver_coherence : |Complex.re μ| = C δS := by
  rw [mu_real_part, abs_neg, _root_.abs_of_nonneg (by positivity),
      silver_coherence]

-- ════════════════════════════════════════════════════════════════════════════
-- Section 7 — Uniqueness
-- The only r > 0 with C(r) = √2/2 are r = δS and r = 1/δS.
-- Proof: C(r) = √2/2 rearranges to √2·r² − 4r + √2 = 0, which factors as
--   √2·(r − δS)·(r − (√2−1)) = 0.  Since √2 ≠ 0, exactly r = δS (= 1+√2) or
--   r = √2−1 = 1/δS.
-- ════════════════════════════════════════════════════════════════════════════

/-- **Algebraic rearrangement**: C(r) = √2/2 iff √2·r² − 4r + √2 = 0  (for r > 0).

    Proof: 2r/(1+r²) = √2/2 ↔ 4r = √2(1+r²) ↔ √2·r² − 4r + √2 = 0. -/
theorem silver_coherence_iff_quadratic (r : ℝ) (hr : 0 < r) :
    C r = Real.sqrt 2 / 2 ↔ Real.sqrt 2 * r ^ 2 - 4 * r + Real.sqrt 2 = 0 := by
  unfold C
  have hpos : (0 : ℝ) < 1 + r ^ 2 := by positivity
  have hsq : Real.sqrt 2 * Real.sqrt 2 = 2 := sqrt2_mul_self
  constructor
  · intro h
    have heq : 2 * r = Real.sqrt 2 / 2 * (1 + r ^ 2) :=
      (div_eq_iff (ne_of_gt hpos)).mp h
    nlinarith [Real.sqrt_nonneg 2]
  · intro h
    rw [div_eq_iff (ne_of_gt hpos)]
    nlinarith [Real.sqrt_nonneg 2]

/-- **Uniqueness theorem**: for r > 0, C(r) = √2/2  if and only if  r = δS  or  r = 1/δS.

    Both solutions come from the quadratic √2·r²−4r+√2 = 0, which factors as
    √2·(r − δS)·(r − (√2−1)) = 0.  Since √2 ≠ 0, the roots are exactly
    r = 1+√2 = δS  (meso, r > 1)  and  r = √2−1 = 1/δS  (micro, r < 1). -/
theorem silver_coherence_unique (r : ℝ) (hr : 0 < r) :
    C r = Real.sqrt 2 / 2 ↔ r = δS ∨ r = 1 / δS := by
  rw [silver_coherence_iff_quadratic r hr]
  have hsq : Real.sqrt 2 * Real.sqrt 2 = 2 := sqrt2_mul_self
  have h2pos : (0 : ℝ) < Real.sqrt 2 := by positivity
  have hδinv : 1 / δS = Real.sqrt 2 - 1 := silverRatio_inv
  constructor
  · intro h
    -- Rearrange: √2·(r − δS)·(r − (√2−1)) = 0
    have hfact : Real.sqrt 2 * (r - δS) * (r - (Real.sqrt 2 - 1)) = 0 := by
      unfold δS; nlinarith [Real.sqrt_nonneg 2]
    rcases mul_eq_zero.mp hfact with h12 | hr2
    · rcases mul_eq_zero.mp h12 with h1 | hr1
      · exact absurd h1 (ne_of_gt h2pos)
      · left; linarith
    · right; rw [hδinv]; linarith
  · intro h
    cases h with
    | inl h =>
      subst h; unfold δS
      nlinarith [Real.sqrt_nonneg 2]
    | inr h =>
      rw [hδinv] at h
      nlinarith [Real.sqrt_nonneg 2]

-- ════════════════════════════════════════════════════════════════════════════
-- Section 8 — Physics at 45°
-- The silver scale δS arises naturally in scattering theory and EM corrections:
--
--   C(δS) = sin(π/4) = √2/2  — amplitude at 45° elastic scattering phase
--   sin²(π/4) = C(δS)² = 1/2 — elastic unitarity: Im(f) = |f|² at 45° phase
--   α_FS/(2π) < C(δS)²       — Schwinger loop contribution sub-threshold
--   coherenceEM(δS) > C(φ²)  — EM-corrected silver coherence stays above Koide
--   π/4 + 3π/4 = π            — silver phase + eigenvalue phase = supplementary
-- ════════════════════════════════════════════════════════════════════════════

/-- **Scattering phase**: C(δS) = sin(π/4) = √2/2.

    In partial-wave scattering, C(δS) equals the amplitude |f_l| = |sin(δ_l)|
    at the 45° elastic phase shift δ_l = π/4.  The silver scale is the coherence
    function evaluated at the standard "45-degree" scattering point. -/
theorem silver_eq_sin_45 : C δS = Real.sin (Real.pi / 4) := by
  rw [Real.sin_pi_div_four, silver_coherence]

/-- **Elastic unitarity at 45°**: sin²(π/4) = C(δS)² = 1/2.

    At the 45° phase shift (δ = π/4), the imaginary part of the partial-wave
    amplitude equals |f|²: Im(f_l) = sin²(δ_l) = 1/2 = C(δS)².
    This is the unitarity condition for maximally coherent elastic scattering. -/
theorem silver_unitarity_elastic_sq : Real.sin (Real.pi / 4) ^ 2 = C δS ^ 2 := by
  rw [← silver_eq_sin_45]

/-- **Schwinger term below 45°-threshold**: α_FS/(2π) < C(δS)².

    The Schwinger one-loop QED contribution to the anomalous magnetic moment is
    α_FS/(2π) ≈ 0.00116 ≪ 1/2 = C(δS)².  Fine-structure corrections lie far below
    the 45°-phase coherence threshold.

    Proof: α_FS/(2π) < 1/2 ↔ α_FS < π ↔ 1/137 < π.  Since π > 3 > 1/137. -/
theorem silver_schwinger_bound : α_FS / (2 * Real.pi) < C δS ^ 2 := by
  rw [silver_coherence_sq]
  unfold α_FS
  have hπ : (3 : ℝ) < Real.pi := Real.pi_gt_three
  rw [div_lt_iff₀ (by positivity : (0 : ℝ) < 2 * Real.pi)]
  linarith

/-- **EM-corrected silver coherence exceeds Koide**: coherenceEM(δS) > C(φ²).

    Even after electromagnetic coupling reduces coherence by factor (1−α_FS),
    the silver-scale coherence (1−α_FS)·C(δS) = (136/137)·(√2/2) ≈ 0.702
    remains strictly above the Koide value 2/3 ≈ 0.667.

    Proof: (136/137)·√2/2 > 2/3 ↔ 408·√2 > 548, verified by 408²·2 > 548². -/
theorem silver_em_stays_above_koide : coherenceEM δS > C (φ ^ 2) := by
  rw [show coherenceEM δS = (1 - α_FS) * C δS from rfl,
      koide_coherence_bridge, silver_coherence]
  unfold α_FS
  have hsq : Real.sqrt 2 * Real.sqrt 2 = 2 := sqrt2_mul_self
  have h2pos : (0 : ℝ) < Real.sqrt 2 := by positivity
  -- Key: 408·√2 > 548, proved via 408²·2 = 332928 > 300304 = 548²
  have hkey : (548 : ℝ) < 408 * Real.sqrt 2 := by
    have : (408 * Real.sqrt 2 + 548) * (408 * Real.sqrt 2 - 548) = 32624 := by
      nlinarith [hsq]
    nlinarith
  linarith

/-- **Supplementary phases**: the silver 45° phase and the eigenvalue 135° phase sum to π.

    The critical eigenvalue μ = exp(I·3π/4) carries phase 3π/4 (135°).
    The silver scale corresponds to the 45° scattering phase (π/4).
    Together: π/4 + 3π/4 = π — they are supplementary, pairing as
    particle/antiparticle amplitudes in 2-body scattering kinematics. -/
theorem silver_phase_complement : Real.pi / 4 + 3 * Real.pi / 4 = Real.pi := by ring

end -- noncomputable section
