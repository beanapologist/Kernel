/-
  ParticleMass.lean — Lean 4 formalization of the Koide lepton mass formula
  and the proton/electron mass ratio, connected to the Kernel μ-eigenvalue /
  coherence framework through the golden ratio.

  The Koide formula (Koide 1982) is the dimensionless empirical relation:
      Q = (me + mμ + mτ) / (√me + √mμ + √mτ)² = 2/3

  The central result is the μ-cycle identification
      C(φ²) = 2/3    where φ = (1+√5)/2 (golden ratio)
  and C is the Kernel coherence function C(r) = 2r/(1+r²).

  This connects the Koide value 2/3 to the same coherence structure that
  governs the μ-eigenvalue orbit in CriticalEigenvalue.lean, via the identity
      1 + φ⁴ = 3φ²   (proved from φ² = φ + 1).

  The key algebraic chain:
      C(φ²) = 2φ² / (1 + φ⁴)   [definition of C]
            = 2φ² / (3φ²)       [1 + φ⁴ = 3φ², from φ² = φ+1]
            = 2/3               ✓

  Thus the Koide value is not an ad-hoc constant: it is the Kernel coherence
  function evaluated at the golden ratio scale.  The μ-orbit always achieves
  C(|μ^n|) = C(1) = 1 (maximum coherence); the Koide scale φ² sits strictly
  below the μ-orbit peak in the meso turbulence regime [1, 100].

  Proton/electron mass ratio
  ──────────────────────────
  The proton/electron mass ratio R ≈ 1836.15 is formalised with the integer
  approximation R = 1836.  Key bounds relate R to the fine structure constant:
      1/α_FS = 137  <  R = 1836   (mass ratio dominates EM coupling scale)
  The reduced mass correction 1/(R+1) ≈ 1/1837 < α_FS is smaller than the
  fine-structure coupling, confirming that proton-recoil is a sub-leading
  effect in hydrogen spectroscopy.

  Sections
  ────────
  1.  Koide quotient definition and algebraic bounds  (1/3 ≤ Q ≤ 1)
  2.  Extremal masses  (equal masses Q = 1/3; attainment proof)
  3.  Golden ratio  (φ = (1+√5)/2; φ > 1; φ² = φ+1; φ⁴ = 3φ+2)
  4.  Koide-coherence bridge  (C(φ²) = 2/3 — the μ-cycle trick)
  5.  μ-orbit Koide connection  (golden scale meso; Koide below μ-orbit peak)
  6.  Proton/electron mass ratio  (R = 1836; R > 1/α_FS; reduced mass)

  Proof status
  ────────────
  All 38 theorems have complete machine-checked proofs.
  No `sorry` placeholders remain.
-/

import CriticalEigenvalue
import Turbulence
import FineStructure

open Complex Real

noncomputable section

-- ════════════════════════════════════════════════════════════════════════════
-- Section 1 — Koide Quotient
-- Q(m₁,m₂,m₃) = (m₁+m₂+m₃) / (√m₁+√m₂+√m₃)²
-- Algebraic bounds: 1/3 ≤ Q ≤ 1 for positive masses.
-- Lower bound (Cauchy-Schwarz):  (√m₁+√m₂+√m₃)² ≤ 3(m₁+m₂+m₃)
-- Upper bound (non-neg cross terms): (√m₁+√m₂+√m₃)² ≥ m₁+m₂+m₃
-- Ref: Koide, Y. (1982). A fermion-boson composite model of quarks and leptons.
--      Phys. Lett. B 120, 161–165.
-- ════════════════════════════════════════════════════════════════════════════

/-- The Koide quotient: Q(m₁,m₂,m₃) = (m₁+m₂+m₃) / (√m₁+√m₂+√m₃)²

    Measures how uniformly a mass triple is distributed.  When masses are
    equal Q = 1/3 (lower bound); when masses are concentrated Q → 1. -/
noncomputable def koideQuotient (m₁ m₂ m₃ : ℝ) : ℝ :=
  (m₁ + m₂ + m₃) / (Real.sqrt m₁ + Real.sqrt m₂ + Real.sqrt m₃) ^ 2

/-- The Koide denominator is positive when at least one mass is positive. -/
theorem koideQuotient_denom_pos (m₁ m₂ m₃ : ℝ) (h₁ : 0 < m₁) :
    0 < (Real.sqrt m₁ + Real.sqrt m₂ + Real.sqrt m₃) ^ 2 := by
  have : 0 < Real.sqrt m₁ := Real.sqrt_pos.mpr h₁
  positivity

/-- The Koide quotient is non-negative for non-negative masses. -/
theorem koideQuotient_nonneg (m₁ m₂ m₃ : ℝ) (h₁ : 0 ≤ m₁) (h₂ : 0 ≤ m₂) (h₃ : 0 ≤ m₃) :
    0 ≤ koideQuotient m₁ m₂ m₃ := by
  unfold koideQuotient
  apply div_nonneg <;> [linarith; positivity]

/-- Cauchy-Schwarz lower bound: Q ≥ 1/3 for positive masses.

    Proof: (√m₁+√m₂+√m₃)² ≤ 3(m₁+m₂+m₃) by the inequality
    (a+b+c)² ≤ 3(a²+b²+c²) with aᵢ = √mᵢ, which is equivalent to
    (a-b)²+(b-c)²+(a-c)² ≥ 0. -/
theorem koideQuotient_lower_bound (m₁ m₂ m₃ : ℝ)
    (h₁ : 0 < m₁) (h₂ : 0 < m₂) (h₃ : 0 < m₃) :
    1 / 3 ≤ koideQuotient m₁ m₂ m₃ := by
  unfold koideQuotient
  have hD : 0 < (Real.sqrt m₁ + Real.sqrt m₂ + Real.sqrt m₃) ^ 2 :=
    koideQuotient_denom_pos m₁ m₂ m₃ h₁
  rw [le_div_iff hD]
  have s1 := Real.sq_sqrt h₁.le
  have s2 := Real.sq_sqrt h₂.le
  have s3 := Real.sq_sqrt h₃.le
  nlinarith [Real.sqrt_nonneg m₁, Real.sqrt_nonneg m₂, Real.sqrt_nonneg m₃,
             sq_nonneg (Real.sqrt m₁ - Real.sqrt m₂),
             sq_nonneg (Real.sqrt m₂ - Real.sqrt m₃),
             sq_nonneg (Real.sqrt m₁ - Real.sqrt m₃)]

/-- AM-QM upper bound: Q ≤ 1 for non-negative masses.

    Proof: (√m₁+√m₂+√m₃)² = m₁+m₂+m₃ + 2·(√m₁√m₂ + √m₂√m₃ + √m₁√m₃)
    ≥ m₁+m₂+m₃ since all cross terms are non-negative. -/
theorem koideQuotient_upper_bound (m₁ m₂ m₃ : ℝ)
    (h₁ : 0 ≤ m₁) (h₂ : 0 ≤ m₂) (h₃ : 0 ≤ m₃)
    (hD : 0 < (Real.sqrt m₁ + Real.sqrt m₂ + Real.sqrt m₃) ^ 2) :
    koideQuotient m₁ m₂ m₃ ≤ 1 := by
  unfold koideQuotient
  rw [div_le_one hD]
  have s1 := Real.sq_sqrt h₁
  have s2 := Real.sq_sqrt h₂
  have s3 := Real.sq_sqrt h₃
  nlinarith [Real.sqrt_nonneg m₁, Real.sqrt_nonneg m₂, Real.sqrt_nonneg m₃,
             mul_nonneg (Real.sqrt_nonneg m₁) (Real.sqrt_nonneg m₂),
             mul_nonneg (Real.sqrt_nonneg m₂) (Real.sqrt_nonneg m₃),
             mul_nonneg (Real.sqrt_nonneg m₁) (Real.sqrt_nonneg m₃)]

-- ════════════════════════════════════════════════════════════════════════════
-- Section 2 — Extremal Masses
-- Equal masses give the Koide lower bound Q = 1/3.
-- The bound is attained at any positive equal-mass triple.
-- ════════════════════════════════════════════════════════════════════════════

/-- Equal masses give the Koide lower bound:  Q(m,m,m) = 1/3.

    When m₁ = m₂ = m₃ = m > 0:
        Q = 3m / (3√m)² = 3m / (9m) = 1/3. -/
theorem koideQuotient_equal_masses (m : ℝ) (hm : 0 < m) :
    koideQuotient m m m = 1 / 3 := by
  unfold koideQuotient
  have hs   : Real.sqrt m ^ 2 = m     := Real.sq_sqrt hm.le
  have hnn  : 0 < Real.sqrt m         := Real.sqrt_pos.mpr hm
  have hsum : Real.sqrt m + Real.sqrt m + Real.sqrt m = 3 * Real.sqrt m := by ring
  have hden : (3 * Real.sqrt m) ^ 2   = 9 * m                          := by nlinarith
  rw [show m + m + m = 3 * m by ring, hsum, hden]
  have h9m : (9 : ℝ) * m ≠ 0 := (mul_pos (by norm_num) hm).ne'
  rw [div_eq_iff h9m]
  ring

/-- The Koide lower bound Q = 1/3 is attained by the unit equal-mass triple. -/
theorem koide_lower_attained :
    ∃ m₁ m₂ m₃ : ℝ, 0 < m₁ ∧ 0 < m₂ ∧ 0 < m₃ ∧ koideQuotient m₁ m₂ m₃ = 1 / 3 :=
  ⟨1, 1, 1, one_pos, one_pos, one_pos, koideQuotient_equal_masses 1 one_pos⟩

-- ════════════════════════════════════════════════════════════════════════════
-- Section 3 — Golden Ratio
-- φ = (1+√5)/2 is the positive root of x² = x + 1.
-- Key identities used in §4:
--   φ² = φ + 1     (defining equation)
--   φ⁴ = 3φ + 2   (derived by squaring φ² = φ+1 twice)
-- From these: 1 + φ⁴ = 3φ²  (the bridge to the Koide value)
-- Ref: Livio, M. (2002). The Golden Ratio. Broadway Books.
-- ════════════════════════════════════════════════════════════════════════════

/-- The golden ratio: φ = (1 + √5) / 2.

    The Unicode name `φ` is used throughout, consistent with the Kernel
    codebase convention of Unicode names for mathematical constants (cf. `μ`
    in CriticalEigenvalue.lean, `α_FS` in FineStructure.lean).
    ASCII alias: `goldenRatio := φ`. -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- ASCII alias for `φ`. -/
noncomputable def goldenRatio : ℝ := φ

/-- φ > 0. -/
theorem goldenRatio_pos : 0 < φ := by
  unfold φ; linarith [Real.sqrt_nonneg 5]

/-- φ > 1: the golden ratio exceeds 1. -/
theorem goldenRatio_gt_one : 1 < φ := by
  unfold φ
  have h5 : (Real.sqrt 5) ^ 2 = 5 := Real.sq_sqrt (by norm_num)
  nlinarith [Real.sqrt_nonneg 5]

/-- The defining quadratic: φ² = φ + 1. -/
theorem goldenRatio_sq : φ ^ 2 = φ + 1 := by
  unfold φ
  have h5  : (Real.sqrt 5) ^ 2 = 5 := Real.sq_sqrt (by norm_num)
  have hnn : 0 ≤ Real.sqrt 5       := Real.sqrt_nonneg 5
  -- ((1+√5)/2)² = (1+√5)/2 + 1.  Clear denominators via field_simp then close
  -- with nlinarith using (√5)²=5: (1+√5)²=6+2√5 = 2(3+√5) = 2((1+√5)/2+1)·2. ✓
  field_simp
  nlinarith

/-- φ² > 0. -/
theorem goldenRatio_sq_pos : 0 < φ ^ 2 := pow_pos goldenRatio_pos 2

/-- φ⁴ = 3φ + 2 (derived by squaring φ² = φ + 1 and substituting again). -/
theorem goldenRatio_fourth : φ ^ 4 = 3 * φ + 2 := by
  have hφ2 : φ ^ 2 = φ + 1 := goldenRatio_sq
  have : φ ^ 4 = (φ ^ 2) ^ 2 := by ring
  rw [this, hφ2]
  nlinarith [goldenRatio_sq]

-- ════════════════════════════════════════════════════════════════════════════
-- Section 4 — Koide-Coherence Bridge
-- The μ-cycle trick: evaluating the Kernel coherence function C(r) = 2r/(1+r²)
-- at r = φ² gives exactly the Koide value 2/3.
--
-- The proof uses the golden-ratio identity 1 + φ⁴ = 3φ²:
--   C(φ²) = 2φ² / (1+(φ²)²) = 2φ² / (1+φ⁴) = 2φ² / (3φ²) = 2/3  ✓
-- ════════════════════════════════════════════════════════════════════════════

/-- Key golden-ratio identity: 1 + φ⁴ = 3φ².

    Proof: 1 + φ⁴ = 1 + (3φ+2) = 3φ+3 = 3(φ+1) = 3φ². -/
theorem one_add_goldenRatio_fourth : 1 + φ ^ 4 = 3 * φ ^ 2 := by
  nlinarith [goldenRatio_sq, goldenRatio_fourth]

/-- *** The μ-cycle trick: C(φ²) = 2/3. ***

    The Kernel coherence function evaluated at the golden ratio squared equals
    the Koide value exactly:
        C(φ²) = 2φ² / (1 + φ⁴) = 2φ² / (3φ²) = 2/3.

    This identifies the Koide value 2/3 as an instance of the same coherence
    structure that governs the μ = exp(I·3π/4) eigenvalue orbit. -/
theorem koide_coherence_bridge : C (φ ^ 2) = 2 / 3 := by
  unfold C
  have hφ2pos : 0 < φ ^ 2         := goldenRatio_sq_pos
  have h14    : 1 + (φ ^ 2) ^ 2 = 3 * φ ^ 2 := by
    rw [show (φ ^ 2) ^ 2 = φ ^ 4 by ring]; exact one_add_goldenRatio_fourth
  rw [h14, div_eq_iff (mul_pos (by norm_num) hφ2pos).ne']
  ring

/-- The Koide value is exactly 2/3 of the μ-orbit peak coherence C(1) = 1. -/
theorem koide_coherence_two_thirds_of_max : C (φ ^ 2) = 2 / 3 * C 1 := by
  rw [koide_coherence_bridge, (coherence_eq_one_iff 1 zero_le_one).mpr rfl, mul_one]

/-- Coherence-symmetry Koide: C(1/φ²) = 2/3.

    Since C(r) = C(1/r) (coherence_symm), the Koide value is also achieved
    at the reciprocal golden-ratio scale 1/φ² ≈ 0.382. -/
theorem koide_coherence_reciprocal : C (1 / φ ^ 2) = 2 / 3 := by
  rw [← koide_coherence_bridge]
  exact (coherence_symm (φ ^ 2) goldenRatio_sq_pos).symm

-- ════════════════════════════════════════════════════════════════════════════
-- Section 5 — μ-Orbit Koide Connection
-- The golden scale φ² lies in the meso turbulence domain [1, 100].
-- The μ-orbit achieves peak coherence C(1) = 1; the Koide scale φ² ≠ 1 sits
-- strictly below the μ-orbit peak.
-- ════════════════════════════════════════════════════════════════════════════

/-- φ² ≠ 1: the golden ratio squared is not the kernel scale. -/
theorem goldenRatio_sq_ne_one : φ ^ 2 ≠ 1 := by
  intro h
  linarith [goldenRatio_sq, goldenRatio_gt_one]

/-- φ² ∈ mesoScaleDomain: the golden scale lies in [1, 100].

    φ² = φ + 1 ≈ 2.618 satisfies 1 < φ² (since φ > 1) and φ² < 3 ≤ 100
    (since φ < 2, proved from √5 < 3). -/
theorem goldenRatio_sq_meso : φ ^ 2 ∈ mesoScaleDomain := by
  refine ⟨?_, ?_⟩
  · -- 1 ≤ φ²: φ² = φ+1 > 1+1 = 2 > 1
    linarith [goldenRatio_sq, goldenRatio_gt_one]
  · -- φ² ≤ 100: φ < 2 gives φ² = φ+1 < 3 ≤ 100
    have hφ_lt2 : φ < 2 := by
      unfold φ
      have h5  : (Real.sqrt 5) ^ 2 = 5 := Real.sq_sqrt (by norm_num)
      have hnn : 0 ≤ Real.sqrt 5        := Real.sqrt_nonneg 5
      -- (3-√5)²≥0 gives 14≥6√5, so √5≤7/3<3, so (1+√5)/2<2
      have hle : Real.sqrt 5 ≤ 7 / 3 := by nlinarith [sq_nonneg (3 - Real.sqrt 5)]
      linarith
    linarith [goldenRatio_sq]

/-- C(φ²) > 0: the Koide coherence is strictly positive. -/
theorem koide_coherence_pos : 0 < C (φ ^ 2) := by
  rw [koide_coherence_bridge]; norm_num

/-- C(φ²) < C(1): the Koide scale has strictly less coherence than the μ-orbit peak.

    φ² ≠ 1, so by coherence_lt_one (every r ≠ 1 gives C(r) < 1 = C(1))
    the Koide coherence 2/3 lies strictly below the maximum. -/
theorem koide_below_mu_orbit_peak : C (φ ^ 2) < C 1 := by
  rw [(coherence_eq_one_iff 1 zero_le_one).mpr rfl]
  exact coherence_lt_one (φ ^ 2) (le_of_lt goldenRatio_sq_pos) goldenRatio_sq_ne_one

/-- The Koide coherence 2/3 is strictly between 0 and 1. -/
theorem koide_coherence_strictly_between : 0 < C (φ ^ 2) ∧ C (φ ^ 2) < 1 :=
  ⟨koide_coherence_pos,
   coherence_lt_one (φ ^ 2) (le_of_lt goldenRatio_sq_pos) goldenRatio_sq_ne_one⟩

/-- The μ-orbit coherence exceeds the Koide coherence at every step:
    C(|μ^n|) = 1 > 2/3 = C(φ²) for all n.

    The μ-orbit always achieves maximum coherence; the Koide scale lies
    strictly below the μ-orbit peak throughout the 8-cycle. -/
theorem mu_orbit_exceeds_koide (n : ℕ) : C (φ ^ 2) < C (Complex.abs (μ ^ n)) := by
  rw [mu_pow_abs, koide_coherence_bridge, (coherence_eq_one_iff 1 zero_le_one).mpr rfl]
  norm_num

-- ════════════════════════════════════════════════════════════════════════════
-- Section 6 — Proton/Electron Mass Ratio
-- R = 1836 is the integer approximation to the CODATA 2018 value 1836.15267343.
-- Key relations to the fine structure constant α_FS = 1/137:
--   1/α_FS = 137  <  R = 1836        (mass scale dominates EM coupling scale)
--   1/(R+1) ≈ 1/1837  <  α_FS       (recoil correction sub-leading)
-- The reduced mass energy E_n^red = E_n · R/(R+1) corrects Rydberg levels for
-- the finite proton mass.
-- Ref: Mohr, P. J. et al. (2016). CODATA recommended values. Rev. Mod. Phys. 88, 035009.
-- ════════════════════════════════════════════════════════════════════════════

/-- The proton/electron mass ratio: R = 1836 (integer approximation to CODATA 2018
    value 1836.15267343).

    Defined as `ℝ` rather than `ℕ` because it is used in arithmetic expressions
    involving real division: R/(R+1) for the reduced mass factor, and comparisons
    with α_FS = 1/137 ∈ ℝ. -/
noncomputable def protonElectronRatio : ℝ := 1836

/-- The proton is heavier than the electron: R > 1. -/
theorem protonElectronRatio_gt_one : 1 < protonElectronRatio := by
  unfold protonElectronRatio; norm_num

/-- The proton/electron ratio exceeds the fine structure constant denominator:
    1/α_FS = 137  <  R = 1836.

    Equivalently, 1/R < α_FS: the relative proton mass correction is smaller
    than the electromagnetic coupling. -/
theorem protonElectronRatio_gt_α_FS_inv : 1 / α_FS < protonElectronRatio := by
  unfold α_FS protonElectronRatio; norm_num

/-- The proton/electron ratio exceeds the μ-orbit period: R > 8. -/
theorem protonElectronRatio_gt_8cycle : 8 < protonElectronRatio := by
  unfold protonElectronRatio; norm_num

/-- Reduced mass factor: f(R) = R/(R+1) ∈ (0,1).

    In hydrogen, the reduced mass μ_red = me·mp/(me+mp) = me · R/(R+1)
    reduces Rydberg energies by the factor R/(R+1) < 1. -/
theorem reducedMassFactor_mem_unit :
    0 < protonElectronRatio / (protonElectronRatio + 1) ∧
    protonElectronRatio / (protonElectronRatio + 1) < 1 := by
  unfold protonElectronRatio
  constructor <;> norm_num

/-- Reduced mass Rydberg energy: E_n^red = E_n · R/(R+1).

    Corrects the Bohr energy for the finite proton mass.  For large R = 1836
    this is a small upward shift: E_n^red > E_n (less negative, less bound). -/
noncomputable def reducedMassEnergy (n : ℕ) (hn : n ≠ 0) : ℝ :=
  rydbergEnergy n hn * (protonElectronRatio / (protonElectronRatio + 1))

/-- Reduced mass Rydberg energy is negative: still a bound state. -/
theorem reducedMassEnergy_neg (n : ℕ) (hn : n ≠ 0) : reducedMassEnergy n hn < 0 := by
  unfold reducedMassEnergy
  exact mul_neg_of_neg_of_pos
    (rydbergEnergy_neg n hn) reducedMassFactor_mem_unit.1

/-- Reduced mass correction lifts Rydberg levels toward zero:
    E_n < E_n^red < 0.

    Since 0 < R/(R+1) < 1 and E_n < 0: multiplying by R/(R+1) makes E_n
    less negative (closer to zero), representing the reduced binding from
    the proton's finite recoil. -/
theorem reducedMassEnergy_gt_rydberg (n : ℕ) (hn : n ≠ 0) :
    rydbergEnergy n hn < reducedMassEnergy n hn := by
  unfold reducedMassEnergy
  have hE           := rydbergEnergy_neg n hn
  have ⟨hfpos, hflt⟩ := reducedMassFactor_mem_unit
  nlinarith

/-- The reduced mass correction 1/(R+1) < α_FS:
    the proton-recoil correction is smaller than the EM fine-structure coupling.

    1/(1836+1) = 1/1837 < 1/137 = α_FS,  so spectroscopic fine structure
    (α_FS-scale) dominates over mass-recoil (1/R-scale) corrections. -/
theorem reducedMassCorrection_lt_α_FS :
    1 / (protonElectronRatio + 1) < α_FS := by
  unfold α_FS protonElectronRatio; norm_num

-- ════════════════════════════════════════════════════════════════════════════
-- Section 7 — Coherence Triality
--
-- The three distinguished coherence scales form a closed geometric structure:
--
--     1/φ²  <  1  <  φ²     (strict ordering of scales)
--
--   with coherence values:
--
--     C(1/φ²) = 2/3 = C(φ²)  <  C(1) = 1
--
-- Geometric mean:  √(1/φ² · φ²) = √1 = 1  (kernel is the geometric mean)
--
-- Physical interpretation
-- ───────────────────────
--   r = 1    Kernel / μ-orbit     — maximum coherence C=1, Floquet 8-period
--   r = φ²   Lepton sector        — Koide ratio 2/3, meso domain [1,100]
--   r = 1/φ² Hadronic/quark sector — same coherence 2/3, micro domain (0,1)
--
-- The two wings (lepton and hadronic) are coherence mirrors of each other
-- through the kernel scale: C(φ²) = C(1/φ²) = 2/3 < C(1) = 1.
--
-- The triality is a consequence of:
--   1.  C(r) = C(1/r)  (coherence_symm)
--   2.  C(1) = 1       (coherence peak)
--   3.  C(φ²) = 2/3   (Koide-coherence bridge, §4)
-- ════════════════════════════════════════════════════════════════════════════

/-- 1/φ² > 0: the reciprocal golden scale is positive. -/
theorem goldenRatio_sq_recip_pos : 0 < 1 / φ ^ 2 :=
  div_pos one_pos goldenRatio_sq_pos

/-- 1/φ² < 1: the reciprocal golden scale lies below the kernel scale.

    φ² > 1 (from φ > 1), so 1/φ² < 1/1 = 1. -/
theorem goldenRatio_sq_recip_lt_one : 1 / φ ^ 2 < 1 := by
  rw [div_lt_one goldenRatio_sq_pos]
  linarith [goldenRatio_sq, goldenRatio_gt_one]

/-- 1/φ² ∈ microScaleDomain: the hadronic triality scale lies in (0, 1).

    The lepton scale φ² sits in the meso turbulence regime [1, 100];
    its coherence mirror 1/φ² sits in the micro regime (0, 1). -/
theorem goldenRatio_sq_recip_micro : 1 / φ ^ 2 ∈ microScaleDomain :=
  ⟨goldenRatio_sq_recip_pos, goldenRatio_sq_recip_lt_one⟩

/-- Triality scale ordering: 1/φ² < 1 < φ².

    The three triality scales are strictly ordered, with the kernel scale
    r=1 lying exactly between the hadronic mirror (1/φ²) and the lepton
    scale (φ²). -/
theorem triality_scale_ordering : 1 / φ ^ 2 < 1 ∧ (1 : ℝ) < φ ^ 2 :=
  ⟨goldenRatio_sq_recip_lt_one,
   by linarith [goldenRatio_sq, goldenRatio_gt_one]⟩

/-- Geometric mean: (1/φ²) · φ² = 1.

    The kernel scale r=1 is the geometric mean of the two triality wings:
    √(1/φ² · φ²) = √1 = 1.  Equivalently, the triality wings are
    multiplicative inverses: (1/φ²) · φ² = 1. -/
theorem triality_geometric_mean : 1 / φ ^ 2 * φ ^ 2 = 1 :=
  div_mul_cancel₀ 1 goldenRatio_sq_pos.ne'

/-- Coherence mirror: C(1/φ²) = C(φ²).

    The two triality wings have identical coherence: the hadronic scale
    1/φ² is the coherence mirror of the lepton scale φ² through r=1.
    This is an instance of the coherence symmetry C(r) = C(1/r). -/
theorem triality_wings_equal_coherence : C (1 / φ ^ 2) = C (φ ^ 2) :=
  (coherence_symm (φ ^ 2) goldenRatio_sq_pos).symm

/-- The hadronic triality scale is below the kernel peak: C(1/φ²) < C(1). -/
theorem triality_recip_below_kernel : C (1 / φ ^ 2) < C 1 := by
  rw [(coherence_eq_one_iff 1 zero_le_one).mpr rfl]
  exact coherence_lt_one (1 / φ ^ 2)
    (le_of_lt goldenRatio_sq_recip_pos) (ne_of_lt goldenRatio_sq_recip_lt_one)

/-- *** Full Coherence Triality: C(1) = 1, C(φ²) = 2/3, C(1/φ²) = 2/3. ***

    The Kernel coherence function has exactly three distinguished values on
    the triality scales:
    - Kernel peak:   C(1)    = 1    (μ-orbit maximum coherence)
    - Lepton wing:   C(φ²)   = 2/3  (Koide lepton mass ratio)
    - Hadronic wing: C(1/φ²) = 2/3  (coherence mirror of lepton wing)

    The two wings are symmetric across the kernel scale by C(r) = C(1/r),
    and both sit strictly below the kernel maximum. -/
theorem coherence_triality :
    C 1 = 1 ∧ C (φ ^ 2) = 2 / 3 ∧ C (1 / φ ^ 2) = 2 / 3 :=
  ⟨(coherence_eq_one_iff 1 zero_le_one).mpr rfl,
   koide_coherence_bridge,
   koide_coherence_reciprocal⟩

/-- The kernel scale strictly maximises coherence over both triality wings:
    C(1/φ²) < C(1)  and  C(φ²) < C(1). -/
theorem triality_kernel_strict_max :
    C (1 / φ ^ 2) < C 1 ∧ C (φ ^ 2) < C 1 :=
  ⟨triality_recip_below_kernel, koide_below_mu_orbit_peak⟩

/-- The μ-orbit coherence exceeds both triality wings at every step:
    C(1/φ²) < C(|μⁿ|) = 1 for all n.

    The μ-orbit always achieves maximum coherence; both the lepton wing φ²
    and the hadronic wing 1/φ² lie strictly below the μ-orbit peak. -/
theorem mu_orbit_exceeds_triality_wings (n : ℕ) :
    C (1 / φ ^ 2) < C (Complex.abs (μ ^ n)) := by
  rw [koide_coherence_reciprocal, mu_pow_abs, (coherence_eq_one_iff 1 zero_le_one).mpr rfl]
  norm_num

end -- noncomputable section
