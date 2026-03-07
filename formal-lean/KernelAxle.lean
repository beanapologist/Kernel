/-
  KernelAxle.lean — Calculates the central axle of the Kernel engine.

  ╔══════════════════════════════════════════════════════════════════════════╗
  ║  The axle of the Kernel engine:                                         ║
  ║                                                                         ║
  ║    μ = exp(i · 3π/4)                                                   ║
  ║                                                                         ║
  ║  Every other module in the framework rotates around this axle.         ║
  ╚══════════════════════════════════════════════════════════════════════════╝

  The axle calculation:

    Angular step per orbit step:   ω = 3π/4  rad
    Steps per complete orbit:      N = 8
    Gear ratio (turns per orbit):  G = N·ω / (2π) = 8·(3π/4) / (2π) = 3

  So the axle makes exactly **3 full rotations** per 8-step orbit.
  The coprimality gcd(3, 8) = 1 means the axle visits all 8 angular
  positions before returning to start — a primitive 8th root of unity.

  Axle cross-section (why 45°):
    |Re(μ)| = |Im(μ)| = √2/2 = C(δS)  (isotropic, equal components)
  The 45° symmetry means the axle cross-section is circular — no preferred
  direction — and the cross-section radius equals the silver-ratio coherence.

  Connection to every engine module:
    • Turbulence   — μ generates the eddy rotation R(3π/4)
    • TimeCrystal  — μ^8 = 1 drives the 8-period Floquet precession
    • FineStructure— α_FS < C(δS)²: the Schwinger term lies below the axle
    • ParticleMass — 3 turns / 8 steps ↔ 3 triality scales {1/φ², 1, φ²}
    • OhmTriality  — μ-orbit runs at kernel Ohm point G = R = 1
    • SilverCoherence — |Im(μ)| = C(δS) = √2/2 (the axle cross-section
                        equals the silver-ratio coherence scale)

  Closing the loop:
    The axle phase 3π/4 is supplementary to the silver phase π/4:
      3π/4 + π/4 = π
    The 45° silver scale is the unique mirror of the 135° axle angle.

  All 20 theorems have complete machine-checked proofs, zero `sorry`.
-/

import SilverCoherence

open Complex Real

noncomputable section

-- ════════════════════════════════════════════════════════════════════════════
-- §1  The axle angular step: ω = 3π/4
-- ════════════════════════════════════════════════════════════════════════════

/-- The axle step is cos(3π/4): the real part of μ equals the cosine of 3π/4. -/
theorem axle_step_cos : Real.cos (3 * Real.pi / 4) = Complex.re μ := by
  have hcos : Real.cos (3 * Real.pi / 4) = -(Real.sqrt 2 / 2) := by
    rw [show (3 : ℝ) * Real.pi / 4 = Real.pi - Real.pi / 4 by ring]
    rw [Real.cos_pi_sub, Real.cos_pi_div_four]
  rw [hcos, mu_real_part]

/-- The axle step is sin(3π/4): the imaginary part of μ equals the sine of 3π/4. -/
theorem axle_step_sin : Real.sin (3 * Real.pi / 4) = Complex.im μ := by
  have hsin : Real.sin (3 * Real.pi / 4) = Real.sqrt 2 / 2 := by
    rw [show (3 : ℝ) * Real.pi / 4 = Real.pi - Real.pi / 4 by ring]
    rw [Real.sin_pi_sub, Real.sin_pi_div_four]
  rw [hsin, mu_imaginary_part]

/-- The axle Euler form: μ = cos(3π/4) + i·sin(3π/4).
    The axle sits at angle 3π/4 on the unit circle. -/
theorem axle_euler_form :
    μ = ↑(Real.cos (3 * Real.pi / 4)) + ↑(Real.sin (3 * Real.pi / 4)) * Complex.I := by
  conv_rhs => rw [axle_step_cos, axle_step_sin]
  exact (Complex.re_add_im μ).symm

-- ════════════════════════════════════════════════════════════════════════════
-- §2  The gear ratio: 3 complete turns per 8-step orbit
-- ════════════════════════════════════════════════════════════════════════════

/-- **The axle gear ratio**: 8 steps × (3π/4 rad/step) = 3 × (2π) = 3 full rotations.
    The Kernel engine's axle makes exactly 3 complete revolutions per 8-step orbit. -/
theorem axle_gear_ratio : (8 : ℝ) * (3 * Real.pi / 4) = 3 * (2 * Real.pi) := by ring

/-- The gear fraction in lowest terms: 3/8 turns per step.
    The axle advances 3 full turns for every 8 orbit steps. -/
theorem axle_gear_fraction : (8 : ℝ) * (3 * Real.pi / 4) / (2 * Real.pi) = 3 := by
  have hpi : Real.pi ≠ 0 := Real.pi_ne_zero
  field_simp [hpi]
  ring

/-- **Coprimality**: gcd(3, 8) = 1.
    The gear ratio 3:8 is in lowest terms — the axle visits all 8 angular
    positions before returning to its starting configuration. -/
theorem axle_gear_coprime : Nat.gcd 3 8 = 1 := by norm_num

/-- **Primitivity**: μ is a primitive 8th root of unity.
    The orbit {μ^0, μ^1, …, μ^7} consists of 8 distinct positions. -/
theorem axle_orbit_primitive : ∀ j k : Fin 8, (j : ℕ) ≠ k → μ ^ (j : ℕ) ≠ μ ^ (k : ℕ) :=
  mu_powers_distinct

/-- **Gear lock**: the axle closes exactly after 8 steps — μ^8 = 1. -/
theorem axle_orbit_closes : μ ^ 8 = 1 := mu_pow_eight

-- ════════════════════════════════════════════════════════════════════════════
-- §3  The axle cross-section: isotropic at 45°
-- ════════════════════════════════════════════════════════════════════════════

/-- **Rigidity**: the axle has unit norm — |μ| = 1.
    The axle neither stretches nor contracts; pure rotation. -/
theorem axle_rigid : Complex.abs μ = 1 := mu_abs_one

/-- **Isotropic cross-section**: |Re(μ)| = Im(μ).
    The axle components have equal magnitude — a circular cross-section. -/
theorem axle_cross_section_isotropic : |Complex.re μ| = Complex.im μ := by
  rw [mu_real_part, mu_imaginary_part, abs_neg, abs_of_pos]
  positivity

/-- **Cross-section value**: the axle radius is √2/2 = C(δS). -/
theorem axle_cross_section_value : Complex.im μ = Real.sqrt 2 / 2 := mu_imaginary_part

/-- **Silver bridge**: the axle cross-section equals the silver-ratio coherence. -/
theorem axle_cross_section_silver : Complex.im μ = C δS := mu_im_eq_silver_coherence

/-- **Unit-circle constraint**: |Re(μ)|² + Im(μ)² = 1.
    The axle cross-section satisfies the Pythagorean identity. -/
theorem axle_unit_constraint : Complex.re μ ^ 2 + Complex.im μ ^ 2 = 1 := by
  rw [mu_real_part, mu_imaginary_part]
  have h2 : Real.sqrt 2 ^ 2 = 2 := Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2)
  have h1 : (Real.sqrt 2 / 2) ^ 2 = 1 / 2 := by nlinarith
  nlinarith [show (-(Real.sqrt 2 / 2)) ^ 2 = (Real.sqrt 2 / 2) ^ 2 from by ring]

-- ════════════════════════════════════════════════════════════════════════════
-- §4  Axle dynamics: coherence along the μ-orbit
-- ════════════════════════════════════════════════════════════════════════════

/-- **Orbit radius**: every power of μ lies on the unit circle. -/
theorem axle_orbit_unit (n : ℕ) : Complex.abs (μ ^ n) = 1 := mu_pow_abs n

/-- **Maximum coherence at every step**: C(|μ^n|) = 1 for all n.
    The axle runs at maximum coherence — the kernel equilibrium — throughout
    its entire 8-step orbit. -/
theorem axle_maximum_coherence (n : ℕ) : C (Complex.abs (μ ^ n)) = 1 := by
  rw [mu_pow_abs]
  exact (coherence_eq_one_iff 1 zero_le_one).mpr rfl

/-- **Kernel equilibrium**: the axle is at the global coherence maximum, C(1) = 1. -/
theorem axle_kernel_equilibrium : C 1 = 1 :=
  (coherence_eq_one_iff 1 zero_le_one).mpr rfl

-- ════════════════════════════════════════════════════════════════════════════
-- §5  The engine loop: how the axle connects all modules
-- ════════════════════════════════════════════════════════════════════════════

/-- **Supplementary phases**: axle angle 3π/4 and silver angle π/4 add to π.
    The axle and the silver scale are antipodal phases — the 135° axle and the
    45° silver point are mirror images about the 90° line. -/
theorem axle_silver_supplementary : (3 : ℝ) * Real.pi / 4 + Real.pi / 4 = Real.pi := by ring

/-- **Triality connection**: the 3-turn gear ratio pairs with the 3-scale coherence triality.
    The engine turns 3 times per 8 steps; the coherence field has 3 distinguished scales. -/
theorem axle_triality_3_scales :
    C 1 = 1 ∧ C (φ ^ 2) = 2 / 3 ∧ C (1 / φ ^ 2) = 2 / 3 := coherence_triality

/-- **Gear turn count**: there exist exactly 3 full turns in the 8-step orbit.
    The gear ratio 8 × (3π/4) / (2π) = 3 — the unique positive integer satisfying
    n × (2π) = 8 × (3π/4). -/
theorem axle_gear_numerator : ∃ (n : ℕ), n = 3 ∧ (n : ℝ) * (2 * Real.pi) = 8 * (3 * Real.pi / 4) :=
  ⟨3, rfl, by ring⟩

/-- **Closing the loop** — the axle theorem unifying all modules:
    μ is the axle because it simultaneously satisfies all engine conditions:
    1. Rigid: |μ| = 1
    2. Gear lock: μ^8 = 1 (closes in 8 steps)
    3. Maximum coherence: C(|μ^n|) = 1 for all n (runs at kernel optimum)
    4. Silver cross-section: Im(μ) = C(δS) (45° isotropic, silver bridge)
    5. Gear ratio: 8 × (3π/4) = 3 × (2π) (3 full turns per orbit) -/
theorem axle_closes_loop :
    Complex.abs μ = 1 ∧
    μ ^ 8 = 1 ∧
    (∀ n : ℕ, C (Complex.abs (μ ^ n)) = 1) ∧
    Complex.im μ = C δS ∧
    (8 : ℝ) * (3 * Real.pi / 4) = 3 * (2 * Real.pi) :=
  ⟨mu_abs_one, mu_pow_eight, axle_maximum_coherence, mu_im_eq_silver_coherence, by ring⟩

end
