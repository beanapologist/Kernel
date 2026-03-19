/-
  GravityQuantumDuality.lean — Lean 4 formalization of the two sides of the
  observer-reality equation F(s, t) = t + i·s.

  ╔══════════════════════════════════════════════════════════════════════════╗
  ║                                                                          ║
  ║   NEGATIVE REAL AXIS  (Re F < 0)  ←→  GRAVITY / TIME                   ║
  ║                                                                          ║
  ║     • Gravitational potential: Φ_N(G, M, r) = −G·M/r  (negative)       ║
  ║     • Gravitational binding energy: E_grav = −G·M·m/r  (negative)      ║
  ║     • Time-coordinate causality: t < 0 encodes the past light cone      ║
  ║     • Gravity intensifies (more negative) as separation decreases       ║
  ║                                                                          ║
  ║   POSITIVE IMAGINARY AXIS  (Im F > 0)  ←→  QUANTUM / DARK ENERGY       ║
  ║                                                                          ║
  ║     • Zero-point energy: E_zp = hbar·ω/2  (strictly positive)          ║
  ║     • Dark energy density: ρ_Λ = Λ·c²/(8πG)  (positive, Planck 2018)  ║
  ║     • Spatial coherence: Im(F) > 0 for all s ∈ spaceDomain             ║
  ║     • Quantum energy grows with frequency; dark energy with Λ           ║
  ║                                                                          ║
  ╚══════════════════════════════════════════════════════════════════════════╝

  The two sides occupy orthogonal directions in the complex plane:
    • Gravity / time → negative real axis  (Re F < 0)
    • Quantum / dark energy / space → positive imaginary axis  (Im F > 0)
  They are coupled through the observer's modulus |F|² = t² + s² ≥ 0.

  Key structural results
  ──────────────────────
  §1.  Gravity (real) and quantum (imaginary) axes are orthogonal — they share
       no component with each other.
  §2.  Physical reality F(s, t) lies in the second quadrant: Re < 0, Im > 0.
  §3.  Newtonian potential Φ_N < 0; gravitational binding energy E_grav < 0.
       Gravity deepens (more negative) as masses approach.
  §4.  Quantum zero-point energy E_zp > 0.
       The quantum energy floor increases with oscillator frequency.
  §5.  Dark energy density ρ_Λ > 0 for Λ > 0 (Planck 2018: Λ > 0).
       Dark energy grows with the cosmological constant.
  §6.  The duality gap s + t measures quantum–gravity competition:
       positive when space > |time|, negative when gravity dominates.
  §7.  The Kernel equilibrium (s = 1, t = −1): |Re F| = Im F = 1.
       At this point the two sides balance exactly; normSq = 2.
  §8.  Sign duality: Re(F) · Im(F) < 0 for all physical coordinates.
       Gravity and quantum energy always have opposing signs.

  Sections
  ────────
  1.  Gravity–quantum orthogonality
  2.  Second-quadrant structure of physical reality
  3.  Gravity / time side: Newtonian potential and gravitational energy
  4.  Quantum side: zero-point energy
  5.  Dark energy as a positive-imaginary quantity
  6.  Duality gap and quantum–gravity competition
  7.  Kernel equilibrium: the unique balance point
  8.  Sign duality: Re(F) · Im(F) < 0

  Proof status
  ────────────
  All 22 theorems have complete machine-checked proofs.
  No `sorry` placeholders remain.

  Physical summary
  ────────────────
  • Gravity anchors the negative-real axis of observer reality.
  • Quantum mechanics and dark energy anchor the positive-imaginary axis.
  • Their unification in F(s, t) = t + i·s is the Kernel observer equation.
  • The two sides are orthogonal and sign-dual: Re · Im < 0 everywhere.
-/

import SpaceTime

open Complex Real

noncomputable section

-- ════════════════════════════════════════════════════════════════════════════
-- §1  Gravity–quantum orthogonality
-- The gravity (real) and quantum (imaginary) sides of F are orthogonal.
-- ════════════════════════════════════════════════════════════════════════════

/-- The gravity/time component of F has zero imaginary part: the time axis
    contributes nothing to the space (quantum) axis.

    Re(↑t) embeds time as a purely real number; its imaginary part is zero. -/
theorem gravity_axis_im_zero (t : ℝ) : ((↑t : ℂ)).im = 0 :=
  Complex.ofReal_im t

/-- The quantum/space component of F has zero real part: the imaginary axis
    contributes nothing to the gravity (time) axis.

    iSpace s = i·s lives purely on the imaginary axis; its real part is zero.-/
theorem quantum_axis_re_zero (s : ℝ) : (iSpace s).re = 0 :=
  iSpace_re s

/-- Gravity and quantum energy occupy perpendicular directions in the
    complex plane:

        Re(iSpace s) = 0   ∧   Im(↑t) = 0

    Machine-checked statement of gravity–quantum orthogonality. -/
theorem gravity_quantum_orthogonal (s t : ℝ) :
    (iSpace s).re = 0 ∧ ((↑t : ℂ)).im = 0 :=
  space_time_orthogonal s t

-- ════════════════════════════════════════════════════════════════════════════
-- §2  Second-quadrant structure of physical reality
-- Physical coordinates (s > 0, t < 0) place F in the second quadrant.
-- ════════════════════════════════════════════════════════════════════════════

/-- Physical reality lies in the second quadrant of the complex plane.

    For any s ∈ spaceDomain (s > 0) and t ∈ timeDomain (t < 0):
        Re F(s, t) < 0   ←→   gravity / time component is negative
        Im F(s, t) > 0   ←→   quantum / dark energy / space is positive -/
theorem reality_second_quadrant_gqd (s t : ℝ)
    (hs : s ∈ spaceDomain) (ht : t ∈ timeDomain) :
    (F s t).re < 0 ∧ 0 < (F s t).im :=
  F_second_quadrant s t hs ht

/-- The gravity/time component (Re F) is strictly negative for physical t. -/
theorem gravity_component_negative (s t : ℝ) (ht : t ∈ timeDomain) :
    (F s t).re < 0 := by
  rw [F_re]; exact ht

/-- The quantum/dark-energy/space component (Im F) is strictly positive for
    physical s. -/
theorem quantum_component_positive (s t : ℝ) (hs : s ∈ spaceDomain) :
    0 < (F s t).im := by
  rw [F_im]; exact hs

-- ════════════════════════════════════════════════════════════════════════════
-- §3  Gravity / time side — Newtonian potential and binding energy
-- ════════════════════════════════════════════════════════════════════════════

/-- Newtonian gravitational potential:
        Φ_N(G, M, r) = −G · M / r  -/
noncomputable def newtonPotential (G M r : ℝ) : ℝ := -(G * M / r)

/-- The Newtonian gravitational potential is strictly negative for positive
    parameters G, M, r.

    Gravity is the negative-real side of the equation: Φ_N < 0 for all
    finite separations r > 0 between positive masses. -/
theorem newtonPotential_neg (G M r : ℝ) (hG : 0 < G) (hM : 0 < M) (hr : 0 < r) :
    newtonPotential G M r < 0 := by
  unfold newtonPotential
  linarith [div_pos (mul_pos hG hM) hr]

/-- Gravitational binding energy:
        E_grav(G, M, m, r) = −G · M · m / r  -/
noncomputable def gravBindingEnergy (G M m r : ℝ) : ℝ := -(G * M * m / r)

/-- Gravitational binding energy is strictly negative for positive parameters.

    Bringing two positive masses together releases energy; the stored energy
    is negative.  This is the macroscopic expression of the negative-real
    direction of the reality map. -/
theorem gravBindingEnergy_neg (G M m r : ℝ)
    (hG : 0 < G) (hM : 0 < M) (hm : 0 < m) (hr : 0 < r) :
    gravBindingEnergy G M m r < 0 := by
  unfold gravBindingEnergy
  linarith [div_pos (mul_pos (mul_pos hG hM) hm) hr]

/-- As separation decreases, the gravitational potential becomes more negative:
    gravity deepens along the negative-real direction as masses approach.

        r₁ < r₂  →  Φ_N(r₁) < Φ_N(r₂)   (more negative at smaller r) -/
theorem newtonPotential_monotone_decreasing (G M r₁ r₂ : ℝ)
    (hG : 0 < G) (hM : 0 < M) (hr₁ : 0 < r₁) (hr₂ : 0 < r₂)
    (h : r₁ < r₂) :
    newtonPotential G M r₁ < newtonPotential G M r₂ := by
  unfold newtonPotential
  have hGM : 0 < G * M := mul_pos hG hM
  have h1 : G * M / r₂ < G * M / r₁ := by
    rw [div_lt_div_iff₀ hr₂ hr₁]
    exact mul_lt_mul_of_pos_left h hGM
  linarith

/-- The gravitational potential is bounded above by zero:
        Φ_N(G, M, r) < 0  for all G, M, r > 0.

    The negative-real direction has no positive contribution from gravity. -/
theorem newtonPotential_negative_everywhere (G M r : ℝ)
    (hG : 0 < G) (hM : 0 < M) (hr : 0 < r) :
    newtonPotential G M r < 0 :=
  newtonPotential_neg G M r hG hM hr

-- ════════════════════════════════════════════════════════════════════════════
-- §4  Quantum side — zero-point energy
-- ════════════════════════════════════════════════════════════════════════════

/-- Quantum zero-point energy of a harmonic oscillator:
        E_zp(hbar, ω) = hbar · ω / 2

    The Heisenberg uncertainty principle mandates E_zp > 0 in every mode. -/
noncomputable def zeroPointEnergy (hbar ω : ℝ) : ℝ := hbar * ω / 2

/-- Zero-point energy is strictly positive for positive hbar and ω.

    Quantum mechanics mandates a positive energy floor even in the ground
    state.  This positive quantity lives on the positive-imaginary side of the
    reality map — the quantum contribution to the complex energy plane. -/
theorem zeroPointEnergy_pos (hbar ω : ℝ) (hhbar : 0 < hbar) (hω : 0 < ω) :
    0 < zeroPointEnergy hbar ω := by
  unfold zeroPointEnergy
  linarith [mul_pos hhbar hω]

/-- Higher-frequency modes carry more zero-point energy:

        ω₁ < ω₂  →  E_zp(hbar, ω₁) < E_zp(hbar, ω₂)

    The quantum energy floor is frequency-dependent. -/
theorem zeroPointEnergy_monotone (hbar ω₁ ω₂ : ℝ) (hhbar : 0 < hbar) (h : ω₁ < ω₂) :
    zeroPointEnergy hbar ω₁ < zeroPointEnergy hbar ω₂ := by
  unfold zeroPointEnergy
  linarith [mul_lt_mul_of_pos_left h hhbar]

-- ════════════════════════════════════════════════════════════════════════════
-- §5  Dark energy as a positive-imaginary quantity
-- Dark energy density ρ_Λ > 0 drives cosmic expansion along the space axis.
-- ════════════════════════════════════════════════════════════════════════════

/-- Dark energy density:
        ρ_Λ(Λ, c, G) = Λ · c² / (8π · G)

    Λ > 0 is the cosmological constant (Planck 2018), c > 0 is the speed
    of light, G > 0 is Newton's constant. -/
noncomputable def darkEnergyDensity (Λ c G : ℝ) : ℝ := Λ * c ^ 2 / (8 * Real.pi * G)

/-- Dark energy density is strictly positive for Λ, c, G > 0.

    The cosmological constant Λ > 0 (measured by Planck 2018 CMB observations)
    gives a positive vacuum energy density.  This positive quantity lives on
    the positive-imaginary (space/quantum) side of the reality map. -/
theorem darkEnergyDensity_pos (Λ c G : ℝ)
    (hΛ : 0 < Λ) (hc : 0 < c) (hG : 0 < G) :
    0 < darkEnergyDensity Λ c G := by
  unfold darkEnergyDensity
  apply div_pos
  · exact mul_pos hΛ (pow_pos hc 2)
  · exact mul_pos (mul_pos (by norm_num : (0:ℝ) < 8) Real.pi_pos) hG

/-- A larger cosmological constant gives greater dark energy density:
        Λ₁ < Λ₂  →  ρ_Λ(Λ₁) < ρ_Λ(Λ₂)

    More dark energy means a stronger push along the positive-imaginary axis
    (more accelerated cosmic expansion). -/
theorem darkEnergyDensity_monotone (Λ₁ Λ₂ c G : ℝ)
    (hc : 0 < c) (hG : 0 < G) (h : Λ₁ < Λ₂) :
    darkEnergyDensity Λ₁ c G < darkEnergyDensity Λ₂ c G := by
  unfold darkEnergyDensity
  have hden : 0 < 8 * Real.pi * G :=
    mul_pos (mul_pos (by norm_num : (0:ℝ) < 8) Real.pi_pos) hG
  have h2 : Λ₁ * c ^ 2 < Λ₂ * c ^ 2 := mul_lt_mul_of_pos_right h (pow_pos hc 2)
  rw [div_lt_div_iff₀ hden hden]
  exact mul_lt_mul_of_pos_right h2 hden

-- ════════════════════════════════════════════════════════════════════════════
-- §6  Duality gap and quantum–gravity competition
-- ════════════════════════════════════════════════════════════════════════════

/-- The duality gap: quantum (space/Im) minus gravity (time/|Re|) magnitude.

        gap(s, t) = s + t   (since t < 0, s − |t| = s + t)

    Positive gap: quantum/space dominates (expanding Universe).
    Negative gap: gravity/time dominates (gravitational collapse). -/
noncomputable def dualityGap (s t : ℝ) : ℝ := s + t

/-- The duality gap equals Im(F) + Re(F): space amplitude plus time amplitude.
    (Since Re F = t < 0, this is effectively Im F − |Re F|.) -/
theorem dualityGap_eq_imF_plus_reF (s t : ℝ) :
    dualityGap s t = (F s t).im + (F s t).re := by
  unfold dualityGap; rw [F_re, F_im]; ring

/-- When s > |t| (space dominates over gravity), the duality gap is positive:
    quantum/dark energy wins the competition. -/
theorem dualityGap_pos_when_space_dominates (s t : ℝ)
    (ht : t ∈ timeDomain) (h : -t < s) :
    0 < dualityGap s t := by
  unfold dualityGap; linarith

/-- When |t| > s (gravity dominates over space), the duality gap is negative:
    gravitational collapse wins the competition. -/
theorem dualityGap_neg_when_gravity_dominates (s t : ℝ)
    (hs : s ∈ spaceDomain) (h : s < -t) :
    dualityGap s t < 0 := by
  unfold dualityGap; linarith

-- ════════════════════════════════════════════════════════════════════════════
-- §7  Kernel equilibrium: the unique balance point
-- ════════════════════════════════════════════════════════════════════════════

/-- At the Kernel equilibrium (s = 1, t = −1), the gravity and quantum sides
    balance exactly: |Re F| = Im F = 1. -/
theorem kernel_equilibrium_balance :
    |(F 1 (-1)).re| = (F 1 (-1)).im := by
  simp [F_re, F_im]

/-- At the Kernel equilibrium the duality gap is zero: neither gravity nor
    quantum energy dominates.

        gap(1, −1) = 1 + (−1) = 0 -/
theorem kernel_equilibrium_gap_zero :
    dualityGap 1 (-1) = 0 := by
  unfold dualityGap; ring

/-- At the Kernel equilibrium the squared modulus equals 2:
        |Re F|² + |Im F|² = (−1)² + 1² = 2

    The equilibrium point F(1, −1) = −1 + i is equidistant from both axes. -/
theorem kernel_equilibrium_normSq :
    Complex.normSq (F 1 (-1)) = 2 := by
  rw [Complex.normSq_apply, F_re, F_im]
  norm_num

-- ════════════════════════════════════════════════════════════════════════════
-- §8  Sign duality: Re(F) · Im(F) < 0
-- ════════════════════════════════════════════════════════════════════════════

/-- The product of the gravity (Re F < 0) and quantum (Im F > 0) components
    is always strictly negative for physical coordinates:

        Re(F) · Im(F) = t · s < 0   (t < 0, s > 0)

    Gravity and quantum energy are sign-dual: one is always negative where
    the other is positive.  They are dual, not equal. -/
theorem reality_sign_duality (s t : ℝ)
    (hs : s ∈ spaceDomain) (ht : t ∈ timeDomain) :
    (F s t).re * (F s t).im < 0 := by
  rw [F_re, F_im]
  exact mul_neg_of_neg_of_pos ht hs

/-- The gravity and quantum sides always have opposite signs for physical
    coordinates: Re F < 0 and Im F > 0. -/
theorem gravity_and_quantum_opposite_signs (s t : ℝ)
    (hs : s ∈ spaceDomain) (ht : t ∈ timeDomain) :
    (F s t).re < 0 ∧ 0 < (F s t).im :=
  reality_second_quadrant_gqd s t hs ht

end
