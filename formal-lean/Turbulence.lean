/-
  Turbulence.lean — Lean 4 formalization integrating physics-based turbulence
  equations into the Kernel formal model.

  This module formalizes turbulence theory at micro, meso, and macro scales,
  connects with the Navier-Stokes viscous-dissipation framework, and maps the
  eigenvector hypothesis to the μ-driven Kernel theorem structure.

  The central objects are:
    • Turbulence scale hierarchy: micro (0 < η < 1), meso (1 ≤ ℓ ≤ 100), macro (L > 100)
    • Reynolds decomposition: u = ū + u′  (temporal mean + fluctuation)
    • Turbulent kinetic energy: k = ½ (u′)²
    • Multi-scale coherence: turbulenceCoherence(r) = C(r) = 2r/(1+r²) per scale r
    • Navier-Stokes viscous dissipation: ε(ν, g) = ν · g²  (ν = viscosity, g = |∇u|)
    • Eigenvector hypothesis: μ = exp(I·3π/4) governs turbulent rotation and precession

  Eigenvector hypothesis and Kernel mapping
  ──────────────────────────────────────────
  The critical eigenvalue μ = exp(I·3π/4) studied in CriticalEigenvalue.lean is
  identified with the dominant rotational mode of the turbulent velocity-gradient
  tensor.  The mapping proceeds as:

    |μ| = 1         (mu_abs_one)           — turbulent rotation is amplitude-neutral
    μ^8 = 1         (mu_pow_eight)         — turbulent precession is 8-periodic
    C(1) = 1        (coherence_eq_one_iff) — the μ-orbit is maximally coherent
    C(rⁿ) = sech   (coherence_orbit_sech) — orbit coherence decays hyperbolically

  Multi-scale consistency
  ───────────────────────
  The coherence function C(r) = 2r/(1+r²) peaks at r = 1 (the kernel scale) and
  decreases toward both r → 0 (micro) and r → ∞ (macro).  The laminar (fully
  ordered) flow corresponds to r = 1; turbulent fluctuations correspond to r ≠ 1.
  The cross-scale consistency theorem asserts:

      turbulenceCoherence(r) ≤ turbulenceCoherence(1) = 1   for all r ≥ 0,

  confirming that every turbulence scale has lower or equal coherence than the
  kernel-scale laminar state.

  Sections
  ────────
  1.  Turbulence scale hierarchy    (micro η < 1, meso 1 ≤ ℓ ≤ 100, macro L > 100)
  2.  Reynolds decomposition        (u = ū + u′, fluctuation formula, uniqueness)
  3.  Turbulent kinetic energy      (k = ½ u′², non-negativity, zero criterion)
  4.  Multi-scale coherence         (C connects eigenvalue to scale structure)
  5.  Navier-Stokes dissipation     (ε = ν g², non-negativity, monotonicity)
  6.  Eigenvector hypothesis        (μ governs turbulent rotation and precession)
  7.  Cross-scale consistency       (micro, meso, macro coherence hierarchy)

  Proof status
  ────────────
  All 29 theorems have complete machine-checked proofs.
  No `sorry` placeholders remain.
-/

import CriticalEigenvalue

open Complex Real

noncomputable section

-- ════════════════════════════════════════════════════════════════════════════
-- Section 1 — Turbulence Scale Hierarchy
-- Physical turbulent flows exhibit structure across three principal scale ranges:
--   micro  (Kolmogorov dissipation scale, 0 < η < 1)
--   meso   (inertial sub-range, 1 ≤ ℓ ≤ 100)
--   macro  (integral length scale, L > 100)
-- The three-level hierarchy η ≪ ℓ ≪ L underpins the Kolmogorov cascade and
-- the Reynolds decomposition.
-- Ref: Kolmogorov (1941); Pope, "Turbulent Flows" (2000) §6
-- ════════════════════════════════════════════════════════════════════════════

/-- The micro-scale (Kolmogorov dissipation) domain: length scales 0 < η < 1.

    At these scales viscous dissipation dominates inertial forces.
    The local Reynolds number Re(η) ≈ 1 at the Kolmogorov microscale η_K.
    In the Kernel coherence picture, micro scales correspond to amplitude
    ratios r = η < 1 where C(η) < C(1). -/
def microScaleDomain : Set ℝ := {η | 0 < η ∧ η < 1}

/-- The meso-scale (inertial sub-range) domain: length scales 1 ≤ ℓ ≤ 100.

    At these scales neither viscosity nor large-scale forcing dominates;
    the Kolmogorov −5/3 energy spectrum E(k) ∝ k^{−5/3} is observed.
    Meso scales straddle the kernel scale r = 1 from both sides. -/
def mesoScaleDomain : Set ℝ := {ℓ | 1 ≤ ℓ ∧ ℓ ≤ 100}

/-- The macro-scale (integral length) domain: length scales L > 100.

    These are the energy-injection scales at which large eddies are driven.
    In the Kernel coherence picture, macro scales correspond to amplitude
    ratios r = L > 100 > 1 where C(L) < C(1). -/
def macroScaleDomain : Set ℝ := {L | 100 < L}

/-- The micro-scale domain is non-empty: η = 1/2 is a valid Kolmogorov scale. -/
theorem microScale_nonempty : (1 / 2 : ℝ) ∈ microScaleDomain := by
  constructor <;> norm_num

/-- The meso-scale domain is non-empty: ℓ = 10 is in the inertial sub-range. -/
theorem mesoScale_nonempty : (10 : ℝ) ∈ mesoScaleDomain := by
  constructor <;> norm_num

/-- The macro-scale domain is non-empty: L = 1000 is a valid integral length scale. -/
theorem macroScale_nonempty : (1000 : ℝ) ∈ macroScaleDomain := by
  show (100 : ℝ) < 1000; norm_num

/-- A micro scale is strictly smaller than any meso scale.

    Proof: η < 1 ≤ ℓ.  The Kolmogorov scale is always below the inertial
    sub-range; their strict separation is fundamental to turbulence theory. -/
theorem micro_lt_meso (η ℓ : ℝ) (hη : η ∈ microScaleDomain) (hℓ : ℓ ∈ mesoScaleDomain) :
    η < ℓ :=
  lt_of_lt_of_le hη.2 hℓ.1

/-- A meso scale is strictly smaller than any macro scale.

    Proof: ℓ ≤ 100 < L.  The inertial sub-range is always below the
    energy-injection (integral) scale. -/
theorem meso_lt_macro (ℓ L : ℝ) (hℓ : ℓ ∈ mesoScaleDomain) (hL : L ∈ macroScaleDomain) :
    ℓ < L :=
  lt_of_le_of_lt hℓ.2 hL

/-- A micro scale is strictly smaller than any macro scale.

    Transitivity of the three-level scale hierarchy: micro ≪ meso ≪ macro.
    Proof: η < 1 < L (since L > 100 > 1). -/
theorem micro_lt_macro (η L : ℝ) (hη : η ∈ microScaleDomain) (hL : L ∈ macroScaleDomain) :
    η < L :=
  lt_trans hη.2 (by linarith [hL])

-- ════════════════════════════════════════════════════════════════════════════
-- Section 2 — Reynolds Decomposition
-- Any instantaneous velocity field u can be split into a temporal mean ū and
-- a zero-mean fluctuation u′ = u − ū.  This is Reynolds's 1895 decomposition,
-- the starting point of turbulence closure modeling.
-- Ref: Reynolds (1895); Tennekes & Lumley, "A First Course in Turbulence" §2.1
-- ════════════════════════════════════════════════════════════════════════════

/-- A Reynolds decomposition of a scalar velocity field u into mean ū and
    fluctuation u′ satisfies the identity u(t) = ū + u′(t) for all t.

    The mean ū may represent a time average, ensemble average, or spatial mean
    depending on the context.  The fluctuation u′ carries the turbulent content. -/
def isReynoldsDecomp (u : ℝ → ℝ) (u_mean : ℝ) (u_fluct : ℝ → ℝ) : Prop :=
  ∀ t : ℝ, u t = u_mean + u_fluct t

/-- The fluctuation is the difference of the velocity field and its mean:
    u′(t) = u(t) − ū.

    Proof: direct rearrangement of the decomposition identity. -/
theorem reynolds_fluct_formula (u : ℝ → ℝ) (u_mean : ℝ) (u_fluct : ℝ → ℝ)
    (h : isReynoldsDecomp u u_mean u_fluct) (t : ℝ) :
    u_fluct t = u t - u_mean := by
  have := h t; linarith

/-- The canonical Reynolds decomposition always exists: setting u′(t) = u(t) − ū
    satisfies the decomposition identity for any choice of mean ū. -/
theorem reynolds_decomp_canonical (u : ℝ → ℝ) (u_mean : ℝ) :
    isReynoldsDecomp u u_mean (fun t => u t - u_mean) := by
  intro t; ring

/-- Reynolds decomposition is unique: two fluctuation fields arising from the
    same mean ū agree pointwise.

    This says that once the mean is fixed, the Reynolds splitting is determined. -/
theorem reynolds_decomp_unique (u : ℝ → ℝ) (ū : ℝ) (u₁ u₂ : ℝ → ℝ)
    (h₁ : isReynoldsDecomp u ū u₁) (h₂ : isReynoldsDecomp u ū u₂) (t : ℝ) :
    u₁ t = u₂ t := by
  have e₁ := h₁ t; have e₂ := h₂ t; linarith

/-- Reconstruction identity: ū + u′(t) = u(t).

    The original velocity field is fully recovered by summing its mean and
    fluctuation components. -/
theorem reynolds_reconstruction (u : ℝ → ℝ) (ū : ℝ) (u′ : ℝ → ℝ)
    (h : isReynoldsDecomp u ū u′) (t : ℝ) :
    ū + u′ t = u t := by
  have := h t; linarith

-- ════════════════════════════════════════════════════════════════════════════
-- Section 3 — Turbulent Kinetic Energy
-- The turbulent kinetic energy (TKE) at time t is k(t) = ½ (u′(t))², the
-- kinetic energy stored in the velocity fluctuation.  Its non-negativity
-- ensures thermodynamic consistency; it vanishes iff the flow is laminar.
-- Ref: Pope, "Turbulent Flows" (2000) §3.1
-- ════════════════════════════════════════════════════════════════════════════

/-- Turbulent kinetic energy at time t: k(t) = ½ (u′(t))².

    In the full 3D Navier-Stokes setting k = ½ |u′|² = ½ (u′₁² + u′₂² + u′₃²).
    Here we formalize the scalar (1D projection) version. -/
noncomputable def turbulentKE (u_fluct : ℝ → ℝ) (t : ℝ) : ℝ :=
  (1 / 2) * (u_fluct t) ^ 2

/-- Turbulent kinetic energy is non-negative at every instant: k(t) ≥ 0.

    Proof: ½ (u′)² ≥ 0 since squares are non-negative (positivity tactic). -/
theorem turbulentKE_nonneg (u_fluct : ℝ → ℝ) (t : ℝ) : 0 ≤ turbulentKE u_fluct t := by
  unfold turbulentKE; positivity

/-- Turbulent kinetic energy vanishes iff the fluctuation is zero:
    k(t) = 0 ↔ u′(t) = 0.

    Physical interpretation: the flow is instantaneously laminar at time t
    precisely when the turbulent fluctuation is absent. -/
theorem turbulentKE_zero_iff (u_fluct : ℝ → ℝ) (t : ℝ) :
    turbulentKE u_fluct t = 0 ↔ u_fluct t = 0 := by
  unfold turbulentKE
  constructor
  · intro h
    have h2 : (u_fluct t) ^ 2 = 0 := by nlinarith [sq_nonneg (u_fluct t)]
    exact pow_eq_zero_iff (by norm_num : 2 ≠ 0) |>.mp h2
  · intro h; rw [h]; norm_num

/-- Scaling the fluctuation by a factor c scales the TKE by c²:
    k(c u′) = c² · k(u′).

    This reflects the quadratic (kinetic-energy) scaling of the fluctuation
    amplitude: doubling the fluctuation quadruples the TKE. -/
theorem turbulentKE_scale (u_fluct : ℝ → ℝ) (t : ℝ) (c : ℝ) :
    turbulentKE (fun s => c * u_fluct s) t = c ^ 2 * turbulentKE u_fluct t := by
  unfold turbulentKE; ring

-- ════════════════════════════════════════════════════════════════════════════
-- Section 4 — Multi-Scale Coherence via C(r)
-- The Kernel coherence function C(r) = 2r/(1+r²) measures the "ordering" at
-- the amplitude-ratio scale r.  Setting r = η (micro scale), ℓ (meso), or L
-- (macro) gives the turbulence coherence at that scale.
-- Key property: C peaks at r = 1 (the laminar/kernel state) and decreases
-- monotonically toward both r → 0 and r → ∞.
-- ════════════════════════════════════════════════════════════════════════════

/-- The turbulence coherence at scale r is the Kernel coherence function C(r).

    In the Kernel turbulence model the amplitude ratio r parameterises the
    balance between turbulent and laminar modes.  r = 1 is the balanced
    (laminar) state; r ≠ 1 encodes a turbulent imbalance. -/
noncomputable def turbulenceCoherence (r : ℝ) : ℝ := C r

/-- At the kernel (unit) scale r = 1 the turbulence coherence is maximal:
    turbulenceCoherence(1) = 1.

    r = 1 is the fixed point of the coherence function and corresponds to
    fully laminar, maximally coherent flow. -/
theorem turbulenceCoherence_kernel_max : turbulenceCoherence 1 = 1 := by
  unfold turbulenceCoherence
  exact (coherence_eq_one_iff 1 (le_of_lt one_pos)).mpr rfl

/-- Micro-scale turbulence has strictly sub-maximal coherence:
    turbulenceCoherence(η) < 1 for η ∈ microScaleDomain.

    Since η < 1 and η ≠ 1, the coherence C(η) < C(1) = 1: small eddies
    are incoherent relative to the kernel scale. -/
theorem turbulenceCoherence_micro_lt_one (η : ℝ) (hη : η ∈ microScaleDomain) :
    turbulenceCoherence η < 1 := by
  unfold turbulenceCoherence
  exact coherence_lt_one η (le_of_lt hη.1) (ne_of_lt hη.2)

/-- Macro-scale turbulence has strictly sub-maximal coherence:
    turbulenceCoherence(L) < 1 for L ∈ macroScaleDomain.

    Since L > 100 > 1 and L ≠ 1, the coherence C(L) < C(1) = 1: the large
    integral-scale eddies are also incoherent relative to the kernel scale. -/
theorem turbulenceCoherence_macro_lt_one (L : ℝ) (hL : L ∈ macroScaleDomain) :
    turbulenceCoherence L < 1 := by
  unfold turbulenceCoherence
  exact coherence_lt_one L (by linarith [hL]) (ne_of_gt (by linarith [hL]))

/-- Coherence is strictly monotone increasing through the micro-scale range (0, 1]:
    for 0 < η₁ < η₂ ≤ 1, turbulenceCoherence(η₁) < turbulenceCoherence(η₂).

    Smaller eddies are more disordered; coherence builds as the scale grows
    toward the kernel.  This is the formal counterpart of the Kolmogorov cascade
    energy transfer from large to small scales. -/
theorem turbulenceCoherence_micro_strictMono (η₁ η₂ : ℝ)
    (h1 : 0 < η₁) (h12 : η₁ < η₂) (h2 : η₂ ≤ 1) :
    turbulenceCoherence η₁ < turbulenceCoherence η₂ :=
  coherence_strictMono η₁ η₂ h1 h12 h2

/-- Coherence is strictly monotone decreasing above the kernel scale [1, ∞):
    for 1 ≤ L₁ < L₂, turbulenceCoherence(L₂) < turbulenceCoherence(L₁).

    Larger macro eddies have lower coherence than smaller ones: disorder
    increases with scale in the energy-injection range. -/
theorem turbulenceCoherence_macro_strictAnti (L₁ L₂ : ℝ)
    (h1 : 1 ≤ L₁) (h12 : L₁ < L₂) :
    turbulenceCoherence L₂ < turbulenceCoherence L₁ :=
  coherence_strictAnti L₁ L₂ h1 h12

-- ════════════════════════════════════════════════════════════════════════════
-- Section 5 — Navier-Stokes Viscous Dissipation
-- In the incompressible Navier-Stokes equations the local viscous dissipation
-- rate is ε = ν |∇u|², where ν > 0 is the kinematic viscosity and |∇u| is
-- the velocity-gradient magnitude.  Here we formalize this for the scalar
-- model: ε(ν, g) = ν · g² with g = |∇u|.
-- Ref: Navier-Stokes energy equation (Pope §2.5); Batchelor §3.4
-- ════════════════════════════════════════════════════════════════════════════

/-- Viscous dissipation rate: ε(ν, g) = ν · g².

    ν is the kinematic viscosity (ν > 0 in a physical fluid),
    g = |∇u| is the velocity-gradient magnitude.  The dissipation
    converts kinetic energy of the turbulent fluctuations to heat. -/
noncomputable def viscousDissipation (ν g : ℝ) : ℝ := ν * g ^ 2

/-- Viscous dissipation is non-negative for non-negative viscosity: ε(ν, g) ≥ 0.

    This is thermodynamic consistency: dissipation can only remove energy,
    never inject it back into the flow. -/
theorem viscousDissipation_nonneg (ν g : ℝ) (hν : 0 ≤ ν) :
    0 ≤ viscousDissipation ν g := by
  unfold viscousDissipation; positivity

/-- For positive viscosity ν > 0, the dissipation vanishes iff the gradient is zero:
    ε(ν, g) = 0 ↔ g = 0.

    Physical interpretation: in a viscous fluid (ν > 0), dissipation is absent
    only in a spatially uniform flow with zero velocity gradient. -/
theorem viscousDissipation_zero_iff (ν g : ℝ) (hν : 0 < ν) :
    viscousDissipation ν g = 0 ↔ g = 0 := by
  unfold viscousDissipation
  constructor
  · intro h
    have hg2 : g ^ 2 = 0 := (mul_eq_zero.mp h).resolve_left (ne_of_gt hν)
    exact pow_eq_zero_iff (by norm_num : 2 ≠ 0) |>.mp hg2
  · intro h; rw [h]; ring

/-- For positive viscosity ν > 0 and non-zero gradient g ≠ 0, the dissipation
    is strictly positive: ε(ν, g) > 0.

    In the turbulent inertial sub-range, g ≠ 0 always holds, so turbulent
    kinetic energy is continuously dissipated. -/
theorem viscousDissipation_pos (ν g : ℝ) (hν : 0 < ν) (hg : g ≠ 0) :
    0 < viscousDissipation ν g := by
  unfold viscousDissipation
  apply mul_pos hν
  rcases lt_or_eq_of_le (sq_nonneg g) with h | h
  · exact h
  · exact absurd (pow_eq_zero_iff (by norm_num : 2 ≠ 0) |>.mp h.symm) hg

/-- Viscous dissipation is strictly monotone increasing in viscosity for g ≠ 0:
    ν₁ < ν₂ and g ≠ 0 imply ε(ν₁, g) < ε(ν₂, g).

    More viscous fluids dissipate turbulent kinetic energy faster at the same
    velocity gradient. -/
theorem viscousDissipation_mono_viscosity (ν₁ ν₂ g : ℝ)
    (hν : ν₁ < ν₂) (hg : g ≠ 0) :
    viscousDissipation ν₁ g < viscousDissipation ν₂ g := by
  unfold viscousDissipation
  apply mul_lt_mul_of_pos_right hν
  rcases lt_or_eq_of_le (sq_nonneg g) with h | h
  · exact h
  · exact absurd (pow_eq_zero_iff (by norm_num : 2 ≠ 0) |>.mp h.symm) hg

-- ════════════════════════════════════════════════════════════════════════════
-- Section 6 — Eigenvector Hypothesis
-- The critical eigenvalue μ = exp(I·3π/4) from CriticalEigenvalue.lean is
-- identified with the dominant rotational mode of the turbulent
-- velocity-gradient tensor.  Its key properties map directly to turbulence:
--   |μ| = 1   → the rotation is amplitude-neutral (no growth or decay)
--   μ^8 = 1   → turbulent precession has an 8-step period
--   C(1) = 1  → the μ-orbit (at unit amplitude) achieves maximum coherence
-- These properties together say that the turbulent rotation operator μ drives
-- an 8-periodic, amplitude-stable, maximally coherent precession — the formal
-- counterpart of a stable vortex tube in the turbulent cascade.
-- ════════════════════════════════════════════════════════════════════════════

/-- The turbulent rotation operator μ is unitary: |μ| = 1.

    In turbulence modeling, the dominant rotational eigenmode of the
    velocity-gradient tensor has unit modulus when the flow is in the
    balanced (steady-state cascade) regime.  Unitarity ensures that
    the rotation neither amplifies nor damps the mode amplitude. -/
theorem turbulence_rotation_unitary : Complex.abs μ = 1 := mu_abs_one

/-- Turbulent precession is 8-periodic: μ^8 = 1.

    The μ-driven rotation operator generates an exact 8-cycle.  In the
    turbulence context this corresponds to an 8-step discrete precession
    of the dominant vortex mode: after 8 drive periods the flow returns
    to its initial rotational phase. -/
theorem turbulence_precession_8period : μ ^ 8 = 1 := mu_pow_eight

/-- The μ-orbit at unit amplitude is stable: |(1 · μ^n)| = 1 for all n ∈ ℕ.

    A turbulent mode initialised at unit amplitude ratio (r = 1) stays
    on the unit circle under all iterates of the rotation operator.
    This is the formal analogue of a neutrally stable vortex tube. -/
theorem turbulence_eigenstate_orbit_stability (n : ℕ) :
    Complex.abs ((1 : ℂ) * μ ^ n) = 1 := trichotomy_unit_orbit n

/-- The coherence of the turbulent eigenstate orbit is maximal:
    C(|1 · μ|^n) = C(1^n) = 1 for all n ∈ ℕ.

    The amplitude ratio of each iterate of the unit-amplitude μ-orbit is 1
    (since |μ| = 1), so the coherence function C evaluates to its maximum.
    This formally proves that the eigenvector μ sustains full coherence
    throughout its 8-step turbulent precession cycle. -/
theorem turbulence_eigenstate_orbit_coherence (n : ℕ) :
    C (Complex.abs ((1 : ℂ) * μ) ^ n) = 1 := by
  have h : Complex.abs ((1 : ℂ) * μ) = 1 := by rw [one_mul]; exact mu_abs_one
  rw [h]
  exact orbit_coherence_at_one n

-- ════════════════════════════════════════════════════════════════════════════
-- Section 7 — Cross-Scale Consistency
-- The coherence function C(r) provides a unified measure across all turbulence
-- scales.  The cross-scale consistency results show that the kernel scale r = 1
-- is the unique global maximum of coherence, and that turbulence at all other
-- scales (micro, meso with r > 1, macro) has strictly lower coherence.
-- This is the Kernel turbulence theorem: the balanced fixed point r = 1
-- (laminar flow, the μ-eigenstate orbit) is the unique state of maximum order.
-- ════════════════════════════════════════════════════════════════════════════

/-- Micro-scale turbulence has strictly lower coherence than the kernel scale:
    turbulenceCoherence(η) < turbulenceCoherence(1) for all η ∈ microScaleDomain.

    Small-scale disordered eddies are strictly less coherent than the balanced
    laminar state.  This is the formal statement that turbulence at micro scales
    represents a departure from maximum coherence. -/
theorem turbulence_micro_below_kernel (η : ℝ) (hη : η ∈ microScaleDomain) :
    turbulenceCoherence η < turbulenceCoherence 1 := by
  rw [turbulenceCoherence_kernel_max]
  exact turbulenceCoherence_micro_lt_one η hη

/-- Macro-scale turbulence has strictly lower coherence than the kernel scale:
    turbulenceCoherence(L) < turbulenceCoherence(1) for all L ∈ macroScaleDomain.

    Large integral-scale eddies are also strictly less coherent than the
    balanced laminar state, from the other side of the scale spectrum. -/
theorem turbulence_macro_below_kernel (L : ℝ) (hL : L ∈ macroScaleDomain) :
    turbulenceCoherence L < turbulenceCoherence 1 := by
  rw [turbulenceCoherence_kernel_max]
  exact turbulenceCoherence_macro_lt_one L hL

/-- Universal coherence bound: the kernel scale (r = 1) is the global maximum.

    For any amplitude ratio r ≥ 0, turbulenceCoherence(r) ≤ turbulenceCoherence(1) = 1.
    This is the master cross-scale consistency theorem:
    no turbulence scale can achieve higher coherence than the laminar state r = 1,
    and the coherence at every scale is bounded by the Kernel maximum C(1) = 1. -/
theorem turbulence_coherence_universal_bound (r : ℝ) (hr : 0 ≤ r) :
    turbulenceCoherence r ≤ turbulenceCoherence 1 := by
  unfold turbulenceCoherence
  rw [(coherence_eq_one_iff 1 (le_of_lt one_pos)).mpr rfl]
  exact coherence_le_one r hr

end -- noncomputable section
