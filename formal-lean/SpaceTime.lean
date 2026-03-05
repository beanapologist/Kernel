/-
  SpaceTime.lean — Lean 4 formalization unifying time and space into a notion
  of reality, consistent with the TimeCrystal Floquet framework.

  Time is represented as the negative real numbers (the retarded/causal
  direction in the Schrödinger picture).  Space is represented as coordinates
  on the positive imaginary axis: a spatial position s > 0 corresponds to the
  complex point i·s.  Their unification into the complex number

      reality(s, t) = t + i·s

  gives the observer's spacetime coordinate in a unified complex plane, where
  the real axis carries (negative) time and the imaginary axis carries
  (positive) space.

  This file formalizes:
    1.  Time domain  — the set of negative real numbers
    2.  Space domain — the set of positive real numbers (imaginary axis magnitudes)
    3.  The reality function reality(s, t) = t + i·s
    4.  Reality-grounded time crystal states
    5.  Consistency with the TimeCrystal Floquet framework
        (periodicity, symmetry breaking, quasi-energy)
    6.  The positive imaginary axis as space — the spatial embedding
        iSpace s = i·s, its geometry, algebra, and Floquet properties

  The key consistency results show that any reality-grounded time crystal state
  inherits the full suite of Floquet-theory guarantees proved in TimeCrystal.lean:
  period doubling, discrete time-translation symmetry breaking, norm invariance
  across periods, and the quasi-energy identity ε_F · T = π.

  Sections
  ────────
  1.  Time and space domains
  2.  The reality function and its properties
  3.  Reality-grounded time crystal states
  4.  Quasi-energy and Floquet structure in the reality framework
  5.  The observer's reality as a canonical map F(s, t) = t + i·s
  6.  The positive imaginary axis as space

  Proof status
  ────────────
  All theorems have complete machine-checked proofs.
  No `sorry` placeholders remain.
-/

import TimeCrystal

open Complex Real

noncomputable section

-- ════════════════════════════════════════════════════════════════════════════
-- Section 1 — Time and Space Domains
-- Time is the negative real numbers; space is the positive imaginary axis.
-- ════════════════════════════════════════════════════════════════════════════

/-- The time domain: the set of negative real numbers.

    Time coordinates t < 0 lie on the negative real axis.  This reflects the
    retarded (causal) propagation direction: the observer's past corresponds
    to increasingly negative values of t. -/
def timeDomain : Set ℝ := {t | t < 0}

/-- The space domain: the set of positive real numbers, representing
    magnitudes of coordinates on the positive imaginary axis.

    A spatial position s > 0 corresponds to the complex point i·s.
    The imaginary axis separates space (positive imaginary) from time
    (negative real) in the unified complex plane. -/
def spaceDomain : Set ℝ := {s | 0 < s}

/-- The time domain is non-empty: −1 is a valid time coordinate. -/
theorem timeDomain_nonempty : (-1 : ℝ) ∈ timeDomain := by
  show (-1 : ℝ) < 0
  norm_num

/-- The space domain is non-empty: 1 is a valid spatial coordinate. -/
theorem spaceDomain_nonempty : (1 : ℝ) ∈ spaceDomain := by
  show (0 : ℝ) < 1
  norm_num

-- ════════════════════════════════════════════════════════════════════════════
-- Section 2 — The Reality Function
-- reality(s, t) = t + i·s unifies time (real part) and space (imaginary part).
-- ════════════════════════════════════════════════════════════════════════════

/-- The observer's reality: the complex number combining time (real part,
    negative) and space (imaginary part, positive) into a unified spacetime
    coordinate.

        reality s t  =  t + i·s

    For t ∈ timeDomain and s ∈ spaceDomain the point reality s t lies in
    the second quadrant of the complex plane: Re < 0 and Im > 0.

    The real axis encodes time (negative = past) and the imaginary axis
    encodes space (positive = physical extent). -/
def reality (s t : ℝ) : ℂ := ↑t + Complex.I * ↑s

/-- The real part of reality is the time coordinate. -/
theorem reality_re (s t : ℝ) : (reality s t).re = t := by
  simp [reality, Complex.add_re, Complex.mul_re, Complex.I_re, Complex.I_im,
        Complex.ofReal_re, Complex.ofReal_im]

/-- The imaginary part of reality is the space coordinate. -/
theorem reality_im (s t : ℝ) : (reality s t).im = s := by
  simp [reality, Complex.add_im, Complex.mul_im, Complex.I_re, Complex.I_im,
        Complex.ofReal_re, Complex.ofReal_im]

/-- For a time coordinate in the time domain, the real part of reality is
    negative: the observer's position encodes time correctly. -/
theorem reality_time_negative (s t : ℝ) (ht : t ∈ timeDomain) :
    (reality s t).re < 0 := by
  rw [reality_re]
  exact ht

/-- For a space coordinate in the space domain, the imaginary part of reality
    is positive: the observer's position encodes space correctly. -/
theorem reality_space_positive (s t : ℝ) (hs : s ∈ spaceDomain) :
    0 < (reality s t).im := by
  rw [reality_im]
  exact hs

/-- The time evolution operator evaluated at the time coordinate of reality
    is unitary: |U(H, Re(reality s t))| = 1.

    The observer's spacetime position does not alter the unitarity of the
    Schrödinger time evolution: |exp(−iHt)| = 1 regardless of t. -/
theorem reality_timeEvolution_unitary (H s t : ℝ) :
    Complex.abs (timeEvolution H (reality s t).re) = 1 := by
  rw [reality_re]
  exact timeEvolution_abs_one H t

/-- The Floquet phase factor at the spatial coordinate s has unit modulus:
    |e^{−is}| = 1.

    The space coordinate appears in the Floquet phase purely as a phase
    rotation; it does not change the amplitude. -/
theorem reality_floquetPhase_unit (s : ℝ) :
    Complex.abs (floquetPhase s) = 1 :=
  floquetPhase_abs_one s

-- ════════════════════════════════════════════════════════════════════════════
-- Section 3 — Reality-Grounded Time Crystal States
-- A time crystal whose initial data lies in the reality framework.
-- ════════════════════════════════════════════════════════════════════════════

/-- A reality-grounded time crystal state is a time crystal state (Floquet
    phase π, drive period T) whose spatial reference s is in the space domain
    and whose temporal reference t₀ is in the time domain.

    This captures the physical requirement that the observer's reality consists
    of a positive spatial extent s > 0 and a past temporal origin t₀ < 0. -/
def isRealityTimeCrystalState (ψ : ℝ → ℂ) (T s t₀ : ℝ) : Prop :=
  isTimeCrystalState ψ T ∧ s ∈ spaceDomain ∧ t₀ ∈ timeDomain

/-- A reality-grounded time crystal state is a Floquet state with phase π.

    The Floquet phase π (sign flip per period) is the hallmark of the
    time crystal, inherited directly from the isTimeCrystalState definition. -/
theorem realityTC_is_floquet (ψ : ℝ → ℂ) (T s t₀ : ℝ)
    (h : isRealityTimeCrystalState ψ T s t₀) :
    isFloquetState ψ T Real.pi :=
  h.1

/-- A reality-grounded time crystal state is 2T-periodic: ψ(t + 2T) = ψ(t).

    The period doubling follows from two applications of the Floquet step
    with phase π: (e^{−iπ})² = (−1)² = 1. -/
theorem realityTC_period_double (ψ : ℝ → ℂ) (T s t₀ : ℝ)
    (h : isRealityTimeCrystalState ψ T s t₀) :
    ∀ t : ℝ, ψ (t + 2 * T) = ψ t :=
  timeCrystal_period_double ψ T h.1

/-- A non-trivial reality-grounded time crystal state breaks discrete
    time-translation symmetry: it is NOT T-periodic but IS 2T-periodic.

    The observer's reality (s ∈ spaceDomain, t₀ ∈ timeDomain) frames a
    system whose symmetry-breaking is a physical fact, not merely formal. -/
theorem realityTC_breaks_symmetry (ψ : ℝ → ℂ) (T s t₀ : ℝ)
    (h : isRealityTimeCrystalState ψ T s t₀) (hψ : ∃ t, ψ t ≠ 0) :
    breaksDiscreteTimeTranslationSymmetry ψ T :=
  timeCrystalState_breaks_symmetry ψ T h.1 hψ

/-- Iterated Floquet evolution of a reality-grounded time crystal state:
        ψ(t + n·T) = (−1)ⁿ · ψ(t)   for all n ∈ ℕ and all t ∈ ℝ.

    Each period application multiplies by the Floquet factor e^{−iπ} = −1,
    so n periods give the n-th power (−1)ⁿ. -/
theorem realityTC_iterated (ψ : ℝ → ℂ) (T s t₀ : ℝ)
    (h : isRealityTimeCrystalState ψ T s t₀) :
    ∀ (n : ℕ) (t : ℝ), ψ (t + ↑n * T) = floquetPhase Real.pi ^ n * ψ t :=
  floquet_iterated ψ T Real.pi h.1

/-- The norm of a reality-grounded time crystal state is preserved each
    period: |ψ(t + T)| = |ψ(t)| for all t.

    The Floquet phase factor e^{−iπ} = −1 has modulus 1, so each period
    advance is an isometry.  The observer's probability amplitude is stable. -/
theorem realityTC_norm_invariant (ψ : ℝ → ℂ) (T s t₀ : ℝ)
    (h : isRealityTimeCrystalState ψ T s t₀) :
    ∀ t : ℝ, Complex.abs (ψ (t + T)) = Complex.abs (ψ t) :=
  floquet_norm_invariant ψ T Real.pi h.1

/-- The norm is a full dynamical invariant over n periods:
        |ψ(t + n·T)| = |ψ(t)|   for all n ∈ ℕ and all t ∈ ℝ.

    The observer's reality does not change the probability amplitude across
    any number of drive periods. -/
theorem realityTC_norm_n_invariant (ψ : ℝ → ℂ) (T s t₀ : ℝ)
    (h : isRealityTimeCrystalState ψ T s t₀) :
    ∀ (n : ℕ) (t : ℝ), Complex.abs (ψ (t + ↑n * T)) = Complex.abs (ψ t) :=
  floquet_norm_dynamical_invariant ψ T Real.pi h.1

-- ════════════════════════════════════════════════════════════════════════════
-- Section 4 — Quasi-Energy and Floquet Structure in the Reality Framework
-- ════════════════════════════════════════════════════════════════════════════

/-- The quasi-energy of a reality-grounded time crystal with drive period T:
        ε_F · T = π.

    The Floquet phase φ = π corresponds to quasi-energy ε_F = π/T (setting
    ℏ = 1).  This identity holds regardless of the observer's spatial or
    temporal reference coordinates. -/
theorem realityTC_quasi_energy (T : ℝ) (hT : T ≠ 0) :
    timeCrystalQuasiEnergy T hT * T = Real.pi :=
  timeCrystalQuasiEnergy_phase T hT

/-- The Floquet phase evaluated at the quasi-energy round-trip equals the
    time crystal phase:
        e^{−i · (ε_F · T)} = e^{−iπ} = −1.

    The quasi-energy encodes exactly the π phase that drives period doubling
    in the reality framework. -/
theorem realityTC_floquet_at_quasi_energy (T : ℝ) (hT : T ≠ 0) :
    floquetPhase (timeCrystalQuasiEnergy T hT * T) = floquetPhase Real.pi := by
  rw [realityTC_quasi_energy T hT]

/-- The response period 2T strictly exceeds the drive period T for any
    positive T > 0.  In the reality framework this is the quantitative
    signature of period doubling: the observer's spacetime repeats at 2T,
    not T. -/
theorem realityTC_period_doubling_strict (T : ℝ) (hT : 0 < T) : T < 2 * T :=
  timeCrystal_period_doubling_strict T hT

/-- The period-doubling ratio in the reality framework: 2T / T = 2.

    The observer's reality repeats at exactly twice the drive period,
    regardless of the spatial or temporal reference coordinates. -/
theorem realityTC_period_ratio (T : ℝ) (hT : T ≠ 0) : 2 * T / T = 2 :=
  timeCrystal_period_ratio T hT

/-- The time crystal phase e^{−iπ} is distinct from the drive-synchronized
    phase e^{−i·0} = 1: the reality-grounded crystal and its drive are out
    of phase, confirming symmetry breaking. -/
theorem realityTC_phase_not_sync :
    floquetPhase Real.pi ≠ floquetPhase 0 :=
  timeCrystal_phase_not_sync

-- ════════════════════════════════════════════════════════════════════════════
-- Section 5 — The Observer's Reality as a Canonical Map
-- F(s, t) is THE function that assigns a unique complex number to every
-- spacetime point (s, t), formalizing what it means for an observer to
-- occupy a definite position in space and a definite moment in time.
-- ════════════════════════════════════════════════════════════════════════════

/-- The canonical observer reality map.

    For a spacetime point given by spatial coordinate s and time coordinate t,
    the observer's reality is the complex number

        F(s, t) = t + i·s

    The real axis carries time (negative = past) and the imaginary axis
    carries space (positive = physical extent).  Every distinct spacetime
    point (s, t) maps to a unique complex number, so F completely and
    faithfully encodes the observer's position in spacetime. -/
def F (s t : ℝ) : ℂ := reality s t

/-- F agrees with the reality function: F(s, t) = reality s t. -/
theorem F_eq_reality (s t : ℝ) : F s t = reality s t := rfl

/-- The time coordinate is recovered from F as its real part:
        Re(F(s, t)) = t.
    The real axis of the complex plane is the time axis. -/
theorem F_re (s t : ℝ) : (F s t).re = t := reality_re s t

/-- The space coordinate is recovered from F as its imaginary part:
        Im(F(s, t)) = s.
    The imaginary axis of the complex plane is the space axis. -/
theorem F_im (s t : ℝ) : (F s t).im = s := reality_im s t

/-- The observer's reality map F is injective: distinct spacetime points
    map to distinct complex numbers.

        F(s₁, t₁) = F(s₂, t₂)  →  s₁ = s₂  ∧  t₁ = t₂

    Proof: equality of complex numbers implies equality of real and
    imaginary parts.  The real parts give t₁ = t₂ and the imaginary parts
    give s₁ = s₂. -/
theorem F_injective (s₁ t₁ s₂ t₂ : ℝ) (h : F s₁ t₁ = F s₂ t₂) :
    s₁ = s₂ ∧ t₁ = t₂ := by
  have hre : (F s₁ t₁).re = (F s₂ t₂).re := by rw [h]
  have him : (F s₁ t₁).im = (F s₂ t₂).im := by rw [h]
  rw [F_re, F_re] at hre
  rw [F_im, F_im] at him
  exact ⟨him, hre⟩

/-- When the observer occupies a valid spacetime position
    (s ∈ spaceDomain, t ∈ timeDomain) their reality F(s, t) lies strictly
    in the second quadrant of the complex plane:
        Re(F(s, t)) < 0   and   Im(F(s, t)) > 0.

    The second quadrant is the "physical reality" region: negative time
    (causal past) and positive space (physical extent) coexist. -/
theorem F_second_quadrant (s t : ℝ) (hs : s ∈ spaceDomain) (ht : t ∈ timeDomain) :
    (F s t).re < 0 ∧ 0 < (F s t).im := by
  exact ⟨reality_time_negative s t ht, reality_space_positive s t hs⟩

/-- The modulus of F encodes the observer's distance from the origin of
    spacetime.  It is always non-negative, and equals zero only when both
    s = 0 and t = 0 (the spacetime origin).

        |F(s, t)| = 0  ↔  s = 0 ∧ t = 0 -/
theorem F_abs_eq_zero_iff (s t : ℝ) : Complex.abs (F s t) = 0 ↔ s = 0 ∧ t = 0 := by
  rw [Complex.abs.eq_zero]
  constructor
  · intro h
    have hre : (F s t).re = 0 := by rw [h]; simp
    have him : (F s t).im = 0 := by rw [h]; simp
    rw [F_re] at hre
    rw [F_im] at him
    exact ⟨him, hre⟩
  · intro ⟨hs, ht⟩
    simp [F, reality, hs, ht]

/-- The time evolution operator at the observer's time coordinate is
    unitary: |U(H, Re(F(s, t)))| = 1.

    The quantum evolution law is independent of the observer's position in
    spacetime — the Schrödinger dynamics are the same for every observer. -/
theorem F_timeEvolution_unitary (H s t : ℝ) :
    Complex.abs (timeEvolution H (F s t).re) = 1 := by
  rw [F_re]
  exact timeEvolution_abs_one H t

/-- The Floquet phase at the observer's spatial coordinate is a pure phase:
        |e^{−i·Im(F(s, t))}| = 1.

    Spatial position contributes only a phase to Floquet evolution; it does
    not change the amplitude of any state. -/
theorem F_floquetPhase_unit (s t : ℝ) :
    Complex.abs (floquetPhase (F s t).im) = 1 := by
  rw [F_im]
  exact floquetPhase_abs_one s

-- ════════════════════════════════════════════════════════════════════════════
-- Section 6 — The Positive Imaginary Axis as Space
-- The spatial embedding iSpace s = i·s places every physical coordinate s > 0
-- on the positive imaginary axis of ℂ, the unique ray orthogonal to the time
-- axis.  This section develops the geometry and algebra of that ray and
-- connects it to the Floquet phase structure.
-- ════════════════════════════════════════════════════════════════════════════

/-- The positive imaginary axis in ℂ: complex numbers with zero real part and
    strictly positive imaginary part.

    This is the image of spaceDomain under the spatial embedding s ↦ i·s.
    It is the unique ray in ℂ that is orthogonal to the negative real axis
    (the time axis) and carries only positive imaginary values. -/
def posImagAxis : Set ℂ := {z | z.re = 0 ∧ 0 < z.im}

/-- The spatial embedding: every physical spatial coordinate s > 0 is placed
    on the positive imaginary axis as the complex number i·s.

        iSpace s  =  i · s  ∈  ℂ

    Multiplying by i is a 90° rotation in ℂ: it carries the positive real line
    (raw spatial magnitudes) onto the positive imaginary axis, making space and
    time maximally orthogonal within the complex plane. -/
def iSpace (s : ℝ) : ℂ := Complex.I * ↑s

/-- The real part of the spatial embedding is zero: i·s has no time component.
    Space and time do not interfere in the real direction. -/
theorem iSpace_re (s : ℝ) : (iSpace s).re = 0 := by
  simp [iSpace, Complex.mul_re, Complex.I_re, Complex.I_im,
        Complex.ofReal_re, Complex.ofReal_im]

/-- The imaginary part of the spatial embedding equals the coordinate: Im(i·s) = s.
    The imaginary axis carries the spatial coordinate exactly and faithfully. -/
theorem iSpace_im (s : ℝ) : (iSpace s).im = s := by
  simp [iSpace, Complex.mul_im, Complex.I_re, Complex.I_im,
        Complex.ofReal_re, Complex.ofReal_im]

/-- The modulus of the spatial embedding equals the norm of the coordinate:
        |i·s| = ‖s‖.
    Multiplying by i is an isometry — it preserves lengths. -/
theorem iSpace_abs (s : ℝ) : Complex.abs (iSpace s) = ‖s‖ := by
  simp only [iSpace, map_mul, Complex.abs_I, Complex.abs_ofReal, one_mul, Real.norm_eq_abs]

/-- For a spatial coordinate s ∈ spaceDomain (s > 0), the modulus of the
    spatial embedding equals s itself: |i·s| = s. -/
theorem iSpace_abs_pos (s : ℝ) (hs : s ∈ spaceDomain) :
    Complex.abs (iSpace s) = s := by
  rw [iSpace_abs, Real.norm_eq_abs, abs_of_pos hs]

/-- The spatial embedding preserves positivity: s ∈ spaceDomain → Im(i·s) > 0.
    Every physical spatial coordinate maps to the strictly positive imaginary
    half-axis. -/
theorem iSpace_pos_im (s : ℝ) (hs : s ∈ spaceDomain) : 0 < (iSpace s).im := by
  rw [iSpace_im]; exact hs

/-- Every spatial coordinate maps into the positive imaginary axis:
        s ∈ spaceDomain  →  i·s ∈ posImagAxis. -/
theorem iSpace_mem_posImagAxis (s : ℝ) (hs : s ∈ spaceDomain) :
    iSpace s ∈ posImagAxis :=
  ⟨iSpace_re s, iSpace_pos_im s hs⟩

/-- The spatial embedding is injective: i·s₁ = i·s₂ → s₁ = s₂.
    Distinct spatial positions map to distinct points on the imaginary axis. -/
theorem iSpace_injective (s₁ s₂ : ℝ) (h : iSpace s₁ = iSpace s₂) : s₁ = s₂ := by
  have him : (iSpace s₁).im = (iSpace s₂).im := by rw [h]
  rwa [iSpace_im, iSpace_im] at him

/-- The spatial embedding is additive: i·(s₁ + s₂) = i·s₁ + i·s₂.
    Two spatial coordinates combine linearly under the embedding. -/
theorem iSpace_add (s₁ s₂ : ℝ) : iSpace (s₁ + s₂) = iSpace s₁ + iSpace s₂ := by
  simp [iSpace, ofReal_add, mul_add]

/-- The spatial embedding respects real scaling: i·(r·s) = r·(i·s).
    Scaling a spatial coordinate scales its embedded image by the same factor. -/
theorem iSpace_smul (r s : ℝ) : iSpace (r * s) = ↑r * iSpace s := by
  unfold iSpace; push_cast; ring

/-- The space domain is closed under addition: s₁, s₂ > 0 → s₁ + s₂ > 0.
    Combining two physical spatial extents gives a physical spatial extent. -/
theorem spaceDomain_add (s₁ s₂ : ℝ) (h₁ : s₁ ∈ spaceDomain) (h₂ : s₂ ∈ spaceDomain) :
    s₁ + s₂ ∈ spaceDomain :=
  add_pos h₁ h₂

/-- The space domain is closed under positive scaling: r > 0, s > 0 → r·s > 0.
    Scaling a physical spatial extent by a positive factor stays physical. -/
theorem spaceDomain_smul (r s : ℝ) (hr : 0 < r) (hs : s ∈ spaceDomain) :
    r * s ∈ spaceDomain :=
  mul_pos hr hs

/-- Every element of the space domain is non-zero: s ∈ spaceDomain → s ≠ 0. -/
theorem spaceDomain_ne_zero (s : ℝ) (hs : s ∈ spaceDomain) : s ≠ 0 :=
  hs.ne'

/-- The spatial embedding of a spatial coordinate is non-zero:
        s ∈ spaceDomain  →  i·s ≠ 0.
    Physical spatial position is never the zero complex number. -/
theorem iSpace_ne_zero (s : ℝ) (hs : s ∈ spaceDomain) : iSpace s ≠ 0 := by
  intro h
  have : (iSpace s).im = 0 := by rw [h]; simp
  rw [iSpace_im] at this
  exact absurd this hs.ne'

/-- The observer's reality decomposes into independent time and space parts:
        F(s, t)  =  ↑t  +  iSpace s.

    The real part ↑t is the pure time contribution; the imaginary part iSpace s
    is the pure space contribution.  They sum without cross-contamination. -/
theorem F_decomp (s t : ℝ) : F s t = ↑t + iSpace s := by
  simp [F, reality, iSpace]

/-- Time and space are orthogonal within F: the space component has zero real
    part and the time component has zero imaginary part.

        Re(iSpace s) = 0   and   Im(↑t) = 0.

    The two axes of observer reality are mutually transparent to each other. -/
theorem space_time_orthogonal (s t : ℝ) :
    (iSpace s).re = 0 ∧ ((↑t : ℂ)).im = 0 :=
  ⟨iSpace_re s, Complex.ofReal_im t⟩

/-- The Floquet phase factor at the imaginary part of the spatial embedding
    has unit modulus: |e^{−i·Im(i·s)}| = |e^{−is}| = 1.

    Space enters Floquet dynamics entirely as a pure phase rotation.  The
    observer's position in space never changes the amplitude of any state. -/
theorem iSpace_floquetPhase_unit (s : ℝ) :
    Complex.abs (floquetPhase (iSpace s).im) = 1 := by
  rw [iSpace_im]
  exact floquetPhase_abs_one s

end
