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

end
