/-
  TimeCrystal.lean — Lean 4 formalization of discrete time crystal theory.

  Time crystals are quantum systems that exhibit periodic oscillation in
  time at energetically stable states, breaking discrete time-translation
  symmetry.  The key mechanism is Floquet driving: a Hamiltonian periodic
  with period T can induce a ground state with period 2T (period doubling).

  This file formalizes:
    1.  Time evolution operator    U(H,t) = exp(−I·H·t)
    2.  Floquet phase factor        e^{−iφ}
    3.  Floquet theorem             ψ(t+T) = e^{−iφ}·ψ(t) and its iterates
    4.  Time crystal states         period-doubling via Floquet phase φ = π
    5.  Symmetry-breaking criteria  formal conditions distinguishing T from 2T
    6.  Quasi-energy                the half-drive-frequency eigenvalue π/T

  Mathematical background:
    Wilczek, F. (2012). Quantum Time Crystals. Phys. Rev. Lett. 109, 160401.
    Khemani et al. (2016). Phase structure of driven quantum systems.
    Sacha & Zakrzewski (2018). Time crystals: a review. Rep. Prog. Phys.

  Sections
  ────────
  1.  Time evolution operator  U(H,t) = exp(−I·H·t)
  2.  Floquet phase factor  e^{−iφ}
  3.  Floquet states: ψ(t+T) = e^{−iφ}·ψ(t) and iterated form
  4.  Time crystal states  (Floquet phase = π → period doubling)
  5.  Discrete time-translation symmetry breaking
  6.  Quasi-energy and period-doubling ratio

  Proof status
  ────────────
  All theorems have complete machine-checked proofs.
  No `sorry` placeholders remain.
-/

import Mathlib.Analysis.SpecialFunctions.Complex.Circle
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Data.Complex.Exponential

open Complex Real

noncomputable section

-- ════════════════════════════════════════════════════════════════════════════
-- Section 1 — Time Evolution Operator
-- The Schrödinger-picture evolution operator for a scalar Hamiltonian.
-- ════════════════════════════════════════════════════════════════════════════

/-- The time evolution operator U(H, t) = exp(−I · H · t) for real Hamiltonian
    parameter H (energy in units where ℏ = 1) and time t.

    In the Schrödinger picture the state evolves as |ψ(t)⟩ = U(t)|ψ(0)⟩.
    Here H is treated as a real scalar (single-mode / one-dimensional case). -/
def timeEvolution (H t : ℝ) : ℂ := Complex.exp (-(Complex.I * ↑H * ↑t))

/-- The time evolution operator at t = 0 is the identity: no evolution occurs
    before any time has passed. -/
theorem timeEvolution_zero (H : ℝ) : timeEvolution H 0 = 1 := by
  unfold timeEvolution
  simp

/-- The time evolution operator is unitary: |U(H, t)| = 1 for all H and t.

    Proof: U = exp(−I·H·t) lies on the unit circle since
    |exp z| = exp(Re z) and Re(−I·H·t) = 0. -/
theorem timeEvolution_abs_one (H t : ℝ) : Complex.abs (timeEvolution H t) = 1 := by
  unfold timeEvolution
  rw [map_exp, Complex.abs_exp]
  simp [Complex.neg_re, Complex.mul_re, Complex.I_re, Complex.I_im]

/-- The time evolution operator satisfies the group law: U(t + s) = U(t) · U(s).

    Proof: exp(−I·H·(t+s)) = exp(−I·H·t + (−I·H·s))
                            = exp(−I·H·t) · exp(−I·H·s). -/
theorem timeEvolution_add (H t s : ℝ) :
    timeEvolution H (t + s) = timeEvolution H t * timeEvolution H s := by
  unfold timeEvolution
  rw [← Complex.exp_add]
  congr 1
  push_cast
  ring

-- ════════════════════════════════════════════════════════════════════════════
-- Section 2 — Floquet Phase Factor
-- The phase acquired by a Floquet quasi-energy state per period T.
-- ════════════════════════════════════════════════════════════════════════════

/-- The Floquet phase factor e^{−iφ} for quasi-energy phase φ.

    In Floquet theory a periodically driven system has quasi-energy states
    that accumulate a phase e^{−iφ} each period T.  The quasi-energy is
    ε_F = φ/T (choosing ℏ = 1). -/
def floquetPhase (φ : ℝ) : ℂ := Complex.exp (-(Complex.I * ↑φ))

/-- The Floquet phase factor lies on the unit circle: |e^{−iφ}| = 1. -/
theorem floquetPhase_abs_one (φ : ℝ) : Complex.abs (floquetPhase φ) = 1 := by
  unfold floquetPhase
  rw [map_exp, Complex.abs_exp]
  simp [Complex.neg_re, Complex.mul_re, Complex.I_re, Complex.I_im]

/-- Floquet phase factors compose: e^{−i(φ₁+φ₂)} = e^{−iφ₁} · e^{−iφ₂}. -/
theorem floquetPhase_add (φ₁ φ₂ : ℝ) :
    floquetPhase (φ₁ + φ₂) = floquetPhase φ₁ * floquetPhase φ₂ := by
  unfold floquetPhase
  rw [← Complex.exp_add]
  congr 1
  push_cast
  ring

/-- The Floquet phase at φ = 0 is the identity (no phase shift). -/
theorem floquetPhase_zero : floquetPhase 0 = 1 := by
  unfold floquetPhase
  simp

/-- The Floquet phase at φ = 2π equals 1 (full rotation = identity).
    This corresponds to a state with integer quasi-energy. -/
theorem floquetPhase_two_pi : floquetPhase (2 * Real.pi) = 1 := by
  unfold floquetPhase
  rw [show -(Complex.I * ↑(2 * Real.pi)) = -(2 * ↑Real.pi * Complex.I) by push_cast; ring]
  rw [Complex.exp_neg, Complex.exp_two_pi_mul_I]
  simp

/-- The Floquet phase at φ = π equals −1 (half rotation = sign flip).

    This is the hallmark of a period-doubling time crystal: the state
    changes sign each period T, returning to itself only after 2T. -/
theorem floquetPhase_pi : floquetPhase Real.pi = -1 := by
  unfold floquetPhase
  rw [show -(Complex.I * ↑Real.pi) = -(↑Real.pi * Complex.I) by ring]
  rw [Complex.exp_neg, Complex.exp_pi_mul_I]
  norm_num

/-- The Floquet phase at φ = π squares to 1: (e^{−iπ})² = e^{−2iπ} = 1.
    This is the algebraic signature of period doubling at the Floquet level. -/
theorem floquetPhase_pi_sq : floquetPhase Real.pi ^ 2 = 1 := by
  rw [floquetPhase_pi]
  norm_num

-- ════════════════════════════════════════════════════════════════════════════
-- Section 3 — Floquet States
-- The Floquet theorem: ψ(t+T) = e^{−iφ}·ψ(t) and derived invariants.
-- ════════════════════════════════════════════════════════════════════════════

/-- A state ψ : ℝ → ℂ is a Floquet state with period T and quasi-phase φ if
    it advances by the Floquet factor each period:
        ψ(t + T) = e^{−iφ} · ψ(t)   for all t ∈ ℝ.

    This is the Floquet theorem for the Schrödinger equation with a
    T-periodic Hamiltonian. -/
def isFloquetState (ψ : ℝ → ℂ) (T φ : ℝ) : Prop :=
  ∀ t : ℝ, ψ (t + T) = floquetPhase φ * ψ t

/-- Iterated Floquet theorem: after n periods the state accumulates n copies
    of the Floquet phase.

        ψ(t + n·T) = e^{−i·n·φ} · ψ(t)

    Proof by induction on n.  Base case: n = 0 gives ψ(t) = 1·ψ(t).
    Inductive step uses the Floquet condition for one additional period. -/
theorem floquet_iterated (ψ : ℝ → ℂ) (T φ : ℝ) (h : isFloquetState ψ T φ) :
    ∀ (n : ℕ) (t : ℝ), ψ (t + ↑n * T) = floquetPhase φ ^ n * ψ t := by
  intro n
  induction n with
  | zero => intro t; simp
  | succ n ih =>
    intro t
    have step : t + (↑(n + 1) : ℝ) * T = (t + ↑n * T) + T := by push_cast; ring
    rw [step, h (t + ↑n * T), ih t]
    -- (e^{−iφ})^n · e^{−iφ} = (e^{−iφ})^(n+1)
    ring

/-- A Floquet state has constant norm per period: |ψ(t + T)| = |ψ(t)|.

    The Floquet phase factor has modulus 1, so each period advance is an
    isometry.  The norm is therefore a period-1 dynamical invariant. -/
theorem floquet_norm_invariant (ψ : ℝ → ℂ) (T φ : ℝ) (h : isFloquetState ψ T φ) :
    ∀ t : ℝ, Complex.abs (ψ (t + T)) = Complex.abs (ψ t) := by
  intro t
  rw [h t, map_mul, floquetPhase_abs_one, one_mul]

/-- The norm is a full dynamical invariant: |ψ(t + n·T)| = |ψ(t)| for all n.

    Proof: combine `floquet_iterated` (accumulates (e^{−iφ})^n) with
    |(e^{−iφ})^n| = |e^{−iφ}|^n = 1^n = 1. -/
theorem floquet_norm_dynamical_invariant (ψ : ℝ → ℂ) (T φ : ℝ)
    (h : isFloquetState ψ T φ) :
    ∀ (n : ℕ) (t : ℝ), Complex.abs (ψ (t + ↑n * T)) = Complex.abs (ψ t) := by
  intro n t
  rw [floquet_iterated ψ T φ h n t, map_mul, map_pow, floquetPhase_abs_one, one_pow, one_mul]

-- ════════════════════════════════════════════════════════════════════════════
-- Section 4 — Time Crystal States
-- Period-doubling: Floquet phase φ = π causes ψ(t+2T) = ψ(t).
-- ════════════════════════════════════════════════════════════════════════════

/-- A time crystal state with drive period T has Floquet phase π:
        ψ(t + T) = −ψ(t)   for all t.

    The sign flip per period T means the state returns to itself only after
    2T — the hallmark of a discrete time crystal breaking T-translation
    symmetry while exhibiting 2T-periodicity. -/
def isTimeCrystalState (ψ : ℝ → ℂ) (T : ℝ) : Prop :=
  isFloquetState ψ T Real.pi

/-- A time crystal state is 2T-periodic: ψ(t + 2T) = ψ(t).

    Proof: two applications of the Floquet step with φ = π give
    ψ(t + 2T) = (e^{−iπ})² · ψ(t) = (−1)² · ψ(t) = ψ(t). -/
theorem timeCrystal_period_double (ψ : ℝ → ℂ) (T : ℝ)
    (h : isTimeCrystalState ψ T) : ∀ t : ℝ, ψ (t + 2 * T) = ψ t := by
  intro t
  have step : t + 2 * T = (t + T) + T := by ring
  rw [step, h (t + T), h t, floquetPhase_pi]
  ring

/-- Time-translation symmetry is broken: the drive period T differs from the
    response period 2T whenever T ≠ 0.

    Proof: T = 2T implies T = 0 by algebra, contradicting T ≠ 0. -/
theorem timeCrystal_symmetry_breaking (T : ℝ) (hT : T ≠ 0) : T ≠ 2 * T := by
  intro heq
  have : T = 0 := by linarith
  exact hT this

/-- A time crystal state with ψ(t₀) ≠ 0 is strictly not T-periodic at t₀:
        ψ(t₀ + T) ≠ ψ(t₀).

    Proof: ψ(t₀ + T) = −ψ(t₀).  If −ψ(t₀) = ψ(t₀) then 2·ψ(t₀) = 0,
    so ψ(t₀) = 0 (since char ℂ = 0), contradicting the hypothesis. -/
theorem timeCrystal_not_T_periodic (ψ : ℝ → ℂ) (T t₀ : ℝ)
    (h : isTimeCrystalState ψ T) (hψ : ψ t₀ ≠ 0) : ψ (t₀ + T) ≠ ψ t₀ := by
  rw [h t₀, floquetPhase_pi]
  intro heq
  apply hψ
  -- From −ψ(t₀) = ψ(t₀) we derive 2·ψ(t₀) = 0, hence ψ(t₀) = 0 (char ℂ = 0)
  have h2 : (-2 : ℂ) * ψ t₀ = 0 := by linear_combination heq
  exact (mul_eq_zero.mp h2).resolve_left (by norm_num)

-- ════════════════════════════════════════════════════════════════════════════
-- Section 5 — Discrete Time-Translation Symmetry Breaking
-- Formal conditions characterising the time crystal phase.
-- ════════════════════════════════════════════════════════════════════════════

/-- Discrete time-translation symmetry breaking at period T.

    A state ψ spontaneously breaks T-translation symmetry when:
    (1) it is NOT T-periodic: ∃ t, ψ(t+T) ≠ ψ(t)
    (2) it IS 2T-periodic:  ∀ t, ψ(t+2T) = ψ(t)

    This captures the defining property of a discrete time crystal:
    the response period (2T) is twice the drive period (T). -/
def breaksDiscreteTimeTranslationSymmetry (ψ : ℝ → ℂ) (T : ℝ) : Prop :=
  (∃ t : ℝ, ψ (t + T) ≠ ψ t) ∧ (∀ t : ℝ, ψ (t + 2 * T) = ψ t)

/-- Every non-trivial time crystal state satisfies the symmetry-breaking
    criterion: it is not T-periodic but is 2T-periodic. -/
theorem timeCrystalState_breaks_symmetry (ψ : ℝ → ℂ) (T : ℝ)
    (h : isTimeCrystalState ψ T) (hψ : ∃ t₀, ψ t₀ ≠ 0) :
    breaksDiscreteTimeTranslationSymmetry ψ T := by
  obtain ⟨t₀, ht₀⟩ := hψ
  exact ⟨⟨t₀, timeCrystal_not_T_periodic ψ T t₀ h ht₀⟩,
         timeCrystal_period_double ψ T h⟩

/-- The time crystal Floquet phase (π) is strictly different from the
    drive-synchronized phase (0): the crystal and the drive are out of sync. -/
theorem timeCrystal_phase_not_sync : floquetPhase Real.pi ≠ floquetPhase 0 := by
  rw [floquetPhase_pi, floquetPhase_zero]
  norm_num

/-- The period-doubling ratio: response period 2T divided by drive period T
    equals exactly 2 for any non-zero T. -/
theorem timeCrystal_period_ratio (T : ℝ) (hT : T ≠ 0) : 2 * T / T = 2 := by
  field_simp

-- ════════════════════════════════════════════════════════════════════════════
-- Section 6 — Quasi-Energy and Period-Doubling Ratio
-- The quasi-energy eigenvalue associated with a time crystal state.
-- ════════════════════════════════════════════════════════════════════════════

/-- The quasi-energy of a time crystal state with drive period T.

    The Floquet phase φ = π corresponds to quasi-energy ε_F = π / T
    (setting ℏ = 1).  This is half the drive frequency ω = 2π/T. -/
noncomputable def timeCrystalQuasiEnergy (T : ℝ) (_ : T ≠ 0) : ℝ := Real.pi / T

/-- Quasi-energy round-trip: ε_F · T = π (the Floquet phase). -/
theorem timeCrystalQuasiEnergy_phase (T : ℝ) (hT : T ≠ 0) :
    timeCrystalQuasiEnergy T hT * T = Real.pi := by
  unfold timeCrystalQuasiEnergy
  field_simp

/-- The response period 2T is strictly larger than the drive period T for
    any positive drive period T > 0.  This is the quantitative signature of
    period doubling. -/
theorem timeCrystal_period_doubling_strict (T : ℝ) (hT : 0 < T) : T < 2 * T := by
  linarith

end
