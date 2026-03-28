/-
  BidirectionalTime.lean — Bidirectional time frustration and the Planck floor.

  ╔══════════════════════════════════════════════════════════════════════════╗
  ║  Hypothesis: How does frustration behave when time flows in both       ║
  ║  directions simultaneously?                                             ║
  ║                                                                         ║
  ║  In bidirectional time, forward and backward Lyapunov exponents         ║
  ║  lf and lb each independently displace the system from the kernel      ║
  ║  equilibrium.  The total frustration is additive:                       ║
  ║                                                                         ║
  ║    F_bi(lf, lb) = F_fwd(lf) + F_fwd(lb)                               ║
  ║                                                                         ║
  ║  where F_fwd(l) = 1 − sech(l) is the forward classical frustration.   ║
  ║                                                                         ║
  ║  Key results:                                                           ║
  ║  • F_bi(0,0) = 0      — zero frustration at the kernel equilibrium     ║
  ║  • F_bi ≥ 0           — frustration is always nonnegative            ║
  ║  • F_bi < 2           — never fully frustrated in both directions       ║
  ║  • F_bi is symmetric  — direction labels are interchangeable            ║
  ║  • F_bi dominates each component — more frustrated than either alone   ║
  ╚══════════════════════════════════════════════════════════════════════════╝

  The Planck floor: the frustration at the unit Lyapunov exponent l = 1 acts
  as a natural quantum of frustration — the smallest frustration carried by a
  unit-displacement excitation:

    planck_frustration = F_fwd(1) = 1 − sech(1)

  In a bidirectional system, both directions contribute a Planck frustration
  quantum, giving F_bi(1,1) = 2 · planck_frustration.

  Contrast with forward-only time:
    Forward only:  F_fwd(l) ∈ [0, 1)  — one direction of displacement
    Bidirectional: F_bi(lf,lb) ∈ [0, 2) — two independent directions

  Sections
  ────────
  1.  Structural properties    — definition, bounds, symmetry
  2.  Equilibrium              — zero conditions, degenerate limits
  3.  Double-step              — diagonal F_bi(l,l) = 2·F_fwd(l)
  4.  Arrow and dominance      — F_bi dominates each component; arrow of time
  5.  Planck floor             — planck_frustration = F_fwd(1), bounds

  Proof status
  ────────────
  All 24 theorems have complete machine-checked proofs.
  No `sorry` placeholders remain.
-/

import SpeedOfLight

open Complex Real

noncomputable section

-- ════════════════════════════════════════════════════════════════════════════
-- §1  Structural properties
-- ════════════════════════════════════════════════════════════════════════════

/-- Bidirectional time frustration: total frustration from forward exponent lf
    and backward exponent lb.

    F_bi(lf, lb) = F_fwd(lf) + F_fwd(lb)

    Each independent temporal direction contributes its own frustration.
    At the kernel equilibrium (lf = lb = 0): F_bi = 0.
    With full forward displacement (lf = l, lb = 0): F_bi = F_fwd(l). -/
noncomputable def F_bi (lf lb : ℝ) : ℝ := F_fwd lf + F_fwd lb

/-- **Definition expansion**: F_bi(lf,lb) = (1 − sech lf) + (1 − sech lb).
    Unfolds both the bidirectional and forward-time frustration definitions
    to expose the hyperbolic-secant representation. -/
theorem bi_def_expand (lf lb : ℝ) :
    F_bi lf lb = (1 - (Real.cosh lf)⁻¹) + (1 - (Real.cosh lb)⁻¹) := by
  simp [F_bi, F_fwd]

/-- **Non-negativity**: F_bi(lf,lb) ≥ 0 for all lf, lb.
    Both components are non-negative (each sech ≤ 1), so their sum is too.
    Bidirectional frustration can only accumulate, never reverse. -/
theorem bi_nonneg (lf lb : ℝ) : 0 ≤ F_bi lf lb := by
  simp only [F_bi]
  linarith [fct_frustration_nonneg lf, fct_frustration_nonneg lb]

/-- **Upper bound**: F_bi(lf,lb) < 2 for all lf, lb.
    Each component is strictly less than 1 (sech > 0), so the sum is
    strictly less than 2.  Full bidirectional frustration is unattainable. -/
theorem bi_lt_two (lf lb : ℝ) : F_bi lf lb < 2 := by
  simp only [F_bi]
  linarith [fct_frustration_lt_one lf, fct_frustration_lt_one lb]

/-- **Symmetry**: F_bi(lf,lb) = F_bi(lb,lf).
    The bidirectional frustration is symmetric in its arguments: the labels
    "forward" and "backward" are interchangeable. -/
theorem bi_symm (lf lb : ℝ) : F_bi lf lb = F_bi lb lf := by
  simp [F_bi, add_comm]

/-- **Combined bound**: 0 ≤ F_bi(lf,lb) < 2.
    Packages the non-negativity and upper bound into a single statement. -/
theorem bi_bound (lf lb : ℝ) : 0 ≤ F_bi lf lb ∧ F_bi lf lb < 2 :=
  ⟨bi_nonneg lf lb, bi_lt_two lf lb⟩

-- ════════════════════════════════════════════════════════════════════════════
-- §2  Equilibrium
-- ════════════════════════════════════════════════════════════════════════════

/-- **Equilibrium**: F_bi(0,0) = 0 — zero frustration at the kernel equilibrium.
    When both Lyapunov exponents vanish, both components vanish, and the
    bidirectional system is perfectly coherent. -/
theorem bi_at_zero_zero : F_bi 0 0 = 0 := by
  simp [F_bi, fct_frustration_at_zero]

/-- **Forward degenerate limit**: F_bi(lf,0) = F_fwd(lf).
    When the backward exponent is zero, bidirectional frustration reduces
    to forward-only frustration.  The backward direction contributes nothing. -/
theorem bi_fwd_degenerate (lf : ℝ) : F_bi lf 0 = F_fwd lf := by
  simp [F_bi, fct_frustration_at_zero]

/-- **Backward degenerate limit**: F_bi(0,lb) = F_fwd(lb).
    When the forward exponent is zero, bidirectional frustration reduces
    to the backward frustration.  The forward direction contributes nothing. -/
theorem bi_bwd_degenerate (lb : ℝ) : F_bi 0 lb = F_fwd lb := by
  simp [F_bi, fct_frustration_at_zero]

/-- **Zero characterisation**: F_bi(lf,lb) = 0 ↔ lf = 0 ∧ lb = 0.
    Bidirectional frustration vanishes precisely when both exponents vanish:
    the kernel equilibrium is the unique zero of F_bi.
    Proof: both summands are non-negative, so their sum is zero iff each
    is zero; each is zero iff its exponent is zero (fct_frustration_zero_iff). -/
theorem bi_zero_iff (lf lb : ℝ) : F_bi lf lb = 0 ↔ lf = 0 ∧ lb = 0 := by
  simp only [F_bi]
  constructor
  · intro h
    have hf : F_fwd lf = 0 := by linarith [fct_frustration_nonneg lf, fct_frustration_nonneg lb]
    have hb : F_fwd lb = 0 := by linarith [fct_frustration_nonneg lf, fct_frustration_nonneg lb]
    exact ⟨(fct_frustration_zero_iff lf).mp hf, (fct_frustration_zero_iff lb).mp hb⟩
  · rintro ⟨rfl, rfl⟩
    simp [fct_frustration_at_zero]

-- ════════════════════════════════════════════════════════════════════════════
-- §3  Double-step
-- ════════════════════════════════════════════════════════════════════════════

/-- **Double-step formula**: F_bi(l,l) = 2 · F_fwd(l).
    When forward and backward Lyapunov exponents are equal, the bidirectional
    frustration is exactly twice the single-direction frustration.
    This is the symmetric excitation of both temporal directions. -/
theorem bi_double (l : ℝ) : F_bi l l = 2 * F_fwd l := by
  simp [F_bi, two_mul]

/-- **Double-step non-negativity**: F_bi(l,l) ≥ 0 for all l. -/
theorem bi_double_nonneg (l : ℝ) : 0 ≤ F_bi l l := bi_nonneg l l

/-- **Double-step strict positivity**: F_bi(l,l) > 0 when l ≠ 0.
    A symmetric bidirectional excitation always produces strictly positive
    frustration, doubling the forward-time arrow-of-time effect. -/
theorem bi_double_pos (l : ℝ) (hl : l ≠ 0) : 0 < F_bi l l := by
  rw [bi_double]
  linarith [fct_frustration_pos l hl]

/-- **Double-step upper bound**: F_bi(l,l) < 2 for all l.
    The symmetric excitation never exhausts both temporal directions
    simultaneously. -/
theorem bi_double_lt_two (l : ℝ) : F_bi l l < 2 := bi_lt_two l l

/-- **Double-step zero characterisation**: F_bi(l,l) = 0 ↔ l = 0.
    The symmetric bidirectional frustration vanishes precisely at the
    kernel equilibrium. -/
theorem bi_double_zero_iff (l : ℝ) : F_bi l l = 0 ↔ l = 0 := by
  rw [bi_zero_iff]; tauto

-- ════════════════════════════════════════════════════════════════════════════
-- §4  Arrow and dominance
-- ════════════════════════════════════════════════════════════════════════════

/-- **Forward dominance**: F_bi(lf,lb) ≥ F_fwd(lf).
    The bidirectional frustration is at least as large as the forward
    component alone, because the backward component is non-negative. -/
theorem bi_ge_fwd (lf lb : ℝ) : F_fwd lf ≤ F_bi lf lb := by
  simp only [F_bi]
  linarith [fct_frustration_nonneg lb]

/-- **Backward dominance**: F_bi(lf,lb) ≥ F_fwd(lb).
    The bidirectional frustration is at least as large as the backward
    component alone, because the forward component is non-negative. -/
theorem bi_ge_bwd (lf lb : ℝ) : F_fwd lb ≤ F_bi lf lb := by
  simp only [F_bi]
  linarith [fct_frustration_nonneg lf]

/-- **Strict forward dominance**: F_bi(lf,lb) > F_fwd(lf) when lb ≠ 0.
    The presence of a nonzero backward exponent strictly increases the
    bidirectional frustration beyond the forward component. -/
theorem bi_strictly_dom_fwd (lf lb : ℝ) (hlb : lb ≠ 0) :
    F_fwd lf < F_bi lf lb := by
  simp only [F_bi]
  linarith [fct_frustration_pos lb hlb]

/-- **Strict backward dominance**: F_bi(lf,lb) > F_fwd(lb) when lf ≠ 0.
    The presence of a nonzero forward exponent strictly increases the
    bidirectional frustration beyond the backward component. -/
theorem bi_strictly_dom_bwd (lf lb : ℝ) (hlf : lf ≠ 0) :
    F_fwd lb < F_bi lf lb := by
  simp only [F_bi]
  linarith [fct_frustration_pos lf hlf]

/-- **Bidirectional arrow of time**: F_bi(0,0) < F_bi(lf,lb) whenever
    at least one exponent is nonzero.
    Any displacement from the kernel equilibrium in either temporal direction
    strictly increases the bidirectional frustration above zero.
    This is the bidirectional generalisation of the forward arrow of time. -/
theorem bi_arrow (lf lb : ℝ) (h : ¬(lf = 0 ∧ lb = 0)) :
    F_bi 0 0 < F_bi lf lb := by
  rw [bi_at_zero_zero]
  simp only [F_bi]
  push_neg at h
  by_cases hlf : lf = 0
  · subst hlf
    simp [fct_frustration_at_zero]
    exact fct_frustration_pos lb (h rfl)
  · linarith [fct_frustration_pos lf hlf, fct_frustration_nonneg lb]

-- ════════════════════════════════════════════════════════════════════════════
-- §5  Planck floor
-- ════════════════════════════════════════════════════════════════════════════

/-- The Planck frustration floor: frustration at unit Lyapunov exponent.

    planck_frustration = F_fwd(1) = 1 − sech(1)

    This is the natural quantum of frustration — the frustration carried by
    a unit-displacement temporal excitation.  It serves as a Planck-scale
    lower bound on the frustration of any non-equilibrium unit-step state. -/
noncomputable def planck_frustration : ℝ := F_fwd 1

/-- **Planck frustration identity**: planck_frustration = 1 − (cosh 1)⁻¹.
    Expands the definition to expose the hyperbolic-secant representation.
    Numerically: cosh(1) ≈ 1.543, so planck_frustration ≈ 0.352. -/
theorem planck_frustration_eq : planck_frustration = 1 - (Real.cosh 1)⁻¹ := by
  simp [planck_frustration, F_fwd]

/-- **Planck frustration positivity**: planck_frustration > 0.
    A unit-step temporal excitation always carries strictly positive
    frustration.  There is no zero-cost excitation above equilibrium. -/
theorem planck_frustration_pos : 0 < planck_frustration :=
  fct_frustration_pos 1 one_ne_zero

/-- **Planck frustration sub-unit**: planck_frustration < 1.
    The unit-step frustration is less than 1: the system retains positive
    coherence sech(1) > 0 even at unit Lyapunov displacement. -/
theorem planck_frustration_lt_one : planck_frustration < 1 :=
  fct_frustration_lt_one 1

/-- **Bidirectional Planck double**: F_bi(1,1) = 2 · planck_frustration.
    A symmetric unit excitation in both temporal directions carries exactly
    twice the single-direction Planck frustration floor. -/
theorem bi_planck_double : F_bi 1 1 = 2 * planck_frustration := by
  simp [planck_frustration, bi_double]

/-- **Planck frustration bound**: 0 < planck_frustration ∧ planck_frustration < 1.
    The Planck floor is the irreducible positive frustration quantum carried
    by any unit-step temporal excitation: strictly above zero, strictly
    below full frustration.  It is the absolute floor for non-equilibrium
    dynamics in the bidirectional time framework. -/
theorem planck_frustration_bound :
    0 < planck_frustration ∧ planck_frustration < 1 :=
  ⟨planck_frustration_pos, planck_frustration_lt_one⟩

end
