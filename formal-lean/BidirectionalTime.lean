/-
  BidirectionalTime.lean — Bidirectional time frustration and the Planck
  frustration floor.

  ╔══════════════════════════════════════════════════════════════════════════╗
  ║  Question: What is the structure of frustration when time runs in       ║
  ║  both forward and backward directions simultaneously?                   ║
  ║                                                                         ║
  ║  ForwardClassicalTime.lean establishes:                                 ║
  ║    F_fwd(l) = 1 − sech(l)  >  0  for l ≠ 0                            ║
  ║  in the forward direction alone.                                        ║
  ║                                                                         ║
  ║  This module extends to bidirectional time:                             ║
  ║    F_bi(l_f, l_b) = F_fwd(l_f) + F_fwd(l_b)                           ║
  ║  where l_f is the forward Lyapunov exponent and l_b is the backward    ║
  ║  Lyapunov exponent.                                                     ║
  ║                                                                         ║
  ║  Key results:                                                           ║
  ║  • F_bi(0, 0) = 0   — zero total frustration at equilibrium            ║
  ║  • F_bi ≥ 0         — bidirectional frustration is non-negative        ║
  ║  • F_bi < 2         — strictly bounded                                 ║
  ║  • F_bi(l, l) = 2·F_fwd(l) — symmetric doubling in matched steps      ║
  ║  • F_bi = 0 ↔ l_f = 0 ∧ l_b = 0  — equilibrium characterisation      ║
  ║                                                                         ║
  ║  The Planck frustration floor:                                          ║
  ║    planck_frustration = F_fwd(1) = 1 − sech(1) > 0                    ║
  ║  is the frustration at the natural-unit Planck time displacement.      ║
  ╚══════════════════════════════════════════════════════════════════════════╝

  The bidirectional frustration F_bi(l_f, l_b) = F_fwd(l_f) + F_fwd(l_b)
  measures the total coherence deficit from both a forward time step of
  Lyapunov exponent l_f and a backward time step of Lyapunov exponent l_b.
  Since F_fwd is even, the backward step contributes the same frustration
  as a forward step of equal magnitude.

  The Planck natural-unit time displacement is t_P = 1 (in natural units
  ℏ = c = G = 1).  The corresponding frustration planck_frustration = F_fwd(1)
  provides the minimum nonzero quantum of frustration at the Planck scale.

  Connection to ForwardClassicalTime:
    BidirectionalTime extends ForwardClassicalTime by combining two
    independent single-direction frustration terms.  The forward-only
    hypothesis was confirmed in ForwardClassicalTime; here we verify that
    bidirectional frustration inherits all the key structural properties.

  Sections
  ────────
  1.  Bidirectional frustration  F_bi(l_f, l_b) = F_fwd(l_f) + F_fwd(l_b)
  2.  Equilibrium and positivity
  3.  Double-step symmetry  F_bi(l, l) = 2·F_fwd(l)
  4.  Arrow-of-time in the bidirectional framework
  5.  Planck frustration floor  planck_frustration = F_fwd(1) = 1 − sech(1)

  Proof status
  ────────────
  All 24 theorems have complete machine-checked proofs.
  No `sorry` placeholders remain.
-/

import SpeedOfLight

open Complex Real

noncomputable section

-- ════════════════════════════════════════════════════════════════════════════
-- Definitions
-- ════════════════════════════════════════════════════════════════════════════

/-- Bidirectional time frustration: the total frustration from a forward
    Lyapunov displacement lf and a backward Lyapunov displacement lb.

      F_bi(lf, lb) = F_fwd(lf) + F_fwd(lb)

    Since F_fwd is even (F_fwd l = F_fwd(−l)), a backward displacement of
    magnitude |lb| contributes the same frustration as a forward displacement
    of the same magnitude. -/
noncomputable def F_bi (lf lb : ℝ) : ℝ := F_fwd lf + F_fwd lb

/-- The Planck natural-unit time displacement.

    In natural units (ℏ = c = G = 1) the Planck time is 1.  The frustration
    at this displacement gives the minimum resolvable quantum of frustration
    at the quantum-gravity scale. -/
noncomputable def planck_time_nat : ℝ := 1

/-- The Planck frustration floor: the forward-time frustration at a single
    natural-unit Planck time displacement.

      planck_frustration = F_fwd(1) = 1 − sech(1) -/
noncomputable def planck_frustration : ℝ := F_fwd planck_time_nat

-- ════════════════════════════════════════════════════════════════════════════
-- Section 1 — Structural properties of bidirectional frustration
-- ════════════════════════════════════════════════════════════════════════════

/-- The bidirectional frustration vanishes at the equilibrium point (0, 0):
    F_bi(0, 0) = 0. -/
theorem fbi_zero : F_bi 0 0 = 0 := by
  unfold F_bi
  rw [fct_frustration_at_zero, add_zero]

/-- Bidirectional frustration is symmetric: swapping forward and backward
    displacements does not change the total frustration.
      F_bi(lf, lb) = F_bi(lb, lf). -/
theorem fbi_symm (lf lb : ℝ) : F_bi lf lb = F_bi lb lf := by
  unfold F_bi; ring

/-- Bidirectional frustration is non-negative: both F_fwd terms are
    non-negative (by fct_frustration_nonneg), so their sum is non-negative. -/
theorem fbi_nonneg (lf lb : ℝ) : 0 ≤ F_bi lf lb := by
  unfold F_bi
  linarith [fct_frustration_nonneg lf, fct_frustration_nonneg lb]

/-- Bidirectional frustration is strictly less than 2: each component
    satisfies F_fwd < 1, so their sum is less than 2. -/
theorem fbi_upper_bound (lf lb : ℝ) : F_bi lf lb < 2 := by
  unfold F_bi
  linarith [fct_frustration_lt_one lf, fct_frustration_lt_one lb]

/-- Bidirectional frustration lies in the half-open interval [0, 2). -/
theorem fbi_mem_interval (lf lb : ℝ) : 0 ≤ F_bi lf lb ∧ F_bi lf lb < 2 :=
  ⟨fbi_nonneg lf lb, fbi_upper_bound lf lb⟩

-- ════════════════════════════════════════════════════════════════════════════
-- Section 2 — Equilibrium characterisation and positivity
-- ════════════════════════════════════════════════════════════════════════════

/-- Bidirectional frustration is zero if and only if both displacements
    are zero:
      F_bi(lf, lb) = 0 ↔ lf = 0 ∧ lb = 0. -/
theorem fbi_zero_iff (lf lb : ℝ) : F_bi lf lb = 0 ↔ lf = 0 ∧ lb = 0 := by
  unfold F_bi
  constructor
  · intro h
    have hf := fct_frustration_nonneg lf
    have hb := fct_frustration_nonneg lb
    exact ⟨(fct_frustration_zero_iff lf).mp (by linarith),
           (fct_frustration_zero_iff lb).mp (by linarith)⟩
  · rintro ⟨rfl, rfl⟩
    simp [fct_frustration_at_zero]

/-- A non-zero forward displacement produces strictly positive bidirectional
    frustration:  lf ≠ 0 → 0 < F_bi(lf, lb). -/
theorem fbi_pos_of_fwd_ne (lf lb : ℝ) (hf : lf ≠ 0) : 0 < F_bi lf lb := by
  unfold F_bi
  linarith [fct_frustration_pos lf hf, fct_frustration_nonneg lb]

/-- A non-zero backward displacement produces strictly positive bidirectional
    frustration:  lb ≠ 0 → 0 < F_bi(lf, lb). -/
theorem fbi_pos_of_bwd_ne (lf lb : ℝ) (hb : lb ≠ 0) : 0 < F_bi lf lb := by
  unfold F_bi
  linarith [fct_frustration_nonneg lf, fct_frustration_pos lb hb]

/-- Bidirectional frustration is positive if and only if at least one
    displacement is non-zero:
      0 < F_bi(lf, lb) ↔ lf ≠ 0 ∨ lb ≠ 0. -/
theorem fbi_pos_iff (lf lb : ℝ) : 0 < F_bi lf lb ↔ lf ≠ 0 ∨ lb ≠ 0 := by
  constructor
  · intro h
    by_contra h_and
    push_neg at h_and
    obtain ⟨hf, hb⟩ := h_and
    have := (fbi_zero_iff lf lb).mpr ⟨hf, hb⟩
    linarith
  · rintro (hf | hb)
    · exact fbi_pos_of_fwd_ne lf lb hf
    · exact fbi_pos_of_bwd_ne lf lb hb

-- ════════════════════════════════════════════════════════════════════════════
-- Section 3 — Double-step symmetry:  F_bi(l, l) = 2 · F_fwd(l)
-- ════════════════════════════════════════════════════════════════════════════

/-- When forward and backward displacements are equal, the bidirectional
    frustration doubles the single-direction frustration:
      F_bi(l, l) = 2 · F_fwd(l). -/
theorem fbi_double (l : ℝ) : F_bi l l = 2 * F_fwd l := by
  unfold F_bi; ring

/-- The double-step frustration is non-negative. -/
theorem fbi_double_nonneg (l : ℝ) : 0 ≤ F_bi l l := by
  rw [fbi_double]
  linarith [fct_frustration_nonneg l]

/-- The double-step frustration is strictly positive for non-zero displacement. -/
theorem fbi_double_pos (l : ℝ) (hl : l ≠ 0) : 0 < F_bi l l := by
  rw [fbi_double]
  linarith [fct_frustration_pos l hl]

/-- The double-step frustration is strictly less than 2. -/
theorem fbi_double_lt_two (l : ℝ) : F_bi l l < 2 := by
  rw [fbi_double]
  linarith [fct_frustration_lt_one l]

/-- The double-step frustration is even:  F_bi(l, l) = F_bi(−l, −l).
    This follows from the even symmetry F_fwd(l) = F_fwd(−l). -/
theorem fbi_double_even (l : ℝ) : F_bi l l = F_bi (-l) (-l) := by
  simp only [fbi_double, ← fct_even l]

-- ════════════════════════════════════════════════════════════════════════════
-- Section 4 — Arrow of time and dominance properties
-- ════════════════════════════════════════════════════════════════════════════

/-- Bidirectional arrow of time: any nonzero displacement (forward or
    backward) produces strictly positive total frustration.
      lf ≠ 0 ∨ lb ≠ 0 → 0 < F_bi(lf, lb)  ← BIDIRECTIONAL ARROW -/
theorem fbi_arrow (lf lb : ℝ) (h : lf ≠ 0 ∨ lb ≠ 0) : 0 < F_bi lf lb :=
  (fbi_pos_iff lf lb).mpr h

/-- The equilibrium (0, 0) is the global minimum of bidirectional frustration:
      F_bi(0, 0) ≤ F_bi(lf, lb)  for all lf, lb. -/
theorem fbi_min_at_equilibrium (lf lb : ℝ) : F_bi 0 0 ≤ F_bi lf lb := by
  rw [fbi_zero]
  exact fbi_nonneg lf lb

/-- The forward component is dominated by the bidirectional frustration:
      F_fwd(lf) ≤ F_bi(lf, lb). -/
theorem fbi_ge_fwd_component (lf lb : ℝ) : F_fwd lf ≤ F_bi lf lb := by
  unfold F_bi
  linarith [fct_frustration_nonneg lb]

/-- The backward component is dominated by the bidirectional frustration:
      F_fwd(lb) ≤ F_bi(lf, lb). -/
theorem fbi_ge_bwd_component (lf lb : ℝ) : F_fwd lb ≤ F_bi lf lb := by
  unfold F_bi
  linarith [fct_frustration_nonneg lf]

/-- Bidirectional frustration expressed as a coherence sum:
      F_bi(lf, lb) = 2 − C(exp lf) − C(exp lb).
    Each component F_fwd(l) = 1 − C(exp l) is the coherence deficit,
    so the total bidirectional frustration is the sum of coherence deficits. -/
theorem fbi_coherence_sum (lf lb : ℝ) :
    F_bi lf lb = 2 - C (Real.exp lf) - C (Real.exp lb) := by
  unfold F_bi
  rw [fct_frustration_eq lf, fct_frustration_eq lb]
  ring

-- ════════════════════════════════════════════════════════════════════════════
-- Section 5 — Planck frustration floor
-- ════════════════════════════════════════════════════════════════════════════

/-- The Planck frustration is strictly positive:
      planck_frustration = F_fwd(1) > 0.
    This follows from fct_frustration_pos since 1 ≠ 0. -/
theorem planck_frustration_pos : 0 < planck_frustration := by
  unfold planck_frustration planck_time_nat
  exact fct_frustration_pos 1 one_ne_zero

/-- The Planck frustration is strictly less than 1. -/
theorem planck_frustration_lt_one : planck_frustration < 1 := by
  unfold planck_frustration planck_time_nat
  exact fct_frustration_lt_one 1

/-- The Planck frustration equals 1 − (cosh 1)⁻¹ = 1 − sech(1).
    Explicit value at the natural-unit Planck time displacement. -/
theorem planck_frustration_eq_sech : planck_frustration = 1 - (Real.cosh 1)⁻¹ := by
  unfold planck_frustration planck_time_nat F_fwd
  ring

/-- The Planck frustration equals the coherence deficit at exp(1):
      planck_frustration = 1 − C(exp 1).
    Direct consequence of the Lyapunov–coherence duality. -/
theorem planck_frustration_coherence_deficit :
    planck_frustration = 1 - C (Real.exp 1) := by
  unfold planck_frustration planck_time_nat
  exact fct_frustration_eq 1

/-- The Planck frustration floor: planck_frustration is strictly positive
    and strictly less than 1.
      0 < planck_frustration ∧ planck_frustration < 1  ← PLANCK FLOOR -/
theorem planck_frustration_floor :
    0 < planck_frustration ∧ planck_frustration < 1 :=
  ⟨planck_frustration_pos, planck_frustration_lt_one⟩

end
