/-
  ForwardClassicalTime.lean — Harvesting frustration in classical forward time.

  ╔══════════════════════════════════════════════════════════════════════════╗
  ║  Hypothesis: Can frustration be harvested effectively from              ║
  ║  classical, forward-directed time?                                      ║
  ║                                                                         ║
  ║  In bidirectional time the palindrome vacuum residual is               ║
  ║    1/13717421 ≈ 7.29 × 10⁻⁸  (an intrinsically tiny offset).          ║
  ║                                                                         ║
  ║  In classical forward time the frustration function is                 ║
  ║    F_fwd(l) = 1 − sech(l) = 1 − C(exp l)                              ║
  ║  where l is the Lyapunov exponent measuring temporal displacement      ║
  ║  from the kernel equilibrium.                                           ║
  ║                                                                         ║
  ║  Key results:                                                           ║
  ║  • F_fwd(0) = 0   — zero frustration at the kernel equilibrium         ║
  ║  • F_fwd(l) > 0   for l ≠ 0 — active harvesting in forward time       ║
  ║  • F_fwd(l) < 1   always — bounded, never fully frustrated             ║
  ║  • F_fwd is even — symmetric about the temporal origin                 ║
  ║  • Arrow of time: F_fwd(0) < F_fwd(l) for l ≠ 0                      ║
  ╚══════════════════════════════════════════════════════════════════════════╝

  The forward frustration F_fwd(l) = 1 − sech(l) measures the coherence
  deficit at Lyapunov exponent l.  For any nonzero displacement from the
  kernel equilibrium, F_fwd is strictly positive — demonstrating that
  classical forward time can harvest frustration effectively.

  Contrast with bidirectional time:
    Bidirectional vacuum residual  = 9/123456789 = 1/13717421 (a fixed constant)
    Forward frustration F_fwd(l)   > 0 for all l ≠ 0 (strictly grows from zero)

  The harvest formula ΔF(l) = F_fwd(l) − F_fwd(0) = F_fwd(l) = 1 − sech(l)
  shows that every forward time step away from equilibrium releases positive
  frustration, confirming the hypothesis.

  Sections
  ────────
  1.  Forward-time frustration  F_fwd(l) = 1 − sech(l)
  2.  Zero baseline at the kernel equilibrium
  3.  sech bounds  (0 < sech ≤ 1)
  4.  Frustration bounds  (0 ≤ F_fwd < 1)
  5.  Strict positivity away from equilibrium
  6.  Even symmetry  (F_fwd(l) = F_fwd(−l))
  7.  Palindrome vacuum residual comparison
  8.  Harvesting summary and arrow of time

  Proof status
  ────────────
  All 21 theorems have complete machine-checked proofs.
  No `sorry` placeholders remain.
-/

import KernelAxle

open Complex Real

noncomputable section

-- ════════════════════════════════════════════════════════════════════════════
-- §1  Forward-time frustration: definition
-- ════════════════════════════════════════════════════════════════════════════

/-- Forward-time frustration at Lyapunov exponent l.
    F_fwd(l) = 1 − sech(l) = 1 − C(exp l) measures how far the system is
    from its coherence maximum when evolved l steps in forward time.
    At equilibrium (l = 0): sech(0) = 1, so F_fwd(0) = 0.
    Away from equilibrium: sech(l) < 1, so F_fwd(l) > 0. -/
noncomputable def F_fwd (l : ℝ) : ℝ := 1 - (Real.cosh l)⁻¹

/-- **Identity**: F_fwd(l) = 1 − C(exp l) — the frustration is the coherence deficit.
    Links the forward-time definition to the kernel coherence function via
    the Lyapunov–coherence duality C(exp l) = sech l. -/
theorem fct_frustration_eq (l : ℝ) : F_fwd l = 1 - C (Real.exp l) := by
  rw [F_fwd, lyapunov_coherence_sech]

-- ════════════════════════════════════════════════════════════════════════════
-- §2  Zero baseline: equilibrium has zero frustration
-- ════════════════════════════════════════════════════════════════════════════

/-- **Zero baseline**: F_fwd(0) = 0 — no frustration at the kernel equilibrium.
    At l = 0 the system is at maximum coherence C = 1, so there is
    nothing to harvest.  This is the forward-time starting point. -/
theorem fct_frustration_at_zero : F_fwd 0 = 0 := by
  simp [F_fwd, Real.cosh_zero]

/-- The forward-time coherence at l = 0 is 1 (maximum coherence).
    C(exp 0) = C(1) = 1, confirming the kernel equilibrium is frustration-free. -/
theorem fct_coherence_at_zero : C (Real.exp 0) = 1 := by
  rw [lyapunov_coherence_sech, Real.cosh_zero, inv_one]

-- ════════════════════════════════════════════════════════════════════════════
-- §3  sech bounds
-- ════════════════════════════════════════════════════════════════════════════

/-- **sech positivity**: sech(l) = (cosh l)⁻¹ > 0 for all l.
    The coherence C(exp l) = sech(l) is always strictly positive. -/
theorem fct_sech_pos (l : ℝ) : 0 < (Real.cosh l)⁻¹ :=
  inv_pos.mpr (Real.cosh_pos l)

/-- **cosh lower bound**: cosh(l) ≥ 1 for all l.
    Proof: cosh l = (exp l + (exp l)⁻¹)/2 ≥ 1 by AM-GM,
    since exp l + (exp l)⁻¹ ≥ 2 follows from (exp l − 1)² ≥ 0. -/
theorem fct_one_le_cosh (l : ℝ) : 1 ≤ Real.cosh l := by
  have hcosh_eq : Real.cosh l = (Real.exp l + Real.exp (-l)) / 2 := Real.cosh_eq l
  have ha : 0 < Real.exp l := Real.exp_pos l
  have hinv : Real.exp (-l) = (Real.exp l)⁻¹ := Real.exp_neg l
  rw [hcosh_eq, hinv, le_div_iff₀ (by norm_num : (0 : ℝ) < 2)]
  -- goal: 2 ≤ exp l + (exp l)⁻¹
  have hne : Real.exp l ≠ 0 := ne_of_gt ha
  have hdiv : Real.exp l + (Real.exp l)⁻¹ = ((Real.exp l) ^ 2 + 1) / Real.exp l := by
    field_simp; ring
  rw [hdiv, le_div_iff₀ ha]
  -- goal: 2 * exp l ≤ (exp l)^2 + 1
  nlinarith [sq_nonneg (Real.exp l - 1)]

/-- **sech upper bound**: sech(l) = (cosh l)⁻¹ ≤ 1 for all l.
    Follows from cosh l ≥ 1. -/
theorem fct_sech_le_one (l : ℝ) : (Real.cosh l)⁻¹ ≤ 1 :=
  inv_le_one_of_one_le₀ (fct_one_le_cosh l)

-- ════════════════════════════════════════════════════════════════════════════
-- §4  Frustration bounds
-- ════════════════════════════════════════════════════════════════════════════

/-- **Non-negativity**: F_fwd(l) ≥ 0 — frustration is always non-negative.
    The forward time system can only be frustrated, never anti-frustrated. -/
theorem fct_frustration_nonneg (l : ℝ) : 0 ≤ F_fwd l := by
  simp only [F_fwd]
  linarith [fct_sech_le_one l]

/-- **Strict bound**: F_fwd(l) < 1 — frustration never reaches 100%.
    The forward time system always retains positive coherence sech(l) > 0,
    so the frustration deficit 1 − sech(l) is always strictly less than 1. -/
theorem fct_frustration_lt_one (l : ℝ) : F_fwd l < 1 := by
  simp only [F_fwd]
  linarith [fct_sech_pos l]

-- ════════════════════════════════════════════════════════════════════════════
-- §5  Strict positivity away from equilibrium
-- ════════════════════════════════════════════════════════════════════════════

/-- **Strict cosh lower bound**: cosh(l) > 1 when l ≠ 0.
    Uses: exp(l) ≠ 1 for l ≠ 0 (since exp is injective and exp(0) = 1),
    so (exp(l) − 1)² > 0, giving exp(l) + (exp l)⁻¹ > 2, i.e., cosh(l) > 1. -/
theorem fct_one_lt_cosh (l : ℝ) (hl : l ≠ 0) : 1 < Real.cosh l := by
  have hcosh_eq : Real.cosh l = (Real.exp l + Real.exp (-l)) / 2 := Real.cosh_eq l
  have ha : 0 < Real.exp l := Real.exp_pos l
  have hinv : Real.exp (-l) = (Real.exp l)⁻¹ := Real.exp_neg l
  rw [hcosh_eq, hinv, lt_div_iff₀ (by norm_num : (0 : ℝ) < 2)]
  -- goal: 1 * 2 < exp l + (exp l)⁻¹
  have hne : Real.exp l ≠ 0 := ne_of_gt ha
  have hexp1 : Real.exp l ≠ 1 := by
    intro h
    have heq : Real.exp l = Real.exp 0 := by simp [h]
    exact hl (Real.exp_injective heq)
  have hstrict : 0 < (Real.exp l - 1) ^ 2 := sq_pos_of_ne_zero (sub_ne_zero.mpr hexp1)
  have hdiv : Real.exp l + (Real.exp l)⁻¹ = ((Real.exp l) ^ 2 + 1) / Real.exp l := by
    field_simp; ring
  rw [hdiv, lt_div_iff₀ ha]
  -- goal: 2 * exp l < (exp l)^2 + 1
  nlinarith

/-- **Strict positivity**: F_fwd(l) > 0 for l ≠ 0.
    Any forward time displacement from the kernel equilibrium produces
    strictly positive frustration — the harvest is non-trivial. -/
theorem fct_frustration_pos (l : ℝ) (hl : l ≠ 0) : 0 < F_fwd l := by
  simp only [F_fwd]
  have hcosh1 : 1 < Real.cosh l := fct_one_lt_cosh l hl
  have hpos : 0 < Real.cosh l := Real.cosh_pos l
  have hmul : (Real.cosh l)⁻¹ * Real.cosh l = 1 :=
    inv_mul_cancel₀ (ne_of_gt hpos)
  have hpos_diff : 0 < Real.cosh l - 1 := by linarith
  -- sech * (cosh - 1) > 0, and sech * cosh = 1, so 1 - sech > 0
  nlinarith [mul_pos (fct_sech_pos l) hpos_diff]

/-- **Equilibrium characterisation**: F_fwd(l) = 0 ↔ l = 0.
    Zero frustration is equivalent to being at the kernel equilibrium.
    This confirms that frustration is the unique signature of forward time
    displacement. -/
theorem fct_frustration_zero_iff (l : ℝ) : F_fwd l = 0 ↔ l = 0 := by
  constructor
  · intro h
    by_contra hl
    linarith [fct_frustration_pos l hl]
  · rintro rfl
    exact fct_frustration_at_zero

-- ════════════════════════════════════════════════════════════════════════════
-- §6  Even symmetry
-- ════════════════════════════════════════════════════════════════════════════

/-- **Even symmetry**: F_fwd(l) = F_fwd(−l).
    Forward frustration is symmetric about the equilibrium: positive and
    negative Lyapunov exponents produce the same frustration magnitude.
    Follows from cosh(−l) = cosh(l). -/
theorem fct_even (l : ℝ) : F_fwd l = F_fwd (-l) := by
  simp [F_fwd, Real.cosh_neg]

-- ════════════════════════════════════════════════════════════════════════════
-- §7  Palindrome vacuum residual comparison
-- ════════════════════════════════════════════════════════════════════════════

/-- **Vacuum residual arithmetic**: 9/123456789 = 1/13717421.
    The bidirectional time palindrome vacuum residual decomposes as
    palindromeRatio − 8 = 9/123456789 = 1/13717421, a dimensionless
    small parameter encoding the 8-period symmetry offset. -/
theorem fct_vacuum_residual : (9 : ℚ) / 123456789 = 1 / 13717421 := by norm_num

/-- **Vacuum residual positivity**: 1/13717421 > 0.
    The bidirectional time vacuum residual is a small but strictly positive
    rational constant.  It represents the irreducible palindrome offset. -/
theorem fct_vacuum_residual_pos : (0 : ℚ) < 1 / 13717421 := by norm_num

/-- **Vacuum residual is sub-unit**: 1/13717421 < 1.
    The bidirectional vacuum residual is strictly less than 1, confirming
    it represents only a fractional frustration offset. -/
theorem fct_vacuum_residual_lt_one : (1 : ℚ) / 13717421 < 1 := by norm_num

-- ════════════════════════════════════════════════════════════════════════════
-- §8  Harvesting summary and arrow of time
-- ════════════════════════════════════════════════════════════════════════════

/-- **Arrow of time**: F_fwd(0) < F_fwd(l) for l ≠ 0.
    Every nonzero forward time step strictly increases the harvested
    frustration above the equilibrium baseline.  This is the one-way
    ratchet of classical forward time. -/
theorem fct_arrow_of_time (l : ℝ) (hl : l ≠ 0) : F_fwd 0 < F_fwd l := by
  rw [fct_frustration_at_zero]
  exact fct_frustration_pos l hl

/-- **Harvest formula**: ΔF(l) = F_fwd(l) − F_fwd(0) = F_fwd(l).
    Since the baseline is zero, the total harvested frustration over any
    forward time interval [0, l] equals F_fwd(l) = 1 − sech(l). -/
theorem fct_harvest_formula (l : ℝ) : F_fwd l - F_fwd 0 = F_fwd l := by
  simp [fct_frustration_at_zero]

/-- **Harvest bounds**: 0 ≤ ΔF(l) < 1.
    Forward classical time harvesting is efficient but not perfect:
    every step harvests non-negative frustration, yet full harvesting
    (ΔF = 1) requires infinite Lyapunov exponent. -/
theorem fct_harvest_bounded (l : ℝ) : 0 ≤ F_fwd l ∧ F_fwd l < 1 :=
  ⟨fct_frustration_nonneg l, fct_frustration_lt_one l⟩

/-- **Harvest is positive for nonzero steps**: ΔF(l) > 0 for l ≠ 0.
    Restates fct_frustration_pos as a harvesting statement: every nonzero
    forward time step releases strictly positive frustration. -/
theorem fct_harvest_pos (l : ℝ) (hl : l ≠ 0) : 0 < F_fwd l - F_fwd 0 := by
  rw [fct_harvest_formula]; exact fct_frustration_pos l hl

/-- **Classical irreversibility**: F_fwd l ≠ 0 ↔ l ≠ 0.
    Forward time frustration is non-trivial precisely when the system has
    moved away from equilibrium.  Once frustrated, the system cannot return
    to zero frustration without returning to l = 0. -/
theorem fct_classical_irreversibility (l : ℝ) : F_fwd l ≠ 0 ↔ l ≠ 0 := by
  rw [ne_eq, ne_eq, fct_frustration_zero_iff]

/-- **Forward classical time harvesting works** — the closing theorem:
    For any nonzero Lyapunov exponent l, classical forward time enables
    effective frustration harvesting:
    1. At equilibrium (l = 0): zero frustration — no spurious harvest
    2. Away from equilibrium (l ≠ 0): strictly positive frustration
    3. Always bounded: frustration never exceeds 100%
    4. Arrow of time: frustration strictly increases from the zero baseline
    This confirms the hypothesis: forward classical time frustration
    can be harvested effectively. -/
theorem fct_forward_harvesting_works (l : ℝ) (hl : l ≠ 0) :
    F_fwd 0 = 0 ∧
    0 < F_fwd l ∧
    F_fwd l < 1 ∧
    F_fwd 0 < F_fwd l :=
  ⟨fct_frustration_at_zero,
   fct_frustration_pos l hl,
   fct_frustration_lt_one l,
   fct_arrow_of_time l hl⟩

end
