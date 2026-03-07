/-
  BidirectionalTime.lean — Lean 4 formalization of bidirectional time using
  the silver ratio δS = 1 + √2.

  This file models time as a bidirectional construct by exploiting the
  inherent properties of the silver ratio.  The forward direction uses δS
  as a scaling factor; the backward direction uses its reciprocal 1/δS = √2−1.
  Their product is 1, reflecting temporal conservation under inversion.

  The time reversal operator T : t ↦ −t maps the past (timeDomain, t < 0)
  to the future (forwardDomain, t > 0) and vice versa.  Under reversal:
    • Forward and backward evolution compose to the identity: U(H,t)·U(H,−t)=1.
    • The Floquet phase is inverted: e^{iφ} = (e^{−iφ})⁻¹.
    • Coherence C(r) is preserved: C(r) = C(1/r)  (even symmetry).
    • The palindrome residual changes sign: Res(1/r) = −Res(r)  (odd symmetry).
    • A time crystal state reversed in time remains a time crystal state.

  The silver ratio's self-similar property δS² = 2·δS + 1 encodes the
  temporal nesting: applying the forward scale twice equals two forward steps
  plus the original coordinate.  The coherence value C(δS) = C(1/δS) = η
  shows that the silver ratio sits at the same coherence level whether
  approached from the forward or backward temporal direction.

  Sections
  ────────
  1.  Time reversal operator and domains
  2.  Silver ratio as bidirectional temporal scale
  3.  Time evolution under reversal
  4.  Coherence under bidirectional flow
  5.  Bidirectional time crystal

  Proof status
  ────────────
  All theorems have complete machine-checked proofs.
  No `sorry` placeholders remain.
-/

import SpaceTime

open Complex Real

noncomputable section

-- ════════════════════════════════════════════════════════════════════════════
-- Section 1 — Time Reversal Operator and Domains
-- The operator T(t) = −t exchanges past and future.
-- timeDomain (t < 0) and forwardDomain (t > 0) are exchanged by T.
-- ════════════════════════════════════════════════════════════════════════════

/-- The time reversal operator: negates a time coordinate, mapping past to future
    and future to past.

    In the causal framework where timeDomain = {t | t < 0} (past), applying T
    to any past instant yields a positive future instant. -/
def timeReversal (t : ℝ) : ℝ := -t

/-- The forward time domain: the set of positive real numbers, representing
    future instants.

    This is the image of timeDomain under time reversal. -/
def forwardDomain : Set ℝ := {t | 0 < t}

/-- Time reversal is an involution: applying T twice returns the original
    coordinate.  Reversing time and then reversing again is the identity. -/
theorem timeReversal_involution (t : ℝ) : timeReversal (timeReversal t) = t := by
  unfold timeReversal; ring

/-- Time reversal maps the past to the future: if t < 0 then −t > 0. -/
theorem timeReversal_past_to_future (t : ℝ) (ht : t ∈ timeDomain) :
    timeReversal t ∈ forwardDomain := by
  unfold timeReversal forwardDomain timeDomain at *
  simp only [Set.mem_setOf_eq] at *
  linarith

/-- Time reversal maps the future to the past: if t > 0 then −t < 0. -/
theorem timeReversal_future_to_past (t : ℝ) (ht : t ∈ forwardDomain) :
    timeReversal t ∈ timeDomain := by
  unfold timeReversal forwardDomain timeDomain at *
  simp only [Set.mem_setOf_eq] at *
  linarith

/-- Every nonzero time coordinate belongs to either the past or the future:
    the two domains partition the nonzero reals. -/
theorem time_domains_partition (t : ℝ) (ht : t ≠ 0) :
    t ∈ timeDomain ∨ t ∈ forwardDomain := by
  simp only [timeDomain, forwardDomain, Set.mem_setOf_eq]
  exact lt_or_gt_of_ne ht

/-- A time coordinate lies in the past if and only if its reversal lies in
    the future.  The two domains are mirror images under T. -/
theorem timeReversal_domain_iff (t : ℝ) :
    t ∈ timeDomain ↔ timeReversal t ∈ forwardDomain := by
  simp only [timeDomain, forwardDomain, Set.mem_setOf_eq, timeReversal]
  constructor <;> intro h <;> linarith

-- ════════════════════════════════════════════════════════════════════════════
-- Section 2 — Silver Ratio as Bidirectional Temporal Scale
-- Forward scale: δS > 1 (stretches time away from origin).
-- Backward scale: 1/δS < 1 (contracts time toward origin).
-- They compose to the identity: applying one then the other returns t.
-- ════════════════════════════════════════════════════════════════════════════

/-- The forward temporal scale: multiplying by δS > 1 stretches time away from
    the origin, corresponding to forward (future-directed) temporal flow. -/
def silverForward (t : ℝ) : ℝ := δS * t

/-- The backward temporal scale: multiplying by 1/δS = √2−1 < 1 contracts time
    toward the origin, corresponding to backward (past-directed) temporal flow. -/
def silverBackward (t : ℝ) : ℝ := (1 / δS) * t

/-- The silver ratio is strictly greater than 1.  Since δS = 1 + √2 and √2 > 0,
    the forward temporal scale genuinely accelerates time. -/
theorem silverRatio_gt_one : 1 < δS := by
  unfold δS
  have : (0 : ℝ) < Real.sqrt 2 := Real.sqrt_pos.mpr (by norm_num)
  linarith

/-- The backward temporal scale 1/δS is positive: backward flow is real. -/
theorem silver_backward_pos : 0 < 1 / δS :=
  div_pos one_pos silverRatio_pos

/-- The backward temporal scale 1/δS is strictly less than 1: backward flow
    contracts time toward the origin. -/
theorem silver_backward_lt_one : 1 / δS < 1 :=
  (div_lt_one silverRatio_pos).mpr silverRatio_gt_one

/-- Silver bidirectional conservation: δS · (1/δS) = 1.
    Applying the forward scale then the backward scale (or vice versa) returns
    to the original time coordinate — the two directions are mutual inverses. -/
theorem silver_bidir_conservation : δS * (1 / δS) = 1 := by
  field_simp [silverRatio_pos.ne']

/-- Applying the backward scale after the forward scale is the identity:
    silverBackward (silverForward t) = t.
    Round-tripping through both directions returns to the start. -/
theorem silver_forward_backward_id (t : ℝ) :
    silverBackward (silverForward t) = t := by
  unfold silverForward silverBackward
  field_simp [silverRatio_pos.ne']

/-- Applying the forward scale after the backward scale is also the identity:
    silverForward (silverBackward t) = t.
    The order of composition does not matter. -/
theorem silver_backward_forward_id (t : ℝ) :
    silverForward (silverBackward t) = t := by
  unfold silverForward silverBackward
  field_simp [silverRatio_pos.ne']

-- ════════════════════════════════════════════════════════════════════════════
-- Section 3 — Time Evolution Under Reversal
-- The time evolution operator U(H,t) = exp(−I·H·t) composed with its
-- time-reversed counterpart U(H,−t) = exp(I·H·t) equals the identity.
-- The Floquet phase e^{−iφ} satisfies the same inversion law.
-- ════════════════════════════════════════════════════════════════════════════

/-- Forward and backward time evolution compose to the identity:
        U(H, t) · U(H, −t) = 1.

    Proof: U(H, t + (−t)) = U(H, 0) = 1 by the group law and zero-time identity. -/
theorem timeEvolution_forward_backward_product (H t : ℝ) :
    timeEvolution H t * timeEvolution H (-t) = 1 := by
  rw [← timeEvolution_add, add_neg_cancel, timeEvolution_zero]

/-- The time-reversed evolution operator is the inverse of the forward one:
        U(H, −t) = U(H, t)⁻¹.

    Proof: U(H,t) is nonzero (it is an exponential), and U(H,t)·U(H,−t) = 1
    by `timeEvolution_forward_backward_product`, so the right factor is the
    multiplicative inverse. -/
theorem timeEvolution_reversal_inverse (H t : ℝ) :
    timeEvolution H (-t) = (timeEvolution H t)⁻¹ := by
  have h    := timeEvolution_forward_backward_product H t
  have hne  : timeEvolution H t ≠ 0 := Complex.exp_ne_zero _
  exact mul_left_cancel₀ hne (h.trans (mul_inv_cancel₀ hne).symm)

/-- Forward and backward Floquet phases compose to the identity:
        e^{−iφ} · e^{i φ} = 1.

    Proof: floquetPhase φ · floquetPhase (−φ) = floquetPhase (φ + (−φ)) = floquetPhase 0 = 1. -/
theorem floquetPhase_forward_backward (φ : ℝ) :
    floquetPhase φ * floquetPhase (-φ) = 1 := by
  rw [← floquetPhase_add, add_neg_cancel, floquetPhase_zero]

/-- The time-reversed Floquet phase is the inverse of the forward one:
        e^{iφ} = (e^{−iφ})⁻¹.

    Proof: analogous to `timeEvolution_reversal_inverse`, using the group law
    for floquetPhase and nonzero-ness of complex exponentials. -/
theorem floquetPhase_neg_inverse (φ : ℝ) :
    floquetPhase (-φ) = (floquetPhase φ)⁻¹ := by
  have h    := floquetPhase_forward_backward φ
  have hne  : floquetPhase φ ≠ 0 := Complex.exp_ne_zero _
  exact mul_left_cancel₀ hne (h.trans (mul_inv_cancel₀ hne).symm)

-- ════════════════════════════════════════════════════════════════════════════
-- Section 4 — Coherence Under Bidirectional Flow
-- The coherence function C(r) = 2r/(1+r²) is even: C(r) = C(1/r).
-- This means the coherence is identical whether the system runs forward
-- (amplitude ratio r) or backward (amplitude ratio 1/r).
-- The palindrome residual Res(r) is odd: Res(1/r) = −Res(r).
-- ════════════════════════════════════════════════════════════════════════════

/-- Coherence is invariant under time reversal: C(r) = C(1/r).
    The coherence of a state is the same whether measured in the forward or
    backward direction.  This is the central symmetry of bidirectional time. -/
theorem coherence_bidir (r : ℝ) (hr : 0 < r) : C r = C (1 / r) :=
  coherence_symm r hr

/-- The coherence at the backward silver scale equals η.
    Since C(δS) = η (coherence_at_silver_is_eta) and C is even,
    C(1/δS) = C(δS) = η: both temporal directions sit at the same
    canonical coherence level 1/√2. -/
theorem silver_coherence_backward : C (1 / δS) = η :=
  (coherence_symm δS silverRatio_pos).symm.trans coherence_at_silver_is_eta

/-- The palindrome residual changes sign under time reversal: Res(1/r) = −Res(r).
    While coherence is even (unchanged by r ↦ 1/r), the palindrome residual
    is odd: reversing the amplitude ratio flips its sign. -/
theorem palindrome_bidir (r : ℝ) (hr : 0 < r) : Res (1 / r) = -Res r :=
  palindrome_residual_antisymm r hr

/-- The palindrome residuals at the forward and backward silver scales cancel:
        Res(δS) + Res(1/δS) = 0.
    The silver ratio and its reciprocal contribute equal and opposite residuals,
    reflecting the perfect bidirectional balance of the silver temporal scale. -/
theorem silver_palindrome_sum_zero : Res δS + Res (1 / δS) = 0 :=
  palindrome_sum_zero δS silverRatio_pos

-- ════════════════════════════════════════════════════════════════════════════
-- Section 5 — Bidirectional Time Crystal
-- A time crystal state remains a time crystal when time is reversed.
-- The silver ratio's self-similar scaling law δS² = 2·δS + 1 governs how
-- doubly-forward-scaled time decomposes into two forward steps plus the
-- original — the temporal analogue of the silver continued-fraction nesting.
-- ════════════════════════════════════════════════════════════════════════════

/-- A time crystal state remains a time crystal under time reversal.

    If ψ(t+T) = −ψ(t) for all t (a standard period-doubling time crystal),
    then the time-reversed state ψ_R(t) = ψ(−t) also satisfies ψ_R(t+T) = −ψ_R(t).

    Proof: ψ_R(t+T) = ψ(−t−T).  From the crystal condition with s = −t−T:
    ψ(−t−T+T) = −ψ(−t−T), i.e., ψ(−t) = −ψ(−t−T), i.e., ψ(−t−T) = −ψ(−t) = −ψ_R(t). -/
theorem timeReversed_is_time_crystal (ψ : ℝ → ℂ) (T : ℝ)
    (h : isTimeCrystalState ψ T) :
    isTimeCrystalState (fun t => ψ (-t)) T := by
  unfold isTimeCrystalState isFloquetState at *
  intro t
  -- Reduce to: ψ (-(t + T)) = floquetPhase Real.pi * ψ (-t)
  show ψ (-(t + T)) = floquetPhase Real.pi * ψ (-t)
  rw [show -(t + T) = -t - T from by ring]
  -- Use the crystal condition at s = −t − T:
  -- h(−t−T) : ψ(−t−T+T) = floquetPhase π * ψ(−t−T)
  -- simplifies to: ψ(−t) = floquetPhase π * ψ(−t−T)
  have step := h (-t - T)
  rw [show -t - T + T = -t from by ring] at step
  -- step : ψ(-t) = floquetPhase π * ψ(-t-T)
  -- Rewrite floquetPhase π = −1 in both step and goal
  rw [floquetPhase_pi] at step ⊢
  -- step : ψ(-t) = -1 * ψ(-t-T)     goal: ψ(-t-T) = -1 * ψ(-t)
  linear_combination step

/-- Silver temporal nesting: applying the forward scale twice decomposes as
    two forward steps plus the original coordinate.

        silverForward (silverForward t) = 2 · silverForward t + t

    Proof: δS · (δS · t) = δS² · t = (2·δS + 1) · t = 2·(δS·t) + t,
    using the silver quadratic identity δS² = 2·δS + 1. -/
theorem bidir_silver_time_scale (t : ℝ) :
    silverForward (silverForward t) = 2 * silverForward t + t := by
  unfold silverForward
  calc δS * (δS * t)
      = δS ^ 2 * t           := by ring
    _ = (2 * δS + 1) * t     := by rw [silverRatio_sq]
    _ = 2 * (δS * t) + t     := by ring

end
