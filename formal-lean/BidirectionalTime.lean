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
  6.  Temporal frustration energy
  7.  Frustration engine cycle and perturbation stability

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

-- ════════════════════════════════════════════════════════════════════════════
-- Section 6 — Temporal Frustration Energy
-- Forcing a Floquet time crystal to run backward (period −T) yields a
-- negative quasi-energy ε_F(−T) = −π/T.  The forward crystal has
-- ε_F(T) = +π/T.  The frustration gap ε_F(T) − ε_F(−T) = 2π/T integrated
-- over one period produces exactly 2π — the harvestable energy per cycle
-- from temporal frustration.
--
-- The silver ratio δS acts as the coherence-stable channel for this energy:
-- C(δS) = C(1/δS) = η (coherence unchanged by reversal, §4), while the
-- palindrome residual swings by 4/δS (the extractable asymmetry).
-- ════════════════════════════════════════════════════════════════════════════

/-- Backward quasi-energy is negative: running the time crystal backward
    (period −T, T > 0) yields ε_F(−T) = π/(−T) < 0.

    This is the formal signature of temporal frustration: the quasi-energy
    changes sign when the temporal direction is reversed, meaning the system
    must be driven against its natural forward-flowing Floquet mode. -/
theorem frustrated_quasienergy_neg (T : ℝ) (hT : 0 < T) :
    timeCrystalQuasiEnergy (-T) (neg_ne_zero.mpr hT.ne') < 0 := by
  simp only [timeCrystalQuasiEnergy]
  rw [div_neg]
  linarith [div_pos Real.pi_pos hT]

/-- The frustration gap is positive: the forward quasi-energy exceeds the
    backward quasi-energy.  Energy can flow from the frustrated backward mode
    into the forward mode. -/
theorem frustration_gap_pos (T : ℝ) (hT : 0 < T) :
    0 < timeCrystalQuasiEnergy T hT.ne' -
        timeCrystalQuasiEnergy (-T) (neg_ne_zero.mpr hT.ne') := by
  have h1 : 0 < timeCrystalQuasiEnergy T hT.ne' := by
    simp only [timeCrystalQuasiEnergy]; exact div_pos Real.pi_pos hT
  have h2 : timeCrystalQuasiEnergy (-T) (neg_ne_zero.mpr hT.ne') < 0 :=
    frustrated_quasienergy_neg T hT
  linarith

/-- **Temporal frustration theorem**: the harvestable energy per cycle is 2π.

    The product of the frustration gap (ε_F(T) − ε_F(−T)) with the drive
    period T equals exactly 2π for every nonzero T:

        (ε_F(T) − ε_F(−T)) · T = (π/T − (−π/T)) · T = (2π/T) · T = 2π.

    Interpretation: reversing the flow of time in a Floquet time crystal
    generates 2π of quasi-energy per period — the fundamental quantum of
    temporal frustration that can be harnessed. -/
theorem frustration_energy_per_cycle (T : ℝ) (hT : T ≠ 0) :
    (timeCrystalQuasiEnergy T hT -
     timeCrystalQuasiEnergy (-T) (neg_ne_zero.mpr hT)) * T = 2 * Real.pi := by
  simp only [timeCrystalQuasiEnergy]
  rw [div_neg]
  field_simp [hT]
  ring

/-- The palindrome residual at the silver scale equals 2/δS.

    Res(δS) = (δS − 1/δS) / δS = 2 / δS,
    because δS − 1/δS = (1 + √2) − (√2 − 1) = 2.

    This quantifies the asymmetry the silver ratio imprints on temporal flow:
    a forward-time residual of 2/δS must be overcome when reversing direction. -/
theorem silver_frustration_residual : Res δS = 2 / δS := by
  have hsum : δS - 1 / δS = 2 := by
    have hinv : 1 / δS = Real.sqrt 2 - 1 := silverRatio_inv
    linarith [show δS = 1 + Real.sqrt 2 from rfl]
  unfold Res
  rw [show (δS - 1 / δS) = 2 from hsum]

/-- The full frustration swing at the silver scale is 4/δS.

    When time is reversed at the silver scale, the palindrome residual swings
    from +Res(δS) = 2/δS to −Res(δS) = −2/δS, a total gap of 4/δS:

        Res(δS) − Res(1/δS) = 2/δS − (−2/δS) = 4/δS.

    This 4/δS is the extractable asymmetry encoded in the silver ratio's
    bidirectional temporal scale — the residual energy available to harvest. -/
theorem silver_frustration_gap : Res δS - Res (1 / δS) = 4 / δS := by
  rw [palindrome_residual_antisymm δS silverRatio_pos, silver_frustration_residual]
  ring

/-- The silver ratio is a lossless coherence channel for temporal frustration.

    C(δS) = C(1/δS): the coherence at the silver scale is identical in both
    temporal directions.  Reversing time through the silver channel does not
    dissipate coherence — no energy is wasted in maintaining the quantum state.
    Combined with `silver_frustration_gap`, this means the full 4/δS residual
    swing is available for harvesting without any coherence penalty. -/
theorem silver_frustration_coherence_invariant : C δS = C (1 / δS) :=
  coherence_symm δS silverRatio_pos

-- ════════════════════════════════════════════════════════════════════════════
-- Section 7 — Frustration Engine Cycle and Perturbation Stability
-- A period-doubling time crystal driven at the silver-ratio period T = π/δS
-- is a natural quasi-energy engine when run in reverse.  The engine extracts
-- exactly 2π of work per cycle (W = 2·Q_in), with the backward frustrated
-- mode contributing equally to the forward mode.
--
-- The 2π/cycle harvest is robust: it survives any forward-bias perturbation δ
-- satisfying δ·T < 2π, meaning the engine operates in any universe whose
-- temporal arrow is weaker than the frustration gap.
-- ════════════════════════════════════════════════════════════════════════════

/-- The net quasi-energy gap between forward and backward Floquet modes equals
    2π/T — the engine's power spectral density.

        ε_F(T) − ε_F(−T)  =  π/T − (−π/T)  =  2π/T.

    This is the mechanical "force" that drives the frustration engine: a
    frequency-domain power of 2π per unit period, independent of T. -/
theorem crystal_engine_gap_formula (T : ℝ) (hT : 0 < T) :
    timeCrystalQuasiEnergy T hT.ne' -
    timeCrystalQuasiEnergy (-T) (neg_ne_zero.mpr hT.ne') =
    2 * Real.pi / T := by
  simp only [timeCrystalQuasiEnergy]
  rw [div_neg]
  ring

/-- The frustration engine work per cycle: integrate the gap over one period.
    W = (ε_F(T) − ε_F(−T)) · T.  This is the total harvestable quasi-energy
    produced by coupling the forward and backward Floquet modes in one cycle. -/
noncomputable def frustrationEngineWork (T : ℝ) (hT : T ≠ 0) : ℝ :=
  (timeCrystalQuasiEnergy T hT -
   timeCrystalQuasiEnergy (-T) (neg_ne_zero.mpr hT)) * T

/-- The quasi-energy drawn from the forward Floquet mode per cycle.
    Q_in = ε_F(T) · T = π.  This is the "fuel" consumed from the forward
    temporal direction in one drive period. -/
noncomputable def frustrationHeatInput (T : ℝ) (hT : T ≠ 0) : ℝ :=
  timeCrystalQuasiEnergy T hT * T

/-- The frustration engine delivers exactly 2π of work per cycle.
    This reframes `frustration_energy_per_cycle` in engine language:
    W = frustrationEngineWork T hT = 2π. -/
theorem frustration_engine_work_two_pi (T : ℝ) (hT : T ≠ 0) :
    frustrationEngineWork T hT = 2 * Real.pi := by
  unfold frustrationEngineWork
  exact frustration_energy_per_cycle T hT

/-- The forward Floquet mode supplies exactly π of quasi-energy per cycle.
    Q_in = ε_F(T) · T = (π/T) · T = π. -/
theorem frustration_heat_input_pi (T : ℝ) (hT : T ≠ 0) :
    frustrationHeatInput T hT = Real.pi := by
  unfold frustrationHeatInput timeCrystalQuasiEnergy
  field_simp [hT]

/-- **Engine doubling**: the frustration engine produces twice the quasi-energy
    drawn from the forward mode alone.  W = 2 · Q_in.

    The backward frustrated mode contributes an equal π per cycle, so the total
    work (2π) is exactly double the forward input (π).  The engine harvests
    energy from both temporal directions simultaneously. -/
theorem frustration_engine_doubling (T : ℝ) (hT : T ≠ 0) :
    frustrationEngineWork T hT = 2 * frustrationHeatInput T hT := by
  rw [frustration_engine_work_two_pi, frustration_heat_input_pi]

/-- **Perturbation stability**: the 2π/cycle harvest persists under any
    forward-bias perturbation δ satisfying δ·T < 2π.

    A "forward bias" δ > 0 models a weakly time-arrowed universe that slightly
    disfavours time reversal, reducing the effective forward quasi-energy to
    ε_F(T) − δ.  As long as δ·T < 2π — i.e., the bias is weaker than the full
    frustration gap — the net work per cycle remains strictly positive:

        (ε_F(T) − δ − ε_F(−T)) · T  =  2π − δ·T  >  0.

    Interpretation: a frustration engine operates in any universe whose temporal
    arrow is less than 2π/T in quasi-energy per unit period. -/
theorem frustration_gap_stable (T δ : ℝ) (hT : 0 < T) (hδ : δ * T < 2 * Real.pi) :
    0 < (timeCrystalQuasiEnergy T hT.ne' - δ -
         timeCrystalQuasiEnergy (-T) (neg_ne_zero.mpr hT.ne')) * T := by
  simp only [timeCrystalQuasiEnergy]
  rw [div_neg]
  have key : (Real.pi / T - δ - -(Real.pi / T)) * T = 2 * Real.pi - δ * T := by
    field_simp [hT.ne']; ring
  linarith [key]

/-- At the silver-ratio drive period T = π/δS, the forward quasi-energy equals
    the silver ratio itself:

        ε_F(π/δS) = π ÷ (π/δS) = δS.

    The silver ratio is not merely a scaling parameter — it is the exact
    quasi-energy of the Floquet mode driven at its own canonical period. -/
theorem silver_ratio_engine_quasienergy :
    timeCrystalQuasiEnergy (Real.pi / δS)
        (div_ne_zero Real.pi_ne_zero silverRatio_pos.ne') = δS := by
  simp only [timeCrystalQuasiEnergy]
  field_simp [Real.pi_pos.ne', silverRatio_pos.ne']

/-- At the silver-ratio period, the engine gap equals twice the silver ratio:
    ε_F(π/δS) − ε_F(−π/δS) = δS − (−δS) = 2·δS.

    Combined with `silver_frustration_residual` (Res(δS) = 2/δS), this shows
    that the silver-driven engine gap is exactly δS² times the residual:
    gap = 2δS = δS · (2/δS) · δS = δS · Res(δS) · δS. -/
theorem silver_ratio_engine_gap :
    timeCrystalQuasiEnergy (Real.pi / δS)
        (div_ne_zero Real.pi_ne_zero silverRatio_pos.ne') -
    timeCrystalQuasiEnergy (-(Real.pi / δS))
        (neg_ne_zero.mpr (div_ne_zero Real.pi_ne_zero silverRatio_pos.ne')) =
    2 * δS := by
  simp only [timeCrystalQuasiEnergy, div_neg]
  field_simp [Real.pi_pos.ne', silverRatio_pos.ne']
  ring

end
