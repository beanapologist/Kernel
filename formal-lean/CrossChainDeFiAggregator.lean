/-
  CrossChainDeFiAggregator.lean — Lean 4 formalization of cross-chain DeFi
  aggregation mechanics for a Polkadot-based multi-parachain platform.

  This module provides machine-checked mathematical foundations for a
  decentralized finance platform that aggregates lending, borrowing, and
  swapping services across multiple parachains and EVM-compatible chains,
  using Polkadot's cross-chain message-passing (XCM) capabilities.

  Core model
  ──────────
  • Automated Market Maker (AMM): the constant-product invariant x · y = k
    governs token swaps on each chain.  The cross-chain aggregator selects
    the route yielding maximum output.

  • Lending / Borrowing: simple-interest model I = P · r · t, where P is
    principal, r is the per-period utilization rate, and t is holding period.
    The aggregator routes capital to the chain offering the highest rate.

  • Rate Aggregation: best_rate r₁ r₂ = max r₁ r₂ selects the dominant rate
    across chains.  Soundness (best_rate ≥ every individual rate) and
    optimality (best_rate is the least upper bound) are machine-checked.

  Sections
  ────────
  1.  AMM constant-product invariant and output formula
  2.  Cross-chain swap price and invariant preservation
  3.  Slippage and price-impact bounds
  4.  Lending / borrowing simple-interest model
  5.  Cross-chain rate aggregation (best-rate selection)
  6.  LP value and monotone-output properties

  Proof status
  ────────────
  All 20 theorems have complete machine-checked proofs.
  No `sorry` placeholders remain.

  Connection to Kernel
  ────────────────────
  The AMM output formula out = y · Δ / (x + Δ) mirrors the Kernel coherence
  function C(r) = 2r/(1+r²): both express a ratio bounded above by a reserve
  quantity and equal to zero at the origin.  The best-rate aggregation is
  structurally identical to the max-coherence selection in KernelAxle.lean,
  and the lending-interest bound parallels the Floquet quasi-energy shift
  formalised in FineStructure.lean.
-/

import SpeedOfLight

open Complex Real

noncomputable section

-- ════════════════════════════════════════════════════════════════════════════
-- Definitions
-- ════════════════════════════════════════════════════════════════════════════

/-- AMM output: the amount of token Y received when swapping Δ units of token X
    into a pool with reserves x (of X) and y (of Y).

    Derived from the constant-product invariant (x + Δ) · (y − out) = x · y:
    solving for out gives  out = y · Δ / (x + Δ). -/
noncomputable def amm_out (x y Δ : ℝ) : ℝ := y * Δ / (x + Δ)

/-- AMM spot price: the marginal price of Y in units of X at pool state (x, y).
    One unit of Y costs x/y units of X at the current margin. -/
noncomputable def amm_price (x y : ℝ) : ℝ := x / y

/-- Simple lending interest: I = P · r · t, where P is principal,
    r is the per-period utilization rate, and t is the holding period. -/
noncomputable def lending_interest (P r t : ℝ) : ℝ := P * r * t

/-- Best rate from two chains: selects the maximum available rate. -/
noncomputable def best_rate (r₁ r₂ : ℝ) : ℝ := max r₁ r₂

/-- Liquidity-provider (LP) value: the geometric mean √(x · y) of the pool
    reserves, proportional to the number of LP shares outstanding. -/
noncomputable def lp_value (x y : ℝ) : ℝ := Real.sqrt (x * y)

-- ════════════════════════════════════════════════════════════════════════════
-- Section 1 — AMM Constant-Product Invariant and Output Formula
-- Invariant: x · y = k (preserved by every swap).
-- Output for swapping Δ of token X: out(x, y, Δ) = y · Δ / (x + Δ).
-- ════════════════════════════════════════════════════════════════════════════

/-- The AMM constant-product k = x · y is strictly positive when both reserves
    are strictly positive. -/
theorem amm_invariant_pos (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
    0 < x * y := mul_pos hx hy

/-- The AMM output is strictly positive for positive reserves and positive
    input amount Δ. -/
theorem amm_out_pos (x y Δ : ℝ) (hx : 0 < x) (hy : 0 < y) (hΔ : 0 < Δ) :
    0 < amm_out x y Δ := by
  unfold amm_out
  exact div_pos (mul_pos hy hΔ) (by linarith)

/-- Zero input yields zero output: swapping nothing returns nothing. -/
theorem amm_out_zero_input (x y : ℝ) (_hx : 0 < x) :
    amm_out x y 0 = 0 := by
  unfold amm_out
  simp

-- ════════════════════════════════════════════════════════════════════════════
-- Section 2 — Cross-chain Swap Price and Invariant Preservation
-- ════════════════════════════════════════════════════════════════════════════

/-- The AMM spot price is strictly positive when both reserves are positive. -/
theorem amm_price_pos (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
    0 < amm_price x y :=
  div_pos hx hy

/-- The constant-product invariant is preserved after a swap: after token X
    increases from x to (x + Δ) while token Y decreases by amm_out, the
    product (x + Δ) · (y − amm_out x y Δ) equals the original x · y. -/
theorem amm_invariant_preserved (x y Δ : ℝ) (hx : 0 < x) (_hy : 0 < y)
    (hΔ : 0 < Δ) :
    (x + Δ) * (y - amm_out x y Δ) = x * y := by
  unfold amm_out
  have hd : (x + Δ) ≠ 0 := by linarith
  field_simp [hd]
  ring

/-- The AMM output is strictly less than the full Y reserve: a finite input
    cannot drain the pool entirely.  This is the fundamental pool-safety
    property of the constant-product AMM. -/
theorem amm_out_bounded (x y Δ : ℝ) (hx : 0 < x) (hy : 0 < y) (hΔ : 0 < Δ) :
    amm_out x y Δ < y := by
  unfold amm_out
  rw [div_lt_iff₀ (by linarith : (0 : ℝ) < x + Δ)]
  nlinarith

-- ════════════════════════════════════════════════════════════════════════════
-- Section 3 — Slippage and Price-Impact Bounds
-- ════════════════════════════════════════════════════════════════════════════

/-- Slippage is always positive: the effective swap rate y · Δ / (x + Δ) is
    strictly less than the pro-rata rate y · Δ / x because x + Δ > x.
    Every finite trade on a constant-product AMM incurs positive price impact. -/
theorem amm_slippage_positive (x y Δ : ℝ) (hx : 0 < x) (hy : 0 < y)
    (hΔ : 0 < Δ) :
    amm_out x y Δ < y * Δ / x := by
  unfold amm_out
  rw [div_lt_div_iff₀ (by linarith : (0 : ℝ) < x + Δ) hx]
  nlinarith [mul_pos (mul_pos hy hΔ) hΔ]

/-- Price impact is strictly less than 1: the pool always retains a positive
    fraction of its Y reserve after any finite trade. -/
theorem amm_price_impact_lt_one (x y Δ : ℝ) (hx : 0 < x) (hy : 0 < y)
    (hΔ : 0 < Δ) :
    amm_out x y Δ / y < 1 := by
  rw [div_lt_one hy]
  exact amm_out_bounded x y Δ hx hy hΔ

/-- AMM output is strictly monotone in the input: a larger trade yields more
    tokens out, all else equal. -/
theorem amm_out_monotone (x y Δ₁ Δ₂ : ℝ) (hx : 0 < x) (hy : 0 < y)
    (hΔ₁ : 0 < Δ₁) (h : Δ₁ < Δ₂) :
    amm_out x y Δ₁ < amm_out x y Δ₂ := by
  unfold amm_out
  have hΔ₂ : 0 < Δ₂ := lt_trans hΔ₁ h
  have hd₁ : (0 : ℝ) < x + Δ₁ := by linarith
  have hd₂ : (0 : ℝ) < x + Δ₂ := by linarith
  rw [div_lt_div_iff₀ hd₁ hd₂]
  nlinarith [mul_pos hy hx]

-- ════════════════════════════════════════════════════════════════════════════
-- Section 4 — Lending / Borrowing Simple-Interest Model
-- ════════════════════════════════════════════════════════════════════════════

/-- Lending interest is non-negative when principal, rate, and time are
    all non-negative. -/
theorem lending_interest_nonneg (P r t : ℝ) (hP : 0 ≤ P) (hr : 0 ≤ r)
    (ht : 0 ≤ t) : 0 ≤ lending_interest P r t := by
  unfold lending_interest
  exact mul_nonneg (mul_nonneg hP hr) ht

/-- Lending interest is strictly positive when principal, rate, and time are
    all strictly positive. -/
theorem lending_interest_pos (P r t : ℝ) (hP : 0 < P) (hr : 0 < r)
    (ht : 0 < t) : 0 < lending_interest P r t := by
  unfold lending_interest
  exact mul_pos (mul_pos hP hr) ht

/-- The total repayment amount P + I strictly exceeds the principal P when
    interest is positive.  A lender is always made whole plus a positive yield. -/
theorem lending_amount_exceeds_principal (P r t : ℝ) (hP : 0 < P) (hr : 0 < r)
    (ht : 0 < t) : P < P + lending_interest P r t :=
  lt_add_of_pos_right P (lending_interest_pos P r t hP hr ht)

/-- Higher rate → more interest: lending_interest is strictly increasing in r.
    The aggregator should route to the chain with the highest lending rate. -/
theorem lending_rate_monotone (P r₁ r₂ t : ℝ) (hP : 0 < P) (ht : 0 < t)
    (hr : r₁ < r₂) : lending_interest P r₁ t < lending_interest P r₂ t := by
  unfold lending_interest
  have h : P * r₁ < P * r₂ := mul_lt_mul_of_pos_left hr hP
  exact mul_lt_mul_of_pos_right h ht

-- ════════════════════════════════════════════════════════════════════════════
-- Section 5 — Cross-chain Rate Aggregation (Best-rate Selection)
-- best_rate r₁ r₂ = max r₁ r₂ is the optimal rate across two chains.
-- ════════════════════════════════════════════════════════════════════════════

/-- The aggregated best rate is at least the first chain's rate. -/
theorem best_rate_ge_left (r₁ r₂ : ℝ) : r₁ ≤ best_rate r₁ r₂ :=
  le_max_left r₁ r₂

/-- The aggregated best rate is at least the second chain's rate. -/
theorem best_rate_ge_right (r₁ r₂ : ℝ) : r₂ ≤ best_rate r₁ r₂ :=
  le_max_right r₁ r₂

/-- Best-rate aggregation is symmetric: the order in which chains are compared
    does not affect the result.  Chain ordering is irrelevant to optimality. -/
theorem best_rate_symm (r₁ r₂ : ℝ) : best_rate r₁ r₂ = best_rate r₂ r₁ :=
  max_comm r₁ r₂

/-- Optimality: any rate that dominates all individual chains also dominates
    the aggregated best rate.  The aggregator is the tightest lower bound. -/
theorem best_rate_optimal (r₁ r₂ r : ℝ) (h₁ : r₁ ≤ r) (h₂ : r₂ ≤ r) :
    best_rate r₁ r₂ ≤ r := max_le h₁ h₂

/-- Idempotence: comparing a rate with itself yields that same rate. -/
theorem best_rate_idempotent (r : ℝ) : best_rate r r = r := max_self r

-- ════════════════════════════════════════════════════════════════════════════
-- Section 6 — LP Value and Monotone-Output Properties
-- ════════════════════════════════════════════════════════════════════════════

/-- LP value is strictly positive when both reserves are positive.
    A pool with positive reserves always has positive depth. -/
theorem lp_value_pos (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
    0 < lp_value x y := by
  unfold lp_value
  exact Real.sqrt_pos.mpr (mul_pos hx hy)

/-- A deeper pool (larger reserve product) has a strictly larger LP value:
    if x < x' and y < y' then lp_value x y < lp_value x' y'.
    Larger reserves provide greater liquidity depth and lower slippage. -/
theorem lp_value_monotone (x y x' y' : ℝ) (hx : 0 < x) (hy : 0 < y)
    (hx' : x < x') (hy' : y < y') :
    lp_value x y < lp_value x' y' := by
  unfold lp_value
  apply Real.sqrt_lt_sqrt (le_of_lt (mul_pos hx hy))
  have hx'pos : 0 < x' := lt_trans hx hx'
  have hy'pos : 0 < y' := lt_trans hy hy'
  calc x * y < x * y' := by nlinarith
    _ < x' * y' := by nlinarith

end
