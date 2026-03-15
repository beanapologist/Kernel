/-
  EthereumTradingBot.lean — Lean 4 formal verification of core invariants
  for an Ethereum automated market maker (AMM) trading bot.

  This module establishes machine-checked mathematical guarantees for a
  trading bot operating on Ethereum's decentralized exchange infrastructure.
  The central primitive is Uniswap's constant-product AMM whose invariant

      x · y = k   (Uniswap v2 constant-product formula)

  underpins price discovery and trade execution on Ethereum. All theorems
  are self-contained real-arithmetic results that do not depend on any
  probabilistic or economic assumptions beyond stated hypotheses.

  Sections
  ────────
  1.  Constant-product invariant  — k = x·y is preserved through every swap
  2.  Output and reserve safety   — trades are bounded and reserves stay positive
  3.  Spot price and price impact — every trade moves the price against the trader
  4.  Liquidity arithmetic        — adding liquidity preserves price and grows k
  5.  Trading strategy bounds     — Kelly sizing and slippage limits

  Proof status
  ────────────
  All 20 theorems have complete machine-checked proofs.
  No `sorry` placeholders remain.

  Limitations
  ───────────
  · Reserves and trade sizes are abstract positive reals; discrete token
    quantities and integer rounding are not modelled.
  · The constant-product formula is fee-free; a fee parameter f ∈ (0,1)
    would replace dx with dx·(1−f) in the output formula, preserving all
    qualitative results.
  · The Kelly criterion theorems use the simple discrete two-outcome model;
    continuous-time generalisations are left for future work.
-/

-- Import SpeedOfLight to maintain the module chain (this module follows SpeedOfLight
-- in the library build order; all Mathlib real-arithmetic machinery is available
-- transitively through the chain).
import SpeedOfLight

open Real

noncomputable section

-- ════════════════════════════════════════════════════════════════════════════
-- §1  Constant-product AMM invariant  (Uniswap v2 core)
-- ════════════════════════════════════════════════════════════════════════════

/-- The constant-product market-maker invariant k = x · y. -/
def amm_k (x y : ℝ) : ℝ := x * y

/-- **Positivity**: k is strictly positive whenever both token reserves are positive. -/
theorem amm_k_pos (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 0 < amm_k x y :=
  mul_pos hx hy

/-- **Symmetry**: the constant-product invariant treats both tokens equally —
    swapping the labels x and y leaves k unchanged. -/
theorem amm_k_symm (x y : ℝ) : amm_k x y = amm_k y x := by
  simp only [amm_k]; ring

/-- The output amount dy obtained by depositing dx tokens of x under the
    constant-product rule:  dy = y · dx / (x + dx). -/
def amm_out (x y dx : ℝ) : ℝ := y * dx / (x + dx)

/-- **Invariant preservation**: a swap with input dx and output dy = amm_out x y dx
    preserves k exactly — the constant product is maintained after every trade. -/
theorem amm_invariant_preserved (x y dx : ℝ) (hx : 0 < x) (hy : 0 < y) (hdx : 0 < dx) :
    amm_k (x + dx) (y - amm_out x y dx) = amm_k x y := by
  simp only [amm_k, amm_out]
  have hxdx : (x + dx) ≠ 0 := by positivity
  field_simp
  ring

-- ════════════════════════════════════════════════════════════════════════════
-- §2  Output amount and reserve safety
-- ════════════════════════════════════════════════════════════════════════════

/-- **Output positivity**: the output dy is strictly positive for any
    strictly positive input dx — every trade produces a real gain. -/
theorem amm_out_pos (x y dx : ℝ) (hx : 0 < x) (hy : 0 < y) (hdx : 0 < dx) :
    0 < amm_out x y dx :=
  div_pos (mul_pos hy hdx) (by linarith)

/-- **Drain bound**: the output dy is strictly less than the full reserve y —
    a single swap can never drain an AMM pool to zero. -/
theorem amm_out_lt_reserve (x y dx : ℝ) (hx : 0 < x) (hy : 0 < y) (hdx : 0 < dx) :
    amm_out x y dx < y := by
  simp only [amm_out]
  rw [div_lt_iff (by linarith : 0 < x + dx)]
  nlinarith

/-- **Reserve safety**: the y-reserve remains strictly positive after every swap.
    No valid trade puts the pool into a state with non-positive reserves. -/
theorem amm_reserve_y_pos (x y dx : ℝ) (hx : 0 < x) (hy : 0 < y) (hdx : 0 < dx) :
    0 < y - amm_out x y dx :=
  sub_pos.mpr (amm_out_lt_reserve x y dx hx hy hdx)

/-- **No free lunch**: zero input produces zero output — the AMM never gives
    tokens away for free. -/
theorem amm_no_free_lunch (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
    amm_out x y 0 = 0 := by
  simp [amm_out]

/-- **Fee-free reversibility**: in a fee-free AMM, buying dy tokens and
    immediately selling them back costs exactly dx — the trade is lossless
    when reversed. -/
theorem amm_fee_free_round_trip (x y dx : ℝ) (hx : 0 < x) (hy : 0 < y) (hdx : 0 < dx) :
    amm_out (y - amm_out x y dx) (x + dx) (amm_out x y dx) = dx := by
  simp only [amm_out]
  have hxdx : (x + dx) ≠ 0 := by positivity
  have hyne : y ≠ 0 := hy.ne'
  field_simp
  ring

-- ════════════════════════════════════════════════════════════════════════════
-- §3  Spot price, effective price, and price impact
-- ════════════════════════════════════════════════════════════════════════════

/-- The AMM spot price: units of x paid per unit of y at current reserves (= x / y). -/
def amm_price (x y : ℝ) : ℝ := x / y

/-- **Price positivity**: the spot price is strictly positive when reserves
    are strictly positive. -/
theorem amm_price_pos (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 0 < amm_price x y :=
  div_pos hx hy

/-- **Price impact**: after buying y tokens, the new spot price equals (x + dx)² / (x · y),
    which strictly exceeds the pre-trade price x / y.
    Every buy trade pushes the price up against the buyer. -/
theorem amm_price_increases_after_buy (x y dx : ℝ) (hx : 0 < x) (hy : 0 < y) (hdx : 0 < dx) :
    amm_price x y < amm_price (x + dx) (y - amm_out x y dx) := by
  have hxdx : 0 < x + dx := by linarith
  have hxdxne : (x + dx) ≠ 0 := hxdx.ne'
  have hxne : x ≠ 0 := hx.ne'
  have hyne : y ≠ 0 := hy.ne'
  -- Simplify the new price to (x + dx)² / (x · y)
  have new_price : amm_price (x + dx) (y - amm_out x y dx) = (x + dx) ^ 2 / (x * y) := by
    simp only [amm_price, amm_out]
    field_simp
    ring
  rw [new_price, amm_price, div_lt_div_iff hy (mul_pos hx hy)]
  nlinarith [mul_pos hx hdx, sq_nonneg dx]

/-- **Effective price exceeds spot**: the execution price dx / dy is always
    strictly greater than the pre-trade spot price x / y.
    Traders always pay more per token than the quoted price. -/
theorem amm_effective_price_gt_spot (x y dx : ℝ) (hx : 0 < x) (hy : 0 < y) (hdx : 0 < dx) :
    amm_price x y < dx / amm_out x y dx := by
  simp only [amm_price, amm_out]
  have hxdx : 0 < x + dx := by linarith
  -- dx / (y·dx/(x+dx)) simplifies to (x+dx)/y
  have h : dx / (y * dx / (x + dx)) = (x + dx) / y := by
    field_simp
    ring
  rw [h]
  exact (div_lt_div_right hy).mpr (by linarith)

/-- **Relative price impact formula**: the relative price impact of a trade of
    size dx in a pool with x-reserve x equals dx / x.
    A 1% trade size causes a 1% price impact. -/
theorem amm_price_impact_formula (x y dx : ℝ) (hx : 0 < x) (hy : 0 < y) (hdx : 0 < dx) :
    (dx / amm_out x y dx - amm_price x y) / amm_price x y = dx / x := by
  simp only [amm_price, amm_out]
  have hxdxne : (x + dx) ≠ 0 := by positivity
  have hxne : x ≠ 0 := hx.ne'
  have hyne : y ≠ 0 := hy.ne'
  -- dx / (y·dx/(x+dx)) = (x+dx)/y
  have h : dx / (y * dx / (x + dx)) = (x + dx) / y := by field_simp; ring
  rw [h]
  field_simp
  ring

/-- **Depth reduces impact**: for the same trade size dx, a deeper pool
    (larger x-reserve) produces strictly smaller relative price impact dx/x. -/
theorem amm_deeper_pool_less_impact (x₁ x₂ dx : ℝ)
    (hx₁ : 0 < x₁) (hx₂ : 0 < x₂) (h : x₁ < x₂) (hdx : 0 < dx) :
    dx / x₂ < dx / x₁ := by
  rw [div_lt_div_iff hx₂ hx₁]
  exact mul_lt_mul_of_pos_left h hdx

-- ════════════════════════════════════════════════════════════════════════════
-- §4  Liquidity arithmetic
-- ════════════════════════════════════════════════════════════════════════════

/-- The total value of an LP position measured in x-tokens:
    x tokens in the pool plus the x-value of the y-tokens at spot price. -/
def amm_lp_value_x (x y : ℝ) : ℝ := x + y * amm_price x y

/-- **LP value formula**: the total x-denominated value of an LP position is
    2 · x — the position is worth exactly twice the x-reserve held in the pool. -/
theorem amm_lp_value_formula (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
    amm_lp_value_x x y = 2 * x := by
  simp only [amm_lp_value_x, amm_price]
  field_simp
  ring

/-- **Balanced addition preserves price**: adding liquidity in proportion
    Δx/x = Δy/y = r > 0 does not move the spot price — price is unchanged. -/
theorem amm_balanced_add_preserves_price (x y r : ℝ)
    (hx : 0 < x) (hy : 0 < y) (hr : 0 < r) :
    amm_price (x * (1 + r)) (y * (1 + r)) = amm_price x y := by
  simp only [amm_price]
  have hr1 : (1 + r) ≠ 0 := by positivity
  have hyne : y ≠ 0 := hy.ne'
  field_simp
  ring

/-- **Liquidity grows k**: balanced liquidity addition with ratio r > 0 strictly
    increases k — the pool invariant scales by (1 + r)². -/
theorem amm_k_increases_with_liquidity (x y r : ℝ)
    (hx : 0 < x) (hy : 0 < y) (hr : 0 < r) :
    amm_k x y < amm_k (x * (1 + r)) (y * (1 + r)) := by
  simp only [amm_k]
  nlinarith [mul_pos hx hy, mul_pos hr (mul_pos hx hy), sq_nonneg r]

-- ════════════════════════════════════════════════════════════════════════════
-- §5  Trading strategy bounds
-- ════════════════════════════════════════════════════════════════════════════

/-- The Kelly optimal bet fraction in a two-outcome game:
      f* = (p · b − (1 − p)) / b
    where p is win probability and b is the odds (win b per 1 unit risked). -/
def kelly_fraction (p b : ℝ) : ℝ := (p * b - (1 - p)) / b

/-- **Kelly positivity**: the Kelly fraction is positive if and only if the
    expected value of the bet is positive — edge is required to bet. -/
theorem kelly_fraction_pos_iff (p b : ℝ) (hb : 0 < b) :
    0 < kelly_fraction p b ↔ 1 - p < p * b := by
  simp only [kelly_fraction]
  constructor
  · intro h
    rcases div_pos_iff.mp h with ⟨h1, _⟩ | ⟨_, h2⟩
    · linarith
    · linarith
  · intro h
    exact div_pos (by linarith) hb

/-- **Kelly ≤ 1**: when the win probability p ≤ 1, the Kelly fraction never
    exceeds 1 — the strategy never bets more than the full bankroll. -/
theorem kelly_fraction_le_one (p b : ℝ) (hb : 0 < b) (hp : p ≤ 1) :
    kelly_fraction p b ≤ 1 := by
  simp only [kelly_fraction]
  rw [div_le_one hb]
  nlinarith [mul_nonneg (show 0 ≤ 1 - p by linarith) (show 0 ≤ b + 1 by linarith)]

/-- **Capital safety**: with Kelly fraction f ≤ 1 and positive capital C,
    the remaining capital C − f · C after a loss is non-negative. -/
theorem kelly_capital_non_negative (f C : ℝ)
    (hf0 : 0 ≤ f) (hf1 : f ≤ 1) (hC : 0 < C) :
    0 ≤ C - f * C := by nlinarith

/-- **Slippage bound**: any trade with dx ≤ s · x satisfies price impact dx/x ≤ s,
    giving the bot a formal guarantee on worst-case slippage. -/
theorem slippage_bound (x dx s : ℝ)
    (hx : 0 < x) (hdx : 0 < dx) (hs : 0 < s) (h : dx ≤ s * x) :
    dx / x ≤ s := by
  rwa [div_le_iff hx]

end
