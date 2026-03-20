/-
  CryptoBridge.lean — Lean 4 formalization of cross-chain crypto bridge
  security properties.

  This module provides machine-checked mathematical foundations for a
  cross-chain bridge protocol.  The bridge enables asset transfers between
  blockchains via a lock-and-mint / burn-and-unlock mechanism, secured by
  over-collateralised relayers and Hash Time-Lock Contracts (HTLCs).

  Core model
  ──────────
  • Lock/Mint:  a depositor locks `amount` on the source chain, paying a
    protocol fee `fee`.  The bridge mints `amount − fee` wrapped tokens on
    the destination chain, conserving net value exactly.

  • Collateral:  relayers post collateral c ≥ locked l.  The collateral
    ratio c/l ≥ 1 is the on-chain solvency invariant; c/l > 1 provides a
    positive safety cushion against slippage and liquidation delays.

  • HTLC:  a Hash Time-Lock Contract locks `amount` with a secret hash.
    The claimant receives `amount − fee` upon revealing the hash preimage;
    the sender recovers the full `amount` after the timeout expires without
    a valid reveal.  Value is conserved: claim + fee = amount.

  • Merkle trees:  the bridge's inclusion-proof scheme uses a complete
    binary Merkle tree of depth d, which has exactly 2^d leaf nodes.
    Proof size grows logarithmically with the number of bridge positions.

  Sections
  ────────
  1.  Lock / mint conservation and fee deduction
  2.  Fee-net positivity and monotonicity
  3.  Collateral solvency and over-collateralisation
  4.  HTLC atomic-swap mechanics
  5.  Merkle tree structure and proof bounds
  6.  Bridge liquidity and supply conservation

  Proof status
  ────────────
  All 20 theorems have complete machine-checked proofs.
  No `sorry` placeholders remain.

  Connection to Kernel
  ────────────────────
  The bridge conservation law (locked = minted) mirrors the Kernel coherence
  condition C(1) = 1: a unit-gain channel preserves amplitude exactly.
  The HTLC conservation identity (claim + fee = amount) parallels the
  energy balance in FineStructure.lean, and the collateral ratio ≥ 1 echoes
  the unit-circle eigenvalue |μ| = 1 from CriticalEigenvalue.lean.
-/

import SpeedOfLight

open Real

-- ════════════════════════════════════════════════════════════════════════════
-- Computable Nat definition (Merkle tree depth → leaf count)
-- ════════════════════════════════════════════════════════════════════════════

/-- merkle_leaf_count: the number of leaf positions in a complete binary
    Merkle tree of depth `d`.  A tree of depth d has exactly 2^d leaves. -/
def merkle_leaf_count (d : ℕ) : ℕ := 2 ^ d

noncomputable section

-- ════════════════════════════════════════════════════════════════════════════
-- Definitions
-- ════════════════════════════════════════════════════════════════════════════

/-- bridge_locked: the amount held in escrow on the source chain after a
    deposit of `amount` with protocol fee `fee` deducted.  The bridge contract
    holds exactly this value until the corresponding burn is finalised. -/
noncomputable def bridge_locked (amount fee : ℝ) : ℝ := amount - fee

/-- bridge_minted: the wrapped tokens created on the destination chain for a
    bridge deposit.  By the conservation invariant, bridge_minted = bridge_locked. -/
noncomputable def bridge_minted (amount fee : ℝ) : ℝ := amount - fee

/-- collateral_ratio: the ratio of relayer collateral to bridge locked amount.
    The on-chain solvency invariant requires collateral_ratio ≥ 1. -/
noncomputable def collateral_ratio (collateral locked : ℝ) : ℝ := collateral / locked

/-- htlc_claim: the net value received by the preimage revealer in a Hash
    Time-Lock Contract.  Equals the locked amount minus the protocol fee. -/
noncomputable def htlc_claim (amount fee : ℝ) : ℝ := amount - fee

/-- htlc_refund: the amount returned to the original sender when the HTLC
    timeout expires without a valid preimage reveal.  The full `amount` is
    returned with no fee deducted. -/
noncomputable def htlc_refund (amount : ℝ) : ℝ := amount

-- ════════════════════════════════════════════════════════════════════════════
-- Section 1 — Lock / Mint Conservation and Fee Deduction
-- bridge_locked = bridge_minted: no value is created or destroyed.
-- ════════════════════════════════════════════════════════════════════════════

/-- Conservation: the amount locked on the source chain exactly equals the
    amount minted on the destination chain.  The bridge neither creates nor
    destroys value. -/
theorem bridge_conservation (amount fee : ℝ) :
    bridge_locked amount fee = bridge_minted amount fee := rfl

/-- A positive protocol fee strictly reduces the minted output below the
    deposited amount.  The bridge is not free to use: users receive less than
    they deposit. -/
theorem bridge_fee_reduces_output (amount fee : ℝ) (hfee : 0 < fee) :
    bridge_minted amount fee < amount := by
  unfold bridge_minted; linarith

/-- Zero fee: a fee-free bridge transfers the full deposited amount.  When
    fee = 0 the bridge is a perfect value conduit with no loss. -/
theorem bridge_zero_fee (amount : ℝ) : bridge_minted amount 0 = amount := by
  unfold bridge_minted; ring

-- ════════════════════════════════════════════════════════════════════════════
-- Section 2 — Fee-Net Positivity and Monotonicity
-- The net transfer is positive when amount > fee, and monotone in amount.
-- ════════════════════════════════════════════════════════════════════════════

/-- Net transfer is strictly positive when the deposit exceeds the fee.  The
    relayer only processes economically viable bridge requests. -/
theorem bridge_locked_pos (amount fee : ℝ) (h : fee < amount) :
    0 < bridge_locked amount fee := by
  unfold bridge_locked; linarith

/-- Net transfer is strictly less than the deposit when the fee is positive.
    Every non-zero fee reduces the output below the full deposit. -/
theorem bridge_locked_lt_amount (amount fee : ℝ) (hfee : 0 < fee) :
    bridge_locked amount fee < amount := by
  unfold bridge_locked; linarith

/-- The bridged output is strictly monotone in the deposit: a larger deposit
    yields a larger net transfer, all fees equal.  Bridge output preserves
    the strict ordering of deposit sizes. -/
theorem bridge_locked_monotone (a₁ a₂ fee : ℝ) (h : a₁ < a₂) :
    bridge_locked a₁ fee < bridge_locked a₂ fee := by
  unfold bridge_locked; linarith

-- ════════════════════════════════════════════════════════════════════════════
-- Section 3 — Collateral Solvency and Over-Collateralisation
-- Relayers post collateral c; the solvency invariant is c / locked ≥ 1.
-- ════════════════════════════════════════════════════════════════════════════

/-- The collateral ratio is strictly positive when both collateral and locked
    amount are positive.  A funded relayer always has a non-trivial ratio. -/
theorem collateral_ratio_pos (c l : ℝ) (hc : 0 < c) (hl : 0 < l) :
    0 < collateral_ratio c l :=
  div_pos hc hl

/-- Solvency: collateral ≥ locked implies collateral_ratio ≥ 1.  The bridge
    is solvent whenever the relayer is at least fully backed. -/
theorem collateral_solvency (c l : ℝ) (hl : 0 < l) (h : l ≤ c) :
    1 ≤ collateral_ratio c l := by
  unfold collateral_ratio
  rw [le_div_iff₀ hl, one_mul]
  exact h

/-- Over-collateralisation: collateral strictly greater than locked gives
    collateral_ratio > 1.  A surplus buffer provides an extra security margin
    against on-chain slippage and liquidation delays. -/
theorem collateral_overcollateralised (c l : ℝ) (hl : 0 < l) (h : l < c) :
    1 < collateral_ratio c l := by
  unfold collateral_ratio
  rw [lt_div_iff₀ hl, one_mul]
  exact h

/-- Surplus from ratio: collateral_ratio > 1 implies the collateral strictly
    exceeds the locked amount.  A ratio above 1 guarantees a positive cushion
    c − l > 0. -/
theorem collateral_surplus_pos (c l : ℝ) (hl : 0 < l)
    (h : 1 < collateral_ratio c l) : l < c := by
  unfold collateral_ratio at h
  rw [lt_div_iff₀ hl, one_mul] at h
  exact h

-- ════════════════════════════════════════════════════════════════════════════
-- Section 4 — HTLC Atomic-Swap Mechanics
-- Hash Time-Lock Contracts: the claimant who reveals the preimage receives
-- htlc_claim; the sender who waits past timeout receives htlc_refund.
-- ════════════════════════════════════════════════════════════════════════════

/-- The HTLC claimant receives a strictly positive amount when the locked
    value exceeds the protocol fee.  Revealing the preimage is worthwhile for
    any amount > fee. -/
theorem htlc_claim_pos (amount fee : ℝ) (h : fee < amount) :
    0 < htlc_claim amount fee := by
  unfold htlc_claim; linarith

/-- The HTLC timeout refund equals the original locked amount: the sender
    recovers the full deposit when the preimage is not revealed before expiry.
    No fee is charged on timeout. -/
theorem htlc_refund_full (amount : ℝ) : htlc_refund amount = amount := rfl

/-- The refund strictly exceeds the claim when the fee is positive.  A sender
    who times out recovers strictly more than a claimant who pays the fee —
    incentivising timely preimage revelation. -/
theorem htlc_refund_exceeds_claim (amount fee : ℝ) (hfee : 0 < fee) :
    htlc_claim amount fee < htlc_refund amount := by
  unfold htlc_claim htlc_refund; linarith

/-- Value conservation: the claim plus the fee exactly equals the locked
    amount.  No value is lost or created in the HTLC protocol. -/
theorem htlc_value_conservation (amount fee : ℝ) :
    htlc_claim amount fee + fee = amount := by
  unfold htlc_claim; ring

-- ════════════════════════════════════════════════════════════════════════════
-- Section 5 — Merkle Tree Structure and Proof Bounds
-- A complete binary Merkle tree of depth d has exactly 2^d leaf nodes.
-- ════════════════════════════════════════════════════════════════════════════

/-- A Merkle tree always has at least one leaf node: 2^d ≥ 1 for all d.
    The bridge inclusion-proof scheme is always well-defined. -/
theorem merkle_leaves_pos (d : ℕ) : 0 < merkle_leaf_count d := by
  unfold merkle_leaf_count
  positivity

/-- Doubling: each additional depth level doubles the leaf count.
    leaves(d+1) = 2 · leaves(d).  The provable-position set grows
    exponentially with tree depth. -/
theorem merkle_leaves_double (d : ℕ) :
    merkle_leaf_count (d + 1) = 2 * merkle_leaf_count d := by
  unfold merkle_leaf_count; ring

/-- Merkle leaf count is strictly monotone in depth: a deeper tree supports
    strictly more bridge positions.  Increasing depth by one level strictly
    expands the number of provable inclusion positions. -/
theorem merkle_leaves_monotone (d₁ d₂ : ℕ) (h : d₁ < d₂) :
    merkle_leaf_count d₁ < merkle_leaf_count d₂ := by
  unfold merkle_leaf_count
  exact pow_lt_pow_right₀ (by norm_num) h

-- ════════════════════════════════════════════════════════════════════════════
-- Section 6 — Bridge Liquidity and Supply Conservation
-- ════════════════════════════════════════════════════════════════════════════

/-- Withdrawal feasibility: if the bridge liquidity pool covers a pending
    withdrawal, the remaining balance is non-negative.  The bridge never
    goes into deficit when withdrawals are within the liquidity bound. -/
theorem bridge_liquidity_nonneg (liquidity withdrawal : ℝ)
    (h : withdrawal ≤ liquidity) : 0 ≤ liquidity - withdrawal := by linarith

/-- Supply conservation: the minted wrapped supply never exceeds the locked
    source supply.  When conservation holds (minted = locked), the bridge
    cannot create tokens beyond what has been deposited. -/
theorem bridge_supply_conservation (locked minted : ℝ)
    (h : minted = locked) : minted ≤ locked := le_of_eq h

/-- Deeper liquidity enables larger withdrawals: a pool with more liquidity
    can service any withdrawal that a shallower pool could service.  Strictly
    increasing the pool size strictly expands the set of feasible withdrawals. -/
theorem bridge_liquidity_monotone (l₁ l₂ w : ℝ) (hl : l₁ < l₂)
    (hw : w ≤ l₁) : w ≤ l₂ := le_trans hw (le_of_lt hl)

end
