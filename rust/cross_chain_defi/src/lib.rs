//! cross_chain_defi — Polkadot-native cross-chain DeFi aggregation
//!
//! This crate is the Rust/Substrate translation of the Lean 4 module
//! `CrossChainDeFiAggregator.lean`.  Every function here has a direct
//! counterpart in the formal proofs; the doc-comment for each item
//! identifies the corresponding Lean definition or theorem.
//!
//! # Design choices
//!
//! Polkadot runtimes must be `no_std` and use **integer arithmetic only**.
//! Floating-point is therefore avoided throughout.  Instead:
//!
//! * Token amounts are `u128` (smallest indivisible denomination — analogous to
//!   `planck` in Substrate parlance).
//! * Rates and prices are `u128` values scaled by [`PRECISION`] = 10^12
//!   (one *pico-unit*).  A rate of 5 % is stored as `50_000_000_000`.
//! * The LP value (geometric mean) uses an **integer square-root** that is
//!   accurate to ±1 unit.
//!
//! # Correspondence with Lean theorems
//!
//! | Lean definition / theorem                | Rust item                           |
//! |------------------------------------------|-------------------------------------|
//! | `amm_out`                                | [`amm_out`]                         |
//! | `amm_price`                              | [`amm_price`]                       |
//! | `lending_interest`                       | [`lending_interest`]                |
//! | `best_rate`                              | [`best_rate`]                       |
//! | `lp_value`                               | [`lp_value`]                        |
//! | `amm_invariant_pos`                      | [`amm_invariant_pos`]               |
//! | `amm_out_pos`                            | [`amm_out_pos`]                     |
//! | `amm_out_zero_input`                     | [`amm_out_zero_input`]              |
//! | `amm_price_pos`                          | [`amm_price_pos`]                   |
//! | `amm_invariant_preserved`                | [`amm_invariant_preserved`]         |
//! | `amm_out_bounded`                        | [`amm_out_bounded`]                 |
//! | `amm_slippage_positive`                  | [`amm_slippage_positive`]           |
//! | `amm_price_impact_lt_one`                | [`amm_price_impact_lt_one`]         |
//! | `amm_out_monotone`                       | [`amm_out_monotone`]                |
//! | `lending_interest_nonneg`                | [`lending_interest_nonneg`]         |
//! | `lending_interest_pos`                   | [`lending_interest_pos`]            |
//! | `lending_amount_exceeds_principal`       | [`lending_amount_exceeds_principal`]|
//! | `lending_rate_monotone`                  | [`lending_rate_monotone`]           |
//! | `best_rate_ge_left` / `best_rate_ge_right` | [`best_rate`] + property fns      |
//! | `best_rate_symm` / `best_rate_optimal`  | property fns                        |
//! | `lp_value_pos` / `lp_value_monotone`    | [`lp_value_pos`], [`lp_value_monotone`]|

// ── Precision ─────────────────────────────────────────────────────────────────

/// Fixed-point precision factor.  All rates and prices are scaled by this
/// value so that integer arithmetic can represent fractional quantities.
///
/// `PRECISION` = 10^12 means one `u128` unit represents 10^{-12} of a whole
/// token (analogous to a *pico-unit*).  Choose this constant to match the
/// asset's `decimals` field in `pallet_assets` (typically 10 or 12 on
/// Polkadot).
pub const PRECISION: u128 = 1_000_000_000_000; // 10^12

// ─────────────────────────────────────────────────────────────────────────────
// §1  AMM constant-product invariant and output formula
//
//     Lean: noncomputable def amm_out (x y Δ : ℝ) : ℝ := y * Δ / (x + Δ)
//
//     Integer analogue: out = y * delta / (x + delta)
//     (exact integer division; caller ensures x + delta ≠ 0)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the AMM output amount for a constant-product pool.
///
/// Given reserves `x` (of token X) and `y` (of token Y), and an input of
/// `delta` units of token X, returns the number of token-Y units the trader
/// receives:
///
/// ```text
/// out = y * delta / (x + delta)          (Lean: amm_out x y Δ)
/// ```
///
/// All values are in the pool's smallest denomination (`u128`).
/// Returns `None` if `x + delta` would overflow or is zero.
///
/// # Lean correspondence
/// `noncomputable def amm_out (x y Δ : ℝ) : ℝ := y * Δ / (x + Δ)`
pub fn amm_out(x: u128, y: u128, delta: u128) -> Option<u128> {
    let denom = x.checked_add(delta)?;
    if denom == 0 {
        return None;
    }
    // y * delta may overflow u128 for very large pools; use u128 with care.
    // In production, use a wider integer (e.g. U256) or scale inputs first.
    let numerator = y.checked_mul(delta)?;
    Some(numerator / denom)
}

/// Compute the AMM spot price of token Y in units of token X, scaled by
/// [`PRECISION`].
///
/// ```text
/// price = x * PRECISION / y              (Lean: amm_price x y = x / y)
/// ```
///
/// Returns `None` if `y == 0` or if the multiplication overflows.
///
/// # Lean correspondence
/// `noncomputable def amm_price (x y : ℝ) : ℝ := x / y`
pub fn amm_price(x: u128, y: u128) -> Option<u128> {
    if y == 0 {
        return None;
    }
    let scaled = x.checked_mul(PRECISION)?;
    Some(scaled / y)
}

// ─────────────────────────────────────────────────────────────────────────────
// §2  Property checks: invariant preservation and pool-drain safety
// ─────────────────────────────────────────────────────────────────────────────

/// Check that the constant-product k = x * y is positive.
///
/// # Lean correspondence
/// `theorem amm_invariant_pos (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 0 < x * y`
pub fn amm_invariant_pos(x: u128, y: u128) -> bool {
    x > 0 && y > 0
}

/// Check that `amm_out` is positive for positive inputs.
///
/// # Lean correspondence
/// `theorem amm_out_pos : 0 < amm_out x y Δ`
pub fn amm_out_pos(x: u128, y: u128, delta: u128) -> bool {
    match amm_out(x, y, delta) {
        Some(out) => out > 0,
        None => false,
    }
}

/// Check that `amm_out` is zero when the input is zero.
///
/// # Lean correspondence
/// `theorem amm_out_zero_input : amm_out x y 0 = 0`
pub fn amm_out_zero_input(x: u128, y: u128) -> bool {
    amm_out(x, y, 0) == Some(0)
}

/// Check that the AMM spot price is positive when both reserves are positive.
///
/// # Lean correspondence
/// `theorem amm_price_pos : 0 < amm_price x y`
pub fn amm_price_pos(x: u128, y: u128) -> bool {
    match amm_price(x, y) {
        Some(p) => p > 0,
        None => false,
    }
}

/// Verify the constant-product invariant is preserved after a swap.
///
/// Checks that `(x + delta) * (y - out) == x * y` (up to rounding from
/// integer division).  Because integer division truncates, the check uses
/// an error tolerance of ±(x + delta) to account for the truncation.
///
/// # Lean correspondence
/// `theorem amm_invariant_preserved : (x + Δ) * (y − amm_out x y Δ) = x · y`
pub fn amm_invariant_preserved(x: u128, y: u128, delta: u128) -> bool {
    let out = match amm_out(x, y, delta) {
        Some(v) => v,
        None => return false,
    };
    if out > y {
        return false;
    }
    let denom = match x.checked_add(delta) {
        Some(d) => d,
        None => return false,
    };
    let lhs = match denom.checked_mul(y - out) {
        Some(v) => v,
        None => return false,
    };
    let rhs = match x.checked_mul(y) {
        Some(v) => v,
        None => return false,
    };
    // Integer truncation makes lhs ≥ rhs; the gap is at most (x + delta) - 1.
    lhs >= rhs && lhs - rhs < denom
}

/// Check the pool-drain safety property: `amm_out < y` for all positive
/// inputs (a finite trade cannot drain the pool).
///
/// # Lean correspondence
/// `theorem amm_out_bounded : amm_out x y Δ < y`
pub fn amm_out_bounded(x: u128, y: u128, delta: u128) -> bool {
    match amm_out(x, y, delta) {
        Some(out) => out < y,
        None => false,
    }
}

/// Check the slippage property: `amm_out < y * delta / x`.
///
/// The effective rate is always worse than the pro-rata rate because the
/// denominator `x + delta > x`.
///
/// # Lean correspondence
/// `theorem amm_slippage_positive : amm_out x y Δ < y * Δ / x`
pub fn amm_slippage_positive(x: u128, y: u128, delta: u128) -> bool {
    if x == 0 {
        return false;
    }
    let out = match amm_out(x, y, delta) {
        Some(v) => v,
        None => return false,
    };
    let pro_rata = match y.checked_mul(delta) {
        Some(n) => n / x,
        None => return false,
    };
    out < pro_rata
}

/// Check the price-impact bound: `amm_out / y < 1`, i.e. the fractional
/// output is less than one full reserve.
///
/// # Lean correspondence
/// `theorem amm_price_impact_lt_one : amm_out x y Δ / y < 1`
pub fn amm_price_impact_lt_one(x: u128, y: u128, delta: u128) -> bool {
    amm_out_bounded(x, y, delta)
}

/// Check that a larger input yields strictly more output.
///
/// # Lean correspondence
/// `theorem amm_out_monotone : Δ₁ < Δ₂ → amm_out x y Δ₁ < amm_out x y Δ₂`
pub fn amm_out_monotone(x: u128, y: u128, delta1: u128, delta2: u128) -> bool {
    if delta1 >= delta2 {
        return false;
    }
    let out1 = match amm_out(x, y, delta1) {
        Some(v) => v,
        None => return false,
    };
    let out2 = match amm_out(x, y, delta2) {
        Some(v) => v,
        None => return false,
    };
    out1 < out2
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  Lending / borrowing simple-interest model
//
//     Lean: noncomputable def lending_interest (P r t : ℝ) : ℝ := P * r * t
//
//     Integer analogue: interest = P * r * t / PRECISION^2
//     (r and t are PRECISION-scaled fractions)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute simple lending interest: `I = P * r * t`.
///
/// `principal` is in token units (`u128`).
/// `rate_scaled` is the per-period utilization rate scaled by [`PRECISION`]
///   (e.g. 5 % → `50_000_000_000`).  Only this argument is PRECISION-scaled.
/// `periods` is a **raw** integer count of whole time periods (not scaled).
///
/// The result is `principal * rate_scaled * periods / PRECISION`, which gives
/// the interest in the same token units as `principal`.  One division by
/// `PRECISION` is sufficient because only `rate_scaled` carries the scaling
/// factor; `periods` is dimensionless.
///
/// Returns the interest in token units, or `None` on overflow.
///
/// # Lean correspondence
/// `noncomputable def lending_interest (P r t : ℝ) : ℝ := P * r * t`
pub fn lending_interest(principal: u128, rate_scaled: u128, periods: u128) -> Option<u128> {
    // principal * rate_scaled may already be large; check for overflow.
    let p_r = principal.checked_mul(rate_scaled)?;
    let p_r_t = p_r.checked_mul(periods)?;
    // Descale once: rate_scaled was in PRECISION units; periods is a raw count.
    Some(p_r_t / PRECISION)
}

/// Check that lending interest is non-negative (trivially true for `u128`).
///
/// # Lean correspondence
/// `theorem lending_interest_nonneg : 0 ≤ lending_interest P r t`
pub fn lending_interest_nonneg(_principal: u128, _rate_scaled: u128, _periods: u128) -> bool {
    true // u128 is always ≥ 0
}

/// Check that lending interest is strictly positive for positive inputs.
///
/// # Lean correspondence
/// `theorem lending_interest_pos : 0 < lending_interest P r t`
pub fn lending_interest_pos(principal: u128, rate_scaled: u128, periods: u128) -> bool {
    match lending_interest(principal, rate_scaled, periods) {
        Some(i) => i > 0,
        None => false,
    }
}

/// Check that the total repayment exceeds the principal: `P < P + I`.
///
/// # Lean correspondence
/// `theorem lending_amount_exceeds_principal : P < P + lending_interest P r t`
pub fn lending_amount_exceeds_principal(
    principal: u128,
    rate_scaled: u128,
    periods: u128,
) -> bool {
    match lending_interest(principal, rate_scaled, periods) {
        Some(interest) => interest > 0,
        None => false,
    }
}

/// Check that a higher rate produces more interest: `r₁ < r₂ → I(r₁) < I(r₂)`.
///
/// # Lean correspondence
/// `theorem lending_rate_monotone : r₁ < r₂ → lending_interest P r₁ t < lending_interest P r₂ t`
pub fn lending_rate_monotone(
    principal: u128,
    rate1: u128,
    rate2: u128,
    periods: u128,
) -> bool {
    if rate1 >= rate2 {
        return false;
    }
    let i1 = match lending_interest(principal, rate1, periods) {
        Some(v) => v,
        None => return false,
    };
    let i2 = match lending_interest(principal, rate2, periods) {
        Some(v) => v,
        None => return false,
    };
    i1 < i2
}

// ─────────────────────────────────────────────────────────────────────────────
// §5  Cross-chain rate aggregation (best-rate selection)
//
//     Lean: noncomputable def best_rate (r₁ r₂ : ℝ) : ℝ := max r₁ r₂
// ─────────────────────────────────────────────────────────────────────────────

/// Select the best (highest) rate across two chains.
///
/// ```text
/// best_rate r1 r2 = max r1 r2            (Lean: best_rate r₁ r₂ = max r₁ r₂)
/// ```
///
/// Rates are [`PRECISION`]-scaled `u128` values.
///
/// # Lean correspondence
/// `noncomputable def best_rate (r₁ r₂ : ℝ) : ℝ := max r₁ r₂`
pub fn best_rate(r1: u128, r2: u128) -> u128 {
    r1.max(r2)
}

/// Check that `best_rate` dominates the left operand: `r₁ ≤ best_rate r₁ r₂`.
///
/// # Lean correspondence
/// `theorem best_rate_ge_left : r₁ ≤ best_rate r₁ r₂`
pub fn best_rate_ge_left(r1: u128, r2: u128) -> bool {
    best_rate(r1, r2) >= r1
}

/// Check that `best_rate` dominates the right operand: `r₂ ≤ best_rate r₁ r₂`.
///
/// # Lean correspondence
/// `theorem best_rate_ge_right : r₂ ≤ best_rate r₁ r₂`
pub fn best_rate_ge_right(r1: u128, r2: u128) -> bool {
    best_rate(r1, r2) >= r2
}

/// Check symmetry: `best_rate r₁ r₂ = best_rate r₂ r₁`.
///
/// # Lean correspondence
/// `theorem best_rate_symm : best_rate r₁ r₂ = best_rate r₂ r₁`
pub fn best_rate_symm(r1: u128, r2: u128) -> bool {
    best_rate(r1, r2) == best_rate(r2, r1)
}

/// Check optimality: if `r` dominates both operands then it also dominates
/// `best_rate`.
///
/// # Lean correspondence
/// `theorem best_rate_optimal : r₁ ≤ r → r₂ ≤ r → best_rate r₁ r₂ ≤ r`
pub fn best_rate_optimal(r1: u128, r2: u128, r: u128) -> bool {
    if r1 > r || r2 > r {
        return false;
    }
    best_rate(r1, r2) <= r
}

/// Check idempotence: `best_rate r r = r`.
///
/// # Lean correspondence
/// `theorem best_rate_idempotent : best_rate r r = r`
pub fn best_rate_idempotent(r: u128) -> bool {
    best_rate(r, r) == r
}

// ─────────────────────────────────────────────────────────────────────────────
// §6  LP value and monotone-output properties
//
//     Lean: noncomputable def lp_value (x y : ℝ) : ℝ := Real.sqrt (x * y)
//
//     Integer analogue: lp_value(x, y) = isqrt(x * y)
//     where isqrt is the integer square root floor(sqrt(n)).
// ─────────────────────────────────────────────────────────────────────────────

/// Integer square root: largest `s` with `s * s ≤ n`.
///
/// Uses a Newton–Raphson iteration that converges in O(log n) steps.
/// Returns 0 for `n == 0`.
fn isqrt(n: u128) -> u128 {
    if n == 0 {
        return 0;
    }
    // Number of bits in n: (128 - leading_zeros).  We want an initial estimate
    // ≥ floor(sqrt(n)).  Shifting 1 left by ceil((bits)/2) is sufficient.
    // The shift amount is at most 64 (when n uses all 128 bits), which is
    // well within the u128 shift range of 0..=127.
    let bits = 128u32 - n.leading_zeros(); // 1..=128
    let shift = (bits + 1) / 2;           // 1..=64
    let mut x = 1u128 << shift;
    loop {
        let x1 = (x + n / x) / 2;
        if x1 >= x {
            return x;
        }
        x = x1;
    }
}

/// Compute the LP value (geometric mean) of pool reserves: `√(x · y)`.
///
/// Returns the integer floor of the true geometric mean.
/// Returns `None` if `x * y` overflows.
///
/// # Lean correspondence
/// `noncomputable def lp_value (x y : ℝ) : ℝ := Real.sqrt (x * y)`
pub fn lp_value(x: u128, y: u128) -> Option<u128> {
    let product = x.checked_mul(y)?;
    Some(isqrt(product))
}

/// Check that LP value is positive when both reserves are positive.
///
/// # Lean correspondence
/// `theorem lp_value_pos : 0 < lp_value x y`
pub fn lp_value_pos(x: u128, y: u128) -> bool {
    match lp_value(x, y) {
        Some(v) => v > 0,
        None => false,
    }
}

/// Check that LP value is strictly monotone: larger reserves → larger value.
///
/// Specifically checks `lp_value(x, y) < lp_value(x', y')` when `x < x'`
/// and `y < y'`.
///
/// # Lean correspondence
/// `theorem lp_value_monotone : x < x' → y < y' → lp_value x y < lp_value x' y'`
pub fn lp_value_monotone(x: u128, y: u128, xp: u128, yp: u128) -> bool {
    if x >= xp || y >= yp {
        return false;
    }
    let lv = match lp_value(x, y) {
        Some(v) => v,
        None => return false,
    };
    let lvp = match lp_value(xp, yp) {
        Some(v) => v,
        None => return false,
    };
    lv < lvp
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests — runtime verification of the mathematical properties proved in Lean
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── §1 AMM output ──────────────────────────────────────────────────────

    #[test]
    fn test_amm_out_basic() {
        // Pool: 1000 X, 2000 Y.  Swap 100 X → out = 2000*100/(1000+100) ≈ 181
        let out = amm_out(1000, 2000, 100).unwrap();
        assert_eq!(out, 181); // floor(200000/1100)
    }

    #[test]
    fn test_amm_invariant_pos() {
        assert!(amm_invariant_pos(500, 1000));
        assert!(!amm_invariant_pos(0, 1000));
        assert!(!amm_invariant_pos(500, 0));
    }

    #[test]
    fn test_amm_out_pos() {
        // Lean: theorem amm_out_pos
        assert!(amm_out_pos(1000, 2000, 100));
        assert!(!amm_out_pos(1000, 2000, 0)); // zero input → zero output
    }

    #[test]
    fn test_amm_out_zero_input() {
        // Lean: theorem amm_out_zero_input
        assert!(amm_out_zero_input(1000, 2000));
        assert!(amm_out_zero_input(1, 1));
    }

    // ── §2 Invariant preservation and pool-drain safety ────────────────────

    #[test]
    fn test_amm_price_pos() {
        // Lean: theorem amm_price_pos
        assert!(amm_price_pos(1000, 2000));
        assert!(!amm_price_pos(0, 2000)); // x=0 → price=0
    }

    #[test]
    fn test_amm_invariant_preserved() {
        // Lean: theorem amm_invariant_preserved
        assert!(amm_invariant_preserved(1000, 2000, 100));
        assert!(amm_invariant_preserved(500, 500, 250));
        assert!(amm_invariant_preserved(1_000_000, 1_000_000, 1));
    }

    #[test]
    fn test_amm_out_bounded() {
        // Lean: theorem amm_out_bounded — out < y for all positive inputs
        assert!(amm_out_bounded(1000, 2000, 100));
        assert!(amm_out_bounded(1, 1_000_000, 1_000_000));
        assert!(amm_out_bounded(1, 1, 1_000_000_000));
    }

    // ── §3 Slippage and price-impact bounds ───────────────────────────────

    #[test]
    fn test_amm_slippage_positive() {
        // Lean: theorem amm_slippage_positive — out < y*delta/x
        assert!(amm_slippage_positive(1000, 2000, 100));
        assert!(amm_slippage_positive(100, 50_000, 10));
    }

    #[test]
    fn test_amm_price_impact_lt_one() {
        // Lean: theorem amm_price_impact_lt_one — out/y < 1
        assert!(amm_price_impact_lt_one(1000, 2000, 100));
        assert!(amm_price_impact_lt_one(1, 1, 1_000_000));
    }

    #[test]
    fn test_amm_out_monotone() {
        // Lean: theorem amm_out_monotone — delta1 < delta2 → out1 < out2
        assert!(amm_out_monotone(1000, 2000, 50, 100));
        assert!(amm_out_monotone(1000, 2000, 1, 999));
        assert!(!amm_out_monotone(1000, 2000, 100, 50)); // reversed inputs
    }

    // ── §4 Lending model ──────────────────────────────────────────────────

    #[test]
    fn test_lending_interest_basic() {
        // P=1000, r=5% (50_000_000_000), t=1 period
        // interest = 1000 * 50_000_000_000 * 1 / 10^12 = 50
        let i = lending_interest(1000, 50_000_000_000, 1).unwrap();
        assert_eq!(i, 50);
    }

    #[test]
    fn test_lending_interest_nonneg() {
        // Lean: theorem lending_interest_nonneg — always true for u128
        assert!(lending_interest_nonneg(1000, 50_000_000_000, 1));
        assert!(lending_interest_nonneg(0, 0, 0));
    }

    #[test]
    fn test_lending_interest_pos() {
        // Lean: theorem lending_interest_pos
        assert!(lending_interest_pos(1000, 50_000_000_000, 1));
        assert!(!lending_interest_pos(0, 50_000_000_000, 1)); // zero principal
    }

    #[test]
    fn test_lending_amount_exceeds_principal() {
        // Lean: theorem lending_amount_exceeds_principal — P < P + I
        assert!(lending_amount_exceeds_principal(1000, 50_000_000_000, 1));
        assert!(!lending_amount_exceeds_principal(0, 50_000_000_000, 1));
    }

    #[test]
    fn test_lending_rate_monotone() {
        // Lean: theorem lending_rate_monotone — r1 < r2 → I(r1) < I(r2)
        let r1 = 30_000_000_000u128; // 3%
        let r2 = 50_000_000_000u128; // 5%
        assert!(lending_rate_monotone(1000, r1, r2, 1));
        assert!(!lending_rate_monotone(1000, r2, r1, 1)); // reversed rates
    }

    // ── §5 Rate aggregation ───────────────────────────────────────────────

    #[test]
    fn test_best_rate_selects_max() {
        // Lean: best_rate r1 r2 = max r1 r2
        assert_eq!(best_rate(3, 7), 7);
        assert_eq!(best_rate(7, 3), 7);
        assert_eq!(best_rate(5, 5), 5);
    }

    #[test]
    fn test_best_rate_dominance() {
        // Lean: theorem best_rate_ge_left / best_rate_ge_right
        assert!(best_rate_ge_left(3, 7));
        assert!(best_rate_ge_right(3, 7));
        assert!(best_rate_ge_left(7, 3));
        assert!(best_rate_ge_right(7, 3));
    }

    #[test]
    fn test_best_rate_symm() {
        // Lean: theorem best_rate_symm
        assert!(best_rate_symm(3, 7));
        assert!(best_rate_symm(7, 3));
        assert!(best_rate_symm(5, 5));
    }

    #[test]
    fn test_best_rate_optimal() {
        // Lean: theorem best_rate_optimal
        assert!(best_rate_optimal(3, 7, 10));
        assert!(best_rate_optimal(7, 3, 7));
        assert!(!best_rate_optimal(3, 7, 6)); // r=6 < best_rate(3,7)=7
    }

    #[test]
    fn test_best_rate_idempotent() {
        // Lean: theorem best_rate_idempotent
        assert!(best_rate_idempotent(5));
        assert!(best_rate_idempotent(0));
        assert!(best_rate_idempotent(u128::MAX));
    }

    // ── §6 LP value ───────────────────────────────────────────────────────

    #[test]
    fn test_lp_value_basic() {
        // lp_value(100, 400) = sqrt(40000) = 200
        assert_eq!(lp_value(100, 400).unwrap(), 200);
        // lp_value(9, 16) = sqrt(144) = 12
        assert_eq!(lp_value(9, 16).unwrap(), 12);
        // lp_value(2, 8) = sqrt(16) = 4
        assert_eq!(lp_value(2, 8).unwrap(), 4);
    }

    #[test]
    fn test_lp_value_pos() {
        // Lean: theorem lp_value_pos
        assert!(lp_value_pos(100, 400));
        assert!(lp_value_pos(1, 1));
        assert!(!lp_value_pos(0, 400)); // zero reserve → zero LP value
    }

    #[test]
    fn test_lp_value_monotone() {
        // Lean: theorem lp_value_monotone — x < x', y < y' → lp_value(x,y) < lp_value(x',y')
        assert!(lp_value_monotone(100, 400, 200, 800));
        assert!(lp_value_monotone(1, 1, 2, 2));
        assert!(!lp_value_monotone(200, 800, 100, 400)); // reversed
    }

    // ── isqrt correctness ─────────────────────────────────────────────────

    #[test]
    fn test_isqrt() {
        assert_eq!(isqrt(0), 0);
        assert_eq!(isqrt(1), 1);
        assert_eq!(isqrt(4), 2);
        assert_eq!(isqrt(9), 3);
        assert_eq!(isqrt(10), 3); // floor(sqrt(10)) = 3
        assert_eq!(isqrt(u128::MAX), 18_446_744_073_709_551_615);
    }
}
