# Kernel — Lean 4 Formal Verification (`formal-lean/`)

This directory contains a [Lean 4](https://leanprover.github.io/) formalization
of the core theorems from the Kernel research project.  The underlying
mathematics is documented in [`../docs/master_derivations.pdf`](../docs/master_derivations.pdf).

---

## Directory layout

```
formal-lean/
├── lakefile.lean          # Lake project config; declares Mathlib dependency
├── lean-toolchain         # Pins the exact Lean 4 version
├── Main.lean              # Executable entry point (prints verified theorems)
├── CriticalEigenvalue.lean # Formalized theorems (see §Contents below)
└── README.md              # This file
```

---

## Prerequisites

1. **Install `elan`** (the Lean version manager):
   ```bash
   curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
   ```
   `elan` will automatically install the Lean version pinned in `lean-toolchain`.

2. **`lake`** is bundled with Lean — no separate install needed.

---

## Quick start

```bash
# 1. Enter the project directory
cd formal-lean/

# 2. (Strongly recommended) Download the pre-built Mathlib cache
#    This avoids recompiling Mathlib from scratch (~1 hour build).
lake exe cache get

# 3. Build the project
lake build

# 4. Run the entry-point executable
lake exe formalLean
```

Expected output of `lake exe formalLean`:
```
===================================================
 Kernel — Lean 4 Formal Verification
===================================================

Theorems verified by the Lean 4 type checker:

  [1] mu_def          : μ = exp(I · 3π/4)
  [2] mu_pow_eight    : μ^8 = 1  (8-cycle closure)
  [3] mu_abs_one      : |μ| = 1  (μ lies on the unit circle)
  [4] rotMat_det      : det R(3π/4) = 1
  [5] rotMat_orthog   : R(3π/4) · R(3π/4)ᵀ = I
  [6] rotMat_pow_eight: R(3π/4)^8 = I
  [7] coherence_le_one: C(r) ≤ 1, with equality iff r = 1
  [8] canonical_norm  : η² + |μ·η|² = 1  (η = 1/√2)

See CriticalEigenvalue.lean for full proof terms.
```

---

## Testing

```bash
# Build and check for proof errors
lake build 2>&1 | grep -E "error|warning|sorry"
```

All 28 theorems in `CriticalEigenvalue.lean` have complete machine-checked proofs (no `sorry`).

---

## Contents

### `CriticalEigenvalue.lean`

**§1–6 Core eigenvalue and coherence structure**

| # | Theorem | Description |
|---|---------|-------------|
| 1 | `mu_eq_cart` | μ = (−1 + i)/√2 in Cartesian form |
| 2 | `mu_abs_one` | \|μ\| = 1 |
| 3 | `mu_pow_eight` | μ⁸ = 1 (8-cycle closure) |
| 4 | `mu_powers_distinct` | {μ⁰,…,μ⁷} pairwise distinct (`IsPrimitiveRoot`, gcd(3,8)=1) |
| 5 | `rotMat_det` | det R(3π/4) = 1 |
| 6 | `rotMat_orthog` | R · Rᵀ = I |
| 7 | `rotMat_pow_eight` | R(3π/4)⁸ = I |
| 8 | `coherence_le_one` | C(r) ≤ 1 for r ≥ 0 (AM–GM) |
| 9 | `coherence_eq_one_iff` | C(r) = 1 ↔ r = 1 |
| 10 | `canonical_norm` | η² + \|μ·η\|² = 1 |

**§7 Silver ratio (Proposition 4)**

| # | Theorem | Description |
|---|---------|-------------|
| 11 | `silverRatio_mul_conj` | δS · (√2−1) = 1 |
| 12 | `silverRatio_sq` | δS² = 2·δS + 1 |
| 13 | `silverRatio_inv` | 1/δS = √2−1 |

**§8 Additional coherence properties (Theorem 11)**

| # | Theorem | Description |
|---|---------|-------------|
| 14 | `coherence_pos` | C(r) > 0 for r > 0 |
| 15 | `coherence_symm` | C(r) = C(1/r) — even symmetry about r = 1 |
| 16 | `coherence_lt_one` | C(r) < 1 for r ≥ 0, r ≠ 1 |

**§9 Palindrome residual (Theorem 12)**

| # | Theorem | Description |
|---|---------|-------------|
| 17 | `palindrome_residual_zero_iff` | R(r) = 0 ↔ r = 1 |
| 18 | `palindrome_residual_pos` | R(r) > 0 for r > 1 |
| 19 | `palindrome_residual_neg` | R(r) < 0 for 0 < r < 1 |
| 20 | `palindrome_residual_antisymm` | R(1/r) = −R(r) — odd anti-symmetry |

**§10 Lyapunov–coherence duality (Theorem 14)**

| # | Theorem | Description |
|---|---------|-------------|
| 21 | `lyapunov_coherence_duality` | C(exp λ) = 2/(exp λ + exp(−λ)) |
| 22 | `lyapunov_coherence_sech` | C(exp λ) = (cosh λ)⁻¹ = sech λ |

**§11 Derived invariant equivalences — machine-discovered connections (Corollary 13)**

| # | Theorem | Description |
|---|---------|-------------|
| 23 | `palindrome_coherence_equiv` | R(r)=0 ↔ C(r)=1 — connecting two independent invariants |
| 24 | `coherence_palindrome_duality` | C even ∧ R odd — dual symmetries about r = 1 |
| 25 | `coherence_max_symm` | C(r)=1 ↔ C(1/r)=1 |
| 26 | `palindrome_zero_self_dual` | R(r)=0 → r = 1/r |
| 27 | `simultaneous_break` | r=1 ↔ C(r)=1 ∧ R(r)=0 |
| 28 | `lyapunov_bound` | C(exp λ) ≤ 1 via the sech route |

---

## Updating Mathlib

```bash
# Update the Mathlib dependency to the latest compatible version
lake update

# Rebuild cache after updating
lake exe cache get
lake build
```

---

## References

- [Lean 4 documentation](https://leanprover.github.io/lean4/doc/)
- [Mathlib4 documentation](https://leanprover-community.github.io/mathlib4_docs/)
- [Mathlib4 on GitHub](https://github.com/leanprover-community/mathlib4)
- [`../docs/master_derivations.pdf`](../docs/master_derivations.pdf) — mathematical background
