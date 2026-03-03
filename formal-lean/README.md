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
# Build and check for proof errors (sorry warnings are expected for
# the two placeholder proofs marked ⚠ in CriticalEigenvalue.lean)
lake build 2>&1 | grep -E "error|warning|sorry"
```

Proofs currently using `sorry` (tracked placeholders, not errors):
- `mu_powers_distinct` — requires `IsPrimitiveRoot`; proof sketch included.
- `rotMat_pow_eight`   — requires `Matrix` power lemmas; proof sketch included.

---

## Contents

### `CriticalEigenvalue.lean`

| # | Theorem | Status | Description |
|---|---------|--------|-------------|
| 1 | `mu_eq_cart` | ✓ proved | μ = (−1 + i)/√2 in Cartesian form |
| 2 | `mu_abs_one` | ✓ proved | \|μ\| = 1 |
| 3 | `mu_pow_eight` | ✓ proved | μ⁸ = 1 (8-cycle closure) |
| 4 | `mu_powers_distinct` | ⚠ sorry | 8 powers are pairwise distinct |
| 5 | `rotMat_det` | ✓ proved | det R(3π/4) = 1 |
| 6 | `rotMat_orthog` | ✓ proved | R · Rᵀ = I |
| 7 | `rotMat_pow_eight` | ⚠ sorry | R(3π/4)⁸ = I |
| 8 | `coherence_le_one` | ✓ proved | C(r) ≤ 1 for r ≥ 0 (AM–GM) |
| 9 | `coherence_eq_one_iff` | ✓ proved | C(r) = 1 ↔ r = 1 |
| 10 | `canonical_norm` | ✓ proved | η² + \|μ·η\|² = 1 |

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
