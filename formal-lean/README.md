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
├── CriticalEigenvalue.lean # 71 theorems on eigenvalue/coherence structure
├── TimeCrystal.lean        # 20 theorems on discrete time crystal theory
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

All 71 theorems in `CriticalEigenvalue.lean` have complete machine-checked proofs (no `sorry`).
All 33 theorems in `TimeCrystal.lean` have complete machine-checked proofs (no `sorry`).

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

**§12 Orbit magnitude and trichotomy (Theorem 10)**

| # | Theorem | Description |
|---|---------|-------------|
| 29 | `mu_pow_abs` | \|μ^n\| = 1 for all n |
| 30 | `scaled_orbit_abs` | \|(r·μ)^n\| = r^n for r ≥ 0 — radial amplitude formula |
| 31 | `trichotomy_unit_orbit` | r = 1: stable unit-circle orbit |
| 32 | `trichotomy_grow` | r > 1: magnitudes strictly increasing (spiral outward) |
| 33 | `trichotomy_shrink` | 0 < r < 1: magnitudes strictly decreasing (spiral inward) |

**§13 Coherence monotonicity**

| # | Theorem | Description |
|---|---------|-------------|
| 34 | `coherence_strictMono` | 0 < r < s ≤ 1 → C(r) < C(s) — increasing toward r=1 |
| 35 | `coherence_strictAnti` | 1 ≤ r < s → C(s) < C(r) — decreasing away from r=1 |

**§14 Palindrome arithmetic**

| # | Theorem | Description |
|---|---------|-------------|
| 36 | `palindrome_comp` | 987654321 = 8 × 123456789 + 9 |
| 37 | `precession_period_factor` | 9 × 13717421 = 123456789 |
| 38 | `precession_gcd_one` | gcd(8, 13717421) = 1 — coprime periods |
| 39 | `precession_lcm` | lcm(8, 13717421) = 8·13717421 — torus super-period |

**§15 Z/8Z rotational memory**

| # | Theorem | Description |
|---|---------|-------------|
| 40 | `z8z_period` | (n + 8) % 8 = n % 8 |
| 41 | `z8z_reconstruction` | addr % 8 + 8 * (addr / 8) = addr |
| 42 | `mu_z8z_period` | μ^(j+8) = μ^j — orbit clock = memory clock |

**§16 Zero-overhead precession**

| # | Theorem | Description |
|---|---------|-------------|
| 43 | `precession_phasor_unit` | \|e^{iθ}\| = 1 for any real θ |
| 44 | `precession_preserves_abs` | \|e^{iθ}·β\| = \|β\| — amplitude invariant |
| 45 | `precession_preserves_coherence` | C(\|e^{iθ}·β\|/\|α\|) = C(\|β\|/\|α\|) — zero overhead |

**§17 Ohm-Coherence circuit identities**

| # | Theorem | Description |
|---|---------|-------------|
| 46 | `geff_reff_one` | (cosh λ)⁻¹ · cosh λ = 1 — single-channel G·R = 1 |
| 47 | `geff_at_zero` | (cosh 0)⁻¹ = 1 — maximal conductance at balance |
| 48 | `parallel_circuit_one` | N parallel channels: G_tot · R_tot = 1 |
| 49 | `series_circuit_one` | M series stages: G_tot · R_tot = 1 |

**§18 Pythagorean coherence identity (machine-discovered)**

| # | Theorem | Description |
|---|---------|-------------|
| 50 | `coherence_pythagorean` | C(r)² + ((r²−1)/(1+r²))² = 1 — coherence on unit circle |
| 51 | `palindrome_amplitude_eq` | δS·r·Res(r) = r²−1 — connects residual to Pythagorean term |

**§19 Orbit Lyapunov connection**

| # | Theorem | Description |
|---|---------|-------------|
| 52 | `orbit_radius_exp` | \|(r·μ)^n\| = exp(n·log r) — Lyapunov-exponent form |
| 53 | `coherence_orbit_sech` | C(rⁿ) = (cosh(n·log r))⁻¹ — full orbit–coherence chain |
| 54 | `coherence_orbit_decay` | r > 1 ∧ n ≥ 1 → C(rⁿ) ≤ C(r) — coherence decays under amplification |
| 55 | `orbit_coherence_at_one` | C(1ⁿ) = 1 — stable fixed point |

**§20 Silver ratio self-similarity**

| # | Theorem | Description |
|---|---------|-------------|
| 56 | `silverRatio_pos` | 0 < δS |
| 57 | `silverRatio_cont_frac` | δS = 2 + 1/δS — continued-fraction fixed point |
| 58 | `silverRatio_minPoly` | δS² − 2δS − 1 = 0 — minimal polynomial over ℚ |

**§21 Phase accumulation and NullSliceBridge coverage**

| # | Theorem | Description |
|---|---------|-------------|
| 59 | `phase_full_cycle` | D · (2π/D) = 2π — full return after D precession steps |
| 60 | `nullslice_channels_distinct` | {3k mod 8 : k ∈ Fin 8} has cardinality 8 |
| 61 | `nullslice_coverage_bijective` | k ↦ 3k is a bijection on ZMod 8 (gcd(3,8)=1) |

**§22 Machine-discovered deep connections**

| # | Theorem | Description |
|---|---------|-------------|
| 62 | `coherence_is_sech_of_log` | C(r) = (cosh(log r))⁻¹ — master Lyapunov link |
| 63 | `coherence_at_silver_is_eta` | C(δS) = η — cross-section discovery: §6∩§7 |
| 64 | `sech_at_log_silverRatio` | (cosh(log δS))⁻¹ = η — corollary |
| 65 | `lyapunov_tanh_residual` | Res(exp λ) = 2·sinh λ/δS — palindrome as sinh |
| 66 | `coherence_lyapunov_pythag` | C(exp λ)² + tanh²λ = 1 — hyperbolic Pythagorean |
| 67 | `coherence_residual_pythagorean` | C²+(δS·r·Res r/(1+r²))²=1 — unified form |
| 68 | `nullslice_involution` | 3·(3·k)=k in ZMod 8 — self-inverse bridge |
| 69 | `orbit_decoherence_rate` | C(rⁿ) ≤ 2/rⁿ — explicit decay bound |
| 70 | `mu_inv_eq_pow7` | μ⁷ = μ⁻¹ — inverse in the 8-cycle |
| 71 | `palindrome_sum_zero` | Res(r)+Res(1/r)=0 — anti-symmetry sum form |

---

### `TimeCrystal.lean`

Formalizes discrete time crystal theory: the phenomenon where a
T-periodically driven quantum system exhibits stable oscillation with
period 2T, spontaneously breaking discrete time-translation symmetry.

**§1 Time evolution operator**

| # | Theorem | Description |
|---|---------|-------------|
| 1 | `timeEvolution_zero` | U(H, 0) = 1 — identity at t = 0 |
| 2 | `timeEvolution_abs_one` | \|U(H,t)\| = 1 — unitarity |
| 3 | `timeEvolution_add` | U(t+s) = U(t)·U(s) — group law |

**§2 Floquet phase factor**

| # | Theorem | Description |
|---|---------|-------------|
| 4 | `floquetPhase_abs_one` | \|e^{−iφ}\| = 1 — unit circle |
| 5 | `floquetPhase_add` | e^{−i(φ₁+φ₂)} = e^{−iφ₁}·e^{−iφ₂} — composition |
| 6 | `floquetPhase_zero` | e^{−i·0} = 1 — trivial phase |
| 7 | `floquetPhase_two_pi` | e^{−i·2π} = 1 — full cycle |
| 8 | `floquetPhase_pi` | e^{−iπ} = −1 — Euler half-cycle |
| 9 | `floquetPhase_pi_sq` | (e^{−iπ})² = 1 — period-2 Floquet factor |

**§3 Floquet theorem**

| # | Theorem | Description |
|---|---------|-------------|
| 10 | `floquet_iterated` | ψ(t+n·T) = e^{−i·n·φ}·ψ(t) — iterated Floquet |
| 11 | `floquet_norm_invariant` | \|ψ(t+T)\| = \|ψ(t)\| — norm conserved per period |
| 12 | `floquet_norm_dynamical_invariant` | \|ψ(t+n·T)\| = \|ψ(t)\| — norm is dynamical invariant |

**§4 Time crystal states**

| # | Theorem | Description |
|---|---------|-------------|
| 13 | `timeCrystal_period_double` | ψ(t+2T) = ψ(t) — 2T-periodicity |
| 14 | `timeCrystal_symmetry_breaking` | T ≠ 2T (for T ≠ 0) — distinct periods |
| 15 | `timeCrystal_not_T_periodic` | ψ(t₀+T) ≠ ψ(t₀) when ψ(t₀) ≠ 0 |

**§5 Discrete time-translation symmetry breaking**

| # | Theorem | Description |
|---|---------|-------------|
| 16 | `timeCrystalState_breaks_symmetry` | non-trivial TC state satisfies DTTS-breaking |
| 17 | `timeCrystal_phase_not_sync` | e^{−iπ} ≠ 1 — crystal phase ≠ drive phase |
| 18 | `timeCrystal_period_ratio` | 2T/T = 2 — period-doubling ratio |

**§6 Quasi-energy and period-doubling**

| # | Theorem | Description |
|---|---------|-------------|
| 19 | `timeCrystalQuasiEnergy_phase` | ε_F · T = π — quasi-energy reconstruction |
| 20 | `timeCrystal_period_doubling_strict` | T > 0 → T < 2T — strict period doubling |

**§7 Kernel eigenvalue recipe for a time crystal**

Bridges `CriticalEigenvalue.lean` (μ, C, η, δS) with the Floquet framework to
give a six-step recipe for constructing the Kernel discrete time crystal.

| # | Theorem | Description |
|---|---------|-------------|
| 21 | `mu_isFloquetFactor` | \|μ\| = 1 — unitarity (restate from §2) |
| 22 | `mu_Hamiltonian_recipe` | H·T = 5π/4 → U(H,T) = μ — drive prescription |
| 23 | `mu_driven_iterated` | ψ(t+n·T) = μⁿ·ψ(t) — iterated μ-evolution |
| 24 | `mu_driven_norm_invariant` | \|ψ(t+T)\| = \|ψ(t)\| — 1-step norm conservation |
| 25 | `mu_driven_norm_n` | \|ψ(t+n·T)\| = \|ψ(t)\| — n-step norm conservation |
| 26 | `mu_driven_8period` | ψ(t+8T) = ψ(t) — 8-fold periodicity from μ^8=1 |
| 27 | `mu_ne_one` | μ ≠ 1 — non-trivial drive (uses mu_powers_distinct) |
| 28 | `mu_driven_not_T_periodic` | ψ(t₀+T) ≠ ψ(t₀) — not T-periodic |
| 29 | `mu_driven_breaks_symmetry` | (∃t, ψ(t+T)≠ψ(t)) ∧ (∀t, ψ(t+8T)=ψ(t)) |
| 30 | `mu_crystal_max_coherence` | C(1) = 1 — maximal coherence at amplitude ratio 1 |
| 31 | `mu_crystal_coherence_stability` | C(\|ψ(t+nT)\|/\|ψ(t)\|) = 1 — coherence maintained |
| 32 | `mu_crystal_canonical_init` | η²+normSq(μ·η)=1 — canonical normalization |
| 33 | `mu_crystal_silver_coherence` | C(δS) = η — silver ratio equals canonical amplitude |

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
