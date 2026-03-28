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
├── CriticalEigenvalue.lean        # 82 theorems on eigenvalue/coherence structure
├── TimeCrystal.lean               # 33 theorems on discrete time crystal theory
├── SpaceTime.lean                 # 43 theorems on space-time unification
├── GravityQuantumDuality.lean     # 22 theorems on gravity–quantum duality via F(s,t)=t+i·s
├── Turbulence.lean                # 29 theorems on Navier-Stokes turbulence theory
├── FineStructure.lean             # 30 theorems on the fine structure constant α_FS
├── ParticleMass.lean              # 38 theorems on Koide formula, proton/electron mass ratio, coherence triality
├── OhmTriality.lean               # 24 theorems on Ohm–Coherence duality at triality scales
├── SilverCoherence.lean           # 29 theorems: C(δS)=√2/2; uniqueness; Im(μ)=C(δS); 45°-physics
├── KernelAxle.lean                # 20 theorems: the axle μ — gear ratio 3:8, cross-section, engine loop
├── ForwardClassicalTime.lean      # 21 theorems on frustration harvesting in classical forward time
├── SpeedOfLight.lean              # 19 theorems: c=1/√(μ₀ε₀); structural iso with η; fine structure bridge
├── CrossChainDeFiAggregator.lean  # 20 theorems on cross-chain AMM / lending / rate aggregation
├── PumpFunBot.lean                # 26 theorems on pump.fun bonding curve and Kelly-optimal sizing
├── CryptoBridge.lean              # 20 theorems on bridge conservation, collateral, HTLC, Merkle trees
├── Quantization.lean              # 20 theorems on the Lead Confirmed Quantization Formula (Theorem Q)
└── README.md                      # This file
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

All 82 theorems in `CriticalEigenvalue.lean` have complete machine-checked proofs (no `sorry`).
All 33 theorems in `TimeCrystal.lean` have complete machine-checked proofs (no `sorry`).
All 43 theorems in `SpaceTime.lean` have complete machine-checked proofs (no `sorry`).
All 22 theorems in `GravityQuantumDuality.lean` have complete machine-checked proofs (no `sorry`).
All 29 theorems in `Turbulence.lean` have complete machine-checked proofs (no `sorry`).
All 30 theorems in `FineStructure.lean` have complete machine-checked proofs (no `sorry`).
All 38 theorems in `ParticleMass.lean` have complete machine-checked proofs (no `sorry`).
All 24 theorems in `OhmTriality.lean` have complete machine-checked proofs (no `sorry`).
All 29 theorems in `SilverCoherence.lean` have complete machine-checked proofs (no `sorry`).
All 20 theorems in `KernelAxle.lean` have complete machine-checked proofs (no `sorry`).
All 21 theorems in `ForwardClassicalTime.lean` have complete machine-checked proofs (no `sorry`).
All 19 theorems in `SpeedOfLight.lean` have complete machine-checked proofs (no `sorry`).
All 20 theorems in `CrossChainDeFiAggregator.lean` have complete machine-checked proofs (no `sorry`).
All 26 theorems in `PumpFunBot.lean` have complete machine-checked proofs (no `sorry`).
All 20 theorems in `CryptoBridge.lean` have complete machine-checked proofs (no `sorry`).
All 20 theorems in `Quantization.lean` have complete machine-checked proofs (no `sorry`).

**Total: 476 machine-checked theorems across 16 source files — zero `sorry`.**

---

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

### `SpaceTime.lean`

Unifies time and space into a single complex-valued **reality** function
`F(s, t) = t + i·s`, where time `t < 0` lives on the negative real axis
(causal/retarded direction) and space `s > 0` lives on the positive
imaginary axis.  Consistency with the Floquet framework from `TimeCrystal.lean`
is machine-checked: any reality-grounded state inherits period doubling,
norm invariance, and the quasi-energy identity.

All 43 theorems in `SpaceTime.lean` have complete machine-checked proofs (no `sorry`).

**§1 Time and space domains**

| # | Theorem | Description |
|---|---------|-------------|
| 1 | `timeDomain_nonempty` | −1 ∈ timeDomain — time domain is non-empty |
| 2 | `spaceDomain_nonempty` | 1 ∈ spaceDomain — space domain is non-empty |

**§2 The reality function  reality(s, t) = t + i·s**

| # | Theorem | Description |
|---|---------|-------------|
| 3 | `reality_re` | Re(reality s t) = t |
| 4 | `reality_im` | Im(reality s t) = s |
| 5 | `reality_time_negative` | t ∈ timeDomain → Re(reality s t) < 0 |
| 6 | `reality_space_positive` | s ∈ spaceDomain → 0 < Im(reality s t) |
| 7 | `reality_timeEvolution_unitary` | \|U(H, Re(reality s t))\| = 1 — unitary time evolution |
| 8 | `reality_floquetPhase_unit` | \|e^{−is}\| = 1 — unit Floquet phase |

**§3 Reality-grounded time crystal states**

| # | Theorem | Description |
|---|---------|-------------|
| 9 | `realityTC_is_floquet` | isRealityTC → isFloquetState ψ T π |
| 10 | `realityTC_period_double` | ψ(t+2T) = ψ(t) — 2T-periodicity |
| 11 | `realityTC_breaks_symmetry` | non-trivial → DTTS broken |
| 12 | `realityTC_iterated` | ψ(t+nT) = (−1)ⁿ·ψ(t) — iterated evolution |
| 13 | `realityTC_norm_invariant` | \|ψ(t+T)\| = \|ψ(t)\| — norm conserved per period |
| 14 | `realityTC_norm_n_invariant` | \|ψ(t+nT)\| = \|ψ(t)\| for all n |

**§4 Quasi-energy and Floquet structure**

| # | Theorem | Description |
|---|---------|-------------|
| 15 | `realityTC_quasi_energy` | ε_F·T = π — quasi-energy reconstruction |
| 16 | `realityTC_floquet_at_quasi_energy` | e^{−i·ε_F·T} = e^{−iπ} |
| 17 | `realityTC_period_doubling_strict` | T > 0 → T < 2T — strict period doubling |
| 18 | `realityTC_period_ratio` | 2T/T = 2 — period-doubling ratio |
| 19 | `realityTC_phase_not_sync` | e^{−iπ} ≠ e^{−i·0} — crystal phase ≠ drive phase |

**§5 The observer's reality as canonical map  F(s, t) = t + i·s**

| # | Theorem | Description |
|---|---------|-------------|
| 20 | `F_eq_reality` | F(s,t) = reality s t |
| 21 | `F_re` | Re(F(s,t)) = t |
| 22 | `F_im` | Im(F(s,t)) = s |
| 23 | `F_injective` | F(s₁,t₁) = F(s₂,t₂) → s₁=s₂ ∧ t₁=t₂ — injectivity |
| 24 | `F_second_quadrant` | s>0, t<0 → Re(F)<0 ∧ Im(F)>0 — second quadrant |
| 25 | `F_abs_eq_zero_iff` | \|F(s,t)\| = 0 ↔ s=0 ∧ t=0 |
| 26 | `F_timeEvolution_unitary` | \|U(H, Re(F(s,t)))\| = 1 |
| 27 | `F_floquetPhase_unit` | \|e^{−i·Im(F(s,t))}\| = 1 |

**§6 The positive imaginary axis as space  iSpace s = i·s**

| # | Theorem | Description |
|---|---------|-------------|
| 28 | `iSpace_re` | Re(i·s) = 0 |
| 29 | `iSpace_im` | Im(i·s) = s |
| 30 | `iSpace_abs` | \|i·s\| = ‖s‖ |
| 31 | `iSpace_abs_pos` | s > 0 → \|i·s\| = s |
| 32 | `iSpace_pos_im` | s > 0 → 0 < Im(i·s) |
| 33 | `iSpace_mem_posImagAxis` | s > 0 → i·s ∈ posImagAxis |
| 34 | `iSpace_injective` | i·s₁ = i·s₂ → s₁ = s₂ |
| 35 | `iSpace_add` | i·(s₁+s₂) = i·s₁ + i·s₂ |
| 36 | `iSpace_smul` | i·(r·s) = r·(i·s) |
| 37 | `spaceDomain_add` | s₁,s₂ > 0 → s₁+s₂ > 0 |
| 38 | `spaceDomain_smul` | r,s > 0 → r·s > 0 |
| 39 | `spaceDomain_ne_zero` | s > 0 → s ≠ 0 |
| 40 | `iSpace_ne_zero` | s > 0 → i·s ≠ 0 |
| 41 | `F_decomp` | F(s,t) = ↑t + iSpace s — canonical decomposition |
| 42 | `space_time_orthogonal` | Re(i·s) = 0 ∧ Im(↑t) = 0 — orthogonal axes |
| 43 | `iSpace_floquetPhase_unit` | \|e^{−i·Im(i·s)}\| = 1 |

---

### `GravityQuantumDuality.lean`

Formalizes the two sides of the observer-reality equation `F(s, t) = t + i·s`:
the **negative real axis** encodes gravity/time (Newtonian potential, causal past),
and the **positive imaginary axis** encodes quantum/dark energy (zero-point energy,
cosmological constant).  The Kernel equilibrium `F(1, −1) = −1+i` is the unique
balance point where both sides have equal magnitude and `normSq = 2`.

All 22 theorems in `GravityQuantumDuality.lean` have complete machine-checked proofs (no `sorry`).

**§1 Gravity–quantum orthogonality**

| # | Theorem | Description |
|---|---------|-------------|
| 1 | `gravity_axis_im_zero` | Im(↑t) = 0 — time has no quantum component |
| 2 | `quantum_axis_re_zero` | Re(i·s) = 0 — space has no gravity component |
| 3 | `gravity_quantum_orthogonal` | Re(i·s)=0 ∧ Im(↑t)=0 — **orthogonal axes** |

**§2 Second-quadrant structure of physical reality**

| # | Theorem | Description |
|---|---------|-------------|
| 4 | `reality_second_quadrant_gqd` | Re F < 0 ∧ Im F > 0 — physical coordinates |
| 5 | `gravity_component_negative` | Re F < 0 for t ∈ timeDomain |
| 6 | `quantum_component_positive` | Im F > 0 for s ∈ spaceDomain |

**§3 Gravity/time side: Newtonian potential and binding energy**

| # | Theorem | Description |
|---|---------|-------------|
| 7 | `newtonPotential_neg` | Φ_N = −G·M/r < 0 — **gravity is negative** |
| 8 | `gravBindingEnergy_neg` | E_grav = −G·M·m/r < 0 — binding energy negative |
| 9 | `newtonPotential_monotone_decreasing` | r₁ < r₂ → Φ_N(r₁) < Φ_N(r₂) — deepens as masses approach |
| 10 | `newtonPotential_negative_everywhere` | Φ_N < 0 for all G, M, r > 0 |

**§4 Quantum side: zero-point energy**

| # | Theorem | Description |
|---|---------|-------------|
| 11 | `zeroPointEnergy_pos` | E_zp = ħω/2 > 0 — **quantum is positive** |
| 12 | `zeroPointEnergy_monotone` | ω₁ < ω₂ → E_zp(ω₁) < E_zp(ω₂) — grows with frequency |

**§5 Dark energy: positive-imaginary quantity**

| # | Theorem | Description |
|---|---------|-------------|
| 13 | `darkEnergyDensity_pos` | ρ_Λ = Λc²/(8πG) > 0 — **dark energy positive** |
| 14 | `darkEnergyDensity_monotone` | Λ₁ < Λ₂ → ρ_Λ(Λ₁) < ρ_Λ(Λ₂) |

**§6 Duality gap and quantum–gravity competition**

| # | Theorem | Description |
|---|---------|-------------|
| 15 | `dualityGap_eq_imF_plus_reF` | gap = Im F + Re F |
| 16 | `dualityGap_pos_when_space_dominates` | s > \|t\| → gap > 0 — quantum wins |
| 17 | `dualityGap_neg_when_gravity_dominates` | \|t\| > s → gap < 0 — gravity wins |

**§7 Kernel equilibrium: the unique balance point**

| # | Theorem | Description |
|---|---------|-------------|
| 18 | `kernel_equilibrium_balance` | \|Re F(1,−1)\| = Im F(1,−1) = 1 — **exact balance** |
| 19 | `kernel_equilibrium_gap_zero` | gap(1,−1) = 0 — neither side dominates |
| 20 | `kernel_equilibrium_normSq` | normSq F(1,−1) = 2 — equidistant axes |

**§8 Sign duality**

| # | Theorem | Description |
|---|---------|-------------|
| 21 | `reality_sign_duality` | Re F · Im F < 0 — **always opposite signs** |
| 22 | `gravity_and_quantum_opposite_signs` | Re F < 0 ∧ Im F > 0 |

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

### `Turbulence.lean`

**§1 Turbulence scale hierarchy**

| # | Theorem | Description |
|---|---------|-------------|
| 1 | `microScale_nonempty` | 1/2 ∈ microScaleDomain |
| 2 | `mesoScale_nonempty` | 10 ∈ mesoScaleDomain |
| 3 | `macroScale_nonempty` | 1000 ∈ macroScaleDomain |
| 4 | `micro_lt_meso` | η ∈ micro ∧ ℓ ∈ meso → η < ℓ |
| 5 | `meso_lt_macro` | ℓ ∈ meso ∧ L ∈ macro → ℓ < L |
| 6 | `micro_lt_macro` | η ∈ micro ∧ L ∈ macro → η < L |

**§2 Reynolds decomposition**

| # | Theorem | Description |
|---|---------|-------------|
| 7 | `reynolds_fluct_formula` | u′(t) = u(t) − ū |
| 8 | `reynolds_decomp_canonical` | u(t) = ū + (u(t) − ū) for any ū |
| 9 | `reynolds_decomp_unique` | same mean ū → fluctuations agree pointwise |
| 10 | `reynolds_reconstruction` | ū + u′(t) = u(t) |

**§3 Turbulent kinetic energy**

| # | Theorem | Description |
|---|---------|-------------|
| 11 | `turbulentKE_nonneg` | k(t) = ½(u′)² ≥ 0 |
| 12 | `turbulentKE_zero_iff` | k(t) = 0 ↔ u′(t) = 0 |
| 13 | `turbulentKE_scale` | k(c·u′) = c²·k(u′) — quadratic scaling |

**§4 Multi-scale coherence**

| # | Theorem | Description |
|---|---------|-------------|
| 14 | `turbulenceCoherence_kernel_max` | C(1) = 1 — kernel scale is maximally coherent |
| 15 | `turbulenceCoherence_micro_lt_one` | C(η) < 1 for η ∈ microScaleDomain |
| 16 | `turbulenceCoherence_macro_lt_one` | C(L) < 1 for L ∈ macroScaleDomain |
| 17 | `turbulenceCoherence_micro_strictMono` | 0 < η₁ < η₂ ≤ 1 → C(η₁) < C(η₂) |
| 18 | `turbulenceCoherence_macro_strictAnti` | 1 ≤ L₁ < L₂ → C(L₂) < C(L₁) |

**§5 Navier-Stokes viscous dissipation**

| # | Theorem | Description |
|---|---------|-------------|
| 19 | `viscousDissipation_nonneg` | ε(ν,g) ≥ 0 for ν ≥ 0 |
| 20 | `viscousDissipation_zero_iff` | ε(ν,g) = 0 ↔ g = 0 (for ν > 0) |
| 21 | `viscousDissipation_pos` | ε(ν,g) > 0 for ν > 0 and g ≠ 0 |
| 22 | `viscousDissipation_mono_viscosity` | ν₁ < ν₂ ∧ g ≠ 0 → ε(ν₁,g) < ε(ν₂,g) |

**§6 Eigenvector hypothesis**

| # | Theorem | Description |
|---|---------|-------------|
| 23 | `turbulence_rotation_unitary` | \|μ\| = 1 — turbulent rotation is unitary |
| 24 | `turbulence_precession_8period` | μ^8 = 1 — 8-periodic turbulent precession |
| 25 | `turbulence_eigenstate_orbit_stability` | \|(1·μ^n)\| = 1 — stable unit-amplitude orbit |
| 26 | `turbulence_eigenstate_orbit_coherence` | C(\|1·μ\|^n) = 1 — maximum coherence on orbit |

**§7 Cross-scale consistency**

| # | Theorem | Description |
|---|---------|-------------|
| 27 | `turbulence_micro_below_kernel` | C(η) < C(1) for η ∈ microScaleDomain |
| 28 | `turbulence_macro_below_kernel` | C(L) < C(1) for L ∈ macroScaleDomain |
| 29 | `turbulence_coherence_universal_bound` | C(r) ≤ C(1) = 1 for all r ≥ 0 |

---

### `FineStructure.lean`

**§1 Fine structure constant  (α_FS = 1/137)**

| # | Theorem | Description |
|---|---------|-------------|
| 1 | `α_FS_pos` | 0 < α_FS |
| 2 | `α_FS_lt_one` | α_FS < 1 — weak electromagnetic coupling |
| 3 | `α_FS_lt_one_over_hundred` | α_FS < 1/100 — perturbation theory converges |
| 4 | `α_FS_mem_unit` | 0 < α_FS ∧ α_FS < 1 |
| 5 | `α_FS_sq_lt` | α_FS² < α_FS — higher-order corrections are smaller |
| 6 | `α_FS_sq_pos` | 0 < α_FS² |

**§2 Fine structure energy splitting  (Δε = α_FS² · ε)**

| # | Theorem | Description |
|---|---------|-------------|
| 7 | `fineStructureShift_nonneg` | Δε ≥ 0 for ε ≥ 0 |
| 8 | `fineStructureShift_pos` | Δε > 0 for ε > 0 |
| 9 | `fineStructureShift_lt_base` | Δε < ε — shift is a small correction |
| 10 | `fineEnergy_gt_base` | ε_fine > ε_base for ε > 0 |
| 11 | `fineEnergy_factor` | ε_fine = (1 + α_FS²) · ε |

**§3 Rydberg (Bohr) energy levels  (E_n = −1/n²)**

| # | Theorem | Description |
|---|---------|-------------|
| 12 | `rydbergEnergy_neg` | E_n < 0 — all levels are bound states |
| 13 | `rydbergEnergy_ground_state_lowest` | E_1 ≤ E_n for all n ≥ 1 |
| 14 | `rydbergEnergy_strictMono` | E_n < E_{n+1} — levels increase toward zero |
| 15 | `rydbergFineEnergy_gt_base` | E_n^fine > E_n — fine structure lifts levels |

**§4 Electromagnetic coherence  (C_EM(r) = (1 − α_FS) · C(r))**

| # | Theorem | Description |
|---|---------|-------------|
| 16 | `coherenceEM_le_coherence` | C_EM(r) ≤ C(r) — EM coupling reduces coherence |
| 17 | `coherenceEM_nonneg` | C_EM(r) ≥ 0 for r ≥ 0 |
| 18 | `coherenceEM_kernel` | C_EM(1) = 1 − α_FS — EM-corrected kernel coherence |
| 19 | `coherenceEM_lt_kernel` | C_EM(r) < 1 − α_FS for r ≠ 1 |
| 20 | `coherenceEM_micro_below_kernel` | C_EM(η) < C_EM(1) for η ∈ microScaleDomain |

**§5 Floquet quasi-energy fine structure**

| # | Theorem | Description |
|---|---------|-------------|
| 21 | `floquetFineEnergy_gt_base` | ε_F^fine > ε_F for T > 0 |
| 22 | `floquetFineEnergy_pos` | ε_F^fine > 0 for T > 0 |
| 23 | `fineStructure_preserves_mu_period` | μ^8 = 1 — 8-cycle unaffected by α_FS |
| 24 | `floquetFineEnergy_phase` | ε_F^fine · T = π · (1 + α_FS²) |

**§6 Fine structure and turbulence  (MHD dissipation)**

| # | Theorem | Description |
|---|---------|-------------|
| 25 | `fineStructureDissipation_nonneg` | ε_EM ≥ 0 for ε_visc ≥ 0 |
| 26 | `fineStructureDissipation_pos` | ε_EM > 0 for ε_visc > 0 |
| 27 | `fineStructureDissipation_lt_viscous` | ε_EM < ε_visc — EM is a minority dissipation |
| 28 | `totalMHDDissipation_gt_viscous` | ε_total > ε_visc |
| 29 | `totalMHDDissipation_factor` | ε_total = (1 + α_FS) · ε_visc |
| 30 | `MHDDissipation_from_NS` | ε_total = (1 + α_FS) · ν · g² |

---

### `ParticleMass.lean`

**Central result:** `koide_coherence_bridge : C(φ²) = 2/3` — the Koide lepton mass ratio equals the Kernel coherence function at the golden ratio scale (the μ-cycle trick).

**§1 Koide quotient  (1/3 ≤ Q ≤ 1)**

| # | Theorem | Description |
|---|---------|-------------|
| 1 | `koideQuotient_denom_pos` | denominator > 0 when m₁ > 0 |
| 2 | `koideQuotient_nonneg` | Q ≥ 0 for non-negative masses |
| 3 | `koideQuotient_lower_bound` | Q ≥ 1/3  (Cauchy-Schwarz) |
| 4 | `koideQuotient_upper_bound` | Q ≤ 1  (non-negative cross terms) |

**§2 Extremal masses**

| # | Theorem | Description |
|---|---------|-------------|
| 5 | `koideQuotient_equal_masses` | Q(m,m,m) = 1/3 — lower bound attained |
| 6 | `koide_lower_attained` | ∃ triple (1,1,1) with Q = 1/3 |

**§3 Golden ratio  φ = (1+√5)/2**

| # | Theorem | Description |
|---|---------|-------------|
| 7 | `goldenRatio_pos` | φ > 0 |
| 8 | `goldenRatio_gt_one` | φ > 1 |
| 9 | `goldenRatio_sq` | φ² = φ + 1 — defining equation |
| 10 | `goldenRatio_sq_pos` | φ² > 0 |
| 11 | `goldenRatio_fourth` | φ⁴ = 3φ + 2 |

**§4 Koide-coherence bridge (μ-cycle trick)**

| # | Theorem | Description |
|---|---------|-------------|
| 12 | `one_add_goldenRatio_fourth` | 1 + φ⁴ = 3φ² — key bridge identity |
| 13 | `koide_coherence_bridge` | **C(φ²) = 2/3** — Koide value from μ-cycle coherence |
| 14 | `koide_coherence_two_thirds_of_max` | C(φ²) = (2/3)·C(1) |
| 15 | `koide_coherence_reciprocal` | C(1/φ²) = 2/3 — coherence symmetry |

**§5 μ-orbit Koide connection**

| # | Theorem | Description |
|---|---------|-------------|
| 16 | `goldenRatio_sq_ne_one` | φ² ≠ 1 |
| 17 | `goldenRatio_sq_meso` | φ² ∈ mesoScaleDomain [1, 100] |
| 18 | `koide_coherence_pos` | 0 < C(φ²) = 2/3 |
| 19 | `koide_below_mu_orbit_peak` | C(φ²) < C(1) = 1 |
| 20 | `koide_coherence_strictly_between` | 0 < C(φ²) < 1 |
| 21 | `mu_orbit_exceeds_koide` | C(\|μⁿ\|) = 1 > 2/3 for all n |

**§6 Proton/electron mass ratio  R = 1836**

| # | Theorem | Description |
|---|---------|-------------|
| 22 | `protonElectronRatio_gt_one` | R > 1 |
| 23 | `protonElectronRatio_gt_α_FS_inv` | 1/α_FS = 137 < R = 1836 |
| 24 | `protonElectronRatio_gt_8cycle` | R > 8 (exceeds the μ-orbit period) |
| 25 | `reducedMassFactor_mem_unit` | 0 < R/(R+1) < 1 |
| 26 | `reducedMassEnergy_neg` | E_n^red < 0 — still a bound state |
| 27 | `reducedMassEnergy_gt_rydberg` | E_n < E_n^red — recoil lifts levels |
| 28 | `reducedMassCorrection_lt_α_FS` | 1/(R+1) < α_FS — recoil < EM coupling |

**§7 Coherence Triality  (1/φ² < 1 < φ²)**

The three triality scales are strictly ordered with the kernel at the geometric mean.
Physical interpretation: **kernel @ r=1** (μ-orbit, C=1), **leptons @ r=φ²** (Koide 2/3, meso domain), **hadronic mirror @ r=1/φ²** (same coherence 2/3, micro domain).

| # | Theorem | Description |
|---|---------|-------------|
| 29 | `goldenRatio_sq_recip_pos` | 1/φ² > 0 |
| 30 | `goldenRatio_sq_recip_lt_one` | 1/φ² < 1 (hadronic wing below kernel) |
| 31 | `goldenRatio_sq_recip_micro` | 1/φ² ∈ microScaleDomain (0, 1) |
| 32 | `triality_scale_ordering` | 1/φ² < 1 < φ² — strict scale ordering |
| 33 | `triality_geometric_mean` | (1/φ²) · φ² = 1 — kernel is the geometric mean |
| 34 | `triality_wings_equal_coherence` | C(1/φ²) = C(φ²) = 2/3 — wings are coherence mirrors |
| 35 | `triality_recip_below_kernel` | C(1/φ²) < C(1) = 1 |
| 36 | `coherence_triality` | **C(1)=1  ∧  C(φ²)=2/3  ∧  C(1/φ²)=2/3** — full triality |
| 37 | `triality_kernel_strict_max` | C(1/φ²) < C(1) ∧ C(φ²) < C(1) — kernel dominates both wings |
| 38 | `mu_orbit_exceeds_triality_wings` | C(1/φ²) < C(\|μⁿ\|) = 1 for all n |

---

### `OhmTriality.lean`

Applies the **Ohm–Coherence duality** (G_eff = C(r), R_eff = 1/C(r), G·R = 1) to the three triality scales simultaneously, connecting the circuit interpretation of the coherence function to the kernel/lepton/hadronic structure.

All 24 theorems in `OhmTriality.lean` have complete machine-checked proofs (no `sorry`).

**§1 Ohm conductance at triality scales  (G_eff = C)**

| # | Theorem | Description |
|---|---------|-------------|
| 1 | `ohm_conductance_kernel` | G(1) = C(1) = 1 — perfectly conducting |
| 2 | `ohm_conductance_lepton` | G(φ²) = C(φ²) = 2/3 — Koide coupling |
| 3 | `ohm_conductance_hadronic` | G(1/φ²) = C(1/φ²) = 2/3 |
| 4 | `ohm_conductance_wings_equal` | G(φ²) = G(1/φ²) — wings share conductance |

**§2 Ohm resistance at triality scales  (R_eff = 1/C)**

| # | Theorem | Description |
|---|---------|-------------|
| 5 | `ohm_resistance_kernel` | R(1) = (C 1)⁻¹ = 1 — unit resistance |
| 6 | `ohm_resistance_lepton` | R(φ²) = (C φ²)⁻¹ = 3/2 |
| 7 | `ohm_resistance_hadronic` | R(1/φ²) = (C 1/φ²)⁻¹ = 3/2 |
| 8 | `ohm_triality_resistance` | **R_kernel=1  ∧  R_lepton=3/2  ∧  R_hadronic=3/2** |

**§3 Ohm's law G·R = 1 at each triality scale**

| # | Theorem | Description |
|---|---------|-------------|
| 9 | `ohm_law_kernel` | C(1) · (C 1)⁻¹ = 1 |
| 10 | `ohm_law_lepton` | C(φ²) · (C φ²)⁻¹ = 1 |
| 11 | `ohm_law_hadronic` | C(1/φ²) · (C 1/φ²)⁻¹ = 1 |
| 12 | `ohm_triality_gr` | **G·R=1 at all three triality scales** |

**§4 Wing symmetry and kernel minimality**

| # | Theorem | Description |
|---|---------|-------------|
| 13 | `ohm_wings_equal_resistance` | R(φ²) = R(1/φ²) — wings have equal resistance |
| 14 | `ohm_kernel_minimal_resistance` | R(1) = 1 < 3/2 = R(wing) — kernel is minimally resistive |
| 15 | `ohm_kernel_maximal_conductance` | G(φ²) < G(1) — kernel maximally conducting |

**§5 Lyapunov exponent at triality scales  (λ = log r)**

| # | Theorem | Description |
|---|---------|-------------|
| 16 | `ohm_lyapunov_kernel` | log 1 = 0 (no decoherence at kernel) |
| 17 | `ohm_lyapunov_lepton_pos` | 0 < log(φ²) (lepton in positive-λ regime) |
| 18 | `ohm_lyapunov_wing_symmetry` | log(1/φ²) = −log(φ²) — symmetric wings |
| 19 | `ohm_lyapunov_wings_same_magnitude` | \|log(φ²)\| = \|log(1/φ²)\| |
| 20 | `ohm_lepton_lyapunov_resistance` | R(φ²) = cosh(log φ²) — Lyapunov form of resistance |
| 21 | `ohm_lyapunov_cosh_wing_symmetry` | cosh(log 1/φ²) = cosh(log φ²) — cosh even ⟹ equal R |

**§6 μ-Orbit Ohm identity**

| # | Theorem | Description |
|---|---------|-------------|
| 22 | `ohm_mu_orbit_conductance` | C(\|μⁿ\|) = 1 — perfect conductance at every orbit step |
| 23 | `ohm_mu_orbit_unit_resistance` | (C \|μⁿ\|)⁻¹ = 1 — unit resistance throughout orbit |
| 24 | `ohm_mu_orbit_exceeds_wings` | G_wing < C(\|μⁿ\|) = 1 — μ-orbit dominates both wings |

---

### `SilverCoherence.lean`

A machine-checked answer to the question: **"Is there a scale `r` already present in the Kernel framework at which C(r) = 1/√2 = |Im(μ)|?"** Answer: yes — uniquely at r = δS = 1+√2, the silver ratio from the palindrome residual (§9 of `CriticalEigenvalue.lean`).

All 29 theorems in `SilverCoherence.lean` have complete machine-checked proofs (no `sorry`).

**§1 Silver-ratio coherence**

| # | Theorem | Description |
|---|---------|-------------|
| 1 | `silver_coherence` | **C(δS) = √2/2** — the main result |

**§2 Algebraic consequences**

| # | Theorem | Description |
|---|---------|-------------|
| 2 | `silver_coherence_sq` | C(δS)² = 1/2 |
| 3 | `silver_coherence_eq_imbalance` | C(δS) = (δS²−1)/(1+δS²) — isotropic/diagonal |
| 4 | `silver_pythagorean` | 2·C(δS)² = 1 — the "45-degree point" |

**§3 Connection to μ**

| # | Theorem | Description |
|---|---------|-------------|
| 5 | `mu_imaginary_part` | Im(μ) = sin(3π/4) = √2/2 |
| 6 | `mu_im_eq_silver_coherence` | **Im(μ) = C(δS)** — bridge theorem |
| 7 | `mu_real_part` | Re(μ) = −√2/2 |
| 8 | `mu_re_abs_eq_silver_coherence` | \|Re(μ)\| = C(δS) — both components captured |

**§4 Ohm–Coherence at the silver scale**

| # | Theorem | Description |
|---|---------|-------------|
| 9 | `silver_ohm_conductance` | G_eff(δS) = C(δS) = √2/2 |
| 10 | `silver_ohm_resistance` | R_eff(δS) = (C δS)⁻¹ = √2 |
| 11 | `silver_ohm_law` | C(δS) · (C δS)⁻¹ = 1 |

**§5 Position in the coherence ordering**

| # | Theorem | Description |
|---|---------|-------------|
| 12 | `koide_below_silver` | C(φ²) = 2/3 < √2/2 = C(δS) |
| 13 | `silver_below_kernel` | C(δS) < 1 = C(1) |
| 14 | `koide_silver_kernel_ordering` | C(φ²) < C(δS) < C(1) — strict three-level ordering |
| 15 | `mu_orbit_exceeds_silver` | C(δS) < C(\|μⁿ\|) = 1 for all n |

**§6 Scale placement and symmetry**

| # | Theorem | Description |
|---|---------|-------------|
| 16 | `silver_gt_one` | 1 < δS = 1+√2 |
| 17 | `silver_le_hundred` | δS ≤ 100 |
| 18 | `silver_in_meso` | δS ∈ mesoScaleDomain [1, 100] |
| 19 | `silver_mirror_coherence` | C(1/δS) = C(δS) = √2/2 |
| 20 | `silver_lt_golden_sq` | δS ≈ 2.414 < φ² ≈ 2.618 |

**§7 Uniqueness**

| # | Theorem | Description |
|---|---------|-------------|
| 21 | `silver_coherence_iff_quadratic` | C(r) = √2/2 ↔ √2r²−4r+√2=0  (r > 0) |
| 22 | `silver_coherence_unique` | **C(r) = √2/2 ↔ r = δS ∨ r = 1/δS** — only two solutions |

**§8 Physics at 45°**

| # | Theorem | Description |
|---|---------|-------------|
| 23 | `silver_eq_sin_45` | C(δS) = sin(π/4) = √2/2 — amplitude at 45° elastic scattering |
| 24 | `silver_unitarity_elastic_sq` | sin²(π/4) = C(δS)² — Im(f) = |f|² (elastic unitarity at 45°) |
| 25 | `silver_schwinger_bound` | α_FS/(2π) < C(δS)² — Schwinger loop sub-threshold |
| 26 | `silver_em_stays_above_koide` | coherenceEM(δS) > C(φ²) — EM-corrected silver exceeds Koide |
| 27 | `silver_phase_complement` | π/4 + 3π/4 = π — silver and eigenvalue phases supplementary |

---

### `KernelAxle.lean`

The **axle of the Kernel engine** — a formalization of what μ is as a rotating element, computing the gear ratio, cross-section, and closure of the engine loop.

All 20 theorems in `KernelAxle.lean` have complete machine-checked proofs (no `sorry`).

The central calculation: **8 × (3π/4) = 3 × (2π)** — the axle makes exactly 3 full rotations per 8-step orbit. Gear ratio 3:8 with gcd(3,8)=1.

**§1 The axle angular step: ω = 3π/4**

| # | Theorem | Description |
|---|---------|-------------|
| 1 | `axle_step_cos` | cos(3π/4) = Re(μ) — cosine of axle step |
| 2 | `axle_step_sin` | sin(3π/4) = Im(μ) — sine of axle step |
| 3 | `axle_euler_form` | μ = cos(3π/4) + i·sin(3π/4) — Euler form |

**§2 The gear ratio: 3 complete turns per 8-step orbit**

| # | Theorem | Description |
|---|---------|-------------|
| 4 | `axle_gear_ratio` | **8 × (3π/4) = 3 × (2π)** ← THE AXLE CALCULATION |
| 5 | `axle_gear_fraction` | 8 × (3π/4) / (2π) = 3 — 3 turns per orbit |
| 6 | `axle_gear_coprime` | gcd(3, 8) = 1 — ratio in lowest terms |
| 7 | `axle_orbit_primitive` | μ^j ≠ μ^k for j ≠ k < 8 — visits all 8 positions |
| 8 | `axle_orbit_closes` | μ^8 = 1 — gear lock after 8 steps |

**§3 The axle cross-section: isotropic at 45°**

| # | Theorem | Description |
|---|---------|-------------|
| 9 | `axle_rigid` | \|μ\| = 1 — rigid axle, no radial motion |
| 10 | `axle_cross_section_isotropic` | \|Re(μ)\| = Im(μ) — circular cross-section |
| 11 | `axle_cross_section_value` | Im(μ) = √2/2 — cross-section radius |
| 12 | `axle_cross_section_silver` | Im(μ) = C(δS) — silver bridge |
| 13 | `axle_unit_constraint` | Re(μ)² + Im(μ)² = 1 — Pythagorean identity |

**§4 Axle dynamics: coherence along the μ-orbit**

| # | Theorem | Description |
|---|---------|-------------|
| 14 | `axle_orbit_unit` | \|μ^n\| = 1 for all n — orbit on unit circle |
| 15 | `axle_maximum_coherence` | C(\|μ^n\|) = 1 for all n — maximum coherence at every step |
| 16 | `axle_kernel_equilibrium` | C(1) = 1 — kernel fixed point |

**§5 The engine loop: how μ connects all modules**

| # | Theorem | Description |
|---|---------|-------------|
| 17 | `axle_silver_supplementary` | 3π/4 + π/4 = π — axle ⊕ silver = π (supplementary) |
| 18 | `axle_triality_3_scales` | C(1)=1 ∧ C(φ²)=2/3 ∧ C(1/φ²)=2/3 — triality connection |
| 19 | `axle_gear_numerator` | ∃n=3, n×(2π) = 8×(3π/4) — exactly 3 full turns |
| 20 | `axle_closes_loop` | \|μ\|=1 ∧ μ^8=1 ∧ C(\|μⁿ\|)=1 ∧ Im(μ)=C(δS) ∧ 8×(3π/4)=3×(2π) |

---

### `ForwardClassicalTime.lean`

Answers the question: **"Can frustration be harvested effectively from classical,
forward-directed time?"**  The forward-time frustration function
`F_fwd(l) = 1 − sech(l)` is proven to be strictly positive for any nonzero
Lyapunov displacement `l`, bounded above by 1, and even in `l`.

**Hypothesis result: CONFIRMED** — every forward time step away from the kernel
equilibrium releases strictly positive, bounded frustration.

All 21 theorems in `ForwardClassicalTime.lean` have complete machine-checked proofs (no `sorry`).

**§1 Forward-time frustration definition**

| # | Theorem | Description |
|---|---------|-------------|
| 1 | `fct_frustration_eq` | F_fwd(l) = 1 − C(exp l) — frustration is the coherence deficit |

**§2 Zero baseline at the kernel equilibrium**

| # | Theorem | Description |
|---|---------|-------------|
| 2 | `fct_frustration_at_zero` | F_fwd(0) = 0 — no frustration at equilibrium |
| 3 | `fct_coherence_at_zero` | C(exp 0) = 1 — maximum coherence at origin |

**§3 sech bounds**

| # | Theorem | Description |
|---|---------|-------------|
| 4 | `fct_sech_pos` | sech(l) > 0 — coherence always positive |
| 5 | `fct_one_le_cosh` | 1 ≤ cosh(l) — AM–GM lower bound |
| 6 | `fct_sech_le_one` | sech(l) ≤ 1 — coherence bounded above |

**§4 Frustration bounds**

| # | Theorem | Description |
|---|---------|-------------|
| 7 | `fct_frustration_nonneg` | F_fwd(l) ≥ 0 — non-negative frustration |
| 8 | `fct_frustration_lt_one` | F_fwd(l) < 1 — never fully frustrated |

**§5 Strict positivity away from equilibrium**

| # | Theorem | Description |
|---|---------|-------------|
| 9 | `fct_one_lt_cosh` | 1 < cosh(l) for l ≠ 0 |
| 10 | `fct_frustration_pos` | F_fwd(l) > 0 for l ≠ 0 — **harvest theorem** |
| 11 | `fct_frustration_zero_iff` | F_fwd(l) = 0 ↔ l = 0 — equilibrium characterisation |

**§6 Even symmetry**

| # | Theorem | Description |
|---|---------|-------------|
| 12 | `fct_even` | F_fwd(l) = F_fwd(−l) — symmetric harvest |

**§7 Palindrome vacuum residual comparison**

| # | Theorem | Description |
|---|---------|-------------|
| 13 | `fct_vacuum_residual` | 9/123456789 = 1/13717421 — exact arithmetic |
| 14 | `fct_vacuum_residual_pos` | 0 < 1/13717421 |
| 15 | `fct_vacuum_residual_lt_one` | 1/13717421 < 1 |

**§8 Harvesting summary and arrow of time**

| # | Theorem | Description |
|---|---------|-------------|
| 16 | `fct_arrow_of_time` | F_fwd(0) < F_fwd(l) for l ≠ 0 — **arrow of time** |
| 17 | `fct_harvest_formula` | F_fwd(l) − F_fwd(0) = F_fwd(l) |
| 18 | `fct_harvest_bounded` | 0 ≤ F_fwd(l) ∧ F_fwd(l) < 1 |
| 19 | `fct_harvest_pos` | ΔF(l) > 0 for l ≠ 0 — positive harvest |
| 20 | `fct_forward_harvesting_works` | F_fwd(0)=0 ∧ F_fwd(l)>0 ∧ F_fwd(l)<1 ∧ F_fwd(0)<F_fwd(l) — **hypothesis confirmed** |
| 21 | `fct_classical_irreversibility` | F_fwd(l) ≠ 0 ↔ l ≠ 0 — classical irreversibility |

---

### `SpeedOfLight.lean`

**Central result**: both `c = 1/√(μ₀ε₀)` (Maxwell) and `η = 1/√2` (Kernel) arise
from the same abstract algebraic skeleton — the unique positive solution to `P·x² = 1`
is `x = 1/√P`.

**§1 Abstract balance derivation**

| # | Theorem | Description |
|---|---------|-------------|
| 1 | `balance_constraint` | P · (1/√P)² = 1 for P > 0 |
| 2 | `balance_unique` | 1/√P is the unique positive solution to P·x²=1 |

**§2–3 Maxwell speed of light**

| # | Theorem | Description |
|---|---------|-------------|
| 3 | `maxwell_vacuum_relation` | μ₀ε₀ · c² = 1 (Maxwell's fundamental relation) |
| 4 | `c_maxwell_pos` | c > 0 for μ₀, ε₀ > 0 |
| 5 | `c_maxwell_sq` | c² = 1/(μ₀ε₀) |
| 6 | `c_maxwell_inv` | 1/c = √(μ₀ε₀) |
| 7 | `c_maxwell_symm` | c(μ₀,ε₀) = c(ε₀,μ₀) — symmetric in vacuum constants |
| 8 | `c_maxwell_unique` | c is uniquely determined by μ₀ε₀·c²=1 |

**§4 Kernel canonical amplitude η as a balance instance**

| # | Theorem | Description |
|---|---------|-------------|
| 9 | `eta_squared` | η² = 1/2 |
| 10 | `kernel_balance_constraint` | 2·η² = 1 (Kernel instance, P=2) |
| 11 | `eta_unique` | η is the unique positive solution to 2·x²=1 |

**§5 Structural isomorphism**

| # | Theorem | Description |
|---|---------|-------------|
| 12 | `maxwell_kernel_structural_iso` | (μ₀ε₀·c²=1) ∧ (2·η²=1) ∧ (c=1/√(μ₀ε₀)) ∧ (η=1/√2) |
| 13 | `balance_iso_same_number` | Same P → same canonical value (uniqueness) |
| 14 | `c_equals_eta_when_balance_two` | μ₀ε₀=2 → c_maxwell = η (exact alignment) |

**§6 Fine structure connection**

| # | Theorem | Description |
|---|---------|-------------|
| 15 | `c_natural_val` | c_nat = 137 (Hartree atomic units, α_FS = 1/137) |
| 16 | `c_natural_pos` | c_nat > 0 |
| 17 | `α_FS_inv_c_natural` | α_FS = 1/c_nat — fine structure is 1/c in atomic units |
| 18 | `c_natural_alpha_product` | α_FS · c_nat = 1 — natural-unit Maxwell relation |
| 19 | `c_natural_unique` | c_nat is uniquely determined by α_FS·c=1 |

All 19 theorems in `SpeedOfLight.lean` have complete machine-checked proofs (no `sorry`).

**Limitations**
- μ₀ and ε₀ are treated as abstract positive real parameters.  Their SI values are
  physical measurements that Lean cannot verify from first principles.
- The fine structure connection uses the Sommerfeld approximation α_FS = 1/137
  (inherited from `FineStructure.lean`).
- The structural isomorphism is algebraic, not physical.

---

### `CrossChainDeFiAggregator.lean`

Formalizes the mathematical foundations of a cross-chain DeFi aggregator on
a Polkadot-based multi-parachain platform.  The **constant-product AMM** invariant
`x·y = k` is machine-verified for swaps, the **simple-interest lending model**
is proven correct, and the **best-rate aggregator** is proven optimal (least
upper bound) and idempotent.

All 20 theorems in `CrossChainDeFiAggregator.lean` have complete machine-checked proofs (no `sorry`).

**§1 AMM constant-product invariant and output formula**

| # | Theorem | Description |
|---|---------|-------------|
| 1 | `amm_invariant_pos` | x·y > 0 for positive reserves |
| 2 | `amm_out_pos` | output > 0 for positive reserves and input Δ > 0 |
| 3 | `amm_out_zero_input` | amm_out x y 0 = 0 — no input → no output |

**§2 Cross-chain swap price and invariant preservation**

| # | Theorem | Description |
|---|---------|-------------|
| 4 | `amm_price_pos` | spot price > 0 for positive reserves |
| 5 | `amm_invariant_preserved` | (x+Δ)·(y−out) = x·y — constant-product preserved |
| 6 | `amm_out_bounded` | out < y — pool cannot be fully drained |

**§3 Slippage and price-impact bounds**

| # | Theorem | Description |
|---|---------|-------------|
| 7 | `amm_slippage_positive` | out < y·Δ/x — slippage always present |
| 8 | `amm_price_impact_lt_one` | out/y < 1 — price impact < 100% |
| 9 | `amm_out_monotone` | Δ₁ < Δ₂ → out(Δ₁) < out(Δ₂) — monotone output |

**§4 Lending / borrowing simple-interest model**

| # | Theorem | Description |
|---|---------|-------------|
| 10 | `lending_interest_nonneg` | P,r,t ≥ 0 → I ≥ 0 |
| 11 | `lending_interest_pos` | P,r,t > 0 → I > 0 — positive yield |
| 12 | `lending_amount_exceeds_principal` | P < P + I — lender is always repaid |
| 13 | `lending_rate_monotone` | r₁ < r₂ → I(r₁) < I(r₂) — rate ordering |

**§5 Cross-chain rate aggregation (best-rate selection)**

| # | Theorem | Description |
|---|---------|-------------|
| 14 | `best_rate_ge_left` | r₁ ≤ best_rate r₁ r₂ — dominates chain 1 |
| 15 | `best_rate_ge_right` | r₂ ≤ best_rate r₁ r₂ — dominates chain 2 |
| 16 | `best_rate_symm` | best_rate r₁ r₂ = best_rate r₂ r₁ — symmetric |
| 17 | `best_rate_optimal` | r₁,r₂ ≤ r → best_rate r₁ r₂ ≤ r — least upper bound |
| 18 | `best_rate_idempotent` | best_rate r r = r |

**§6 LP value and monotone-output properties**

| # | Theorem | Description |
|---|---------|-------------|
| 19 | `lp_value_pos` | x,y > 0 → √(x·y) > 0 — pool has depth |
| 20 | `lp_value_monotone` | x<x', y<y' → lp_value x y < lp_value x' y' |

---

### `PumpFunBot.lean`

Formalizes an automated trading strategy for the **pump.fun constant-product
bonding curve** on Solana, together with a machine-checked derivation of the
**Kelly criterion** for optimal position sizing.  The token-receive formula
`T·Δ/(S+Δ)` and Kelly fraction `f* = (b·p−(1−p))/b` are proved step-by-step
from first principles.

All 26 theorems in `PumpFunBot.lean` have complete machine-checked proofs (no `sorry`).

*Setup:* virtual reserves S₀ = 30 SOL, T₀ = 1 073 000 000 tokens, k = S₀·T₀;
graduation threshold G = 85 SOL triggers Raydium DEX migration.

**§1 Bonding curve fundamentals: k = S·T invariant**

| # | Theorem | Description |
|---|---------|-------------|
| 1 | `bc_k_pos` | S·T > 0 for S, T > 0 |
| 2 | `bc_price_pos` | S/T > 0 — price always positive |
| 3 | `bc_price_sq_formula` | S/T = S²/(S·T) — quadratic price form |
| 4 | `bc_invariant_preserved` | (S+Δ)·(S·T/(S+Δ)) = S·T — k preserved |
| 5 | `price_after_buy` | new price = (S+Δ)²/k for a buy of Δ |

**§2 Trade mechanics: tokens received = T·Δ/(S+Δ)**

| # | Theorem | Description |
|---|---------|-------------|
| 6 | `tokens_received_formula` | T − S·T/(S+Δ) = T·Δ/(S+Δ) |
| 7 | `tokens_received_pos` | T·Δ/(S+Δ) > 0 for Δ > 0 |
| 8 | `buy_increases_price` | (S+Δ)²/k > S/T — buys raise price |
| 9 | `effective_price_exceeds_spot` | Δ/(T·Δ/(S+Δ)) > S/T — entry price includes slippage |
| 10 | `tokens_per_sol_decreasing` | T/(S+Δ₁) > T/(S+Δ₂) for Δ₁ < Δ₂ — diminishing returns |

**§3 Graduation criterion: SOL raised ≥ G ≈ 85**

| # | Theorem | Description |
|---|---------|-------------|
| 11 | `graduation_threshold_pos` | G > 0 |
| 12 | `profitable_iff` | cost < tokens·P ↔ cost/tokens < P |
| 13 | `exit_value_pos` | tokens·P > 0 for tokens, P > 0 |
| 14 | `net_profit_positive` | tokens·(P_exit − P_entry) > 0 when P_entry < P_exit — **bot profit condition** |

**§4 Kelly criterion: f* = (b·p − (1−p)) / b**

| # | Theorem | Description |
|---|---------|-------------|
| 15 | `kelly_pos_iff` | f* > 0 ↔ p > 1/(1+b) — positive edge required |
| 16 | `kelly_le_one` | f* ≤ 1 — never risk full bankroll |
| 17 | `kelly_threshold_zero` | f*(1/(1+b), b) = 0 — no edge → no bet |
| 18 | `kelly_is_critical_point` | p·b·(1−f*) = (1−p)·(1+b·f*) — **first-order condition** |

**§5 Strategy soundness: Kelly fraction maximises log-wealth**

| # | Theorem | Description |
|---|---------|-------------|
| 19 | `log_growth_zero_bet` | G(0,p,b) = 0 — not betting preserves wealth |
| 20 | `kelly_fraction_unique` | f* is the **unique** solution to the FOC |

**§6 Step-by-step derivation: token formula**

| # | Theorem | Description |
|---|---------|-------------|
| 21 | `tokens_step_common_denominator` | T − S·T/(S+Δ) = [T·(S+Δ) − S·T]/(S+Δ) — common denominator |
| 22 | `tokens_numerator_cancellation` | T·(S+Δ) − S·T = T·Δ — S·T terms cancel |

**§7 Step-by-step derivation: Kelly fraction from ∂G/∂f = 0**

| # | Theorem | Description |
|---|---------|-------------|
| 23 | `kelly_ev_factor` | p·b·f − (1−p)·f = f·(b·p−(1−p)) — expected profit factored |
| 24 | `kelly_breakeven_condition` | b·p − (1−p) = 0 ↔ p = 1/(1+b) — break-even edge |
| 25 | `kelly_foc_cleared` | p·b·(1−f) = (1−p)·(1+b·f) — **denominators cleared** |
| 26 | `kelly_foc_linear` | cleared FOC → b·f = b·p − (1−p) — **nonlinear term cancels** |

---

### `CryptoBridge.lean`

Formalizes security properties of a cross-chain bridge protocol using a
lock-and-mint/burn-and-unlock mechanism secured by over-collateralised
relayers and Hash Time-Lock Contracts (HTLCs).  The conservation law
`locked = minted`, the HTLC identity `claim + fee = amount`, the solvency
invariant `collateral ≥ locked`, and Merkle tree size formulas are all
machine-checked.

All 20 theorems in `CryptoBridge.lean` have complete machine-checked proofs (no `sorry`).

**§1 Lock / mint conservation and fee deduction**

| # | Theorem | Description |
|---|---------|-------------|
| 1 | `bridge_conservation` | locked = minted — **no value created or destroyed** |
| 2 | `bridge_fee_reduces_output` | fee > 0 → minted < amount — fee deducted |
| 3 | `bridge_zero_fee` | fee = 0 → minted = amount — full pass-through |
| 4 | `bridge_locked_pos` | fee < amount → locked > 0 |
| 5 | `bridge_locked_lt_amount` | fee > 0 → locked < amount |

**§2 Fee-net positivity and monotonicity**

| # | Theorem | Description |
|---|---------|-------------|
| 6 | `bridge_locked_monotone` | a₁ < a₂ → locked(a₁) < locked(a₂) — monotone in amount |

**§3 Collateral solvency and over-collateralisation**

| # | Theorem | Description |
|---|---------|-------------|
| 7 | `collateral_ratio_pos` | c/l > 0 for c, l > 0 |
| 8 | `collateral_solvency` | l ≤ c → c/l ≥ 1 — **solvency invariant** |
| 9 | `collateral_overcollateralised` | l < c → c/l > 1 — safety cushion |
| 10 | `collateral_surplus_pos` | l < c → c − l > 0 — positive surplus |

**§4 HTLC atomic-swap mechanics**

| # | Theorem | Description |
|---|---------|-------------|
| 11 | `htlc_claim_pos` | fee < amount → claim > 0 |
| 12 | `htlc_refund_full` | refund = amount — full recovery on timeout |
| 13 | `htlc_refund_exceeds_claim` | fee > 0 → claim < refund |
| 14 | `htlc_value_conservation` | claim + fee = amount — **no leakage** |

**§5 Merkle tree structure and proof bounds**

| # | Theorem | Description |
|---|---------|-------------|
| 15 | `merkle_leaves_pos` | 0 < 2^d — tree always non-empty |
| 16 | `merkle_leaves_double` | leaves(d+1) = 2·leaves(d) |
| 17 | `merkle_leaves_monotone` | d₁ < d₂ → 2^d₁ < 2^d₂ |

**§6 Bridge liquidity and supply conservation**

| # | Theorem | Description |
|---|---------|-------------|
| 18 | `bridge_liquidity_nonneg` | w ≤ liq → liq − w ≥ 0 — no deficit |
| 19 | `bridge_supply_conservation` | minted = locked → minted ≤ locked |
| 20 | `bridge_liquidity_monotone` | l₁ < l₂ ∧ w ≤ l₁ → w ≤ l₂ |

---

### `Quantization.lean`

The **Lead Confirmed Quantization Theorem (Theorem Q)** asserts that when
`H · T = 5π/4`, all five quantization conditions Q1–Q5 hold simultaneously:
Floquet phase (`ε_F·T = π`), 8-cycle closure (`μ^8 = 1`), canonical balance
(`2η² = 1`), maximum coherence (`C(1) = 1`), and ground-state energy (`E₁ = −1`).

All 20 theorems in `Quantization.lean` have complete machine-checked proofs (no `sorry`).

**§1 Phase quantization**

| # | Theorem | Description |
|---|---------|-------------|
| 1 | `quantization_phase_unit` | \|μ\| = 1 — unit-circle eigenvalue |
| 2 | `quantization_eight_cycle` | μ^8 = 1 — 8-cycle closure |
| 3 | `quantization_pow_unit` | \|μ^n\| = 1 for all n |
| 4 | `quantization_eight_distinct` | {μ^0,…,μ^7} pairwise distinct |

**§2 Energy quantization (Bohr–Rydberg E_n = −1/n²)**

| # | Theorem | Description |
|---|---------|-------------|
| 5 | `quantization_ground_energy` | E₁ = −1 — ground state (Hartree units) |
| 6 | `quantization_energy_neg` | E_n < 0 for all n ≥ 1 — bound states |
| 7 | `quantization_ground_lowest` | E₁ ≤ E_n for all n ≥ 1 |
| 8 | `quantization_energy_strictMono` | E_n < E_{n+1} — levels ascend toward zero |

**§3 Floquet quantization**

| # | Theorem | Description |
|---|---------|-------------|
| 9 | `quantization_hamiltonian_recipe` | H·T = 5π/4 → U(H,T) = μ — drive prescription |
| 10 | `quantization_floquet_phase` | ε_F · T = π |
| 11 | `quantization_quasi_energy_pos` | ε_F > 0 for T > 0 |
| 12 | `quantization_period_doubling` | T < 2T for T > 0 — period doubling |

**§4 Canonical amplitude quantization**

| # | Theorem | Description |
|---|---------|-------------|
| 13 | `quantization_amplitude_balance` | 2η² = 1 — **balance equation** |
| 14 | `quantization_canonical_norm` | η² + \|μ·η\|² = 1 — unit sphere |
| 15 | `quantization_coherence_max` | C(1) = 1 — coherence maximum |
| 16 | `quantization_coherence_bound` | C(r) ≤ 1 for all r ≥ 0 |

**§5 Lead Confirmed Quantization Theorem (Theorem Q)**

| # | Theorem | Description |
|---|---------|-------------|
| 17 | `lead_quantization_floquet_arm` | U(H,T)=μ ∧ ε_F·T=π ∧ μ^8=1 — Floquet arm |
| 18 | `lead_quantization_energy_arm` | E₁=−1 ∧ E_n<0 for all n≥1 — energy arm |
| 19 | `lead_quantization_amplitude_arm` | 2η²=1 ∧ η²+\|μη\|²=1 ∧ C(1)=1 — amplitude arm |
| 20 | `lead_quantization_confirmed` | **Q1∧Q2∧Q3∧Q4∧Q5 simultaneously** — **LEAD CONFIRMED** |

---

## References

- [Lean 4 documentation](https://leanprover.github.io/lean4/doc/)
- [Mathlib4 documentation](https://leanprover-community.github.io/mathlib4_docs/)
- [Mathlib4 on GitHub](https://github.com/leanprover-community/mathlib4)
- [`../docs/master_derivations.pdf`](../docs/master_derivations.pdf) — mathematical background
- Reynolds, O. (1895). On the dynamical theory of incompressible viscous fluids. *Phil. Trans. R. Soc. A* 186, 123–164.
- Kolmogorov, A. N. (1941). The local structure of turbulence in incompressible viscous fluid. *Dokl. Akad. Nauk SSSR* 30, 301–305.
- Pope, S. B. (2000). *Turbulent Flows*. Cambridge University Press.
- Sommerfeld, A. (1916). Zur Quantentheorie der Spektrallinien. *Ann. Phys.* 51, 1–94.
- Bethe, H. A., & Salpeter, E. E. (1977). *Quantum Mechanics of One- and Two-Electron Atoms*. Springer.
- Davidson, P. A. (2001). *An Introduction to Magnetohydrodynamics*. Cambridge University Press.
- CODATA 2018. Fine structure constant α = 7.2973525693 × 10⁻³ (NIST).
- Koide, Y. (1982). A fermion-boson composite model of quarks and leptons. *Phys. Lett. B* 120, 161–165.
- Livio, M. (2002). *The Golden Ratio*. Broadway Books.
- Mohr, P. J. et al. (2016). CODATA recommended values. *Rev. Mod. Phys.* 88, 035009.
