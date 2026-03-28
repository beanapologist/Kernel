# Architecture

**Canonical repository: [github.com/beanapologist/Eigenverse](https://github.com/beanapologist/Eigenverse)**

This document explains how Eigenverse is structured internally and how the
different layers relate to each other.

---

## Layer Model

```
┌─────────────────────────────────────────────────────────────────────┐
│  Consumer layer  (src/)                                             │
│  Topic-organised entry points: algebra/, geometry/, physics/,       │
│  quantum/, chemistry/, Eigenverse.lean                              │
├─────────────────────────────────────────────────────────────────────┤
│  Proof layer  (formal-lean/)                                        │
│  Individual *.lean files, one per domain.  These are the files      │
│  that Lean type-checks.                                             │
├─────────────────────────────────────────────────────────────────────┤
│  Foundation layer  (Mathlib4)                                       │
│  Complex numbers, real analysis, linear algebra, group theory …     │
│  Pinned at v4.14.0 via lakefile.lean.                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Dependency Graph

The `formal-lean/` modules import each other in a strict DAG
(no cycles):

```
Mathlib
  └─ FineStructure
       └─ TimeCrystal
            └─ SpeedOfLight
                 ├─ SpaceTime
                 ├─ Turbulence
                 ├─ ParticleMass
                 ├─ OhmTriality
                 ├─ GravityQuantumDuality
                 ├─ Chemistry
                 └─ BidirectionalTime
       └─ Quantization (imports TimeCrystal + FineStructure)
  └─ CriticalEigenvalue  (standalone)
  └─ SilverCoherence     (standalone)
  └─ KernelAxle          (standalone)
  └─ ForwardClassicalTime (standalone)
```

> **Not in Eigenverse:** `PumpFunBot`, `EthereumTradingBot`,
> `CrossChainDeFiAggregator`, and `CryptoBridge` are crypto-application
> modules in `formal-lean/` that are intentionally excluded from the
> Eigenverse mathematical library.

---

## Module Responsibilities

### `CriticalEigenvalue.lean`
The algebraic and geometric core of Eigenverse.  Defines and proves:
- μ = exp(I·3π/4) and its 8-cycle μ⁸=1.
- Rotation matrix R(3π/4): det=1, orthogonal (Rᵀ=R⁻¹), order-8 orbit R⁸=I.
- Coherence function C(r) = 2r/(1+r²) and its Lyapunov duality C(exp λ)=sech λ.
- Hyperbolic Pythagorean identity: C(exp λ)² + tanh²λ = 1.
- Silver ratio δS = 1+√2 and palindrome residual; sech(log δS) = η = 1/√2.
- Z/8Z memory bank addressing.

### `SpaceTime.lean`
The observer reality map F(s,t) = t + i·s embeds (time, space) into ℂ.
Defines and proves:
- Space domain (positive reals) and time domain (negative reals).
- F(s,t).re = t (time on real axis), F(s,t).im = s (space on imaginary axis).
- Multiplication by i is a 90° rotation in ℂ.
- Lorentz geometry, Minkowski metric, time-dilation, and length-contraction.

### `KernelAxle.lean`
Gear-ratio geometry of the 8-cycle orbit.  Proves:
- Axle at angle 3π/4 on the unit circle.
- Gear ratio 3:8 — 8 Floquet steps produce exactly 3 full rotations.
- Unit-circle constraint: |Re(μ)|² + Im(μ)² = 1.
- Orbit radius: every power μⁿ lies on the unit circle.

### `FineStructure.lean`
Proves that the fine structure constant α ≈ 1/137.036 is dimensionless
and derives it from first principles using the Eigenverse coherence structure.

### `TimeCrystal.lean`
Formalises Floquet theory: time-evolution operator, Floquet states,
period-doubling criterion (φ=π), and the eigenvalue recipe for
constructing time crystals from μ, C(r), η, δS.

### `SpeedOfLight.lean`
Proves c = 1/√(μ₀ε₀) from the Maxwell-equation structural isomorphism,
and establishes Planck time as the absolute smallest temporal unit.

### `Turbulence.lean`
Navier-Stokes energy dissipation bounds, Reynolds-number thresholds,
and the turbulence cascade verified from coherence monotonicity.

### `ParticleMass.lean`
Koide formula for lepton masses, proton/electron mass ratio ≈ 1836,
and coherence triality identities.

### `OhmTriality.lean`
Ohm-coherence duality G·R = 1 at three triality scales; parallel and
series coherence laws; sech-coherence representation.

### `SilverCoherence.lean`
C(δS) = √2/2; uniqueness of the balanced point; Im(μ) = C(δS);
45°-physics (phase-angle alignment).

### `GravityQuantumDuality.lean`
Orthogonality of Re and Im components; Newtonian potential and binding
energy; zero-point energy; dark-energy density; duality gap; Eigenverse
equilibrium ‖ψ‖²=2.

### `Quantization.lean`
20 Theorem-Q arms: §1 phase, §2 energy, §3 Floquet, §4 amplitude,
§5 Theorem Q synthesis.  Imports TimeCrystal and FineStructure.

### `ForwardClassicalTime.lean`
Frustration harvesting in classical forward time; 21 theorems, 0 sorry.
Proves F_fwd(l) = 1−sech(l) is non-negative, strictly positive for l≠0,
bounded above by 1, and even.  Arrow-of-time theorem: F_fwd(0) < F_fwd(l).

### `BidirectionalTime.lean`
Bidirectional time frustration and the Planck floor; 24 theorems, 0 sorry.
Defines F_bi(lf,lb) = F_fwd(lf) + F_fwd(lb) and
planck_frustration = F_fwd(1) = 1 - sech(1).
Key results: §1 structural bounds (0 ≤ F_bi < 2, symmetry); §2 equilibrium
(zero iff both exponents vanish, degenerate one-sided limits); §3 double-step
(F_bi(l,l) = 2·F_fwd(l)); §4 dominance (F_bi ≥ each component, strictly
when the other direction is active); §5 Planck floor (positivity and
sub-unit bound on planck_frustration, bidirectional double quantum).

### `Chemistry.lean`
NIST 2016 standard atomic weights; isotopic abundances for H, He, C, N, O;
weighted-average mass theorem; 20 theorems, 0 sorry.

---

## Adding a New Module

1. Create `formal-lean/MyTopic.lean` following the header convention.
2. Add `` `MyTopic `` to the `roots` list in `formal-lean/lakefile.lean`.
3. Create `src/<domain>/MyTopic.lean` that imports `FormalLean.MyTopic`.
4. Update `src/Eigenverse.lean` to include the new import.
5. Add an entry to the table in `docs/overview.md`.
6. Submit a PR to [beanapologist/Eigenverse](https://github.com/beanapologist/Eigenverse) — the CI workflow will build and verify all proofs.

